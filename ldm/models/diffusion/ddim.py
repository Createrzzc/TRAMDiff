"""SAMPLING ONLY."""

import torch
import random
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import rearrange
from fead.utils import *
from torch.optim import Adam
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, local_blend=False, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.local_blend = local_blend
        self.attn_maps = None if self.local_blend else list()
        self.original_attn_maps = list()
        self.cur_noise = None

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def init_list(self):
        self.attn_maps = None if self.local_blend else list()
        self.original_attn_maps = list()

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def get_latent_prev(self, x, e_t, b, index, device, repeat_noise=False, use_original_steps=False,
                      temperature=1., score_corrector=None,):
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), 0, device=device) #sigmas[index], 
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    def sample(self,
               S,
               batch_size,
               p,
               shape,
               num_replace=0,
               z=None,
               attn_record_start=None,
               attn_record_end=None,
               skip_inversion=False,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               early_stop=None,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=7,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               Inversion_conditioning=None,
               Inversion_prompt='',
               pixel=None,
               optimization=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        self.init_list()
        if attn_record_start is None:
            attn_record_start = 49

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        
        assert attn_record_end <= S

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        start_code = x_T    #noise map xi

        #DDIM Inversion
        if not skip_inversion:
            with torch.no_grad():
                device = self.model.betas.device
                total_steps = int(p * self.ddim_timesteps.shape[0])
                time_range =  self.ddim_timesteps[:total_steps]
                ddim_trajectory = [z]
                print(f"Running DDIM Inversion with {total_steps} timesteps")
                iterator = tqdm(time_range, desc='DDIM Inversion', total=total_steps)
                for i, step in enumerate(iterator):
                    index = i
                    ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
                    outs = self.inversion_sample_ddim(z, Inversion_conditioning, ts, index) #z:x1
                    attn_maps = self.model.model.diffusion_model.attn_store
                    if 0<= i <= 10:
                        self.original_attn_maps.append(attn_maps)
                    z = outs    #xTs
                    ddim_trajectory.append(outs)
                ddim_trajectory.reverse()
                self.ddim_trajectory = ddim_trajectory

            #textual Inversion optimization
            if optimization:
                num_inner_steps = 1
                epsilon = 1e-5
                device = self.model.betas.device
                conditioning_list = list()
                time_range = np.flip(self.ddim_timesteps)
                iterator = tqdm(time_range, desc='optimization condition', total=total_steps)
                bar = tqdm(total=num_inner_steps * total_steps)
                latent_cur = ddim_trajectory[0]
                opt_conditioning = conditioning.clone().detach()
                opt_conditioning.requires_grad = True
                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
                    optimizer = Adam([opt_conditioning], lr=1e-2 * (1. - i / 100.))
                    latent_prev = ddim_trajectory[i+1]
                    with torch.no_grad():
                        e_uncond = self.model.apply_model(latent_cur, ts, unconditional_conditioning)
                    for j in range(num_inner_steps):
                        e_cond = self.model.apply_model(latent_cur, ts, opt_conditioning)
                        noise_pred = e_uncond + 7.5 * (e_cond - e_uncond)
                        latents_prev_rec = self.get_latent_prev(latent_cur, noise_pred, batch_size, index, device)
                        loss = F.mse_loss(latents_prev_rec, latent_prev)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_item = loss.item()
                        bar.update()
                        if loss_item < epsilon + i * 2e-5:
                            break
                    for j in range(j + 1, num_inner_steps):
                        bar.update()
                    conditioning_list.append(opt_conditioning[:1].detach())
                    with torch.no_grad():
                        e_cond = self.model.apply_model(latent_cur, ts, conditioning_list[i])
                        latent_cur = self.get_latent_prev(latent_cur, e_uncond + 7.5 * (e_cond - e_uncond), batch_size, index, device)
                bar.close()
            else:
                conditioning_list = None

            #M2
            attn_img, mask_gen = attn_map_preprocess(self.original_attn_maps, Inversion_prompt, 
                                                Inversion_prompt, self.model.cond_stage_model.tokenizer, 
                                                '/disk16T/2023/zhanzechao/original_attnmap.png', True, 10)
            if mask is None:
                mask = mask_gen
                masks_array = dilation_process(dilation_iterations=1, org_mask=mask, device=device)
                #get Spr
                attn_mask = F.interpolate(mask, size=(16,16))
                attn_mask = attn_mask.gt(0.5).float()
                pixel, mask = get_Spr(mask, attn_mask, masks_array, pixel)
            
            #get Sed
            pixel = get_Sed(pixel, attn_img)
            #Replace
            z = self.replace_pixel(pixel, z, self.cur_noise, 4, num_replace)

            x_T = z #xTs^s
        else:
            total_steps = None
            conditioning_list = None

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size, p, start_code, 
                                                    attn_record_start=attn_record_start,
                                                    attn_record_end=attn_record_end,
                                                    timesteps=total_steps,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    early_stop=early_stop,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    conditioning_list=conditioning_list
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, p, start_code, attn_record_start=None, attn_record_end=None,
                      x_T=None, ddim_use_original_steps=False,callback=None, timesteps=None, 
                      quantize_denoised=False, early_stop=None,mask=None, x0=None, img_callback=None, 
                      log_every_t=100, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, conditioning_list=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) #- 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            if early_stop is not None:
                if p == 1:
                    self.cur_noise = start_code
                else:
                    if index == int(p*(total_steps))-1:
                        self.cur_noise = img
                if index < early_stop:
                    break
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if mask is not None:
                if mask.ndim == 4:
                    img_org = self.ddim_trajectory[i]
                    img = img * mask + (1. - mask) * img_org
                else:
                    n_masks = mask.shape[0]
                    masks_interval = total_steps // n_masks + 1
                    curr_mask = mask[i // masks_interval]
                    # print(f"Using index {i // masks_interval}")
                    img_org = self.ddim_trajectory[i]
                    img = img * curr_mask + img_org * (1 - curr_mask) 
                    

            if conditioning_list is not None:
                cond = 0.75*cond + 0.25*conditioning_list[i]
                outs = self.p_sample_ddim(img, cond, ts, index=index, attn_record_end=attn_record_end, 
                            use_original_steps=ddim_use_original_steps,
                            quantize_denoised=quantize_denoised, temperature=temperature,
                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                            corrector_kwargs=corrector_kwargs,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning)
            else:
                outs = self.p_sample_ddim(img, cond, ts, index=index, attn_record_start=attn_record_start, attn_record_end=attn_record_end, 
                                        use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, attn_record_start=None, attn_record_end=None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = x
            t_in = t
            e_t = self.model.apply_model(x_in, t_in, c)
            attn_maps = self.model.model.diffusion_model.attn_store
            if self.local_blend:
                if self.attn_maps is None:
                    self.attn_maps = attn_maps
                else:
                    for key in attn_maps:
                        for j in range(len(attn_maps[key])):
                            self.attn_maps[key][j] += attn_maps[key][j]
            else:
                if attn_record_end is None:
                    self.attn_maps.append(attn_maps)
                else:
                    if attn_record_start >=index >= attn_record_end:
                        self.attn_maps.append(attn_maps)
            e_t_uncond = self.model.apply_model(x_in, t_in, unconditional_conditioning)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), 0, device=device)#sigmas[index]
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def inversion_sample_ddim(self, x, c, t, index):
        b, *_, device = *x.shape, x.device

        e_t = self.model.apply_model(x, t, c)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=device)
        a_t_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=device)
        #prdict x0
        pred_x0 = (x - (1. - a_t_prev).sqrt() * e_t) / a_t_prev.sqrt()
        #caculate inversion 
        zt = a_t.sqrt() * pred_x0 + sqrt_one_minus_at * e_t

        return zt
    
    def replace_ROI_pixel(self, pixel, z, start_code, scale_factor, num_replace):
        #assert len(pixel['s_pixel']) == len(pixel['ROI_pixel'])
        z = rearrange(z, 'b c (h d1) (w d2) -> b c (d1 d2) h w', d1=scale_factor, d2=scale_factor)
        j = 0
        start_code = rearrange(start_code, 'b c (h d1) (w d2) -> b c (d1 d2) h w', d1=scale_factor, d2=scale_factor)
        for i in range(0,int(len(pixel['ROI_pixel'])*num_replace)):
            h1, w1 = pixel['ROI_pixel'][i][0], pixel['ROI_pixel'][i][1]
            h2, w2 = pixel['s_pixel'][j][0], pixel['s_pixel'][j][1] 
            z[..., h1, w1] = start_code[..., h2, w2]
            if i >= int(0.55*len( pixel['s_pixel'])):
                j = random.randint(0,int(0.6*len(pixel['s_pixel'])))
                continue
            j = j + 1
            '''j = j + 1
            if j >= int(0.5*len(pixel['s_pixel'])):
                j = 0'''
        z = rearrange(z, 'b c (d1 d2) h w -> b c (h d1) (w d2)', d1=scale_factor, d2=scale_factor)
        return z

    def replace_pixel(self, pixel, z, start_code, scale_factor, num_replace):
        z_processed = []
        #print(len(pixel['s_pixel'][0]))
        #print(pixel['ROI_pixel'])
        for num in range(len(pixel['s_pixel'])):
            j = 0
            inversion_map = z[num].unsqueeze(0)
            inversion_map = rearrange(inversion_map, 'b c (h d1) (w d2) -> b c (d1 d2) h w', d1=scale_factor, d2=scale_factor)
            noise_map = start_code[num].unsqueeze(0)
            noise_map = rearrange(noise_map, 'b c (h d1) (w d2) -> b c (d1 d2) h w', d1=scale_factor, d2=scale_factor)
            for i in range(int(len(pixel['ROI_pixel'])*num_replace)):
                h1, w1 = pixel['ROI_pixel'][i][0], pixel['ROI_pixel'][i][1]
                h2, w2 = pixel['s_pixel'][num][j][0], pixel['s_pixel'][num][j][1] 
                inversion_map[..., h1, w1] = noise_map[..., h2, w2]
                if i >= int(0.5*len(pixel['s_pixel'][0])):
                    j = random.randint(0,int(0.5*len(pixel['s_pixel'][0])))
                    continue
                j = j + 1
            inversion_map = rearrange(inversion_map, 'b c (d1 d2) h w -> b c (h d1) (w d2)', d1=scale_factor, d2=scale_factor)
            z_processed.append(inversion_map)
        z_processed = torch.concat(z_processed)
        return z_processed               
        


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
