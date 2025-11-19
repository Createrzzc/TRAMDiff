"""SAMPLING ONLY."""

import torch
import random
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import rearrange
from fead.mask.utils import *
from torch.optim import Adam
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        #
        self.attn_maps = list()
        self.qkv_timesteps = list()
        self.cur_noise = None

    def init_list(self):
        self.attn_maps = list()
        self.skin_pixel = list()
        self.qkv_timesteps = list()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

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
               z=None,
               z_hf=None,
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
               mode=None,
               is_qkv_injected=False,
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

        #init noise map
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        start_code = x_T    
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
                    if is_qkv_injected:
                        self.qkv_timesteps.append(self.model.model.diffusion_model.qkv_store) # [{'q':[], 'k':[], 'v':[]}, {'q':[], 'k':[], 'v':[]}, ...]
                    z = outs    #xTs
                    ddim_trajectory.append(outs)
                ddim_trajectory.reverse()
                self.qkv_timesteps.reverse()
                self.ddim_trajectory = ddim_trajectory

                if z_hf is not None:
                    hf_trajectory = [z_hf]
                    print(f"Running HF Inversion with {total_steps} timesteps")
                    iterator = tqdm(time_range, desc='HF Inversion', total=total_steps)
                    for i, step in enumerate(iterator):
                        index = i
                        ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
                        outs = self.inversion_sample_ddim(z_hf, Inversion_conditioning, ts, index) #z:x1
                        hf_trajectory.append(outs)
                    hf_trajectory.reverse()
                    self.hf_trajectory = hf_trajectory
            
            #allocate pixel in different place
            z = self.allocate_pixel(pixel, z, self.cur_noise, mode)

            x_T = z #xTs^s
        else:
            total_steps = None

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
                                                    unconditional_conditioning=unconditional_conditioning
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, p, start_code, attn_record_start=None, attn_record_end=None,
                      x_T=None, ddim_use_original_steps=False,callback=None, timesteps=None, 
                      quantize_denoised=False, early_stop=None,mask=None, x0=None, img_callback=None, 
                      log_every_t=100, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
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

            #get x_S^i, only in stage1
            if early_stop is not None:
                if p == 1:
                    self.cur_noise = start_code
                elif index == int(p*(total_steps))-1:
                    self.cur_noise = img
                if index < early_stop:
                    break
    
            #mask blended
            if mask is not None:
                img_org = self.ddim_trajectory[i]
                hf_org = self.hf_trajectory[i] if 10 < i < 30 else 0
                img = img * mask + (1. - mask) * img_org + 0.5*hf_org * mask

            #run ddim sampling        
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            injected_period = 0.3
            if index <= total_steps * injected_period: 
                qkv_injected = self.qkv_timesteps[i] if self.qkv_timesteps != [] else None
            else:
                qkv_injected = None
            outs = self.p_sample_ddim(img, cond, ts, index=index, attn_record_start=attn_record_start, attn_record_end=attn_record_end, 
                                    use_original_steps=ddim_use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    qkv_injected=qkv_injected)
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
                      unconditional_guidance_scale=1., unconditional_conditioning=None, qkv_injected=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, 
                                                     qkv_injected=qkv_injected).chunk(2)
            attn_maps = self.model.model.diffusion_model.attn_store
            if attn_record_end is None:
                self.attn_maps.append(attn_maps)    #[{'down_attn':[], 'mid_attn':[], 'up_attn':[]}, {'down_attn':[], 'mid_attn':[], 'up_attn':[]}, ....]
            else:
                if attn_record_start >=index >= attn_record_end:
                    self.attn_maps.append(attn_maps)
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
            if i >= int(0.5*len( pixel['s_pixel'])):
                j = random.randint(0,int(0.5*len(pixel['s_pixel'])))
                continue
            j = j + 1
            '''j = j + 1
            if j >= int(0.5*len(pixel['s_pixel'])):
                j = 0'''
        z = rearrange(z, 'b c (d1 d2) h w -> b c (h d1) (w d2)', d1=scale_factor, d2=scale_factor)
        return z

    def allocate_pixel(self, pixel, z, cur_noise, mode):
        #pixel:{"s_pixel": [...], "ROI_pixel": [...], "RM_pixel": [...]}
        z_processed = []
        noise = torch.randn_like(z)
        for num in range(len(pixel['s_pixel'])):
            #init map and pixel
            inversion_map = z[num].unsqueeze(0)
            noise_map = cur_noise[num].unsqueeze(0)  
            #only color mode          
            if mode == "color":
                print("color")
                #ROI_pixel replaced by s_pixel
                j = 0
                for i in range(len(pixel['ROI_pixel'])):
                    h1, w1 = pixel['ROI_pixel'][i][0], pixel['ROI_pixel'][i][1]
                    h2, w2 = pixel['s_pixel'][num][j][0], pixel['s_pixel'][num][j][1]
                    inversion_map[..., h1, w1] = noise_map[..., h2, w2]
                    if i >= int(0.5*len(pixel['s_pixel'][0])):
                        j = random.randint(0,int(0.5*len(pixel['s_pixel'][0])))
                        continue
                    j = j + 1
            #only shape mode
            elif mode == "shape":
                print("shape")
                #AD_pixel replaced by comb_pixel
                j = 0
                noise = torch.randn_like(inversion_map)
                for i in range(len(pixel['AD_pixel'])):
                    h1, w1 = pixel['AD_pixel'][i][0], pixel['AD_pixel'][i][1]
                    h2, w2 = pixel['comb_pixel'][j][0], pixel['comb_pixel'][j][1]
                    inversion_map[..., h1, w1] = noise[..., h1, w1]#inversion_map[..., h2, w2]
                    '''if i >= int(1*len(pixel['comb_pixel'])):
                        j = random.randint(0,int(1*len(pixel['comb_pixel'])))
                        continue
                    j = j + 1'''
                #RM_pixel replaced by skin_pixel
                l = 0
                for k in range(len(pixel['RM_pixel'])):
                    h3, w3 = pixel['RM_pixel'][k][0], pixel['RM_pixel'][k][1]
                    if pixel['skin_pixel'] is not None:
                        h4, w4 = pixel['skin_pixel'][l][0], pixel['skin_pixel'][l][1]
                        inversion_map[..., h3, w3] = inversion_map[..., h4, w4]
                        if k >= int(0.8*len(pixel['skin_pixel'])):
                            l = random.randint(0,int(0.8*len(pixel['skin_pixel'])))
                            continue
                        l = l + 1
                    else:
                        inversion_map[..., h3, w3] = noise[..., h3, w3]
            #shape&color mode
            else:
                print("comb")
                #ROI_pixel replaced by s_pixel
                j = 0
                for i in range(len(pixel['ROI_pixel'])):
                    h1, w1 = pixel['ROI_pixel'][i][0], pixel['ROI_pixel'][i][1]
                    h2, w2 = pixel['s_pixel'][num][j][0], pixel['s_pixel'][num][j][1]
                    inversion_map[..., h1, w1] = noise_map[..., h2, w2]
                    if i >= int(0.5*len(pixel['s_pixel'][0])):
                        j = random.randint(0,int(0.5*len(pixel['s_pixel'][0])))
                        continue
                    j = j + 1
                #RM_pixel replaced by skin_pixel
                l = 0
                for k in range(len(pixel['RM_pixel'])):
                    h3, w3 = pixel['RM_pixel'][k][0], pixel['RM_pixel'][k][1]
                    h4, w4 = pixel['skin_pixel'][l][0], pixel['skin_pixel'][l][1]
                    inversion_map[..., h3, w3] = inversion_map[..., h4, w4]
                    if k > int(0.8*len(pixel['skin_pixel'])):
                        l = random.randint(0, int(0.8*len(pixel['skin_pixel'])))
                        continue
                    l = l + 1
            
            z_processed.append(inversion_map)
        z_processed = torch.concat(z_processed)
        return z_processed 


    def AD_process(self, AD, Source, inversion_map):

        def pixel_replace(AD_item, Source_item, inversion_map):
            j = 0
            for i in range(len(AD_item)):
                h1, w1 = AD_item[i][0], AD_item[i][1]
                h2, w2 = Source_item[j][0], Source_item[j][1]
                inversion_map[..., h1, w1] = inversion_map[..., h2, w2]
                if i >= len(Source_item):
                    j = random.randint(0,len(Source_item))
                    continue
                j = j + 1
            return inversion_map

        AD_sorted = sorted(AD, key=lambda x: (x[0], x[1]))
        AD_list = []
        current_value = AD_sorted[0][0]
        current_group = []

        for sublist in AD_sorted:
            if sublist[0] == current_value:
                current_group.append(sublist)
            else:
                AD_list.append(current_group)
                current_group = [sublist]
                current_value = sublist[1]
        AD_list.append(current_group)

        for AD_item in AD_list:

            if len(AD_item) <= 6:
                continue

            Source_item = [sublist for sublist in Source if sublist[0] == AD_item[0][0]]

            horizon_exist = True if len(Source_item) >= 8 else False
            if horizon_exist == True:
                edge = None
                for i in range(len(AD_item)):
                    if AD_item[i+1][1] - AD_item[i][1] >= 5:
                        edge = i
                        break
                if edge is not None:
                    inversion_map = pixel_replace(AD_item[:edge+1], Source_item[:4], inversion_map)
                    inversion_map = pixel_replace(AD_item[edge+1:], Source_item[-4:], inversion_map)
                else:
                    inversion_map = pixel_replace(AD_item, Source_item[:6], inversion_map)
            else:
                for i in range(len(AD_item)):
                    j = random.randint(0,len(Source))
                    h1, w1 = AD_item[i][0], AD_item[i][1]
                    h2, w2 = Source[j][0], Source[j][1]
                    inversion_map[..., h1, w1] = inversion_map[..., h2, w2]
        
        return inversion_map              
        


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

    @torch.no_grad()
    def skin_inversion(self, p, z, batch_size, inversion_skin):
        device = self.model.betas.device
        total_steps = int(p * self.ddim_timesteps.shape[0])
        time_range =  self.ddim_timesteps[:total_steps]
        print(f"Running DDIM Inversion with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='DDIM Inversion', total=total_steps)
        for i, step in enumerate(iterator):
            index = i
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            outs = self.inversion_sample_ddim(z, inversion_skin, ts, index) #z:x1
            attn_maps = self.model.model.diffusion_model.attn_store
            if 10<= i <= 20:
                self.original_attn_maps.append(attn_maps)
            z = outs    #xTs

        attn_img = attn_map_preprocess(self.original_attn_maps, 'skin', 'skin', 
                                       self.model.cond_stage_model.tokenizer, 
                                       True, "attn_img/M2.png", 10)

        self.skin_pixel = get_skin_pixel(attn_img)
        print('Already get skin_pixel')

        