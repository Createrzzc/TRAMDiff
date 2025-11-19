import argparse, os, sys, glob
sys.path.append(".")
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from scipy.ndimage import binary_dilation
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config, get_word_inds
from fead.mask.ddim import DDIMSampler

from transformers import CLIPTokenizer
from fead.mask.utils import *


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def image_process(image_path, batch_size, device):
    image = Image.open(image_path)

    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]
    image = Image.fromarray(img)
    image = np.array(image).astype(np.uint8)
    image = torch.tensor(image, dtype=torch.float32).to(device)
    image = torch.unsqueeze(image,0)
    image = rearrange(image, 'b h w c -> b c h w')

    if image.shape[2] != 512:
        image = F.interpolate(image, size=(512,512), mode='bicubic')  
    image = (image / 127.5 - 1.0)
    image_in = torch.cat(batch_size*[image])
    return image_in

'''def local_blend(words, threshold, sampler, text, batch_size=1, k=1, device='cuda'):
    alpha_layers = torch.zeros(batch_size,  1, 1, 1, 1, 77)
    tokenizer = CLIPTokenizer.from_pretrained("/root/.cache/huggingface/clip-vit-large-patch14")
    ind = get_word_inds(text, words, tokenizer)
    alpha_layers[:, :, :, :, :, ind] = 1
    alpha_layers = alpha_layers.to(device)
    maps = sampler.attn_maps["down_attn"][4:6] + sampler.attn_maps["up_attn"][0:3]
    maps = [item.reshape(alpha_layers.shape[0], -1, 1, 16, 16, 77) for item in maps]
    maps = torch.cat(maps, dim=1)
    maps = (maps * alpha_layers).sum(-1).mean(1)
    mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
    mask = F.interpolate(mask, size=([64, 64]))
    mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
    mask = mask.gt(threshold).float()
    return mask'''

def read_mask(mask_path, batch_size, device, threshold, img_size=(64,64)):
    org_mask = Image.open(mask_path).convert("L")
    org_mask = org_mask.resize(img_size, Image.LANCZOS)
    org_mask = np.array(org_mask).astype(np.float32) / 255.0
    org_mask = torch.from_numpy(org_mask).unsqueeze(0).to(device)
    mask = torch.stack(batch_size*[org_mask])
    mask = mask.gt(threshold).float()
    mask = mask.to(device)
    return mask

def dilation_process(dilation_iterations, org_mask, device):
    #print(org_mask.shape)
    assert org_mask.ndim == 4
    batch_size = org_mask.shape[0]
    mask = np.array(org_mask[0].squeeze(0).cpu())
    masks_array = []
    for i in reversed(range(dilation_iterations)):
        k_size = 10 + 2 * i
        dilation_mask = binary_dilation(mask, structure=np.ones((k_size, k_size)))
        dilation_mask = torch.stack(batch_size*[torch.from_numpy(dilation_mask).unsqueeze(0)])
        masks_array.append(dilation_mask.to(device))
    masks_array.append(org_mask)

    masks_array = torch.stack(masks_array)
    
    return masks_array

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="dress.",
        help="the prompt to render"
    )
    parser.add_argument(
        "--prior_prompt",
        type=str,
        nargs="?",
        help="something you want to change in the image",
        default="",
    )    
    parser.add_argument(
        "--Inversion_prompt",
        type=str,
        nargs="?",
        help="some details you want to delete",
        default=""
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="output"
    ) 
    parser.add_argument(
        "--image_path",
        type=str,
        nargs="?",
        help="path to input image",
        default="output/source_image.png"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        nargs="?",
        help="dir to mask",
        default="output/mask.png"
    )
    parser.add_argument(
        "--seg_path",
        type=str,
        nargs="?",
        help="dir to seg",
        default="output/seg.png"
    )
    parser.add_argument(
        "--skin_path",
        type=str,
        nargs="?",
        help="dir to seg",
        default="output/skin.png"
    )  
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--skip_inversion",
        action='store_true',
        help="use ddim inversion",
    )
    parser.add_argument(
        "--inversion_steps",
        type=float,
        help="p",
        default=0.8
    )
    parser.add_argument(
        "--attn_record_end",
        type=int,
        help="record attention maps until this iter",
        default=35
    )
    parser.add_argument(
        "--attn_record_start",
        type=int,
        help="record attention maps from this iter",
        default=45
    )  
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/sdv1_5.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    #init parser, model, sampler
    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    #Data Preparation
    prompt = opt.prompt
    prior_prompt = opt.prior_prompt if opt.prior_prompt is not None else opt.prompt
    img = image_process(opt.image_path, batch_size, device)
    assert prompt is not None
    data = [[batch_size * [prompt], img]]   #for pipeline2
    prior_data = [batch_size*[prior_prompt]]  #for pipeline1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    #pipeline1
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(prior_data, desc="prior_data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [''])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         p=opt.inversion_steps,
                                                         early_stop=opt.attn_record_end,
                                                         attn_record_start=opt.attn_record_start,
                                                         attn_record_end=opt.attn_record_end,
                                                         skip_inversion=True,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)


    #M1                                                   
    attn_img = attn_map_preprocess(sampler.attn_maps, opt.prior_prompt, 
                    opt.prior_prompt, model.cond_stage_model.tokenizer, True,
                    'attn_img/M1.png', 10) #b 1 16 16


    #read mask and seg
    mask = read_mask(opt.mask_path, batch_size, device, 0.007)
    seg = read_mask(opt.seg_path, batch_size, device, 0.005)
    skin = read_mask(opt.skin_path, batch_size, device, 0.4)
    mask_in = (mask + seg).gt(0).float() 
    # get Spr and mask_in
    pixel = get_Spr(mask, seg, skin, attn_img)
    
    #pipeline2
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts, img in tqdm(data, desc="data"):
                        uc = None
                        encoder_posterior = model.encode_first_stage(img)
                        z = model.get_first_stage_encoding(encoder_posterior).detach()
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [''])
                            Inversion_c = model.get_learned_conditioning(batch_size * [opt.Inversion_prompt])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                         p=opt.inversion_steps,
                                                         z=z,
                                                         attn_record_start=opt.attn_record_start,
                                                         attn_record_end=opt.attn_record_end,
                                                         skip_inversion=opt.skip_inversion,
                                                         mask=mask_in,
                                                         conditioning=c,
                                                         unconditional_conditioning=uc,
                                                         Inversion_conditioning=Inversion_c,                                                         
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         Inversion_prompt=opt.Inversion_prompt,
                                                         eta=opt.ddim_eta,
                                                         pixel=pixel,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_samples_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_samples_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(outpath, "result.png"))

                        '''if not opt.skip_ddim_inter:
                            ddim_inter = torch.concat(intermediates['pred_x0'], 0)
                            ddim_inter = model.decode_first_stage(ddim_inter)
                            ddim_inter = torch.clamp((ddim_inter + 1.0) / 2.0, min=0.0, max=1.0)
                            ddim_inter = ddim_inter.cpu().permute(0, 2, 3, 1).numpy()
                            ddim_inter = torch.from_numpy(ddim_inter).permute(0, 3, 1, 2)
                            grid = make_grid(ddim_inter, nrow=n_rows)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                            grid_count += 1
                        
                        if not opt.skip_grid:
                            all_samples.append(x_samples_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1'''

                #select_attn_map = attn_map_preprocess(sampler.attn_maps, opt.prompt, "red", model.cond_stage_model.tokenizer, '/disk16T/2023/zhanzechao/attnmap2.png', 4)
                toc = time.time()
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
