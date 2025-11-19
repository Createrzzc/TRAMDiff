import sys
sys.path.append("./")
sys.path.append(".")
import torch
import argparse
import os
import PIL
import cv2
import glob
import time
import yaml
import ollama
import json
#import jsonlines
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange

from tqdm import tqdm, trange
from itertools import islice
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from transformers import CLIPTokenizer
from scipy.ndimage import binary_dilation, binary_erosion

from accelerate import Accelerator

from madiff.mask.utils import *
from madiff.sar.ddim import DDIMSampler
from madiff.sar.utils import *
from ldm.util import instantiate_from_config, get_word_inds
from mask_est.model import Mask_Est_Model
from mask_est.prior_model.graphonomy.run_mask import run_graphonomy
from mask_est.prior_model.openpose.run_pose import run_openpose
from mask_est.prior_model.densepose.run_dp import run_dp
from decision import run_llama, get_mode, get_seg_prompt

def run_mask_unet(opt, mask_model, device, hp, outdir, input_image=None):

    def pre_process(image_path, output_path, output_cloth, output_body, output_pose, cloth_type='', dense=True):    
        os.makedirs(output_path, exist_ok=True)
        run_graphonomy(image_path, output_path, output_cloth, output_body, cloth_type)
        if dense:
            run_dp(image_path, output_path, output_pose)
        else:
            run_openpose(image_path, output_path, output_pose)

    def load_input_image(input_path, output_body, output_pose, device, size=64, interpolation="bicubic"):

        interpolation = {
                        "bilinear": PIL.Image.BILINEAR,
                        "bicubic": PIL.Image.BICUBIC,
                        "lanczos": PIL.Image.LANCZOS,
                        }[interpolation]

        def read_process_image(image_path, interpolation, device, size=64):
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            img = np.array(image).astype(np.uint8)
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]
            image = Image.fromarray(img)
            if size is not None:
                image = image.resize((size, size), resample=interpolation)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            image = torch.from_numpy(image).to(device)
            image = rearrange(image, 'h w c -> c h w')
            return image

        body_path = os.path.join(input_path, output_body)
        pose_path = os.path.join(input_path, output_pose)
        image_path = os.path.join(input_path, 'source_image.png')

        body_mask = read_process_image(body_path, interpolation, device, size)[0].unsqueeze(0)
        pose_map = read_process_image(pose_path, interpolation, device, size)
        image = read_process_image(image_path, interpolation, device, size)

        seg_input_map = image
        mask_input_map = torch.concat([body_mask, pose_map], dim=0)
        
        return mask_input_map.unsqueeze(0), seg_input_map.unsqueeze(0)

    def postprocess(mask_path, image_path, save_path):
        mask_image = Image.open(mask_path).convert('L')
        mask_tensor = TF.to_tensor(mask_image)
        mask_tensor = mask_tensor.expand(3, -1, -1)
        image = Image.open(image_path)
        image_tensor = TF.to_tensor(image)

        result_tensor = mask_tensor * image_tensor

        result_numpy = TF.to_pil_image(result_tensor)
        result_numpy.save(save_path)
        print('Image saved successfully.')

    def get_mask(model, input_map, input_text, outdir, name, threshold=0.25):
        outputs = model(input_map, input_text)
        outputs = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)
        outputs = F.interpolate(outputs, size=(256,256), mode='bilinear', align_corners=False)
        outputs = torch.gt(outputs, threshold).float()
        for x_sample in outputs:
            #x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            x_sample = 255. * x_sample[0].cpu().numpy()
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(outdir, name))

    mask_text = [opt.mask_prompt]
    seg_text = [opt.seg_prompt]
    image_path = opt.image_path if input_image is None else input_image
    pre_process(image_path, outdir, '', 'body.png', 'pose.png')
    mask_input_map, seg_input_map = load_input_image(outdir, 'body.png', 'pose.png', device, size=128)

    get_mask(mask_model, mask_input_map, mask_text, outdir, 'mask.png', hp[0])
    run_graphonomy(image_path, outdir, 'seg.png', None, opt.seg_prompt)

def run_editing(opt, model, sampler, device, hp, outdir=None, save_name='result.png'):

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

    def read_mask(mask_path, batch_size, device, threshold, img_size=(64,64)):
        org_mask = Image.open(mask_path).convert("L")
        org_mask = org_mask.resize(img_size, Image.LANCZOS)
        org_mask = np.array(org_mask).astype(np.float32) / 255.0
        org_mask = torch.from_numpy(org_mask).unsqueeze(0).to(device)
        mask = torch.stack(batch_size*[org_mask])
        mask = mask.gt(threshold).float()
        mask = mask.to(device)
        return mask

    def dilation_process(org_mask, device):
        k_size = 6
        assert org_mask.ndim == 4
        batch_size = org_mask.shape[0]
        mask = np.array(org_mask[0].squeeze(0).cpu())
        dilation_mask = binary_dilation(mask, structure=np.ones((k_size, k_size)))
        dilation_mask = torch.stack(batch_size*[torch.from_numpy(dilation_mask).unsqueeze(0)])
        dilation_mask = dilation_mask.to(device)
        
        return dilation_mask.float()
    
    seed_everything(opt.seed)
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)

    batch_size = opt.n_samples

    #Data Preparation
    prompt = opt.prompt
    prior_prompt = opt.prompt  
    img = image_process(os.path.join(outdir, "source_image.png"), batch_size, device)
    assert prompt is not None
    data = [[batch_size * [prompt], img]]   #for pipeline2
    prior_data = [batch_size*[prior_prompt]]  #for pipeline1
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
                                                         x_T=start_code,
                                                         is_qkv_injected=False)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_samples_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    if not opt.skip_save:
                        for x_sample in x_samples_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save("cc.png")

    #M1                                                   
    attn_img = attn_map_preprocess(sampler.attn_maps, opt.prior_prompt, 
                    opt.prior_prompt, model.cond_stage_model.tokenizer, False,
                    'attn_img/M1.png', 10) #b 1 16 16


    #read mask and seg
    mask = read_mask(os.path.join(outdir, 'mask.png'), batch_size, device, hp[3])
    seg = read_mask(os.path.join(outdir, 'seg.png'), batch_size, device, hp[4])
    skin = None
    #skin = read_mask(os.path.join(outdir, 'skin.png'), batch_size, device, hp[5])
    # get Spr and mask_in according to different mode
    if opt.mode == "shape":
        print("shape mode")
        mask_in = (mask + seg).gt(0).float() - seg * mask
        mask_in = dilation_process(mask_in, device)
        pixel = get_Spr(mask, seg, skin, attn_img)
    elif opt.mode == "color":
        print("color mode")
        mask_in = seg
        pixel = get_Spr(seg, seg, skin, attn_img)
    else: 
        print("shape & color mode")
        mask_in = (mask + seg).gt(0).float() 
        pixel = get_Spr(mask, seg, skin, attn_img)
    
    #pixel = get_Spr(mask, seg, skin, attn_img)
    
    #pipeline2
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts, img in tqdm(data, desc="data"):
                        uc = None
                        encoder_posterior = model.encode_first_stage(img)
                        z = model.get_first_stage_encoding(encoder_posterior).detach()
                        ###############################
                        if determine_file_type_by_extension(opt.image_path) == 'PNG':
                            ms = calculate_masked_variance(opt.image_path, os.path.join(outdir, 'seg.png'))
                        else:
                            ms = calculate_masked_variance(os.path.join(outdir, "source_image.png"), os.path.join(outdir, 'seg.png'))
                        #ms = 2 * ms if ms > 0.1 else ms
                        ##############################
                        hf_img = high_pass_filter(img)
                        encoder_posterior_hf = model.encode_first_stage(hf_img)
                        z_hf = model.get_first_stage_encoding(encoder_posterior_hf).detach()
                        ################################
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [''])
                            Inversion_c = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                         p=opt.inversion_steps,
                                                         z=z,
                                                         ms=ms,
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
                                                         Inversion_prompt="",
                                                         eta=opt.ddim_eta,
                                                         pixel=pixel,
                                                         x_T=start_code,
                                                         mode=opt.mode,
                                                         is_qkv_injected=True)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_samples_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)    

                        out_image_path = os.path.join(outpath, save_name)
                        if not opt.skip_save:
                            for x_sample in x_samples_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(out_image_path)

                        if ms > 0.1:
                            image1 = Image.open(os.path.join(outdir, "source_image.png")).convert('RGB')
                            image2 = Image.open(os.path.join(outpath, save_name)).convert('RGB')
                            image1 = image1.resize((512,512), Image.BICUBIC)
                            mask = Image.open(os.path.join(outdir,"seg.png"))
                            mask = mask.resize((512,512), Image.NEAREST)
                            transform =  transforms.ToTensor()
                            tensor1 = transform(image1).unsqueeze(0).to(device)
                            tensor2 = transform(image2).unsqueeze(0).to(device)
                            mask = transform(mask).unsqueeze(0).to(device)
                            mask = erode_tensor(mask, kernel_size=15)

                            low_pass = gaussian_blur(tensor1)

                            high_pass = tensor1 - low_pass
                            high_pass = torch.clamp(high_pass, min=-0.1)

                            enhanced_tensor = tensor2 + 1*high_pass*mask

                            save_image(enhanced_tensor, out_image_path)
                        
                        
    return out_image_path


def determine_file_type_by_extension(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.jsonl':
        return "JSONL"
    elif file_extension == '.png':
        return "PNG"
    elif file_extension == '.json':
        return "JSON"
    else:
        return "Unknown"

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
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

def load_model(checkpoint_path, model_config, device):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    model = Mask_Est_Model(model_config, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to yaml config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    return argparse.Namespace(**cfg)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable




def main():
    hyper_param = {"color": [0.25, 0.25, 0.5, 0.05, 0.05, 0.5],
                   "shape": [0.25, 0.25, 0.5, 0.23, 0.23, 0.5],
                   "comb": [0.25, 0.25, 0.5, 0.05, 0.05, 0.5]}

    opt = parse_args()
    
    #init parser, model, sampler
    seed_everything(opt.seed)
    accelerator = Accelerator()
    device = accelerator.device

    dconfig = OmegaConf.load(f"{opt.dconfig}")
    diff_model = load_model_from_config(dconfig, f"{opt.ckpt}")
    diff_model = diff_model.to(device).to(torch.float16)
    sampler = DDIMSampler(diff_model)

    mconfig = OmegaConf.load(f"{opt.mconfig}")

    mask_model = load_model(mconfig.mask_config.checkpoints, mconfig.mask_config, device)

    #seg_model = load_model(mconfig.seg_config.checkpoints, mconfig.seg_config, device)
    opt.mask_prompt = run_llama(opt.prompt) if opt.mask_prompt=="" else opt.mask_prompt
    opt.prior_prompt = opt.prompt if opt.prior_prompt == "" else opt.prior_prompt


    outdir = "output"
    
    file_type = determine_file_type_by_extension(opt.image_path)
    if file_type == 'PNG':
        opt.mode = get_mode(opt.prompt) if opt.mode not in ["color", "shape", "comb"] else opt.mode
        opt.seg_prompt = get_seg_prompt(opt.prompt) if not opt.seg_prompt else opt.seg_prompt
        run_mask_unet(opt, mask_model, device, hyper_param[opt.mode], outdir)
        run_editing(opt, diff_model, sampler, device, hyper_param[opt.mode], outdir)
    elif file_type == 'JSON' or file_type == 'JSONL':
        with open(opt.image_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f if line.strip()]

        world_size = accelerator.num_processes
        process_index = accelerator.process_index
        per_process_size = len(data_list) // world_size
        start = process_index * per_process_size
        end = start + per_process_size if process_index < world_size - 1 else len(data_list)
        local_data = data_list[start:end]

        outdir = os.path.join("output", f"rank{process_index}")
        os.makedirs(outdir, exist_ok=True)

        outjson = os.path.join("output", f"res{process_index}.json")


        for i, data in tqdm(enumerate(local_data)):
            try:
                opt.prompt = data["prompt"]
                opt.seg_prompt = get_seg_prompt(opt.prompt)
                opt.mode = get_mode(opt.prompt)
                image_path = os.path.join(opt.image_dir, data["image"])
                run_mask_unet(opt, mask_model, device, hyper_param[opt.mode], outdir, image_path)
                outpath = run_editing(opt, diff_model, sampler, device, hyper_param[opt.mode], outdir, f'''{data["class"]}_{i}_{data["image"]}''')
                data["edited_image"] = outpath
                with open(outjson, "a", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.write("\n")
            except Exception as e:
                print(str(e))
    else:
        print("unknow files")

if __name__ == "__main__":
    main()