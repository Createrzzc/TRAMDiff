import sys
sys.path.append("./")
import torch
import argparse
import os
import PIL
import jsonlines
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
from mask_est.model import Mask_Est_Model
from prior_model.graphonomy.run_mask import run_graphonomy
from prior_model.openpose.run_pose import run_openpose
from prior_model.densepose.run_dp import run_dp


def load_model(checkpoint_path, model_config, device):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    model = Mask_Est_Model(model_config, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

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
    result_tensor = (1-mask_tensor) * image_tensor

    result_numpy = TF.to_pil_image(result_tensor)
    result_numpy.save(save_path)
    print('Image saved successfully.')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask_prompt",
        type=str,
        nargs="?",
        default="t-shirt with short sleeves.",
        help="input text"
    )
    parser.add_argument(
        "--seg_prompt",
        type=str,
        nargs="?",
        default="SHIRTS.",
        help="input text"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        nargs="?",
        default="/home/ac/data/2023/zhanzechao/image_editing/dataset/fashiongen/valid/00445.png",
        help="the path of the input image"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="mask_est/inference.yaml",
        help="config for inference"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="?",
        default="output",
        help="file storage path"
    )
    parser.add_argument(
        "--output_body",
        type=str,
        nargs="?",
        default="body.png",
        help="the name of the body mask file"
    )
    parser.add_argument(
        "--output_pose",
        type=str,
        nargs="?",
        default="pose.png",
        help="the name of the pose file"
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    def get_mask(model, input_map, input_text, name):
        outputs = model(input_map, input_text)
        outputs = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)
        outputs = F.interpolate(outputs, size=(256,256), mode='bilinear', align_corners=False)
        outputs = torch.gt(outputs, 0.25).float()
        for x_sample in outputs:
            #x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            x_sample = 255. * x_sample[0].cpu().numpy()
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(opt.output_path, name))
    
    opt = parser.parse_args()
    config = OmegaConf.load(f"{opt.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_model = load_model(config.mask_config.checkpoints, config.mask_config, device)
    seg_model = load_model(config.seg_config.checkpoints, config.seg_config, device)

    mask_text = [opt.mask_prompt]
    seg_text = [opt.seg_prompt]

    pre_process(opt.image_path, opt.output_path, '', opt.output_body, opt.output_pose)
    mask_input_map, seg_input_map = load_input_image(opt.output_path, opt.output_body, opt.output_pose, device, size=128)

    get_mask(mask_model, mask_input_map, mask_text, 'mask.png')
    get_mask(seg_model, seg_input_map, seg_text, 'seg.png')

    #postprocess(os.path.join(opt.output_path, 'mask.png'), os.path.join(opt.output_path, 'll.png'), os.path.join(opt.output_path, 'comb.png'))    

if __name__ == "__main__":
    main()



