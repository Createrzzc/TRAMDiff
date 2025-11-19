import torch
import os
import numpy as np
import PIL
import json
import random
import jsonlines
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange


class MaskSet(Dataset):
    def __init__(self, jsonl_file, body_mask_root='', cloth_mask_root='', pose_root='', size=256, 
                 interpolation="bicubic", flip_p=0.5, gender=True
                ):
        self.data = list(jsonlines.open(jsonl_file)) # list of dict: [{"image_file": "1.png", "text": "red shirt"}]
        self.body_mask_root = body_mask_root
        self.cloth_mask_root = cloth_mask_root
        self.pose_root = pose_root
        self.size = size
        self.interpolation = {
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def read_process_image(self, root, file_name):
        image_path = os.path.join(root, file_name)
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image)
        image = rearrange(image, 'h w c -> c h w')
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        file_name = item["image"]
        body_mask = self.read_process_image(self.body_mask_root, file_name)[0].unsqueeze(0)
        cloth_mask = self.read_process_image(self.cloth_mask_root, file_name)[0].unsqueeze(0)
        pose_map = self.read_process_image(self.pose_root, file_name)
        input_map = torch.concat([body_mask, pose_map], dim=0)
        return {"input_map" : input_map, 
                "cloth_mask" : cloth_mask,
                "text" : text
                }


class MaskTrain(MaskSet):
    def __init__(self, **kwargs):
        super().__init__(jsonl_file='/home/ac/data/2023/zhanzechao/ours/dataset/mask_train.jsonl', body_mask_root='/home/ac/data/2023/zhanzechao/image_editing/dataset/fashion_mask/body_mask/train', 
                         cloth_mask_root='/home/ac/data/2023/zhanzechao/image_editing/dataset/fashion_mask/cloth_mask/train', pose_root='/home/ac/data/2023/zhanzechao/image_editing/dataset/fashion_mask/densepose/train', **kwargs)


class MaskValid(MaskSet):
    def __init__(self, **kwargs):
        super().__init__(jsonl_file='/home/ac/data/2023/zhanzechao/ours/dataset/mask_valid.jsonl', body_mask_root='/home/ac/data/2023/zhanzechao/image_editing/dataset/fashion_mask/body_mask/valid', 
                         cloth_mask_root='/home/ac/data/2023/zhanzechao/image_editing/dataset/fashion_mask/cloth_mask/valid', pose_root='/home/ac/data/2023/zhanzechao/image_editing/dataset/fashion_mask/densepose/valid', **kwargs)