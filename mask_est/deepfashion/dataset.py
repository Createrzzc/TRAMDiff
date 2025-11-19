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
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        file_name = item["image"]

        body_mask = self.read_process_image(self.body_mask_root, file_name)
        cloth_mask = self.read_process_image(self.cloth_mask_root, file_name)
        pose_map = self.read_process_image(self.pose_root, file_name)
        input_map = torch.concat([body_mask, pose_map], dim=0)
        return {"input_map" : input_map, 
                "cloth_mask" : cloth_mask,
                "text" : text
                }


class MaskTrain(MaskSet):
    def __init__(self, **kwargs):
        super().__init__(jsonl_file='/disk16T/2023/zhanzechao/stable-diffusion/mask_est/train_data.jsonl', body_mask_root='/disk16T/2023/zhanzechao/body_mask_train', 
                         cloth_mask_root='/disk16T/2023/zhanzechao/cloth_mask_train', pose_root='/disk16T/2023/zhanzechao/body_pose_train', **kwargs)


class MaskValid(MaskSet):
    def __init__(self, **kwargs):
        super().__init__(jsonl_file='/disk16T/2023/zhanzechao/stable-diffusion/mask_est/train_data.jsonl', body_mask_root='/disk16T/2023/zhanzechao/body_mask_train', 
                         cloth_mask_root='/disk16T/2023/zhanzechao/cloth_mask_train', pose_root='/disk16T/2023/zhanzechao/body_pose_train', **kwargs)