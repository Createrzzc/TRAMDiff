import torch
import math
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms

def gaussian_blur(img, kernel_size=15, sigma=5):
    _, channels, _, _ = img.shape
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float().to('cuda')
    x = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel1d = x / x.sum()
    kernel2d = torch.outer(kernel1d, kernel1d).unsqueeze(0).unsqueeze(0)
    kernel2d = kernel2d.repeat(channels, 1, 1, 1)

    blurred = F.conv2d(img, kernel2d, padding=kernel_size // 2, groups=channels)
    return blurred

def erode_tensor(tensor, kernel_size=3):
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    eroded_tensor = -F.max_pool2d(-tensor, kernel_size, stride=1, padding=kernel_size // 2)
    return eroded_tensor

def high_pass_filter(tensor, mask=None, max_threshold=0.5):

    low_pass = gaussian_blur(tensor)
    if mask is not None:
        '''if tensor.shape[-1] >= 32:
            mask = erode_tensor(mask, kernel_size=2)'''
        target_resolution = (tensor.shape[-2], tensor.shape[-1])
        downsampled_mask = F.interpolate(mask, size=target_resolution, mode='bilinear', align_corners=False)
        threshold = 0.5
        binary_mask = (downsampled_mask > threshold).float()
        '''if binary_mask.shape[-1] >= 32:
            binary_mask = erode_tensor(binary_mask, kernel_size=3)'''
        high_pass = (tensor - low_pass) * binary_mask
    else:
        high_pass = tensor - low_pass
    high_pass = torch.clamp(high_pass, max=1*max_threshold)
    #high_pass = torch.clamp(high_pass, max=0.1)
    high_pass[high_pass < -0.1] = -0.1
    
    return high_pass

def calculate_masked_variance(image_path, mask_path):
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")  
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    mask_tensor = transform(mask)
    
    mask_tensor = (mask_tensor > 0.5).float()

    masked_image = image_tensor * mask_tensor

    mean_val = masked_image[mask_tensor == 1].mean()
    variance = ((masked_image[mask_tensor == 1] - mean_val) ** 2).mean().item()
    
    return math.sqrt(variance)