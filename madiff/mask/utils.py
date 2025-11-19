import torch
import random
import numpy as np
import torch.nn.functional as F
from transformers import CLIPTokenizer
from einops import rearrange
from math import sqrt, ceil, floor
from PIL import Image
from torchvision.utils import save_image
from scipy.ndimage import binary_dilation

def rectangle_preprocess(scale_factor, rectangle_list):
  for i in range(len(rectangle_list)):
      for j in range(len(rectangle_list[i])):
          rectangle_list[i][j] = round(rectangle_list[i][j]/scale_factor)
  return rectangle_list

def get_text_tokens(text, tokenizer):
  encodings = tokenizer(text)
  encoded_sequence = encodings['input_ids']
  decoded_list = list()
  for i in encoded_sequence:
    decoded_encodings = tokenizer.decode([i])
    decoded_list.append(decoded_encodings)
  return decoded_list

def get_sub_tokens_place(whole_tokens, sub_tokens):
  for i in range(len(whole_tokens)):
    if whole_tokens[i] == sub_tokens[1]:
      if whole_tokens[i : i+len(sub_tokens)-2] == sub_tokens[1:-1]:
        replace_id = i
        break
      else:
        pass
    else:
      pass
    sub_tokens_length = len(sub_tokens)-2
  return replace_id, sub_tokens_length

def get_ROI(mask, size):
  s = torch.sum(mask[0])
  ROI = []
  for i in range(size):
    for j in range(size):
      if mask[0, 0, j, i] == 1:
        ROI.append([j,i])
  return ROI, s

def get_RM(mask, seg, size):
  RM = []
  for i in range(size):
    for j in range(size):
      if mask[0, 0, j, i] == 0 and seg[0, 0, j, i] == 1:
        RM.append([j,i])
  return RM

def get_AD(mask, seg, size):
  AD = []
  for i in range(size):
    for j in range(size):
      if mask[0, 0, j, i] == 1 and seg[0, 0, j, i] == 0:
        AD.append([j,i])
  return AD

def get_comb(mask, seg, size):
  comb = []
  for i in range(size):
    for j in range(size):
      if mask[0, 0, j, i] == 1 and seg[0, 0, j, i] == 1:
        comb.append([j,i])
  return comb

def get_threshold(select_attn_map, s):
  select_attn_map = rearrange(select_attn_map, "b 1 H W -> b (H W)")
  Amin = torch.sort(select_attn_map, dim=-1).values[:,-int(s)]
  return Amin

def top_s_pixel_id(attn_map, Amin):
  s_pixel = []
  for j in range(attn_map.shape[-2]):
    for i in range(attn_map.shape[-1]):
      if attn_map[..., j, i] >= Amin.item():
        if 0<=i<64 and 0<=j<64:
          s_pixel.append([j, i]) 
  return s_pixel

def attn_map_preprocess(attn_maps, text, words, tokenizer, save_img, attn_map_name, attn_map_id=0):
    assert isinstance(attn_maps, list)
    for i in range((attn_map_id+1)): 
      add_maps = attn_maps[i]["down_attn"][-2:] + attn_maps[i]["up_attn"][:3] #取16*16的Attention Map
      if i == 0:
        maps = add_maps
      else:
        for j in range(len(maps)):
          maps[j] = maps[j] + add_maps[j]
    for j in range(len(maps)):
      maps[j] = maps[j]/(attn_map_id+1)
    attn_map = torch.concat(maps,dim=0)
    device = attn_map.device
    attn_map = torch.unsqueeze(attn_map, dim=0)
    replace_id, sub_tokens_length = get_sub_tokens_place(get_text_tokens(text,tokenizer), get_text_tokens(words,tokenizer))
    attn_map = torch.mean(attn_map, dim=1)  #(b,HW,l)
    select_attn_map = torch.sum(attn_map[..., replace_id : replace_id+sub_tokens_length], dim=-1)
    select_attn_map = rearrange(select_attn_map, 'b (H W) -> b 1 H W', H=int(sqrt(attn_map.shape[1])))
    attn_img = F.interpolate(select_attn_map, size=([64, 64]), mode='bicubic')   
    #visualization
    if save_img:
      img = torch.zeros([3, 64, 64], device='cuda')
      img = img + attn_img[0]
      img = img/torch.max(img)
      img = 255. * rearrange(img.cpu().numpy(), 'c h w -> h w c')
      image = Image.fromarray(img.astype(np.uint8))
      image.save(attn_map_name)
    return attn_img

def get_Spr(mask, seg, skin, attn_img):
  assert attn_img.shape[-1] == attn_img.shape[-2]
  ROI_pixel, s = get_ROI(mask, attn_img.shape[-1])
  if skin is not None:
    skin_pixel, _ = get_ROI(skin, attn_img.shape[-1])
  else:
    skin_pixel = None
  RM_pixel = get_RM(mask, seg, attn_img.shape[-1])
  AD_pixel = get_AD(mask, seg, attn_img.shape[-1])
  comb_pixel = get_comb(mask, seg, attn_img.shape[-1])
  pixel = {'s_pixel': [], 'ROI_pixel': ROI_pixel, 'RM_pixel': RM_pixel, 
          'AD_pixel': AD_pixel, 'comb_pixel': comb_pixel, 'skin_pixel': skin_pixel}
  for i in range(attn_img.shape[0]):
    Amin = get_threshold(attn_img[i].unsqueeze(0), s)
    s_pixel = top_s_pixel_id(attn_img[i].unsqueeze(0), Amin)
    pixel['s_pixel'].append(s_pixel)
  return pixel

def get_skin_pixel(attn_img):
  skin_pixel = []
  for i in range(attn_img.shape[0]):
    Amin = get_threshold(attn_img[i].unsqueeze(0), 16)
    s_pixel = top_s_pixel_id(attn_img[i].unsqueeze(0), Amin)
    skin_pixel.append(s_pixel)
  return skin_pixel
