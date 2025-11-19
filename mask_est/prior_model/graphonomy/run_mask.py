import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict
sys.path.append('./')
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2


# Custom includes
from mask_est.prior_model.graphonomy.networks import deeplab_xception_transfer, graph
from mask_est.prior_model.graphonomy.dataloaders import custom_transforms as tr

#
import argparse
import torch.nn.functional as F
import jsonlines
from tqdm import tqdm

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

def get_cloth_mask(image_array, name, img_type):
    if img_type == '':
        return None
    else:
        mask_array = np.zeros(image_array.shape[:2], dtype=np.uint8)
        if img_type in ['SHIRTS', 'TOPS']:
            target_colors = [[255,85,0]]
        if img_type in ['SWEATERS']:
            target_colors = [[255,85,0], [0, 119, 221]]
        if img_type in ['JEANS', 'PANTS']:
            target_colors = [[0, 85, 85]]
        if img_type in ['SHORTS', 'SKIRTS']:
            target_colors = [[0, 85, 85], [0, 128, 0], [0, 0, 85]]
        if img_type in ['DRESSES']:
            target_colors = [[0, 0, 85]]
        if img_type in ['HATS']:
            target_colors = [[128, 0, 0]]
        if img_type in ["HAIRS"]:
            target_colors = [[255, 0, 0]]
        if img_type in ["SHOES"]:
            target_colors = [[255, 255, 0], [255, 170, 0]]
        for color in target_colors:
            target_color = np.array(color)
            color_mask = np.all(image_array == target_color, axis=-1)
            mask_array[color_mask] = 255
        mask_image = Image.fromarray(mask_array)
        mask_image.save(name)

def get_body_mask(image_array, name):
    mask_array = np.any(image_array != [0, 0, 0], axis=-1).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_array)
    mask_image.save(name)

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

def inference(net, img_path='', output_path='./', output_cloth='f', output_body='f', cloth_type='f', use_gpu=True):
    '''

    :param net:
    :param img_path:
    :param output_path:
    :return:
    '''
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    img = read_img(img_path)
    img.save(os.path.join(output_path, 'source_image.png'))
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if use_gpu >= 0:
                inputs = inputs.cuda()
            # outputs = net.forward(inputs)
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()
    vis_res = decode_labels(results)
    if output_cloth:
        get_cloth_mask(vis_res[0], os.path.join(output_path, output_cloth), cloth_type)
    if output_body:
        get_body_mask(vis_res[0], os.path.join(output_path, output_body))
    parsing_im = Image.fromarray(vis_res[0])
    parsing_im.save(os.path.join(output_path, "parse.png"))
    #parsing_im.save(output_path+'/{}.png'.format(output_name))
    #cv2.imwrite(output_path+'/{}_gray.png'.format(output_name), results[0, :, :])

    end_time = timeit.default_timer()
    #print('time used for the multi-scale image inference' + ' is :' + str(end_time - start_time))

def run_graphonomy(image_path, output_path, output_cloth, output_body, cloth_type):

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
    x = torch.load("mask_est/prior_model/graphonomy/data/pretrain/inference.pth")
    net.load_source_model(x)

    net.cuda()
    use_gpu = True

    inference(net=net, img_path=image_path, output_path=output_path, output_cloth=output_cloth, output_body=output_body, cloth_type=cloth_type, use_gpu=use_gpu)