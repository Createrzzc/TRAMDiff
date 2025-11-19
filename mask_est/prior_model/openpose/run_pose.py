import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import jsonlines
import os

from mask_est.prior_model.openpose.src import model
from mask_est.prior_model.openpose.src import util
from mask_est.prior_model.openpose.src.body import Body
from mask_est.prior_model.openpose.src.hand import Hand
from tqdm import tqdm

def run_openpose(image_path, output_file, output_pose):
    body_estimation = Body('mask_est/prior_model/openpose/pretrain/body_pose_model.pth')
    #hand_estimation = Hand('model/hand_pose_model.pth')
    name = os.path.join(output_file, output_pose)
    oriImg = cv2.imread(image_path)  # B,G,R order
    height, width, _ = oriImg.shape
    blackImg = np.zeros((height, width, 3), np.uint8)
    candidate, subset = body_estimation(oriImg)
    canvas = blackImg#copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset, name)