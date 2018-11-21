from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math
import cv2

import torch
import torch.utils.data as data
import imageio as io
import json

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *

import sys
from unreal.arm import get_joint_vertex_2d
import unreal.virtual_db.vdb as vdb
 
def read_jpg(file_dir):   
    L, F = [], []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':  
                L.append(os.path.join(root, file)) 
                F.append(file)
    return L,F 

class Arm(data.Dataset):
    def __init__(self, img_folder, meta_dir ,random_bg_dir, cam_name, inp_res=224, train=True, training_set_percentage = 0.9):
        self.actor_name = "RobotArmActor_3"
        self.img_folder = img_folder    # root image folders
        self.meta_dir = meta_dir
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.cam_name = cam_name
        self.dataset = vdb.Dataset(img_folder)
        ids = self.dataset.get_ids()
        self.color = self.dataset.get_annotation_color()[self.actor_name]
        split = round(len(ids)*training_set_percentage) #90% for training, 10% for validation
        # create train/val split
        self.train = ids[:split]
        self.valid = ids[split:] 
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = './datasets/arm/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
            
        return meanstd['mean'], meanstd['std']

    def load_angles(self, img_path):
        file_basename = os.path.splitext(os.path.basename(img_path))[0] + '.json'
        file_path = os.path.join(self.img_folder, 'angles', file_basename)
        with open(file_path, 'r') as fp:
            obj = json.load(fp)

        angles = np.array(obj[0:4], detype = np.float32)
        return torch.from_numpy(angles)

    def __getitem__(self, index):

        if self.is_train:
            ids = self.train[index]
        else:
            ids = self.valid[index]

        images = self.dataset.get_image([self.cam_name], [ids])
        img_path = images[0]

        img = load_image(img_path)  #CxHxW
        target = self.load_angles(img_path)

        original_size = np.array((img.shape[2], img.shape[1]))

        segmasks = self.dataset.get_seg([self.cam_name], [ids])
        segmask = io.imread(segmasks[0])

        binary_arm = vdb.get_obj_mask(segmask, self.color)
        bb = vdb.seg2bb(binary_arm)
        x0, x1, y0, y1 = bb
        

        c = np.array([(x0+x1), (y0+y1)])/2
        #s = np.sqrt((y1-y0)*(x1-x0))/120.0
        s = np.sqrt((y1-y0)*(x1-x0))/60.0
        r = 0

        #s = max(x1-x0, y1-y0)/125
        if self.is_train:
            c = c + np.array([-30 + 60*random.random() ,-30 + 60*random.random()]) #random move
            s *= 0.6*(1+2*random.random())#random scale
        
            rf = 15
            r = -rf + 2*random.random()*rf#random rotation
            #r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Color
            im_rgb = im_to_numpy(img)
            im_lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)
            im_lab[:,:,0] = np.clip(im_lab[:,:,0]*(random.uniform(0.3, 1.3)), 0, 255)
            img = im_to_torch(cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB))

            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()

        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        return inp, target


    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

    def reset(self):
        ''' just for format consisitency among datasets. Do nothing in arm.'''
        pass 

