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
from pose.utils.evaluation import final_preds, transform_preds #for debug only

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
    def __init__(self, img_folder, meta_dir, random_bg_dir, cam_name, anno_type, inp_res=256, out_res=64, train=True, sigma=1, training_set_percentage = 0.9,
                 label_type='Gaussian',  scales = [0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6], multi_scale = False, ignore_invis_pts = False, replace_bg = False):
        self.scales = scales
        #self.actor_name = "RobotArmActor_3"
        #self.actor_name = "owi535"
        self.img_folder = img_folder    # root image folders
        self.meta_dir = meta_dir
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.label_type = label_type
        self.cam_name = cam_name
        self.replace_bg = replace_bg
        anno_type = anno_type.lower()
        assert anno_type == '3d' or anno_type == '2d' or anno_type == 'none'

        self.anno_type = anno_type

        if self.anno_type == '3d':
            self.dataset = vdb.Dataset(img_folder)
            ids = self.dataset.get_ids()
            annotation_colors = self.dataset.get_annotation_color()
            if annotation_colors.__contains__("owi535"):
                self.actor_name = "owi535" #new version
            else:
                self.actor_name = "RobotArmActor_3" #old version

            self.color = annotation_colors[self.actor_name]

            split = round(len(ids)*training_set_percentage) #90% for training, 10% for validation
            self.train = ids[:split]
            self.valid = ids[split:] 
            with open(os.path.join(meta_dir, 'lr_pairs.json'),'r') as f:
                self.lr_pairs = json.load(f)

        if self.anno_type == '2d':
            if os.path.isfile(os.path.join(img_folder, 'valid_img_list.json')):
                    with open(os.path.join(img_folder, 'valid_img_list.json')) as f:
                        self.dataset = json.load(f)
            else:
                    _, self.dataset = read_jpg(os.path.join(img_folder, 'imgs'))

            ids = list(range(len(self.dataset)))
            with open(os.path.join(img_folder, 'pts.json')) as f:
                self.bbox_anno = json.load(f)

            split = round(len(ids)*training_set_percentage) #90% for training, 10% for validation
            self.train = ids[:split]
            self.valid = ids[split:] 
            with open(os.path.join(meta_dir, 'lr_pairs.json'),'r') as f:
                self.lr_pairs = json.load(f)

        if self.anno_type == 'none':
            L, F = read_jpg(img_folder)
            ids = range(1, len(L)+1)
            split = round(len(ids)*training_set_percentage)
            self.train = ids[:split]
            self.valid = ids[split:] 
            # self.train = []
            # self.valid = ids
            self.dataset = L
            self.F = F
            self.anno = None
            if isfile(os.path.join(img_folder, 'pts.json')):
                self.anno = json.load(open(os.path.join(img_folder, 'pts.json')))

        self.multi_scale = multi_scale
        self.ignore_invis_pts = ignore_invis_pts

        if self.replace_bg:
            self.background_replace = background_replace(random_bg_dir)
        
        # create train/val split
        
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = '../datasets/arm/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                ids = [index]
                cams = [self.cam_name]
                images = self.dataset.get_image(cams, ids, 'synthetic')
                img_path = images[0]
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)

        # if self.is_train:
        
        #     print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        #     print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        #actor_name = "RobotArmActor_3"
        #color = [0, 255, 63]
        scale_factor = 60.0

        if self.multi_scale:
            scale = self.scales[index % len(self.scales)]
            index = index // len(self.scales)

        if self.anno_type == '3d' or self.anno_type == '2d':

            if self.is_train:
                ids = self.train[index]
            else:
                ids = self.valid[index]

            if self.anno_type == '3d':
                joint_2d, vertex_2d, img_path = get_joint_vertex_2d(self.dataset, ids, self.cam_name, self.actor_name)
                joint_2d = joint_2d[1:] #discard the first joint, as we do not predict it.

                with open(os.path.join(self.meta_dir, 'vertex.json'),'r') as f: #from raw vertexs to final keypoints
                    vertex_seq = json.load(f)

                num_vertex = len(vertex_seq)
                pts = np.zeros((num_vertex, 2))

                for i in range(num_vertex):
                    pts[i] = np.average(vertex_2d[vertex_seq[i]], axis = 0)
                    #pts[i] = (vertex_2d[2*i]+vertex_2d[2*i+1])/2

                
                pts = np.concatenate((joint_2d, pts), axis = 0)
            
            if self.anno_type == '2d': #data with only 2d annotations
                img_path = os.path.join(self.img_folder, 'imgs', self.dataset[index])
                with open(os.path.join(self.img_folder, 'd3_preds', os.path.splitext(os.path.basename(img_path))[0] + '.json'), 'r') as f:
                    obj = json.load(f)
                    pts = np.transpose(np.array(obj['reprojection'])) 

                    if self.ignore_invis_pts and 'visibility' in obj:
                        visibility = obj['visibility'][:-2]
                        pts[np.invert(visibility), :] = -1.0

            # For single-person pose estimation with a centered/scaled figure
            nparts = pts.shape[0]

            if not self.replace_bg:
                img = load_image(img_path)  # CxHxW
            else:
                img = im_to_torch(cv2.cvtColor(self.background_replace.replace(cv2.imread(img_path), 'white'), cv2.COLOR_BGR2RGB))

            original_size = np.array((img.shape[2], img.shape[1]))

            if self.anno_type == '3d':
                segmasks = self.dataset.get_seg([self.cam_name], [ids])
                segmask = io.imread(segmasks[0])
        
                binary_arm = vdb.get_obj_mask(segmask, self.color)
                bb = vdb.seg2bb(binary_arm)
                x0, x1, y0, y1 = bb

            if self.anno_type == '2d':
                bb = self.bbox_anno[os.path.basename(img_path)]
                x0, x1, y0, y1 = bb[0][0], bb[1][0], bb[0][1], bb[1][1]

            c = np.array([(x0+x1), (y0+y1)])/2
            s = np.sqrt((y1-y0)*(x1-x0))/scale_factor
            r = 0

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
                    pts[:, 0] = img.size(2) - pts[:, 0]
                    for pair in self.lr_pairs:
                        pts[[pair[0], pair[1]]] = pts[[pair[1], pair[0]]]
                    c[0] = img.size(2) - c[0]

            if self.multi_scale:
                s = s * scale

            inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
            inp = color_normalize(inp, self.mean, self.std)

            # print(pts)

            tpts = pts.copy()
            target = torch.zeros(nparts, self.out_res, self.out_res)
            for i in range(nparts):
                # if tpts[i, 2] > 0: # This is evil!!
                if tpts[i, 1] > 0:
                    tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, [self.out_res, self.out_res], rot=r))
                    target[i] = draw_labelmap(target[i], tpts[i], self.sigma, type=self.label_type)

            # print(transform_preds(torch.from_numpy(tpts), c, s, [64, 64]))
                    
            # Meta info
            meta = {'index' : index, 'pts' : pts, 'tpts' : tpts, 'center':c, 'original_size':original_size,
            'scale':s, 'img_name': os.path.splitext(os.path.basename(img_path))[0]}

            return inp, target, meta

        if self.anno_type == 'none':
            img_path = self.dataset[index]   

            
            if not self.replace_bg:
                img = load_image(img_path)  # CxHxW
            else:
                img = im_to_torch(cv2.cvtColor(self.background_replace.replace(cv2.imread(img_path), 'white'), cv2.COLOR_BGR2RGB))      

            original_size = np.array((img.shape[2], img.shape[1]))

            inp = img

            if self.anno is not None:
                joints = self.anno[self.F[index]]
                x0, y0, x1, y1 = joints[0][0], joints[0][1], joints[1][0], joints[1][1]
                c = np.array([(x0+x1), (y0+y1)])/2
                s = np.sqrt((y1-y0)*(x1-x0))/scale_factor
                if self.multi_scale:
                    s = s*scale

            else:
                c = np.array([img.shape[2]/2, img.shape[1]/2])
                s = 5.0

            r = 0 

            inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
            
            inp = color_normalize(inp, self.mean, self.std)

            meta = {'index' : index, 'pts' : [], 'tpts' : [], 'center':c, 'original_size':original_size,
            'scale':s, 'img_name': os.path.splitext(os.path.basename(img_path))[0]}

            return inp, [], meta


    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            if self.multi_scale:
                return len(self.scales) * len(self.valid)
            else:
                return len(self.valid)

    def reset(self):
        ''' just for format consisitency among datasets. Do nothing in arm.'''
        pass 

