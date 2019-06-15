from __future__ import absolute_import

import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch
import cv2
import json
import time

from .misc import *
from .imutils import *


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def flip_back(flip_output, meta_dir):

    """
    flip output map
    """
    '''
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    elif dataset == 'arm_17':
        matchedParts = ([4,5],[7,9],[8,10])
    else:
        print('Not supported dataset: ' + dataset)
    '''
    with open(os.path.join(meta_dir, 'lr_pairs.json'), 'r') as f:
        matchedParts = json.load(f)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    # print(scale)
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    #new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    # return new_pt[:2].astype(int) + 1
    return (new_pt[:2] + 0.5).astype(int)


def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, 1, 0))
    return coords

def crop_bbox(img, bbox, res):
    img = im_to_numpy(img)

    # Upper left point
    ul = np.array(bbox[0:2])
    # Bottom right point
    br = np.array(bbox[2:4])

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    new_img = img[ul[1]:br[1], ul[0]:br[0]]

    new_img = im_to_torch(scipy.misc.imresize(new_img, res))

    return new_img

def multi_scale_merge(imgs, scales):

    score = np.amax(imgs, axis = (0,2,3)).tolist()

    sf = scales[-1]/scales[0]


    new_img = np.zeros([int(imgs.shape[2]*sf/2)*2, int(imgs.shape[3]*sf/2)*2, imgs.shape[1]]) #create a big image

    res = [int(imgs.shape[2]*sf/2)*2, int(imgs.shape[3]*sf/2)*2]

    counter = np.zeros([int(imgs.shape[2]*sf/2)*2, int(imgs.shape[3]*sf/2)*2, imgs.shape[1]])

    c = [int(imgs.shape[2]*sf/2), int(imgs.shape[3]*sf/2)] #center

    for i in range(imgs.shape[0]): #paste all images to the big one
        sf = scales[i]/scales[0]
        new_ht = int(imgs.shape[2]*sf/2)
        new_wd = int(imgs.shape[3]*sf/2)

        img = np.transpose(imgs[i], (1, 2, 0))

        for j in range(imgs.shape[1]):     
            img_resized = cv2.resize(img[:,:,j], (new_ht*2, new_wd*2))
            new_img[c[0]-new_ht:c[0]+new_ht,c[1]-new_wd:c[1]+new_wd,j] = new_img[c[0]-new_ht:c[0]+new_ht,c[1]-new_wd:c[1]+new_wd,j] + img_resized

        counter[c[0]-new_ht:c[0]+new_ht,c[1]-new_wd:c[1]+new_wd,:] = counter[c[0]-new_ht:c[0]+new_ht,c[1]-new_wd:c[1]+new_wd,:] + 1.0

    new_img = new_img/counter #normalize
     
    return im_to_torch(new_img), res, score


def align_back(score_map, center, scale, original_size, rot = 0):

    #print(center)

    #print(original_size)
    # score_map: C * W * H
    res = [score_map.shape[1], score_map.shape[2]]
    ht, wd = res[0], res[1]

    score_map = score_map.numpy()
    center = center.numpy().astype(np.int32)
    scale = scale.numpy()
    original_size = original_size.numpy()

    img = np.zeros((score_map.shape[0], original_size[1], original_size[0]))

    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1

    new_ht = int(np.math.floor(ht * sf))
    new_wd = int(np.math.floor(wd * sf))

    # new_score_map = np.zeros((score_map.shape[0], new_ht, new_wd))

    new_score_map = cv2.resize(np.transpose(score_map, axes=[1,2,0]), (new_ht, new_wd))
    new_score_map = np.transpose(new_score_map, axes=[2,0,1])

    ul = [new_ht // 2 - center[0], new_wd // 2 - center[1]]
    br = [new_ht // 2 - center[0] + original_size[0], new_wd // 2 - center[1] + original_size[1]]

    new_x = max(0, -ul[0]), min(br[0], new_ht) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], new_wd) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(new_ht, br[0])
    old_y = max(0, ul[1]), min(new_wd, br[1])

    img[:, new_y[0]:new_y[1], new_x[0]:new_x[1]] = new_score_map[:, old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return img

def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = center * 1.0 / sf
            scale = scale / sf

    # print(scale)

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # print([ul, br])

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    return new_img
