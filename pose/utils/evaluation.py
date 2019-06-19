from __future__ import absolute_import

import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import torch

from .misc import *
from .transforms import transform, transform_preds

__all__ = ['accuracy', 'AverageMeter']

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum().numpy() / dists.ne(-1).sum().numpy()
    else:
        return -1

def accuracy(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/4.0
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1], thr=thr)
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:  
        acc[0] = avg_acc / cnt
    return acc

def final_preds_bbox(output, bbox, res):
    preds = get_preds(output) # float typ
    preds = preds.numpy()

    for i in range(preds.shape[0]):
        width = bbox[2][i] - bbox[0][i]
        height = bbox[3][i] - bbox[1][i]
        for j in range(preds.shape[1]):    
            preds[i, j, :] = preds[i, j, :] / res * np.array([width, height]) + np.array([bbox[0][i], bbox[0][i]])

    return torch.from_numpy(preds)

def final_preds(output, center, scale, res):
    coords = get_preds(output) # float type

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords[:, :, 0] += 0.5
    coords[:, :, 1] -= 0.5

    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

def d3_acc(preds, gts, percent = .5):
    num_samples = len(preds)

    acc = np.zeros_like(preds[0])

    hit = 0

    # miss_list = []
    max_error_list = [] #max angle error for each image
    res_list = []

    for i in range(num_samples):
        pred = np.array(preds[i])
        gt = np.array(gts[i])

        res = np.abs(pred - gt)

        res[0:7] = np.abs((res[0:7] + 180.0) % 360.0 - 180.0)
        max_error_list.append(np.max(res[0:4]))
        res_list.append(res)


        # if not np.any(res[0:4]>10): #false prediction
        #     acc += res
        #     hit = hit + 1

        # else:
        #     miss_list.append(i)

    top_n = int(percent * num_samples) #take top N images with smallesr error.
    sorted_list = np.argsort(max_error_list)

    for i in range(top_n):
        acc += res_list[sorted_list[i]]
             
    return (acc/top_n)[:4]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
