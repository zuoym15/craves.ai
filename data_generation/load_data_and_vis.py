import sys
sys.path.append('..')

from unreal.arm import get_joint_2d
import unreal.virtual_db as vdb
import numpy as np
import cv2
import torch
import time

import pose.datasets as datasets
from pose.utils.imutils import *
from pose.utils.transforms import multi_scale_merge
from pose.utils.evaluation import *

if __name__ == "__main__":

    db_root_dir = '../data/20181107/'
    meta_dir = '../data/meta/17_vertex'
    bg_dir = '../background_img'
    cams = 'FusionCameraActor3_2'

    Arm = datasets.Arm(db_root_dir, meta_dir, bg_dir, cams, anno_type = '3d', train = True, training_set_percentage = 0.9)

    for i in np.random.choice(len(Arm), 1):

        inp, target, meta = Arm.__getitem__(i)

        meanstd_file = '../datasets/arm/mean.pth.tar'
        meanstd = torch.load(meanstd_file)
        mean = meanstd['mean']

        for t, m in zip(inp, mean):
            t.add_(m)

        im = sample_with_heatmap(inp, target)
        im = im[:, :, [2,1,0]] #to BGR
        cv2.imshow('sample_with_heatmap', im)
        cv2.waitKey(0)
