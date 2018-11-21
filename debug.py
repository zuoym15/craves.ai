import sys
from unreal.arm import get_joint_2d
import unreal.virtual_db as vdb
import numpy as np
import cv2
import torch
import time

import pose.datasets as datasets
from pose.utils.imutils import *
from pose.utils.transforms import multi_scale_merge

if __name__ == "__main__":


    db_root_dir = './data/gan_20181006/'
    #db_root_dir = 'C:/Users/Yiming/Documents/OWIMap'
    db_root_dir_2 = './data/ft_20181105'
    meta_dir = './data/meta/17_vertex'
    bg_dir = 'C:/Users/Yiming/Documents/test2017'

    cams = 'FusionCameraActor3'

    Arm = datasets.Arm(db_root_dir, meta_dir, bg_dir, cams, use_bbox = False, train = True, scales = [0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6], multi_scale = False, real_img_finetune = False, training_set_percentage = 0.9)

    Arm2 = datasets.Arm(db_root_dir_2, meta_dir, bg_dir, cams, use_bbox = False, train = True, scales = [0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6], multi_scale = False, real_img_finetune = True, training_set_percentage = 1.0, replace_bg=True)

    #Arm = datasets.Concat(datasets = (Arm, Arm2), ratio = None)


    '''

    inps = []
    scales = []

    for i in range(7):
        inp, target, meta = Arm.__getitem__(i)

        

        #imsave('/home/zym/arm-pose/example_images/target.jpg', inp.numpy())

    # exit(0)

        
        inps.append(inp)
        scales.append(meta['scale'])

    inp = np.stack(inps)
    #print(inps)
    inp, _ = multi_scale_merge(inp, scales)
    #print(inp)
    print(inp.shape)


    meanstd_file = '/home/zym/arm-pose/data/arm/mean.pth.tar'
    meanstd = torch.load(meanstd_file)
    mean = meanstd['mean']



    for t, m in zip(inp, mean):
        t.add_(m)







    #inp = im_to_torch(inp)
    imsave('/home/zym/arm-pose/example_images/target.jpg', inp.numpy())
    exit(0)

    '''
    for i in np.random.choice(len(Arm), 10):
        #cur = time.time()

        inp, target, meta = Arm.__getitem__(i)

        meanstd_file = './datasets/arm/mean.pth.tar'
        meanstd = torch.load(meanstd_file)
        mean = meanstd['mean']

        for t, m in zip(inp, mean):
            t.add_(m)

        #imsave('/home/zym/arm-pose/example_images/target.jpg', inp.numpy())
        scipy.misc.imsave('./visualization/'+str(i)+'.jpg', sample_with_heatmap(inp, target))
        #print(time.time() - cur)

    #imsave(sample_with_heatmap(inp, target), '/home/zym/arm-pose/example_images/target.jpg')

