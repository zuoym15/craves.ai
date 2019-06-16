import imageio as io
from unrealcv import client
from unrealcv.util import read_png, read_npy
import cv2
import json
import os
import numpy as np

#### 3d visualization ####

def set_camrea(preds):
    rotation, base, elbow, wrist, pitch, yaw, roll ,x, y, z, = preds
    gripper = 0
    res = client.request('vset /arm {rotation} {base} {elbow} {wrist} {gripper}'.format(**locals()))
    #print(res)

    res = client.request('vset /camera/1/location {x} {y} {z}'.format(**locals()))
    #print(res)

    pitch = -pitch
    yaw = yaw-180
    roll = -roll
    res = client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**locals()))
    #print(res)

    print('{rotation} {base} {elbow} {wrist} {gripper} {x} {y} {z} {pitch} {yaw} {roll}'.format(**locals()))

def get_img_from_preds(preds):
    set_camrea(preds)
    i = 1
    data = client.request('vget /camera/{i}/lit png'.format(**locals()))
    im = read_png(data)
    return im

def visualize_from_d3_preds_single_frame(d3_pred, raw_img): 
    #d3 pred in format " rotation, base, elbow, wrist, pitch, yaw, roll ,x, y, z "
    #raw image should be in OpenCV format
    client.request('vset /camera/1/fov 66.5')
    
    H, W, _ = raw_img.shape

    syn_img = get_img_from_preds(d3_pred)
    syn_img = syn_img[:, :, (2,1,0)] #rgb to bgr
    syn_h, syn_w, _ =  syn_img.shape
    if syn_h < H or syn_w < W:
        print('warning: input image too large, should be smaller than the sampler resolution')
        return syn_img

    syn_img = syn_img[0:H, 0:W, :] #cut the image too the same size as the raw image
    return syn_img

#### 2d visualization ####

def draw_keypoints_2d(im, keypoints):
    for idx in range(keypoints.shape[1]):
        im = cv2.circle(im, (int(keypoints[0, idx]), int(keypoints[1, idx])), radius = 5, color = (255, 0, 0))

    return im

def visualize_single_img_2d(img_dir, pred_dir):
    with open(pred_dir) as f:
        obj = json.load(f)

    pred = np.array(obj['d2_key']).transpose()
    im = cv2.imread(img_dir)
    draw_keypoints_2d(im, pred)
    return im

def visualize_folder(img_folder, pred_folder, save_result_folder):
    img_list = []
    pred_list = []
    save_list = []

    for root, dirs, files in os.walk(img_folder, topdown=False):
        for name in files:
            img_list.append(os.path.join(root, name))
            base_name = name.split('.')[0]
            pred_list.append(os.path.join(pred_folder, base_name + '.json'))
            save_list.append(os.path.join(save_result_folder, name))

    for i in range(len(img_list)):
        im = visualize_single_img_2d(img_list[i], pred_list[i])
        cv2.imwrite(save_list[i], im)

    return 

if __name__ == "__main__":
    client.connect()
    d3_pred = [-52.024386454729466, 36.58394668769975, -35.36400807989878, 20.63523739719251, 50.30914871062571, 178.79913628630686, -6.490615005698983, -713.9431734765288, 322.53760729596786, 665.4524751196102]
    im = visualize_from_d3_preds_single_frame(d3_pred, cv2.imread('../data/youtube_20181105/imgs/0Cuo_W5c1fw_0110.jpg'))
    cv2.imshow('vis', im)
    cv2.waitKey(0)