# Weichao Qiu @ 2018
# Use unrealcv client to control the arm.

# pip install unrealcv
import imageio as io
from unrealcv import client
from unrealcv.util import read_png, read_npy
import time
import random
import cv2
import json
import os
import numpy as np
from numpy import sin,cos,pi
import shutil

def data_sampler(num_data, rand_cam = True):
    ###############################################
    #for regular data generation
    ###############################################
    #num_data = 100

    dist = 600 #distance from cam to origin

    data_dir = 'C:\\Users\\Yiming\\Documents\\OWIMap'
    if not os.path.isdir(os.path.join(data_dir,'angles')):
        print('creating path: ' + os.path.join(data_dir,'angles'))
        os.mkdir(os.path.join(data_dir,'angles'))

    client.request('vset /camera/1/fov 90')

    sample_id = 0

    # res = client.request('vset /arm/random_pose true')
    while True:
        if sample_id >= num_data:
            break

        rotation = random.randint(-130,130)
        base = random.randint(-90, 60)
        elbow = random.randint(-60, 90)
        wrist = random.randint(-45, 45)
        # if base < -70:
        #     min_elbow = -base-20
        # elif base < -50:
        #     min_elbow = -base-30
        # elif base < -0:
        #     min_elbow = -base-40
        # else:
        #     min_elbow = -40
        # elbow = random.randint(min_elbow,70)
        # wrist = random.randint(-30,30)
        gripper = 0
        client.request('vset /arm {rotation} {base} {elbow} {wrist} {gripper}'.format(**locals()))

        if rand_cam:

            pitch = -random.randint(10,60)
            yaw = random.randint(0,360)
            roll = 0
            client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**locals()))

            x = int(-dist*cos(pitch*pi/180)*cos(yaw*pi/180))
            y = int(-dist*cos(pitch*pi/180)*sin(yaw*pi/180))
            z = int(-dist*sin(pitch*pi/180))
            client.request('vset /camera/1/location {x} {y} {z}'.format(**locals()))

        else:
            pitch = -30
            yaw = 180
            roll = 0
            client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**locals()))

            x = int(-dist*cos(pitch*pi/180)*cos(yaw*pi/180))
            y = int(-dist*cos(pitch*pi/180)*sin(yaw*pi/180))
            z = int(-dist*sin(pitch*pi/180))
            client.request('vset /camera/1/location {x} {y} {z}'.format(**locals()))

        tip_z = float(client.request('vget /arm/tip_pose').split()[2])

        if tip_z > 0:

            obj = [rotation, base, elbow, wrist, -pitch, yaw-180, -roll, x, y, z]

            with open(os.path.join(data_dir,'angles',str(sample_id).zfill(8)+'.json'),'w') as f:
                json.dump(obj, f)

            client.request('vset /data_capture/capture_frame')
            print("#{sample_id}: {rotation}, {base}, {elbow}，{wrist}".format(**locals()))
            time.sleep(0.5)

            sample_id+=1

def data_sampler_dense_pose():
    ################################################
    #for generating dense pose data
    ################################################
    counter = 0

    # res = client.request('vset /arm/random_pose true')
    for r in range(-90, 91, 20): #rotation
        for b in range(-70, 71, 20): #base
            for e in range(max(-30, -b-40), 71, 20): #elbow
                rotation = r
                base = b
                elbow = e 
                wrist = 0
                gripper = 0
                client.request('vset /arm {rotation} {base} {elbow} {wrist} {gripper}'.format(**locals()))
                client.request('vset /data_capture/capture_frame')
                print("#{counter}: {r}, {b}, {e}".format(**locals()))
                counter += 1
                time.sleep(0.5)

def visualizer(preds):
    rotation, base, elbow, wrist, pitch, yaw, roll ,x, y, z, = preds
    #################################################
    #for visualization
    #################################################
    #rotation = rotation + 90
    #base = base - 90
    #elbow = elbow - base
    #wrist = wrist - elbow
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

    client.request('vset /data_capture/capture_frame')

    '''

    i = 1
    data = client.request('vget /camera/{i}/lit png'.format(**locals()))
    im = read_png(data)
    io.imsave('cam_%d.png' % i, im)
    print('image saved')
    return im
    '''


def main(data_dir, meta_dir, img_type = 'video'):
    client.connect()
    
    with open(os.path.join(meta_dir,'camera_parameter.json'), 'r') as f:
        camera_parameter = json.load(f)
    
    camera_parameter = camera_parameter[img_type]
    transform = [camera_parameter['PrincipalPoint'][0] - camera_parameter['ImageSize'][0]/2,
                 camera_parameter['PrincipalPoint'][1] - camera_parameter['ImageSize'][1]/2]
    M = np.float32([[1,0,transform[0]],[0,1,transform[1]]])


    with open(os.path.join(data_dir,'d3_pred.json'), 'r') as f:
        d3_pred = json.load(f)

    for file_name, preds in d3_pred.items():
        img_dir = os.path.join(data_dir, 'imgs', os.path.splitext(file_name)[0] + '.jpg')
        img_raw = cv2.imread(img_dir)
        preds = preds['preds']

        rows,cols = img_raw.shape[0:2]

        img_syn = visualizer(preds)
        img_syn = cv2.warpAffine(img_syn,M,(cols,rows))

        img = img_raw/2+img_syn[:,:,0:3]/2

        cv2.imwrite(os.path.join(data_dir, 'syn', os.path.splitext(file_name)[0] + '.jpg'), img)

        time.sleep(0.5)

def training_data_from_d3_preds(d3_pred_dir, img_dir, training_data_dir):
    client.request('vset /camera/1/fov 66.5')

    dir_list = ['FusionCameraActor3_2', 'FusionCameraActor3_2/real']
    for dir in dir_list:
        if not os.path.isdir(os.path.join(training_data_dir, dir)):
            os.mkdir(os.path.join(training_data_dir, dir))

    real_dir = os.path.join(training_data_dir, dir_list[-1])

    with open(d3_pred_dir, 'r') as f:
        obj = json.load(f)
        hit, d3_pred, file_name_list = obj['hit'], obj['d3_pred'], obj['file_name_list']

    cnt = 0

    for file_name in file_name_list:
        pred = d3_pred[file_name]['preds']
        file, _ = os.path.splitext(file_name)
        file_path = os.path.join(img_dir, file + '.jpg')
        newfile_path = os.path.join(real_dir, str(cnt).zfill(8) + '.jpg')
        shutil.copyfile(file_path, newfile_path)
        visualizer(pred)
        print('finished: '+ str(cnt))
        cnt = cnt + 1

def align(img_raw, img_syn, aligned_dir, meta_dir, img_type = 'video'):
    img_raw = cv2.imread(img_raw)
    img_syn = cv2.imread(img_syn)
    
    with open(os.path.join(meta_dir,'camera_parameter.json'), 'r') as f:
        camera_parameter = json.load(f)
    
    camera_parameter = camera_parameter[img_type]
    transform = [camera_parameter['PrincipalPoint'][0] - camera_parameter['ImageSize'][0]/2,
                 camera_parameter['PrincipalPoint'][1] - camera_parameter['ImageSize'][1]/2]
    M = np.float32([[1,0,transform[0]],[0,1,transform[1]]])


    rows,cols = img_raw.shape[0:2]

    img_syn = cv2.warpAffine(img_syn,M,(cols,rows))
    

    img = img_raw/2+img_syn[:,:,0:3]/2
    cv2.imwrite(os.path.join(aligned_dir,'aligned.jpg'), img)

    return


if __name__ == "__main__":
    
    
    #align('C:\\Users\\Yiming\\Desktop\\test\\mix\\test8.jpg', 'C:\\Users\\Yiming\\Desktop\\test\\mix\\00000000.png', 
    #'C:\\Users\\Yiming\\Desktop\\test\\mix', 'C:\\Users\\Yiming\\Desktop\\arm_miscs\\annotation\\auto_test\\20180817\\meta')
    
    
    #main(data_dir = 'C:\\Users\\Yiming Zuo\\Desktop\\Arm\\sample_img\\20180819\\real_0817', meta_dir = 'C:\\Users\\Yiming Zuo\\Desktop\\Arm\\sample_img\\meta_20180814')
    client.connect()
    training_data_from_d3_preds('C:/Users/Yiming/Desktop/arm-pose/visualization/20181106/youtube_20181105/d3_pred.json', 'C:/Users/Yiming/Desktop/arm-pose/data/youtube_20181105/imgs', 
    'C:\\Users\\Yiming\\Documents\\OWIMap')
    # data_sampler(1000, False)
    #visualizer((0, 0, 0, 0, 34.37299479, 159.74876758, 4.87261678, -459.43353766, 144.24237348, 335.00689175))
    #im = visualizer((0,0,0,0,30,0,0,426,0,250))
    #io.imsave('cam_1.png', im)
    

    
