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

def data_sampler(num_data, data_dir):
    ###############################################
    #for regular data generation
    ###############################################
    '''num_data: how many images to generate'''
    '''data_dir: where to place the generated data'''
    print(data_dir)

    dist = 600 #distance from cam to origin

    client.request('vset /camera/1/fov 90')
    sample_id = 0

    while True:
        if sample_id >= num_data:
            break

        #random texture and lightning

        texture_filename = os.path.abspath('../background_img/000000000001.jpg')
        client.request('vset /env/floor/texture %s' % texture_filename)
        client.request('vset /env/sky/texture %s' % texture_filename)
        client.request('vset /env/random_lighting')

        rotation = random.randint(-130,130)
        base = random.randint(-90, 60)
        elbow = random.randint(-60, 90)
        wrist = random.randint(-45, 45)
        gripper = 0
        client.request('vset /arm/owi535/pose {rotation} {base} {elbow} {wrist} {gripper}'.format(**locals()))

        pitch = -random.randint(10,60)
        yaw = random.randint(0,360)
        roll = 0
        client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**locals()))

        x = int(-dist*cos(pitch*pi/180)*cos(yaw*pi/180))
        y = int(-dist*cos(pitch*pi/180)*sin(yaw*pi/180))
        z = int(-dist*sin(pitch*pi/180))
        client.request('vset /camera/1/location {x} {y} {z}'.format(**locals()))

        # get tip pose, ensuring the tip not underground 
        tip_z = float(client.request('vget /arm/owi535/tip_pose').split()[2])
        if tip_z > 0:

            client.request('vset /data_capture/capture_frame ' + data_dir)
            print("#{sample_id}: {rotation}, {base}, {elbow}，{wrist}".format(**locals()))
            time.sleep(0.1)

            sample_id+=1

if __name__ == "__main__":

    '''
    please change the data_dir to the folder where you want captured data to be saved
    '''

    data_dir = "../data/new_data"
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        print('creating path: ' + data_dir)
        os.mkdir(data_dir)

    client.connect()
    data_sampler(100, data_dir)