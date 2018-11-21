import cv2
import numpy as np
import os

from numpy import sin, cos, pi
import json

def make_rotation(pitch, yaw, roll):
    ryaw = [
        [-cos(yaw), sin(yaw), 0],
        [-sin(yaw), -cos(yaw), 0],
        [0, 0, 1]
    ]
    rpitch = [
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ]
    rroll = [
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ]
    T = np.matrix(ryaw) * np.matrix(rpitch) * np.matrix(rroll)
    return T

def get_A(meta_dir, cam_type):
    with open(os.path.join(meta_dir, 'camera_parameter.json')) as f:
        camera_parameter = json.load(f)
    FocalLength = camera_parameter[cam_type]['FocalLength']
    PrincipalPoint = camera_parameter[cam_type]['PrincipalPoint']

    A = np.matrix(
        [[PrincipalPoint[0], FocalLength[0], 0],
        [PrincipalPoint[1], 0, -FocalLength[1]],
        [1, 0, 0]])  

    return A

def read_img(file_dir):   
    L, F = [], []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':
                L.append(os.path.join(root, file)) 
                F.append(file)
    return L,F 

def create_mask(img, color):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    if color == 'pink':
        threshold = [(0, 144, 114), (255, 183, 139)]
    elif color == 'blue':
        threshold = [(0, 60, 141), (255, 140, 173)]
    else:
        raise Exception('Color undefined')
    
    mask = cv2.inRange(img, threshold[0], threshold[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask =  cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask =  cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = mask>0

    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)

    thres_img = np.zeros_like(img, np.uint8)
    thres_img[mask] = img[mask]

    binary_img = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    binary_img[mask] = 255

    return thres_img, binary_img

def get_object(img, color, threshold):
    detection_result = []
    _, binary_img = create_mask(img, color)

    final_detection = np.zeros_like(binary_img, np.uint8)
    # Perform the operation
    output = cv2.connectedComponentsWithStats(binary_img, 8)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    for i in range(num_labels):
        mask = labels==i
        if stats[i, 4] > threshold and all(binary_img[mask] > 0):
            detection_result.append({'num_pixel':stats[i, 4], 'centroid':centroids[i]})
            final_detection[mask] = binary_img[mask]

    return detection_result, final_detection

def get_d3_object(img, color, threshold, estimation, meta_dir, half_height = 15.0, num_joints = 4, cam_type = 'video'):
    result = []
    d2_object, _ = get_object(img, color, threshold)
    R = make_rotation(estimation[num_joints]*pi/180, estimation[num_joints+1]*pi/180, estimation[num_joints+2]*pi/180).getI()
    A = get_A(meta_dir, cam_type)
    t = -R*np.matrix([estimation[num_joints+3],estimation[num_joints+4],estimation[num_joints+5]]).transpose()
    for i in range(len(d2_object)):
        uv = np.matrix([d2_object[i]['centroid'][0], d2_object[i]['centroid'][1], 1]).transpose()
        left_hand_side = R.getI()*A.getI()*uv
        s = float(((R.getI()*t)[2] + half_height)/left_hand_side[2])
        x = float((s*left_hand_side - R.getI()*t)[0])
        y = float((s*left_hand_side - R.getI()*t)[1])
        result.append([x,y,half_height])

    return result
    
if __name__ == '__main__':
    file_dir = './'
    color = 'pink'
    L, _ = read_img(file_dir)
    for img_dir in L:
        #print(img_dir)
        img = cv2.imread(img_dir)
        detection_result = get_d3_object(img, 'pink', 100, (0,0,0,0,30,0,0,426,0,250), 'C:\\Users\\Yiming Zuo\\Desktop\\Arm\\sample_img\\meta_20180814', cam_type = 'synthetic')



