import json
import numpy as np
from numpy import sin, cos
import cv2
import math
from unrealcv import client
from unrealcv.util import read_png, read_npy

def corners_reorder(img, corners, num_x, num_y):  # reorder corners detected by get_extrinsic_parameters() so that the order is consistent among frames
    last_starting = np.array([1280,720])
    #last_starting = np.array([0,720])

    mask = np.zeros_like(img, dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(corners[[0, 6, 48, 42],:]), (255, ))
    avg_color = np.average(img[mask!=0]) #take the average color of chessborad as threshold

    sample1 = (corners[0] + corners[8]) / 2
    sample2 = (corners[48] + corners[40]) / 2

    if img[int(sample1[1]), int(sample1[0])] > avg_color and img[int(sample2[1]), int(sample2[0])] > avg_color:  # white
        corners = np.reshape(corners, (num_x, num_y, 2))
        for x in range(num_x):
            corners[x, :, :] = corners[x, ::-1, :]
        corners = np.reshape(corners, (num_x * num_y, 2))

    # if corners[0,0] > corners[-1,0]: #use the left black square as first square
    #     corners = corners[::-1,:]

    if np.linalg.norm(corners[0] - last_starting) > np.linalg.norm(corners[-1] - last_starting):
        corners = corners[::-1, :]

    #last_starting[:] = corners[0]

    if np.cross(corners[0] - corners[-1], corners[0] - corners[num_x - 1]) < 0:  # scanning every row then column
        corners = np.reshape(corners, (num_x, num_y, 2), order='F')
        corners = np.reshape(corners, (num_x * num_y, 2))

    return corners


def get_extrinsic_parameters(img, num_x, num_y, A, grid_size = 1.0):  # caliberate camera
    # img should be in bgr format

    objp = np.zeros((num_x * num_y, 3), np.float32)
    objp[:, :2] = grid_size * np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)
    if not ret:
        raise RuntimeError('fail to find chessboard!')

    corners = np.squeeze(corners)
    corners = corners_reorder(gray, corners, num_x, num_y)
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, A, None)

    if not ret:
        raise RuntimeError('fail to get camera extrinsic parameters!')

    return rvecs, tvecs, corners

def retrieve_eular_angles(rvec, tvec):
    R = np.zeros((3, 3))
    cv2.Rodrigues(rvec, R)
    R = np.matrix(R)

    #print(R)

    cam_pos = -R.getI() * np.matrix(tvec)
    euler_angles_radians = rotation_matrix_to_attitude_angles(R)
    euler_angles_degrees = 180 * euler_angles_radians/np.pi

    return euler_angles_degrees, cam_pos, R

def rotation_matrix_to_attitude_angles(R):

    R = np.matrix([[0,0,1],[1,0,0],[0,-1,0]]) * R
    R = R.T
    
    print(R)
    cos_beta = math.sqrt(R[2,1] * R[2,1] + R[2,2] * R[2,2])
    validity = cos_beta < 1e-6
    if not validity:
        alpha = -math.atan2(R[1,0], R[0,0])    # yaw   [z]
        beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
        gamma = math.atan2(R[2,1], R[2,2])    # roll  [x]
    else:
        alpha = -math.atan2(R[1,0], R[0,0])    # yaw   [z]
        beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
        gamma = 0                             # roll  [x]

    print(make_rotation(beta, alpha , gamma))

    #print(np.array())

    print(np.array([beta, alpha, gamma])*180/np.pi)

    # return np.array([-beta, alpha, -gamma])

    # return np.array([alpha, beta, gamma])

    return np.array([-beta, alpha + np.pi/2, -gamma])

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
    print(res)

    res = client.request('vset /camera/1/location {x} {y} {z}'.format(**locals()))
    print(res)

    #pitch = -pitch
    #yaw = yaw-180
    #roll = -roll
    res = client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**locals()))
    print(res)

    print('{rotation} {base} {elbow} {wrist} {gripper} {x} {y} {z} {pitch} {yaw} {roll}'.format(**locals()))

    client.request('vset /data_capture/capture_frame')

def make_rotation(pitch, yaw, roll):
    # Convert from degree to radius
    # pitch = pitch / 180.0 * np.pi
    # yaw = yaw / 180.0 * np.pi
    # roll = roll / 180.0 * np.pi
    # pitch = -pitch
    # yaw = yaw # ???!!!
    # roll = -roll # Seems UE4 rotation direction is different
    # from: http://planning.cs.uiuc.edu/node102.html
    ryaw = [
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    rpitch = [
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ]
    rroll = [
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ]
    T = np.matrix(ryaw) * np.matrix(rpitch) * np.matrix(rroll)
    return T

def main():
    grid_size = 15.2 #15mm per grid

    with open("./camera_parameters.json", 'r') as f:
        A = np.matrix(json.load(f))

        A[0, 2] = 640

    # client.connect()

    name = 'batch_009'

    for i in range(1):

        img = cv2.imread('./left/imgs/'+name+'.jpg')

        rvecs, tvecs, corners = get_extrinsic_parameters(img, 7, 7, A, grid_size)

        cv2.drawChessboardCorners(img, (7,7), corners, True)

        euler_angles_degrees, cam_pos ,R = retrieve_eular_angles(rvecs, tvecs)

        #origin_trans = [10, 60, 5] #right
        origin_trans = [10-120-82, 60, 5] #left

        #visualizer([0,0,0,0, euler_angles_degrees[0], euler_angles_degrees[1], euler_angles_degrees[2], cam_pos[1,0] + origin_trans[0], cam_pos[0,0] + origin_trans[1], cam_pos[2,0] + origin_trans[2]])
        #visualizer([0,0,0,0, -euler_angles_degrees[2], euler_angles_degrees[1] + 180, -euler_angles_degrees[0], -cam_pos[0,0]-100, cam_pos[1,0]+60, cam_pos[2,0] + 5])
        # img_syn = cv2.imread('C:/Users/Yiming/Documents/OWIMap/FusionCameraActor3_2/lit/'+str(i).zfill(8)+'.png')

        # transform = [A[0,2] - 640, 0]
        # M = np.float32([[1,0,transform[0]],[0,1,transform[1]]])

        # rows,cols = img.shape[0:2]

        # img_syn = cv2.warpAffine(img_syn,M,(cols,rows))

        # img = (img/2+img_syn[:,:,0:3]/2).astype(np.uint8)

        cv2.imshow('img', img)
        cv2.waitKey(0)

        cam_parameters = {"cam_controller":{
            "location":{"x":float(cam_pos[1,0] + origin_trans[0]), "y":float(cam_pos[0,0] + origin_trans[1]), "z":float(cam_pos[2,0] + origin_trans[2])}, 
            "rotation":{"pitch":float(euler_angles_degrees[0]), "yaw":float(euler_angles_degrees[1]), "roll":float(euler_angles_degrees[2])}, 
            "fov":66.5}}

        with open('./left/annos/'+name+'.json', 'w') as f:
            json.dump(cam_parameters, f)
        
if __name__ == "__main__":
    main()

    

