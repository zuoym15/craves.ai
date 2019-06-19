import json
import os
import numpy as np
from numpy import pi, sin, cos
import random
from scipy.optimize import root, least_squares
import cv2

def read_json(file_dir):   
    L, F = [], []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file)) 
                F.append(file)
    return L, F

def cam_est(ang, scale = 500): #calculate the position of the camera given rotation
    cam_est = scale*np.array([cos(ang[0])*cos(ang[1]), cos(ang[0])*sin(ang[1]), sin(ang[0])])
    return cam_est

def make_rotation(pitch, yaw, roll): #return a the rotation matrix given roatation angles
    # Convert from degree to radius
    #pitch = pitch / 180.0 * np.pi
    #yaw = yaw / 180.0 * np.pi
    #roll = roll / 180.0 * np.pi
    pitch = pitch
    yaw = yaw # ???!!!
    roll = roll # Seems UE4 rotation direction is different
    # from: http://planning.cs.uiuc.edu/node102.html
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

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian 
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def Opt(x,num_joints,A,ang,cam,c,d,uv,meta,estimate_cam, estimate_intrinsic, Reprojection): #The Optimization function. Plz refer to the paper for loss defination. 
    if estimate_intrinsic:
        A = np.matrix(
            [[x[num_joints+6], x[num_joints+8], 0],
                [x[num_joints+7], 0, -x[num_joints+8]],
                [1, 0, 0]]) 

    num_thetas = num_joints #first num_theta element of x correspond to theta
    if estimate_cam:
        R = make_rotation(x[num_joints], x[num_joints+1], x[num_joints+2]).getI()
        t = -R*np.matrix([x[num_joints+3],x[num_joints+4],x[num_joints+5]]).transpose()
    else:
        R = make_rotation(ang[0], ang[1], ang[2]).getI()
        t = -R*cam
    num_keypoints = d.shape[1]
    mat_t = np.matrix(np.zeros((3, num_keypoints)))
    mat_s = np.matrix(np.zeros((num_keypoints, num_keypoints)))
    s = np.zeros(num_keypoints)
    W = np.matrix(np.zeros((3, num_keypoints)))

    for i in range(num_keypoints):
        parent = int(meta[str(i)]['parent']) 
        
        if parent == -1:
            mat_t[:,i] = t
            W[:,i] = c + d[:,i]
        elif parent == 0: 
            R_joint = np.matrix([
            [cos(x[parent+1])*cos(x[0]), -sin(x[0]), -sin(x[parent+1])*cos(x[0])],
            [cos(x[parent+1])*sin(x[0]), cos(x[0]) , -sin(x[parent+1])*sin(x[0])],
            [sin(x[parent+1])          , 0         ,  cos(x[parent+1])          ]
            ])
            mat_t[:,i] = t
            W[:,i] = c + R_joint*d[:,i]
        else:
            R_joint = np.matrix([
            [cos(x[parent+1])*cos(x[0]), -sin(x[0]), -sin(x[parent+1])*cos(x[0])],
            [cos(x[parent+1])*sin(x[0]), cos(x[0]) , -sin(x[parent+1])*sin(x[0])],
            [sin(x[parent+1])          , 0         ,  cos(x[parent+1])          ]
            ])
            W[:,i] = R_joint*d[:,i]

    right_hand_side = A * R * W + A * mat_t

    for i in range(num_keypoints):
        parent = int(meta[str(i)]['parent']) 
        if parent == 0 or parent == -1: 
            s[i] = right_hand_side[2, i]
        else:
            s[i] = right_hand_side[2, i]
            s[i] += s[parent]

    for i in range(num_keypoints):
        parent = int(meta[str(i)]['parent']) 
        if parent == 0 or parent == -1: 
            mat_s[i,i] = s[i]
        else:
            mat_s[parent, i] = -s[parent]
            mat_s[i,i] = s[i]

    loss = uv[0:2, :] * mat_s - right_hand_side[0:2, :]
    Reprojection[:,:right_hand_side.shape[1]] = right_hand_side[0:2, :]*mat_s.getI()
    loss = np.ravel(loss)
    return loss

def estimate(x0, cam, ang, uv, estimate_cam, estimate_intrinsic, cam_type, Reprojection, valid_keypoint_list, meta_dir): #Read the data for a frame and solve Opt() with lm alogrithm.
    with open(os.path.join(meta_dir, 'skeleton.json')) as f:
        meta = json.load(f)

    num_joints = meta['num_joints']

    if valid_keypoint_list is None:
        valid_keypoint_list = list(range(len(meta)-1))

    valid_meta = {'num_joints': meta['num_joints']}
    for i in range(len(valid_keypoint_list)):
        valid_meta[str(i)] = meta[str(valid_keypoint_list[i])]

    valid_uv = uv[:, valid_keypoint_list]

    c = np.matrix(meta['0']['offset']).transpose()

    #rotating joints
    num_keypoints = len(meta)-1 
    d = np.matrix([0, 0, 0]).transpose()
    for i in range(1, num_keypoints):
        d = np.concatenate((d, np.matrix(meta[str(i)]['offset']).transpose()), axis = 1)

    valid_num_keypoints = len(valid_meta)-1 
    valid_d = np.matrix([0, 0, 0]).transpose()
    for i in range(1, valid_num_keypoints):
        valid_d = np.concatenate((valid_d, np.matrix(valid_meta[str(i)]['offset']).transpose()), axis = 1)

    if not estimate_intrinsic:
        with open(os.path.join(meta_dir, 'camera_parameter.json')) as f:
            camera_parameter = json.load(f)
        FocalLength = camera_parameter[cam_type]['FocalLength']
        PrincipalPoint = camera_parameter[cam_type]['PrincipalPoint']

        A = np.matrix(
            [[PrincipalPoint[0], FocalLength[0], 0],
                [PrincipalPoint[1], 0, -FocalLength[1]],
                [1, 0, 0]]) 

    else:
        A = None  

    res = least_squares(Opt, x0, args = (num_joints, A, ang, cam, c, valid_d, valid_uv, valid_meta, estimate_cam, estimate_intrinsic, Reprojection), method = 'lm', max_nfev = 500)

    avg_error = np.average(np.sqrt(np.sum(np.power(valid_uv - Reprojection[:,:valid_uv.shape[1]], 2), axis = 0)))

    _ = Opt(res.x, num_joints, A, ang, cam, c, d, uv, meta, estimate_cam, estimate_intrinsic, Reprojection) #Get the reprojection result of the full keypoint list

    return res, Reprojection, avg_error

def heatmap_vis(heatmap): #for visualization only
    for i in range(len(heatmap)):
        vis = (heatmap[i] * 255).astype(np.uint8)

        cv2.imshow('img', vis)
        cv2.waitKey(0)
 
def uv_from_heatmap(heatmap, labelmap = None): #calculate the max responding position from a numpy matrix
    if labelmap is not None:
        heatmap = np.multiply(heatmap, labelmap)

    score = np.amax(heatmap, axis=(1,2))

    # for i in range(heatmap.shape[0]):
    #     print('joint{}:{}'.format(i, len(np.nonzero(heatmap[i] > 0.7 * score[i])[0])))

    h, w = heatmap.shape[1], heatmap.shape[2]

    heatmap = np.reshape(heatmap, (heatmap.shape[0], h * w))

    uv = np.matrix(np.unravel_index(np.argmax(heatmap, axis=1), (h, w)))[[1,0], :]

    return uv, score

def get_pred(data_dir, filename, pred_from_heatmap): #read the 2d keypoint prediction from files
    #shape of uv: 2 * 17
    pred_dir = os.path.join(data_dir, 'preds', filename)
    with open(pred_dir,'r') as f:
            obj = json.load(f)

    if obj.__contains__('cam_info'):
        cam_info = obj['cam_info']
        cam = np.matrix(cam_info[0:3]).transpose()
        ang = np.array(cam_info[3:6])*np.pi/180
    else: 
        cam = None
        ang = None

    heatmap = None

    if pred_from_heatmap:
        body = os.path.splitext(filename)[0]
        heatmap_dir = os.path.join(data_dir, 'heatmaps', body + '.npy')
        heatmap = np.load(heatmap_dir) #C * H * W

        uv, score = uv_from_heatmap(heatmap)

    else:
        uv = np.matrix(obj['d2_key']).transpose()

        if obj.__contains__('score'):
            score = np.array(obj['score']) #load confident score when testing on real imgs
        else:
            score = np.ones(uv.shape[1])

    #print(uv)

    #print(np.matrix(obj['d2_key']).transpose())

    return uv, score, cam, ang, heatmap

def d2tod3(data_dir, meta_dir, estimate_cam = True, estimate_intrinsic = False, num_joints = 4, cam_type = 'video', keypoint_list = list(range(17)), init = None, pred_from_heatmap = False, em_test = False):
    num_hit = []
    
    thres = 2 if cam_type == 'synthetic' else 20 #threshold for ending loop

    result_save_dir = os.path.join(data_dir, 'd3_preds')
    if not os.path.isdir(result_save_dir):
        os.mkdir(result_save_dir)
        
    pred_dir = os.path.join(data_dir, 'preds')
    file_dir_list, file_name_list = read_json(pred_dir)
    num_sample = len(file_dir_list)

    d3_pred = {}

    x = np.zeros(num_joints + 6)

    for i in range(0, num_sample):
       
        min_error = -1
        hit = False
        # with open(file_dir_list[i],'r') as f:
        #     obj = json.load(f)

        # uv = np.matrix(obj['d2_key']).transpose()

        # if obj.__contains__('score'):
        #     score = np.array(obj['score']) #load confident score when testing on real imgs
        # else:
        #     score = np.ones(uv.shape[1])

        # if obj.__contains__('cam_info'):
        #     cam_info = obj['cam_info']
        #     cam = np.matrix(cam_info[0:3]).transpose()
        #     ang = np.array(cam_info[3:6])*np.pi/180
        # else: 
        #     cam = None
        #     ang = None

        

        uv, score, cam, ang, heatmap = get_pred(data_dir, file_name_list[i], pred_from_heatmap)

        valid_keypoint_list = []

        kp_number = 15 #use the first 15 keypoints for 3d reconstruction

        # for id in score.argsort()[-kp_number:][::-1]:
        #     valid_keypoint_list.append(id)

        for j in range(uv.shape[1]):
            if score[j] > 0.15 and j in keypoint_list:
                valid_keypoint_list.append(j)

        print("testing: {}".format(file_name_list[i]))
        # print("valid key point number: {}".format(len(valid_keypoint_list)))

        Reprojection = np.zeros((2,uv.shape[1]))

        if len(valid_keypoint_list) < 12:
            # print("key point not enough!")
            x_deg = np.zeros(num_joints+6)
            min_error = 100
            d3_pred[file_name_list[i]] = {'preds':x_deg.tolist(), 'error':min_error, 'num_valid_key':len(valid_keypoint_list), 'x_raw':x_deg.tolist()}
            with open(os.path.join(result_save_dir, file_name_list[i]), 'w') as f:
                json.dump({'preds':x_deg.tolist(), 'error':min_error, 'num_valid_key':len(valid_keypoint_list), 'reprojection':Reprojection.tolist()}, f)

            continue
            #x = np.zeros(num_joints+6)
            #min_error = -1

        if em_test:
            em_steps = 5
        else:
            em_steps = 1

        for em_step in range(em_steps):
            # print(x)
            last_x = x

            if em_step != 0: #calculate new uv based on last reprojection result
                #valid_keypoint_list = keypoint_list #use all heatmaps
                

                labelmap = np.zeros_like(heatmap)
                for channel in range(len(labelmap)):
                    draw_labelmap(labelmap[channel], Reprojection[:, channel], sigma = 3)

                uv, _ = uv_from_heatmap(heatmap, labelmap)

            for j in range(10):
                
                if i != 0 or em_step != 0:#try use the last frame prediction as initial value
                    x0 = x
                    res, Reprojection, avg_error = estimate(x0, cam, ang, uv, estimate_cam, estimate_intrinsic,cam_type, Reprojection, valid_keypoint_list, meta_dir)
                    if avg_error < thres:
                        # print('estimation hit!')
                        min_error = avg_error
                        x = res.x
                        break

                if init is not None and j == 0:
                    x0 = init
                    res, Reprojection, avg_error = estimate(x0, cam, ang, uv, estimate_cam, estimate_intrinsic,cam_type, Reprojection, valid_keypoint_list, meta_dir)
                    if avg_error < thres:
                        # print('estimation hit!')
                        min_error = avg_error
                        x = res.x
                        break

                if estimate_intrinsic:
                    x0 = np.random.rand(num_joints+9)
                    if num_joints == 4:
                        x0[0] = -x0[0]*180
                        x0[1] = x0[1]*180
                        x0[2:num_joints] = -90+x0[2:num_joints]*180
                    x0[num_joints] = 70
                    #x0[num_joints+1] = 360*x0[num_joints+1]
                    x0[num_joints+1] = -30
                    x0[num_joints+2] = 0
                    x0 = x0*np.pi/180
                    x0[num_joints+3:num_joints+6] = cam_est(x0[num_joints:num_joints+3], 700)
                    x0[num_joints+6:num_joints+9] = np.array([983,521,1453])
                    #print(x0)
                    #exit(0)

                elif estimate_cam:
                    x0 = np.random.rand(num_joints+6)
                    if num_joints == 4:
                        x0[0] = -x0[0]*180
                        x0[1] = x0[1]*180
                        x0[2:num_joints] = -90+x0[2:num_joints]*180
                    x0[num_joints] = random.randint(10,60)
                    #x0[num_joints+1] = 360*x0[num_joints+1]
                    x0[num_joints+1] = 360*random.random()
                    x0[num_joints+2] = 0
                    x0 = x0*np.pi/180
                    x0[num_joints+3:] = cam_est(x0[num_joints:num_joints+3], 500+500*random.random())
                    
                else:
                    x0 = np.zeros(num_joints)
                    x0 = np.random.rand(num_joints)
                    x0[0] = -x0[0]*180
                    x0[1] = x0[1]*180
                    x0[2:] = -90+x0[2:]*180
                    x0 = x0*np.pi/180

                #print(x0)

                res, Reprojection, avg_error = estimate(x0, cam, ang, uv, estimate_cam, estimate_intrinsic,cam_type, Reprojection, valid_keypoint_list, meta_dir)
            
                if j == 0 or avg_error < min_error:
                    min_error = avg_error
                    x = res.x
                if min_error < thres:
                    break

            if np.linalg.norm(x - last_x) < 1:
                break


        #print(avg_error)
        if(min_error < thres and len(valid_keypoint_list)>=12):
            hit = True
        
        num_hit.append(hit)

        x_deg = x.copy()
        x_deg[0:7] = x[0:7]*180/pi
        x_deg[0] = x_deg[0] + 90
        x_deg[1] = x_deg[1] - 90
        x_deg[3] = x_deg[3] - x_deg[2]
        x_deg[2] = x_deg[2] - x_deg[1]
        # print(x_deg) 
        print('{}:{}, error:{}'.format(i+1,hit,min_error))

        d3_pred[file_name_list[i]] = {'preds':x_deg.tolist(), 'error':min_error, 'num_valid_key':len(valid_keypoint_list), 'x_raw':x.tolist()}
        with open(os.path.join(result_save_dir, file_name_list[i]), 'w') as f:
            json.dump({'preds':x_deg.tolist(), 'error':min_error, 'num_valid_key':len(valid_keypoint_list), 'reprojection':Reprojection.tolist()}, f)


    #print('acc:{}'.format(sum(num_hit)/len(num_hit)))

    with open(os.path.join(data_dir, 'd3_pred.json'), 'w') as f:
        json.dump({'hit':num_hit, 'd3_pred':d3_pred, 'file_name_list':file_name_list}, f)

    return num_hit, d3_pred, file_name_list
