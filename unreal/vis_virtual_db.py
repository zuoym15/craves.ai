# Visualize a virtual dataset generated from unrealcv

import virtual_db as vdb
import imageio as io
import matplotlib.pyplot as plt
import numpy as np

def test_frame_data():
    db_root_dir = '/mnt/c/qiuwch/ue4_project/CarAct/TestHuman'
    camera_name = 'FusionCameraActor_1'
    dataset = vdb.Dataset(db_root_dir)
    ids = dataset.get_ids() # The frame ids we have.
    # print(ids)
    # print(cams)
    # cams = dataset.get_cams()
    camera_name = 'FusionCameraActor_1'
    cams = [camera_name]

    images = dataset.get_image(cams, ids)
    segmask = dataset.get_segmask(cams, ids)
    depth = dataset.get_depth(cams, ids)
    annotation_color = dataset.get_annotation_color()

    video_filename = 'test.mp4'
    vdb.compress_video(video_filename, images)

    plt.imshow(io.imread(images[0]))
    plt.show()

def test_joint_data():
    db_root_dir = '/mnt/c/qiuwch/ue4_project/CarAct/TestHuman'
    # db_root_dir = '/mnt/c/qiuwch/ue4_project/CarAct/OWIMap'

    # cams = dataset.get_cams()
    # print(cams)
    cams = ['FusionCameraActor_1']
    # cams = ['FusionCameraActor2']

    dataset = vdb.Dataset(db_root_dir)

    ids = dataset.get_ids() # The frame ids we have.
    print('Number of images %d' % len(ids))
    ids = [ids[0]]

    joint_name = vdb.meta.skel1
    print(joint_name)

    joint = dataset.get_d3_skeleton(ids)
    joint = joint[0]['Trooper_Full_NoKnife_2']['WorldJoints']
    joint = dict({k.lower(): v for (k, v) in joint.items()}) # Convert key to lower case

    # valid_name = [k for (k, v) in joint.items() if k in joint_name]
    # print('Valid joint names: ' + str(valid_name))

    # world_space = np.array([(v['X'], v['Y'], v['Z']) for (k, v) in joint.items() if k in joint_name])
    valid_joint = []
    valid_flag = []
    for k in joint_name:
        v = joint.get(k)
        if v is None:
            valid_joint.append([0, 0, 0])
            valid_flag.append(False)
        else:
            valid_joint.append([v['X'], v['Y'], v['Z']])
            valid_flag.append(True)
        
    world_space = np.array(valid_joint)
    print(world_space.shape)
    
    # filtered_joint = {(k,v) for (k,v) in joint.items() if k in joint_name}
    # print(joint)
    # print(filtered_joint)
    print(world_space)
    points3d = world_space

    cam_info = dataset.get_cam_info(cams, ids)
    cam_info = cam_info[0]
    loc = cam_info['Location']
    rot = cam_info['Rotation']

    cam = vdb.d3.CameraPose(loc['X'], loc['Y'], loc['Z'], 
        rot['Pitch'], rot['Yaw'], rot['Roll'], 
        cam_info['FilmWidth'], cam_info['FilmHeight'], cam_info['FilmWidth']/2)
    print(cam_info)
    points2d = cam.project_to_2d(points3d)
    print(points2d)


    images = dataset.get_image(cams, ids)
    plt.imshow(io.imread(images[0]))
    plt.plot(points2d[:,0], points2d[:,1], '*')
    for edge in vdb.meta.skel_edge:
        if valid_flag[edge[0]] and valid_flag[edge[1]]:
            plt.plot(points2d[edge,0], points2d[edge,1], 'r-')

    plt.show()

if __name__ == '__main__':
    # test_frame_data()
    test_joint_data()
