# Weichao Qiu @ 2018
# Visualize arm dataset

import unreal.virtual_db as vdb
import imageio as io
import matplotlib.pyplot as plt
import numpy as np

joint_name = vdb.meta.arm_skel

def fov2f(fov, width):
    # fov = 2 * arctan(w / 2f)
    fov = fov * np.pi / 180
    return width / (2 * np.tan(fov / 2))

def f2fov(f, width):
    # fov = 2 * arctan(w / 2f)
    fov = 2 * np.arctan(width / (2 * f))
    fov = fov * 180 / np.pi
    return fov

def get_vertex_2d_raw(dataset, ids, cam, actor_name):
    vertexs = dataset.get_d3_vertex(ids)
    #actor_name = "OWI-arm-baked-pivot_2"; 
    if isinstance(vertexs[0], dict): #fit to the new data generation tool
        vertex = vertexs[0][actor_name] 
    else:
        vertex = vertexs[0]
    points3d = np.array([[v['X'], v['Y'], v['Z']] for v in vertex])
    points2d = cam.project_to_2d(points3d)
    return points2d

def get_vertex_2d(dataset, frame_id, cam_name, actor_name):
    cams = [cam_name]
    ids = [frame_id]

    images = dataset.get_image(cams, ids); image_dir = images[0]
    cams_info = dataset.get_cam_info(cams, ids); cam_info = cams_info[0]

    loc = cam_info['Location']
    rot = cam_info['Rotation']
    cam = vdb.d3.CameraPose(loc['X'], loc['Y'], loc['Z'], 
        rot['Pitch'], rot['Yaw'], rot['Roll'], 
        cam_info['FilmWidth'], cam_info['FilmHeight'], cam_info['FilmWidth']/2)

    points2d = get_vertex_2d_raw(dataset, ids, cam, actor_name)
    return points2d, image_dir

def get_joint_2d_raw(dataset, ids, cam, actor_name):
    #joint_name = vdb.meta.arm_skel
    joints = dataset.get_d3_skeleton(ids)
    #actor_name = "OWI-arm-baked-pivot_2"; 
    joint = joints[0][actor_name]['WorldJoints'] # This usage might change
    # joint = dict({k.lower(): v for (k, v) in joint.items()}) # Convert key to lower case
    # Convert to np array
    points3d = np.array([[joint[v]['X'], joint[v]['Y'], joint[v]['Z']] for v in joint_name])
    points2d = cam.project_to_2d(points3d)
    return points2d

def get_joint_2d(dataset, frame_id, cam_name, actor_name):
    ids = [frame_id]
    cams = [cam_name]

    images = dataset.get_image(cams, ids); image_dir = images[0]

    # Project to 2D joint 
    # ---------------------------------------------------------------------
    cams_info = dataset.get_cam_info(cams, ids); cam_info = cams_info[0]

    loc = cam_info['Location']
    rot = cam_info['Rotation']
    cam = vdb.d3.CameraPose(loc['X'], loc['Y'], loc['Z'], 
        rot['Pitch'], rot['Yaw'], rot['Roll'], 
        cam_info['FilmWidth'], cam_info['FilmHeight'], cam_info['FilmWidth']/2)

    #print(points3d)
    points2d = get_joint_2d_raw(dataset, ids, cam, actor_name)

    return points2d, image_dir

def get_joint_vertex_2d(dataset, frame_id, cam_name, actor_name):
    ids = [frame_id]
    cams = [cam_name]

    images = dataset.get_image(cams, ids); image_dir = images[0]

    # Project to 2D joint 
    # ---------------------------------------------------------------------
    cams_info = dataset.get_cam_info(cams, ids); cam_info = cams_info[0]

    if cam_info.__contains__('Fov'):
        f = fov2f(cam_info['Fov'], cam_info['FilmWidth'])
    else:
        f = fov2f(90, cam_info['FilmWidth'])

    loc = cam_info['Location']
    rot = cam_info['Rotation']
    cam = vdb.d3.CameraPose(loc['X'], loc['Y'], loc['Z'], 
        rot['Pitch'], rot['Yaw'], rot['Roll'], 
        cam_info['FilmWidth'], cam_info['FilmHeight'], f)

    joint_2d = get_joint_2d_raw(dataset, ids, cam, actor_name)
    vertex_2d = get_vertex_2d_raw(dataset, ids, cam, actor_name)
    return joint_2d, vertex_2d, image_dir

def show_frame(dataset, frame_id, cam_name, actor_name):
    ids = [frame_id]
    cams = [cam_name]

    points2d, _ = get_joint_2d(dataset, frame_id, cam_name, actor_name)

    # Visualize 2D joint
    # ---------------------------------------------------------------------
    images = dataset.get_image(cams, ids); image = images[0]

    plt.subplot(121)
    plt.imshow(io.imread(image))
    plt.plot(points2d[:,0], points2d[:,1], '*')
    for edge in vdb.meta.arm_skel_edge:
        plt.plot(points2d[edge,0], points2d[edge,1], 'r-')

    # Load 3D vertex data and project to 2d
    # ---------------------------------------------------------------------
    points2d = get_vertex_2d(dataset, frame_id, cam_name, actor_name)

    plt.subplot(122)
    plt.imshow(io.imread(image))
    plt.plot(points2d[:,0], points2d[:,1], '*')
    
    # Show plot
    plt.show()

def main():
    db_root_dir = '/mnt/c/qiuwch/ue4_project/CarAct/OWIMap'

    cams = ['FusionCameraActor_1', 'FusionCameraActor2',
        'FusionCameraActor3', 'FusionCameraActor4']

    dataset = vdb.Dataset(db_root_dir)
    ids = dataset.get_ids() # The frame ids we have.
    print('Number of images %d' % len(ids))

    show_frame(dataset, ids[0], cams[3])


if __name__ == '__main__':
    main()
