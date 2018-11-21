# Weichao Qiu @ 2018
# Project 3D human keypoints to 2D using camera information
# And also preserve useful information, throw away not in use kps
import glob, os, json
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from diva.projection import CamPose
from diva.util import read_jsonlist

class CocoDatasetBuilder:
    def __init__(self):
        self.person_category_id = 1
        self.kp_id = 0

        person_category = {
                "supercategory": "person",
                "id": self.person_category_id,
                "name": "person",
                "keypoints": [
                    "nose",
                    "left_eye", "right_eye",
                    "left_ear", "right_ear", 
                    "left_shoulder", "right_shoulder", 
                    "left_elbow", "right_elbow", 
                    "left_wrist", "right_wrist", 
                    "left_hip", "right_hip",
                    "left_knee", "right_knee",
                    "left_ankle", "right_ankle"
                  ],
                  "skeleton": [
                      [16,14],[14,12],[17,15],[15,13],[12,13],
                      [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],
                      [9,11],[2,3],[1,2],[1,3],[2,4],
                      [3,5],[4,6],[5,7]
                      ]
                }

        data = dict()
        data['info'] = {
                "description": "DIVA human dataset",
                "url": "weichaoqiu.com",
                "version": "0.1",
                "contributor": "Weichao Qiu",
                "year": "2018",
                "date_created": "2018/05/28",
                }
        data['licenses'] = []
        data['categories'] = [person_category]
        data['images'] = []
        data['annotations'] = []
        self.data = data

    def add(self, frame_id, im_filename, sensor_filename, kp_filename):
        # Make this API stable accross file format change.

        sensors = self.parse_sensor_file(sensor_filename)
        humans = self.parse_3d_kps(kp_filename) 

        h = humans[0] 
        sensor = sensors[0]

        coco_image = CocoImage(frame_id, im_filename)

        self.data['images'].append(coco_image.tojson())
        for points_3d in humans:
            points_2d = sensor.project_to_2d(points_3d[:,0:3])
            vis_flag =  points_3d[:,3] 

            # print(vis_flag)

            points_2d = np.concatenate((points_2d, vis_flag[:,np.newaxis]), axis = 1)
            keypoints = points_2d.flatten().tolist()

            coco_kp = CocoKp(self.kp_id, frame_id, self.person_category_id, keypoints)
            self.kp_id += 1
            self.data['annotations'].append(coco_kp.tojson())

    def save(self, json_filename):
        with open(json_filename, 'w') as f:
            json.dump(self.data, f, indent = 2)

    def parse_3d_kps(self, kp_filename):
        data = read_jsonlist(kp_filename)

        humans = []
        # alison_eric_bones = [
        #         "foot_r", "lowerleg_r",
        #         "upperleg_r", "upperleg_l",
        #         "lowerleg_l", "foot_l",
        #         "hip", "spine_02",
        #         "head", "head_end",
        #         "hand_r", "lowerarm_r",
        #         "upperarm_r", "upperarm_l",
        #         "lowerarm_l", "hand_l"
        #     ]
        alison_eric_bones = [
                "nose", 
                "left_eye", "right_eye", 
                "left_ear", "right_ear",
                "upperarm_l", "upperarm_r", 
                "lowerarm_l", "lowerarm_r",
                "hand_l", "hand_r", 
                "upperleg_l", "upperleg_r", 
                "lowerleg_l", "lowerleg_r",
                "foot_l", "foot_r", 
                # "hip", "spine_02",
                # "head", "head_end",
            ]
        bone_mapping = {
            'alison': alison_eric_bones,
            'eric': alison_eric_bones,
            }
        for v in data:
            actor_name = v['ActorName']
            world_joints = v['WorldJoints']
            # if not actor_name.startswith('alison') or actor_name.startswith('eric'):
            #     return
            required_bones = None
            all_bones = world_joints.keys()
            for model_type, model_required_bones in bone_mapping.items():
                # TODO: this check is invalid due to problems in the data, so use another slow check instead
                # if actor_name.startswith(model_type):
                #     required_bones = model_required_bones
                #     break

                # if all([v in all_bones for v in model_required_bones]):
                #     required_bones = model_required_bones
                #     break

                required_bones = model_required_bones
                break

            if not required_bones: 
                print('Actor {actor_name} is not supported yet, skip'.format(**locals()))
                continue
            else:
                # print('Parsing actor {actor_name}'.format(**locals()))
                pass

            N = len(required_bones)
            points = np.zeros((N, 4))
            for i, bone_name in enumerate(required_bones):
                bone_data = world_joints.get(bone_name)
                if bone_data is None:
                    # print('Bone {bone_name} can not be found'.format(**locals()))
                    vis_flag = 0
                    X, Y, Z = 0, 0, 0
                else:
                    vis_flag = 2
                    X, Y, Z = bone_data['X'], bone_data['Y'], bone_data['Z']
                points[i,0] = X; points[i,1] = Y; points[i,2] = Z; points[i,3] = vis_flag

            # [x2d, y2d] = project_to_2d(points, camera_pose)
            # angle = compute_orientation(bone_data, camera_pose)
            humans.append(points)

        return humans

    def parse_sensor_file(self, sensor_filename):
        data = read_jsonlist(sensor_filename)
        cams = []
        for cam_data in data:
            loc = cam_data['Location']
            rot = cam_data['Rotation']
            x, y, z = float(loc['X']), float(loc['Y']), float(loc['Z'])
            pitch, yaw, roll = float(rot['Pitch']), float(rot['Yaw']), float(rot['Roll'])
            width = float(cam_data['FilmWidth'])
            height = float(cam_data['FilmHeight'])
            f = float(cam_data['FilmWidth']) / 2 
            cam_pose = CamPose(x, y, z, pitch, yaw, roll, width, height, f)
            cams.append(cam_pose)
        return cams

class CocoImage:
    def __init__(self, image_id, file_name):
        self.id = image_id
        im = io.imread(file_name)
        self.height = im.shape[0]
        self.width = im.shape[1]
        self.file_name = os.path.basename(file_name)

    def tojson(self):
        # Convert to json object
        keys = ['id', 'file_name', 'height', 'width']
        data = {}
        for k in keys: data[k] = self.__dict__[k]
        return data

class CocoKp:
    def __init__(self, kp_id, image_id, category_id, keypoints):
        self.id = kp_id
        self.image_id = image_id
        self.category_id = category_id

        self.keypoints = keypoints
        visible_flag = [v for v in keypoints[2::3]]
        # http://cocodataset.org/#format-data
        # v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible
        self.num_keypoints = len([v != 0 for v in visible_flag])

        self.segmentation = []
        self.area = 0
        self.iscrowd = 0
        self.bbox = []

    def tojson(self):
        keys = ['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id']
        data = {}
        for k in keys: data[k] = self.__dict__[k]
        return data



def check_filelist(filelist):
    for f in filelist: 
        if not os.path.isfile(f): print('File {f} can not found'.format(**locals()))


def build_coco():
    data_root = './humans'
    files = glob.glob(os.path.join(data_root, '*_cameras.json'))
    frame_ids = [int(os.path.basename(v).replace('_cameras.json', '')) for v in files]

    im_filename_tpl = os.path.join(data_root, '0/{frame_id}.png')
    kp_filename_tpl = os.path.join(data_root, '{frame_id}_bones.json')
    sensor_filename_tpl = os.path.join(data_root, '{frame_id}_cameras.json')

    # Make a parameter set for each frame
    frame_ids = [int(os.path.basename(v).replace('_cameras.json', '')) for v in files]

    dataset_builder = CocoDatasetBuilder()
    for i, fid in enumerate(frame_ids):
        if i > 10: break

        im_filename = im_filename_tpl.format(frame_id = fid)
        sensor_filename = sensor_filename_tpl.format(frame_id = fid)
        kp_filename = kp_filename_tpl.format(frame_id = fid)

        dataset_builder.add(fid, im_filename, sensor_filename, kp_filename)

    dataset_builder.save('diva_human.json')

if __name__ == '__main__':
    build_coco()
