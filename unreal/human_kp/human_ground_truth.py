# Weichao Qiu @ 2018
import glob, os, re, logging, time
import imageio
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from util import read_jsonlist, CameraPose, project_to_2d
from dataset import MPII, COCO

class ImageUtil:
    ''' Image utility to perform common tasks '''
    def __init__(self):
        pass

    def mask_im(self, im, mask):
        im[~mask] = 0
        return im

class ObjWorld:
    ''' Object information in the 3D world space '''
    def __init__(self, world, name):
        self.world = world
        self.name = name
        self.category = None
        self.R = 0; self.G = 0; self.B = 0
        
        # May not have values
        self.WorldKeypoints = dict()
        self.WorldJoints = dict()

    def view_in_cam(self, cam):
        # TODO: This seems strange
        obj_cam = ObjCam(cam, self)
        return obj_cam

    def get_obj_id(self):
        return self.R * (256 ** 2) + self.G * 256 + self.B

    def get_3d_kp(self):
        # Load car keypoints
        if self.category == 'car':
            world_joints = self.WorldKeypoints
            bones = world_joints.keys()
            X = np.array([world_joints[bone_name]['X'] for bone_name in bones])
            Y = np.array([world_joints[bone_name]['Y'] for bone_name in bones])
            Z = np.array([world_joints[bone_name]['Z'] for bone_name in bones])
            N = len(bones)
            points = np.ones((N, 4))
            points[:, 0] = X; points[:, 1] = Y; points[:, 2] = Z
            return points

        if self.category == 'human':
            world_joints = self.WorldJoints
            bones = world_joints.keys()
            X = np.array([world_joints[bone_name]['X'] for bone_name in bones])
            Y = np.array([world_joints[bone_name]['Y'] for bone_name in bones])
            Z = np.array([world_joints[bone_name]['Z'] for bone_name in bones])
            N = len(bones)
            points = np.ones((N, 4))
            points[:, 0] = X; points[:, 1] = Y; points[:, 2] = Z
            return points

        logging.warning('Object %s is not a human or car, so no associted kp', self.name)
        return

class ObjCam:
    ''' Object information in the 2D camera space '''
    def __init__(self, cam, obj_world):
        # Use ObjWorld.view_in_cam() instead
        self.cam = cam
        self.name = None
        self.mask = None
        self.bb = None
        self.area = None
        # annotation color
        self.R = 0; self.G = 0; self.B = 0

        self.world_info = obj_world
        self.__dict__.update(obj_world.__dict__)


    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_name(self):
        return self.name

    def get_area(self):
        if self.area:
            return self.area
        
        if self.get_mask():
            return self.get_mask().sum()

        logging.warning("Can not compute the area of obj {name}, the area is None", **self.__dict__)
        return None

    def get_mask(self):
        ''' Get binary object mask of an object '''
        if self.mask is None:
            seg_mask = self.cam.get_seg_mask()
            # r, g, b = self.R, self.G, self.B
            # self.mask = (seg_mask[:, :, 0] == r) & (seg_mask[:, :, 1] == g) & (seg_mask[:, :, 2] == b)
            val = self.R * (256 ** 2) + self.G * 256 + self.B 
            self.mask = (seg_mask == val)
            # This ma.masked_equal is less efficient
            # self.mask = ma.masked_equal(seg_mask, val)

        return self.mask 

    def get_bb(self):
        ''' Get object bounding box from seg mask 
        the format is [xmin, xmax, ymin, ymax] '''
        if self.bb is None:
            obj_mask = self.get_mask()
            if obj_mask is None:
                logging.warning("Can not get obj bb for {name}", **locals())
                return None

            area = obj_mask.sum()
            if area == 0:
                logging.warning("Object {name} is not visible", **locals())
                return None

            y, x = np.where(obj_mask == True)
            self.bb = [x.min(), x.max(), y.min(), y.max()]

        return self.bb

    def get_bb_from_kps(self):
        ''' Get object bounding box from keypoints (kp), to get non-occluded mask '''
        kps = self.get_2d_kp()
        x = kps[:, 0]
        y = kps[:, 1]
        bb = [x.min(), x.max(), y.min(), y.max()]
        return bb

    def get_2d_kp(self):
        ''' Get 2D keypoint of a human '''
        kps = self.world_info.get_3d_kp()
        if kps is None: return None

        cam_pose = self.cam.get_cam_pose()
        xs, ys = project_to_2d(kps, cam_pose)
        N = kps.shape[0]
        points = np.ones((N, 2))
        points[:, 0] = xs.squeeze()
        points[:, 1] = ys.squeeze()

        return points

class FrameCamGT:
    ''' Parse ground truth of a frame captured by a camera '''
    def __init__(self, data_root, frame_id, cam_name):
        self.image_util = ImageUtil()
        self.data_root = data_root
        self.frame_id = frame_id
        self.seg_filename = os.path.join(data_root, cam_name, \
                        '{frame_id}_mask.png'.format(**locals()))
        self.lit_filename = os.path.join(data_root, cam_name, \
                        '{frame_id}_lit.png'.format(**locals()))
        self.cam_name = cam_name
        self.world = FrameWorldGT(data_root, frame_id)
        # Do not assign the functions of the world to the CamGT
        # since these two objects have very different contexts

        self.lit_im = None
        self.seg_im = None
        self.seg_mask = None
        self.obj_list = None

    def get_obj_list_in_view(self, sort_by_area=False):
        ''' Get a list of visible objects in this frame '''
        # A fast version without extracting seg mask
        if self.obj_list is None:
            self.obj_list = []
            seg_mask = self.get_seg_mask()
            unique, counts = np.unique(seg_mask, return_counts=True)
            world_obj_list = self.world.get_obj_list()
            total_area = 0
            for obj in world_obj_list:
                obj_id = obj.get_obj_id()
                if obj_id in unique:
                    cam_obj = obj.view_in_cam(self)
                    cam_obj.area = counts[np.where(unique==obj_id)] 
                    self.obj_list.append(cam_obj)
                    total_area += cam_obj.area
            
            if total_area != sum(counts):
                logging.warning("Some objects in the seg_mask are not covered by the annotation")

        # TODO: Find a faster implementation, use the unique values in the seg mask
        # if self.obj_list is None:
        #     world_obj_list = self.world.get_obj_list()
        #     self.obj_list = []
        #     for obj in world_obj_list:
        #         cam_obj = obj.view_in_cam(self)
        #         area = cam_obj.get_area()
        #         if area > 0:
        #             self.obj_list.append(cam_obj)
        if sort_by_area:
            self.obj_list = sorted(self.obj_list, key=lambda o: o.area, reverse=True)

        return self.obj_list
    
    def get_cam_pose(self):
        camera_filename = os.path.join(self.data_root, '%d_sensors.json' % self.frame_id)

        cams = read_jsonlist(camera_filename)

        cam = None
        for cam in cams:
            if cam['SensorName'] == self.cam_name:
                break

        cam_pose = CameraPose(
            cam['Location']['X'], cam['Location']['Y'], cam['Location']['Z'], 
            cam['Rotation']['Pitch'], cam['Rotation']['Yaw'], cam['Rotation']['Roll'], 
            cam['FilmWidth'], cam['FilmHeight'], cam['FilmWidth'] / 2
        )
        return cam_pose

    def get_lit_filename(self, relative=False):
        lit_filename = self.lit_filename
        if relative:
            lit_filename = lit_filename.replace(self.data_root, '', 1)
        return lit_filename
    
    def get_lit(self):
        ''' Get lit image '''
        if self.lit_im is None:
            self.lit_im = imageio.imread(self.lit_filename)
        return self.lit_im

    # def get_crop_lit(self, obj_name):
    #     ''' Get the cropped region of an object '''
    #     im = self.get_lit()
    #     bb = self.get_obj_bb(obj_name)
    #     crop = self.image_util.crop_bb(im, bb)
    #     return crop
    def get_color_seg_mask(self):
        ''' Get color coded seg mask '''
        if self.seg_im is None: # Lazy load
            self.seg_im = imageio.imread(self.seg_filename)
        return self.seg_im
    
    def get_seg_mask(self):
        ''' Get seg mask '''
        if self.seg_mask is None: # Lazy load
            self.seg_im = imageio.imread(self.seg_filename)
            self.seg_mask = np.array(self.seg_im[:,:,0] * (256 ** 2) + self.seg_im[:,:,1] * 256 + self.seg_im[:,:,2])
        return self.seg_mask
    
class FrameWorldGT:
    ''' A container for all frame annotation in the world level '''
    # Consider to check annotation format using JsonSchema here
    def __init__(self, data_root, frame_id):
        self.data_root = data_root
        self.frame_id = frame_id
        self.obj_list = None
        self.cars = None
        self.humans = None

    def get_obj_list(self):
        ''' Get all objects in the virtual scene '''
        if self.obj_list is None: # Lazy load
            logging.info('Load data for frame {frame_id}', **self.__dict__)
            objlist_frame_id = self.frame_id
            # Parse segmentation mask
            obj_filename = os.path.join(self.data_root, '%d_objects.json' % objlist_frame_id)
            while not os.path.isfile(obj_filename) and objlist_frame_id > 0:
                objlist_frame_id -= 1
                obj_filename = os.path.join(self.data_root, '%d_objects.json' % objlist_frame_id)

            if not os.path.isfile(obj_filename):
                logging.warning("Can not find obj list file {obj_filename}", **locals())
                return {}

            json_data = read_jsonlist(obj_filename)[0]
            # Convert dict to a list of objects
            self.obj_list = []
            for (k, v) in json_data.items():
                obj = ObjWorld(self, k)
                obj.__dict__.update(v) # v is annotation color data
                self.obj_list.append(obj)

            # Apply car and human category label and extra data
            cars = {}
            car_kp_filename = os.path.join(self.data_root, '%d_cars.json' % self.frame_id)
            if os.path.isfile(car_kp_filename):
                json_data = read_jsonlist(car_kp_filename)
                cars = {c['ActorName']:c for c in json_data}
            else:
                logging.warning('Can not find car annotation data from %s', car_kp_filename)

            humans = {}
            human_json_filename = os.path.join(self.data_root, '%d_humans.json' % self.frame_id)
            if os.path.isfile(human_json_filename):
                json_data = read_jsonlist(human_json_filename)
                humans = {h['ActorName']:h for h in json_data}
            else:
                logging.warning('Can not find human annotation data from %s', human_json_filename)

            for obj in self.obj_list:
                if obj.name in humans.keys():
                    obj.category = 'human'
                    obj.__dict__.update(humans[obj.name])
                if obj.name in cars.keys():
                    obj.category = 'car'
                    obj.__dict__.update(cars[obj.name])

        return self.obj_list

    

class Activity:
    def __init__(self, actor_name, activity_name):
        self.actor_name = actor_name
        self.activity_name = activity_name
        self.start_frame = -1
        self.end_frame = -1

class SeqGT:
    def __init__(self, data_root):
        self.data_root = data_root
        if not os.path.isdir(data_root):
            print('Can not find data folder: {data_root}'.format(**locals()))
        self.camera_folders = glob.glob(os.path.join(data_root, '*/'))
        self.event_files = glob.glob(os.path.join(data_root, '*_events.json'))
        self.cars = glob.glob(os.path.join(data_root, '*_cars.json'))
        self.humans = glob.glob(os.path.join(data_root, '*_humans.json'))
        self.sensor_filenames = glob.glob(os.path.join(data_root, '*_sensors.json'))
        self.clip_range = None # No range

        ''' Return all the frame numbers in this video sequence '''
        re_pattern = re.compile(r'.*[/\\]([0-9]+)_sensors.json')
        matches = [re_pattern.match(filename) for filename in self.sensor_filenames]
        frame_ids = [int(match.group(1)) for match in matches if match is not None]
        self.all_frame_ids = sorted(frame_ids)
        self.frame_ids = self.all_frame_ids # By default it is all frames without clip 
    
    def stat(self):
        ''' Print dataset statistics '''
        msg = ''
        msg += 'Frame # with cars: %d \n' % len(self.cars)
        msg += 'Frame # with humans: %d \n' % len(self.humans)
        msg += 'Frame # with events: %d \n' % len(self.event_files)
        msg += '# of cameras: %d \n' % len(self.camera_folders)
        return msg

    def get_frames(self):
        return self.frame_ids

    def get_cams(self):
        cam_names = [os.path.basename(os.path.dirname(f)) for f in self.camera_folders]
        assert(len(cam_names) == len(self.camera_folders))
        return cam_names 

    def clip(self, clip_range):
        # clip_range: [0, num(frames)]
        # Clip the sequence to a limited range
        self.clip_range = clip_range
        self.frame_ids =  [self.all_frame_ids[v] for v in clip_range]

    def check_format(self):
        # Check whether the ground truth format is consistent with definition
        # Check the object names 
        pass

    def get_activity(self):
        # Return all activities in this video sequence.
        if self.frame_range is None:
            event_files = self.event_files
        else:
            start_frame = self.frame_range[0]
            end_frame = self.frame_range[1]
            event_files = self.event_files[start_frame:end_frame]

        events = []
        active_events = []
        for event_file in event_files:
            frame_event_data_list = read_jsonlist(event_file)
            for event_data in frame_event_data_list:
                if event_data['EventState'] == 'Begin':
                    # Create a new event and append to all events of this sequence
                    event = Activity(event_data['ActorName'], event_data['EventName'])
                    event.start_frame = int(event_data['FrameNumber'])
                    events.append(event)
                    active_events.append(event)
                if event_data['EventState'] == 'End':
                    for event in active_events:
                        if event.actor_name == event_data['ActorName'] \
                            and event.activity_name == event_data['EventName']:
                            break
                        event.end_frame = int(event_data['FrameNumber']) 
                        active_events.remove(event)

        print('Num. of recorded events: ', len(events))
        print('Num. of active events (unfinished): ', len(active_events))
        print('Num. of finished events: ', len(set(events) - set(active_events)))
                
        complete_events = set(events) - set(active_events)
        # Check to make sure all the events are valid
        for event in complete_events:
            if event.start_frame == -1:
                print('start_frame is invalid')
            if event.end_frame == -1:
                print('end_frame is invalid')
        
        return complete_events
        
    def get_cameras(self):
        # Return camera ids
        pass

class SeqViewer:
    # Visualize a sequence generated from the simulator
    def vis_activity(self, events):
        activity_abbr = {
            'OpenTrunk': 'OT',
            'CloseTrunk': 'CT',
            'Enter': 'EN',
            'Exit': 'EX',
            'EnterCar': 'ENC',
            'ExitCar': 'EXC',
        }

        actor_names = []
        for event in events:
            actor_names.append(event.actor_name)

        unique_actor_names = set(actor_names)
        print('Num. of actors in all events: ', len(actor_names))
        print('Num. of unique actors in all events: ', len(unique_actor_names))

        bar_viewer = BarViewer()
        min_x = 1e10 
        max_x = 0 
        actor_id = 0
        for actor_name in unique_actor_names:
            bar_viewer.draw_row_name(actor_id, actor_name)

            events_to_plot = [event for event in events \
                                if event.actor_name == actor_name]
            for event in events_to_plot:
                min_x = min(min_x, event.start_frame)
                max_x = max(max_x, event.end_frame)
                activity_name = event.activity_name

                if activity_abbr.get(activity_name):
                    activity_name = activity_abbr.get(activity_name)

                bar_viewer.draw_bar(actor_id, [event.start_frame, event.end_frame], 'r', \
                    name=activity_name)

            actor_id += 1
        bar_viewer.set_xlim([min_x, max_x])
        print('Set x lim to [{min_x}, {max_x}]'.format(**locals()))
        bar_viewer.show()

class Frame3DViewer:
    def __init__(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

    def plot_car_kp(self, kp):
        self.ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2])
        self.ax.axis('equal')

    def show(self):
        plt.show()

class FrameViewer:
    def __init__(self, fig=None, ax=None, title=None):
        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig

        if ax is None:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = ax

        if title is not None:
            self.fig.canvas.set_window_title(title)

        self.im_handle = None
        self.im = None

    def get_size(self):
        size = self.fig.get_size_inches() * self.fig.dpi
        return size

    def plot_im(self, im):
        self.im_handle = self.ax.imshow(im)
        self.im = im

    def plot_bb(self, obj_bb, obj_name = None, color = None):
        if obj_bb is None:
            return
        xmin, xmax, ymin, ymax = obj_bb
        if color is None:
            color = 'r'

        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor = color, facecolor='none')
        self.ax.add_patch(rect)
        if obj_name is not None:
            self.ax.text(xmin, ymin, obj_name)

    def plot_kp(self, kp, clip=True):
        xs = kp[:,0]
        ys = kp[:,1]
        if self.im is not None:
            h = self.im.shape[0]
            w = self.im.shape[1]
            if clip:
                valid = (xs>=0) & (xs < w) & (ys>=0) & (ys < h)
                xs = xs[valid]
                ys = ys[valid]
        # print(xs.shape, ys.shape)
        self.ax.scatter(xs, ys)
        
    def show(self):
        plt.show()
    
def main():
    benchmark()

class Timer:
    def __init__(self):
        self.start = None

    def __enter__(self):
        self.start = time.time()
    
    def __exit__(self, tp, value, tb):
        print(time.time() - self.start)

def benchmark():
    data_root = 'data/car_detection_training_0326'
    cam_name = 'FusionCameraActor_1'
    seq_gt = SeqGT(data_root)
    
    frames = seq_gt.get_frames()
    frame_id = frames[0]
    frame_cam_gt = FrameCamGT(data_root, frame_id, cam_name)
    frame_world_gt = FrameWorldGT(data_root, frame_id)

    with Timer():
        for _ in range(100):
            frame_cam_gt.get_lit()

    with Timer():
        for _ in range(100):
            frame_cam_gt.get_seg_mask()

    with Timer():
        for _ in range(100):
            frame_world_gt.get_obj_list()
    
    with Timer():
        for _ in range(100):
            frame_cam_gt.get_obj_list_in_view()

if __name__ == '__main__':
    main()