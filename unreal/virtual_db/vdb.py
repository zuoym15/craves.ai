import glob, os, json
import logging
import numpy as np

logger = logging.Logger("vdb")

class Dataset:
    def __init__(self, db_root_dir):
        self.db_root_dir = db_root_dir
        pass

    def get_ids(self):
        ''' Get frame ids of this dataset '''
        filenames = glob.glob(os.path.join(self.db_root_dir, '*', 'lit', '*.png'))
        filenames = filenames + glob.glob(os.path.join(self.db_root_dir, '*', 'lit', '*.jpg'))
        ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in filenames]
        if len(ids) == 0:
            logger.warning("The dataset is empty, root_dir: %s" % self.db_root_dir)
            print("The dataset is empty")
        return ids
    
    def get_cams(self):
        ''' Get camera names of this dataset '''
        cams = [os.path.basename(v) for v in glob.glob(os.path.join(self.db_root_dir, '*')) if os.path.isdir(v)]
        # Remove special data folders
        # cams = list(set(cams) - set(['summary']))
        cams = [v for v in cams if v.lower().find('cam') != -1]
        return cams

    def get_annotation_color(self):
        ''' Return annotation color as a dict '''
        # The annotation color can be get from the first frame scene data
        ids = self.get_ids()
        ids = [ids[0]]
        scene_info = self.get_scene_info(ids)[0]

        annotation_color = { k : v['AnnotationColor'] for (k, v) in scene_info.items()}
        # annotation_filename = os.path.join(self.db_root_dir, 'annotation_color.json')
        # with open(annotation_filename) as f:
        #     data = json.load(f)
        # return data
        return annotation_color

    def get_scene_info(self, ids):
        ''' Get scene info '''
        joint_info = [] 
        for id in ids:
            json_filename = os.path.join(self.db_root_dir, 'scene/{id:08}.json'.format(**locals()))
            data = json.load(open(json_filename))
            joint_info.append(data)
        return joint_info

    def get_d3_skeleton(self, ids):
        ''' Get 3d annotation of skeleton '''
        joint_info = [] 
        for id in ids:
            json_filename = os.path.join(self.db_root_dir, 'joint/{id:08}.json'.format(**locals()))
            data = json.load(open(json_filename))
            joint_info.append(data)
        return joint_info

    def get_d3_vertex(self, ids):
        ''' Get 3d annotation of skeleton '''
        vertex_info = [] 
        for id in ids:
            # json_filename = os.path.join(self.db_root_dir, 'vertex/{id:08}.json'.format(**locals()))
            json_filename = os.path.join(self.db_root_dir, 'puppeteer/{id:08}.json'.format(**locals()))
            '''in the new version of the generation tool, the vertex info are saved in puppeteer subfolder rather than vertex subfolder.'''
            data = json.load(open(json_filename))
            vertex_info.append(data)
        return vertex_info

    def get_cam_info(self, cams, ids):
        ''' Get camera information, location, rotation '''
        cam_info = [] 
        for (cam, id) in [(cam, id) for cam in cams for id in ids]:
            json_filename = os.path.join(self.db_root_dir, '{cam}/caminfo/{id:08}.json'.format(**locals()))
            if os.path.isfile(json_filename):
                data = json.load(open(json_filename))
                cam_info.append(data)
        return cam_info

    def get_image(self, cams, ids):
        extension = ['.png', '.jpg']
        folder = 'lit'
        # folder = 'lit' if img_type == 'synthetic' else 'real' #real image is saved in '../real' folder

        ''' Get images from a camera '''
        files = [] 
        for (cam, id) in [(cam, id) for cam in cams for id in ids]:
            for ext in extension:
                if os.path.isfile(os.path.join(self.db_root_dir, ('{cam}/'+folder+'/{id:08}'+ext).format(**locals()))):
                    files.append(os.path.join(self.db_root_dir, ('{cam}/'+folder+'/{id:08}'+ext).format(**locals())))
                    break
        return files

    def get_seg(self, cams, ids):
        ''' Get segmentation mask as rgb file '''
        files = [] 
        for (cam, id) in [(cam, id) for cam in cams for id in ids]:
            files.append(os.path.join(self.db_root_dir, '{cam}/seg/{id:08}.png'.format(**locals())))
        return files

    def get_depth(self, cams, ids):
        ''' Get depth '''
        files = [] 
        for (cam, id) in [(cam, id) for cam in cams for id in ids]:
            files.append(os.path.join(self.db_root_dir, '{cam}/depth/{id:08}.npy'.format(**locals())))
        return files

    def get_normal(self, cams, ids):
        ''' Get surface normal '''
        files = []
        for (cam, id) in [(cam, id) for cam in cams for id in ids]:
            files.append(os.path.join(self.db_root_dir, '{cam}/normal/{id:08}.png'.format(**locals())))
        return files

    def get_meta(self, cams, ids):
        ''' Get surface normal '''
        files = []
        for (cam, id) in [(cam, id) for cam in cams for id in ids]:
            files.append(os.path.join(self.db_root_dir, '{cam}/caminfo/{id:08}.json'.format(**locals())))
        return files


def seg2bb(obj_mask):
    ''' Convert binary seg mask of object to bouding box, (x0, y0, x1, y1) format '''
    y, x = np.where(obj_mask == True)
    bb = [x.min(), x.max(), y.min(), y.max()]
    return bb

def get_obj_mask(seg_im, color):
    ''' Get object binary mask from a color coded mask '''
    seg_mask = np.array(seg_im[:,:,0] * (256 ** 2) + seg_im[:,:,1] * 256 + seg_im[:,:,2])
    if isinstance(color, list):
        R, G, B = color
    if isinstance(color, dict):
        R, G, B = color['R'], color['G'], color['B']

    val = R * (256 ** 2) + G * 256 + B 
    obj_mask = np.equal(seg_mask, val)
    return obj_mask
