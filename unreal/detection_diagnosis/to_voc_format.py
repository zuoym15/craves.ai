# Convert the UnrealCV annotation format to PASCAL annotation format
import re, glob, sys
sys.path.append('../unrealcv/unrealcv/client')
import ue4cv
from scipy import misc # for imread
import numpy as np

# %load -s get_color_mapping ipynb_util.py
def get_color_mapping(client, object_list):
    ''' Get the color mapping for specified objects '''
    color_mapping = {}
    for objname in object_list:
        color_mapping[objname] = Color(client.request('vget /object/%s/color' % objname))
    return color_mapping

# %load -s Color,subplot_color ipynb_util.py
class Color(object):
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str


# %load -s match_color,compute_instance_mask ipynb_util.py
def match_color(color_image, target_color, tolerance=3): # Tolerance is used to solve numerical issue
    match_region = np.ones(color_image.shape[0:2], dtype=bool)
    for c in range(3): # Iterate over three channels
        min_val = target_color[c]-tolerance; max_val = target_color[c]+tolerance
        channel_region = (color_image[:,:,c] >= min_val) & (color_image[:,:,c] <= max_val)
        # channel_region = color_image[:,:,c] == target_color[c]
        match_region &= channel_region
    return match_region

def compute_instance_mask(object_mask, color_mapping, objects):
    if isinstance(object_mask, str):
        object_mask = misc.imread(object_mask)

    # dic_instance_mask = {}
    dic_instance_bbox = {}
    for object_name in objects:
        color = color_mapping[object_name]
        region = match_color(object_mask, [color.R, color.G, color.B], tolerance=3)
        if region.sum() != 0: # Present in the image
            [ys, xs] = np.where(region)
            # dic_instance_mask[object_name] = region
            bbox = {'ymin':ys.min(), 'ymax':ys.max(), 'xmin':xs.min(), 'xmax':xs.max()}
            dic_instance_bbox[object_name] = bbox
    # return dic_instance_mask
    return dic_instance_bbox

def to_xml(struct):
    xml_str = ''
    if isinstance(struct, dict):
        for (k, v) in struct.iteritems():
            xml_value = to_xml(v)
            if xml_value.find('\n') != -1:
                lines = xml_value.strip().split('\n')
                # Add indent to a sub level
                xml_str += '<{key}>\n{value}\n</{key}>\n'.format(key = k, value = '\n'.join(['    %s' % v for v in lines]))
            else:
                xml_str += '<{key}>{value}</{key}>\n'.format(key = k, value = xml_value)
    if isinstance(struct, list):
        for v in struct:
            xml_str += to_xml(v)
    if isinstance(struct, str) or isinstance(struct, int) or isinstance(struct, float):
        xml_str = str(struct)
    return xml_str

def voc_xml_file(objects):
    folder = 'image'
    filename = 'test.png'
    database = 'RealisticRendering'
    annotation = 'UnrealCV'
    owner = 'Epic Game'
    width = 100
    height = 100
    depth = 3
    segmented = 1
    annotation = {
        "annotation":
        [
            {"folder": folder},
            {"filename": filename},
            {"source":
                {
                    "database": database,
                    "annotation": annotation,
                }
            },
            {"owner": owner},
            {"size":
                {
                    "width": width,
                    "height": height,
                    "depth": depth,
                }
            },
            {"segmented": segmented},
        ]
    }

    for name, bbox in objects.iteritems():
        print name, bbox
        object_info = {'object': {
            'name': name,
            # This name is actually object category
            'bndbox': {
                "xmin": bbox['xmin'],
                "ymin": bbox['ymin'],
                "xmax": bbox['xmax'],
                "ymax": bbox['ymax']
                }
            }
        }
        annotation['annotation'].append(object_info)
    print annotation


    print to_xml(annotation)

def convert_seg_file(segfile, color_mapping, scene_objects):
    # Read segfile and Parse it into bounding box
    dic_instance_bbox = compute_instance_mask(segfile, color_mapping, scene_objects)
    voc_xml_file(dic_instance_bbox)

    # Save to xml file
    pass

if __name__ == '__main__':
    seg_files = glob.glob('./seg/*.png')
    ue4cv.client.connect()
    if not ue4cv.client.isconnected():
        print 'Can not connect to the game'
    else:
        # Get a list of all objects in the scene
        scene_objects = ue4cv.client.request('vget /objects').split(' ')
        print 'There are %d objects in this scene' % len(scene_objects)
        color_mapping = get_color_mapping(ue4cv.client, scene_objects)

        convert_seg_file(seg_files[0], color_mapping, scene_objects)
    # voc_xml_file()
