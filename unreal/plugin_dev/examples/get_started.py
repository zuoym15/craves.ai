from __future__ import division, absolute_import, print_function
import os
import sys
import time
import re
import json
import numpy as np
import matplotlib.pyplot as plt

# Load some python libraries The dependencies for this tutorials are PIL, Numpy, Matplotlib
imread = plt.imread

def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im


def read_png(res):
    import StringIO
    import PIL.Image
    img = PIL.Image.open(StringIO.StringIO(res))
    return np.asarray(img)


def read_npy(res):
    import StringIO
    return np.load(StringIO.StringIO(res))

def save_im(im, filename):
    plt.imsave(filename, im)

def save_depth(depth, filename):
    plt.figure()
    plt.imshow(depth)
    plt.savefig(filename)

class Frame:
    def __init__(self):
        pass

def capture_frame(client):
    res = client.request('vget /sensor/0/lit png')
    lit = read_png(res)
    save_im(lit, 'lit.png')

    res = client.request('vget /sensor/0/object_mask png')
    seg_mask = read_png(res)
    save_im(seg_mask, 'seg_mask.png')

    res = client.request('vget /sensor/0/depth npy')
    depth = read_npy(res)
    save_depth(depth, 'depth.png')

    res = client.request('vget /sensor/0/normal png')
    normal = read_png(res)
    save_im(normal, 'normal.png')

    frame = dict(lit = lit, depth = depth, \
        normal = normal, seg_mask = seg_mask)
    return frame

def get_obj_list(client):
    scene_objects = client.request('vget /objects').strip().split(' ')
    print('Number of objects in this scene:', len(scene_objects))

    class Color:
        ''' A utility class to parse color value '''
        regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
        def __init__(self, color_str):
            self.color_str = color_str
            match = self.regexp.match(color_str)
            (self.R, self.G, self.B, self.A) = [
                int(match.group(i)) for i in range(1, 5)]

        def __repr__(self):
            return self.color_str

    obj_list = []
    
    for obj_id in scene_objects:
        res = client.request('vget /object/%s/color' % obj_id)
        color = Color(res)
        obj = dict(id = obj_id, color = color)
        obj_list.append(obj)

    return obj_list

def main():
    # Connect to the game
    # Load unrealcv python client, do `pip install -U unrealcv` first.
    from unrealcv import client
    client.connect()
    if not client.isconnected():
        print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
        sys.exit(-1)

    # Make sure the connection works well
    res = client.request('vget /unrealcv/status')
    # The image resolution and port is configured in the config file.
    print(res)

    obj_list = get_obj_list(client)

    frame = capture_frame(client)
    seg_mask = frame['seg_mask']
    lit = frame['lit']
    plt.imshow(seg_mask)
    plt.show()

    covered_sum = 0
    for obj in obj_list:
        obj_color = obj['color']
        obj_mask = (seg_mask[:,:,0] == obj_color.R)  \
                 & (seg_mask[:,:,1] == obj_color.G) \
                 & (seg_mask[:,:,2] == obj_color.B)
        if obj_mask.sum() > 0:
            covered_sum += obj_mask.sum()
            print('Size of %s is %d' % (obj['id'], obj_mask.sum()))
    img_sum = lit.shape[0] * lit.shape[1]
    print('Total pixels of the image %d, covered pixels %d' % (img_sum , covered_sum))


    # # Visualize the captured ground truth
    # plt.imshow(object_mask)
    # plt.figure()
    # # plt.imshow(normal)

    # plt.imshow(depth)



    # def match_color(object_mask, target_color, tolerance=3):
    #     match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    #     for c in range(3):  # r,g,b
    #         min_val = target_color[c] - tolerance
    #         max_val = target_color[c] + tolerance
    #         channel_region = (object_mask[:, :, c] >= min_val) & (
    #             object_mask[:, :, c] <= max_val)
    #         match_region &= channel_region

    #     if match_region.sum() != 0:
    #         return match_region
    #     else:
    #         return None

    #     id2mask = {}
    #     for obj_id in scene_objects:
    #         color = id2color[obj_id]
    #         mask = match_color(object_mask, [color.R, color.G, color.B], tolerance=3)
    #         if mask is not None:
    #             id2mask[obj_id] = mask
    #     # This may take a while
    #     # TODO: Need to find a faster implementation for this

    # with open('object_category.json') as f:
    #     id2category = json.load(f)
    # categories = set(id2category.values())
    # # Show statistics of this frame
    # image_objects = id2mask.keys()
    # print('Number of objects in this image:', len(image_objects))
    # print('%20s : %s' % ('Category name', 'Object name'))
    # for category in categories:
    #     objects = [v for v in image_objects if id2category.get(v) == category]
    #     if len(objects) > 6:  # Trim the list if too long
    #         objects[6:] = ['...']
    #     if len(objects) != 0:
    #         print('%20s : %s' % (category, objects))


if __name__ == '__main__':
    main()
