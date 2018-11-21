import time
class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, tb):
        print(time.time() - self.start)

import numpy as np
import PIL.Image
from io import BytesIO
import imageio
# StringIO module is removed in python3, use io module

def read_png(res):
    '''
    Return a numpy array from binary bytes of png format

    Parameters
    ----------
    res : bytes
        For example, res = client.request('vget /camera/0/lit png')

    Returns
    -------
    numpy.array
        Numpy array
    '''
    img = None
    try:
        PIL_img = PIL.Image.open(BytesIO(res))
        img = np.asarray(PIL_img)
    except:
        print('Read png can not parse response %s' % str(res[:20]))
    return img

def read_npy(res):
    '''
    Return a numpy array from binary bytes of numpy binary file format

    Parameters
    ----------
    res : bytes
        For example, res = client.request('vget /camera/0/depth npy')

    Returns
    -------
    numpy.array
        Numpy array
    '''
    # res is a binary buffer
    arr = None
    try:
        arr = np.load(BytesIO(res))
    except:
        print('Read npy can not parse response %s' % str(res[:20]))
    return arr

def cmd_save_png(client, cmd, png_filename):
    res = client.request(cmd)
    if str(res).startswith('error'):
        assert False, 'cmd = %s, res = %s' % (cmd, res)

    im = read_png(res)
    imageio.imsave(png_filename, im)

def cmd_save_npy(client, cmd, npy_filename):
    res = client.request(cmd)
    if str(res).startswith('error'):
        assert False, 'cmd = %s, res = %s' % (cmd, res)

    npy_data = read_npy(res)
    np.save(npy_filename, npy_data)

def set_cam_pose(client, cam_id, loc, rot):
    [x, y, z] = loc
    [pitch, yaw, roll] = rot
    cmd = 'vset /camera/{cam_id}/location {x} {y} {z}'.format(**locals())
    res = client.request(cmd)
    assert str(res) == 'ok', 'cmd = {cmd}, res = {res}'.format(**locals())
    cmd = 'vset /camera/{cam_id}/rotation {pitch} {yaw} {roll}'.format(**locals())
    res = client.request(cmd)
    assert str(res) == 'ok', 'cmd = {cmd}, res = {res}'.format(**locals())
    cmd = 'vget /camera/{cam_id}/location'.format(**locals())
    res = client.request(cmd); print(res)
    cmd = 'vget /camera/{cam_id}/rotation'.format(**locals())
    res = client.request(cmd); print(res)
    