import numpy as np
import torch
import scipy
from utils.transforms import crop, color_normalize

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))

def get_training_image(img_path, bbox = None, inp_res = 256, mean = (0.6419, 0.6292, 0.5994), std = (0.2311, 0.2304, 0.2379)):
    img = load_image(img_path)

    if bbox is not None:
        x0, y0, x1, y1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
        c = np.array([(x0+x1), (y0+y1)])/2 #center
        s = np.sqrt((y1-y0)*(x1-x0))/60.0 #scale
    
    else:
        c = np.array([img.shape[2]/2, img.shape[1]/2])
        s = 5.0 #THIS HAS TO BE FIXED !!!
    
    r = 0 #rotation
    
    inp = crop(img, c, s, [inp_res, inp_res], rot=r)
    inp = color_normalize(inp, mean, std)

    meta = {'center':c, 'scale':s}

    return inp, meta


    