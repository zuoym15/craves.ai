from unrealcv import client
from unrealcv.util import read_png
import imageio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('img_filename')
args = parser.parse_args(); fname = args.img_filename

client.connect()
client.request('vset /arm/owi535/random_pose')
# Set the arm to a random pose
# Or set the arm to a specific pose with 
# client.request('vset /arm/owi535/pose 20 20 20 20 20')

client.request('vset /camera/0/location 320 0 300')
client.request('vset /camera/0/rotation -30 180 0')
res = client.request('vget /camera/0/lit png')
im = read_png(res)

texture_filename = os.path.abspath('./000000000139.jpg')
client.request('vset /env/sky/texture %s' % texture_filename)
client.request('vset /env/floor/texture %s' % texture_filename)
client.request('vset /env/random_lighting')

imageio.imwrite(fname, im)
    
keypoints = client.request('vget /arm/owi535/keypoints')
print(keypoints)
# Dense 3D keypoint in the world space.
