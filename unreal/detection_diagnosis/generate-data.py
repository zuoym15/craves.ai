# Use UnrealCV to generate data
import sys, math, time, shutil
sys.path.append('../unrealcv/unrealcv/client')
import ue4cv

files = []

ue4cv.client.connect()
if not ue4cv.client.isconnected():
    print 'Can not connect to the game'
else:
    req = ue4cv.client.request
    target_name = 'Couch_13'

    target_location = [float(v) for v in req('vget /object/%s/location' % target_name).split(' ')]
    target_location[2] = 20
    print 'The location of target is ', target_location
    # for radius in range(10, 100, 10):
    # for deg_pitch in range(0, 91, 45):
    for deg_pitch in range(0, 61, 30):
        for deg_yaw in range(90, 271, 45):
        # for deg_yaw in [180]:
            # time.sleep(0.1)
            for radius in range(200, 300, 10):
                # time.sleep(0.1)
            # radius = 20
                yaw = math.radians(deg_yaw)
                pitch = math.radians(deg_pitch)

                offx = radius * math.cos(pitch) * math.cos(yaw)
                offy = radius * math.cos(pitch) * math.sin(yaw)
                offz = radius * math.sin(pitch)
                # print 'offset:', [offx, offy, offz]
                cam_loc = [target_location[0] + offx, target_location[1] + offy, target_location[2] + offz]

                cmd = 'vset /camera/0/location ' + ' '.join(['%.2f' % v for v in cam_loc])
                print req(cmd)
                cmd = 'vset /camera/0/rotation {pitch} {yaw} 0'.format(yaw=180+deg_yaw, pitch=-deg_pitch)
                print req(cmd)
                lit = req('vget /camera/0/lit')
                object_mask = req('vget /camera/0/object_mask')
                frame = dict(image=lit, object_mask=object_mask)

                shutil.copyfile(frame['image'], 'image/%d_%d_%d.png' % (deg_pitch, deg_yaw, radius))
                shutil.copyfile(frame['object_mask'], 'seg/%d_%d_%d.png' % (deg_pitch, deg_yaw, radius))
