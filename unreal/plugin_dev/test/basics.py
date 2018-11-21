# from testutil import read_png, read_npy, Timer
from testutil import cmd_save_npy, cmd_save_png, Timer, set_cam_pose
from unrealcv import client
# Unrealcv basic features are described in - basic features page

def r(cmd):
    res = client.request(cmd)
    if str(res).startswith('error'):
        assert False, 'cmd = %s, res = %s' % (cmd, res)
    return res

def save(filename, im):
    # plt.imsave(filename, im)
    imageio.imsave(filename, im)

def test_control_camera():
    res = r('vget /camera/0/location')
    print(res)
    res = r('vget /camera/0/rotation')
    print(res)
    res = r('vget /camera/0/lit png') # Use lit to check camera location and rotation

    # Move camera to four locations and take pictures
    cam_poses = [
        [[-100, 440, 100], [0, -90, 0]],
        [[-100, 440, 100], [0, 0, 0]],
        [[-100, 440, 100], [0, 90, 0]],
        [[-100, 440, 100], [0, 180, 0]],
        [[-100, 300, 100], [0, -90, 0]],
        [[-100, 200, 100], [0, -90, 0]],
    ]
    cam_id = 0
    for i in range(len(cam_poses)):
        cam_pose = cam_poses[i]
        print(cam_pose)
        set_cam_pose(client, cam_id, cam_pose[0], cam_pose[1])
        cmd = 'vget /camera/{cam_id}/lit png'.format(**locals())
        filename = 'temp/cam_{cam_id}_view{i}.png'.format(**locals())
        cmd_save_png(client, cmd, filename)

def test_move_to():
    cam_poses = [
        ['moveto', [-100, -600, 100], [0, -90, 0]], # will be blocked
        ['setpos_outside', [-100, -600, 100], [0, -90, 0]], # will fly outside
        ['setpos_reset', [-100, 440, 100], [0, -90, 0]], # reset location
    ]
    cam_id = 0

    for i in range(len(cam_poses)):
        cam_pose = cam_poses[i]
        print(cam_pose)
        cam_config = cam_pose[0]
        if cam_config == 'moveto':
            [x, y, z] = cam_pose[1]
            cmd = 'vset /camera/{cam_id}/moveto {x} {y} {z}'.format(**locals())
            res = client.request(cmd)
            assert str(res) == 'ok', 'cmd = {cmd}, res = {res}'.format(**locals())
        else:
            set_cam_pose(client, cam_id, cam_pose[1], cam_pose[2])
        cmd = 'vget /camera/{cam_id}/lit png'.format(**locals())
        filename = 'temp/cam_{cam_id}_{cam_config}.png'.format(**locals())
        cmd_save_png(client, cmd, filename)


def test_ground_truth():
    modes = ['lit', 'normal', 'object_mask', 'lit_fast', 'depth_stencil']
    for mode in modes:
        cmd = 'vget /camera/0/{mode} png'.format(**locals())
        filename = 'temp/{mode}.png'.format(**locals())
        cmd_save_png(client, cmd, filename)
        print('Save result to %s' % filename)

    # Save depth as npy file
    cmd = 'vget /camera/0/depth npy'
    cmd_save_npy(client, cmd, 'temp/depth.npy')

def test_multi_camera():
    print(client.request('vget /cameras'))
    camera_ids = [0, 1]
    for cam_id in camera_ids:
        cmd = 'vget /camera/{cam_id}/lit png'.format(**locals())
        filename = 'temp/cam_{cam_id}.png'.format(**locals())
        cmd_save_png(client, cmd, filename)


def main():
    client.connect()
    # test_control_camera()
    # test_move_to()
    # test_multi_camera()
    test_ground_truth()

    # lit = read_png(r('vget /sensor/0/lit png'))
    # save(lit, 'temp/lit.png')


if __name__ == '__main__':
    main()