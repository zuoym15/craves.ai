from unrealcv import client
from testutil import cmd_save_png

def test_depth_stencil():
    cmd_save_png(client, 'vget /sensor/0/depth_stencil png', 'temp/stencil.png')

def main():
    client.connect()
    test_depth_stencil()


if __name__ == '__main__':
    main()
