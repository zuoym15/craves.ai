# Weichao Qiu @ 2018
# Check - http://docs.unrealcv.org/en/latest/reference/commands.html
# Make sure all commands run as expected

from unrealcv import client

def r(cmd):
    return client.request(cmd)

def init(client):
    client.connect()

def v0.2(client):
    
    r('vget /camera/0/location')
    r('vget /camera/0/rotation')
    r('vset /camera/0/location 0 0 0')
    r('vset /camera/0/rotation 0 0 0')
    r('vget /viewmode') 

def main():
    pass

if __name__ == '__main__':
    binary_path = ''
    main()
