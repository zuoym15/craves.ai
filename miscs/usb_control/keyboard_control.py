from roboarm import Arm
import numpy as np
from pynput.keyboard import Key, Listener

arm = Arm()

move_dict = {
        'e':[arm.wrist.up, arm.wrist.stop],
        'd':[arm.wrist.down, arm.wrist.stop],
        'r':[arm.elbow.up, arm.elbow.stop],
        'f':[arm.elbow.down, arm.elbow.stop],
        'u':[arm.shoulder.up, arm.shoulder.stop],
        'j':[arm.shoulder.down, arm.shoulder.stop],
        'i':[arm.base.rotate_clock, arm.base.stop],
        'k':[arm.base.rotate_counter, arm.base.stop]
        }

def on_press(key):
    k = str(key).replace('\'', '')

    print('{0} press'.format(
        k))
    if k in move_dict.keys():
        move_dict[k][0](0.05)

def on_release(key):
    k = str(key).replace('\'', '')

    print('{0} release'.format(
        k))

    if k in move_dict.keys():
        move_dict[k][1]()
    
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()