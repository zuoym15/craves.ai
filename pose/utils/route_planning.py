import numpy as np
from numpy import sin, cos, pi
import json

def tip2angle(x, y, z, x0 = -45.21, y0 = -22.97, l1 = 89.68, l2 = 113.58, l3 = 75.0, h1 = 67.66, alpha = 0.0):
    #see https://docs.google.com/document/d/1VgmlxTxL5Gy_7Gs2rrF7Axosbcyzhjpy9Q3a6_KBHHA/edit for detail
    #all angles are in rad
    #height of the base is 45mm.
    dx = x - x0
    dy = y - y0
    rotation = -np.arctan(dx/dy)

    if abs(alpha) > pi:
        alpha = alpha * pi / 180.0 # convert to rad

    d = np.sqrt(dx**2 + dy**2)
    lm = np.sqrt((z+l3*sin(alpha)-h1)**2 + (d-l3*cos(alpha))**2)

    if abs((l1**2 + lm**2 - l2**2)/(2*l1*lm))>1 or abs((l1**2 + l2**2 - lm**2)/(2*l1*l2))>1:
        return False, [0,0,0,0]

    theta_1 = np.arccos((l1**2 + lm**2 - l2**2)/(2*l1*lm)) + np.arctan((z+l3*sin(alpha)-h1)/(d-l3)) - np.pi/2
    theta_2 = np.arccos((l1**2 + l2**2 - lm**2)/(2*l1*l2)) - np.pi/2
    theta_3 = - theta_1 - theta_2 - alpha

    result = np.array([rotation, theta_1, theta_2, theta_3]) * 180 / np.pi

    result[3] = np.clip(result[3], -45, 45)

    return True, result.tolist()

def angle2tip(angles, l1 = 89.68, l2 = 113.58, l3 = 75.0, h1 = 67.66):
    assert(len(angles) == 4)
    angles = list(angles)
    for i in range(len(angles)):
        if abs(angles[i]) > pi:
            angles[i] = angles[i] * pi / 180.0
    
    x0 = - l1 * sin(angles[1]) + l2 * cos(angles[1] + angles[2]) + l3 * cos(angles[1] + angles[2] + angles[3])
    z0 = l1 * cos(angles[1]) + l2 * sin(angles[1] + angles[2]) + l3 * sin(angles[1] + angles[2] + angles[3])

    x = x0 * cos(angles[0])
    y = x0 * sin(angles[0])
    z = z0 + h1

    return x, y, z

if __name__ == "__main__":
    print(angle2tip((0.0, -16.16488672529152, -12.105047815587557, 28.269934540879078)))
    # obj = {}
    # for dist in [(100, 20), (150,0), (200,0), (250,0)]:
    #     res, angle = tip2angle(0, dist[0], 100, x0 = 0, y0 = 0, alpha=dist[1])
    #     obj[str(dist[0])] = [angle[1], angle[2], angle[3]]
    
    # with open('C:/Users/Yiming/Desktop/pose.json', 'w') as f:
    #     json.dump(obj, f)

