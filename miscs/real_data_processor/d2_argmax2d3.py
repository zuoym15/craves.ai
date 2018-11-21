import os
import json
import numpy as np

def find_json(file_dir):   
    L = []
    F = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file)) 
                F.append(file)
    return L, F

preds_dir = 'C:\\Users\\Yiming\\Desktop\\arm-pose\\data\\ft_20181106'
L, F = find_json(os.path.join(preds_dir, 'preds'))
for i in range(len(L)):
    with open(L[i], 'r') as f:
        obj = json.load(f)
        anno = np.array(obj['d2_key'])
        anno = np.transpose(anno).tolist()
    with open(os.path.join(preds_dir, 'd3_preds', F[i]), 'w') as f:
        json.dump({'reprojection': anno}, f)

valid_img_list = []

for id in np.random.choice(len(L), 2394):
    valid_img_list.append(os.path.splitext(F[id])[0] + '.jpg')

with open(os.path.join(preds_dir, 'valid_img_list.json'), 'w') as f:
    json.dump(valid_img_list, f)

    