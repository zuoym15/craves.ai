import os
import json

def find_json(file_dir):   
    L = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file)) 
    return L

preds_dir = 'C:\\Users\\Yiming\\Desktop\\arm-pose\\visualization\\20180819\\real_20181010_bug_fixed'
L = find_json(os.path.join(preds_dir, 'd3_preds'))
res_15 = []
res_10 = []
for json_path in L:
    with open(json_path, 'r') as f:
        obj = json.load(f)
        if obj['error'] <= 15.0 and obj['num_valid_key'] >= 14:
            res_15.append(os.path.splitext(os.path.basename(json_path))[0] + '.jpg')
        if obj['error'] <= 10.0 and obj['num_valid_key'] >= 15:
            res_10.append(os.path.splitext(os.path.basename(json_path))[0] + '.jpg')

with open(os.path.join(preds_dir, 'valid_img_list_10.json'), 'w') as f:
    json.dump(res_10 ,f)
    print('len res_10: {}'.format(len(res_10)))

with open(os.path.join(preds_dir, 'valid_img_list_15.json'), 'w') as f:
    json.dump(res_15 ,f)
    print('len res_15: {}'.format(len(res_15)))
        



