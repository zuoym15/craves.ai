# Weichao Qiu @ 2018
# Coco annotation file is too long and not indented, so it is hard to learn.
# This script will split coco dataset into a smaller one to make it easier to analyze

import json, time, pdb


def chunk_instance_data():
    instance_json_filename = './instances_val2017.json'

    start = time.time()
    data = json.load(open(instance_json_filename, 'r'))
    print(time.time() - start)

    print(len(data))
    print(data.keys())
    # pdb.set_trace()
    # info
    # licenses
    # images, 5000 -> 100
    # annotations, 36781 -> 100, each image can contain multiple seg box, the seg run-length encoding takes a lot of space. 
    # categories

    chunked_data = dict()
    chunked_data['info'] = data['info']
    chunked_data['licenses'] = data['licenses']
    chunked_data['categories'] = data['categories']
    num = 100
    chunked_data['images'] = data['images'][:num]
    chunked_data['annotations'] = data['annotations'][:num]

    chunked_json_filename = instance_json_filename.replace('.json', '_100.json')
    with open(chunked_json_filename, 'w') as f:
        json.dump(chunked_data, f, indent = 2)


def chunk_kp_data():
    kp_json_filename = './person_keypoints_val2017.json'
    data = json.load(open(kp_json_filename, 'r'))
    chunked_data = dict()
    chunked_data['info'] = data['info']
    chunked_data['licenses'] = data['licenses']
    chunked_data['categories'] = data['categories']
    num = 100
    chunked_data['images'] = data['images'][:num]
    chunked_data['annotations'] = data['annotations'][:num]

    chunked_json_filename = kp_json_filename.replace('.json', '_100.json')
    with open(chunked_json_filename, 'w') as f:
        json.dump(chunked_data, f, indent = 2)

chunk_kp_data()
