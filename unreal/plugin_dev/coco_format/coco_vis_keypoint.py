import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


def vis_coco_default():
    # dataDir = './annotations_trainval2017/'
    dataDir = '.'
    dataType = 'val2017'

    # annFile = '{dataDir}/annotations/instances_{dataType}.json'.format(**locals())
    # coco = COCO(annFile)

    annFile = '{dataDir}/person_keypoints_{dataType}.json'.format(**locals())
    coco_kps = COCO(annFile)

    catIds = coco_kps.getCatIds(catNms = ['person', 'dog', 'skateboard'])
    imgIds = coco_kps.getImgIds(imgIds = [324158])
    img = coco_kps.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    I = io.imread(img['coco_url'])

    # Show keypoint
    plt.imshow(I)
    plt.axis('off')
    annIds = coco_kps.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = None)
    anns = coco_kps.loadAnns(annIds)

    coco_kps.showAnns(anns)
    plt.show()

def vis_diva_human():
    annFile = 'diva_human.json'
    img_folder = './humans/0/'
    coco_kps = COCO(annFile)

    catIds = coco_kps.getCatIds(catNms = ['person', 'dog', 'skateboard'])
    imgIds = coco_kps.getImgIds(catIds=catIds)

    # imgIds = coco_kps.getImgIds(imgIds = [324158])
    img = coco_kps.loadImgs(imgIds[0])[0]
    I = io.imread(os.path.join(img_folder, img['file_name']))

    # Show keypoint
    plt.imshow(I)
    plt.axis('off')
    annIds = coco_kps.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = None)
    anns = coco_kps.loadAnns(annIds)

    coco_kps.showAnns(anns)
    plt.show()

vis_diva_human()
