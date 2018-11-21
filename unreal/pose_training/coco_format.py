# Weichao Qiu @ 2018
# Support convert the keypoint to coco data format to be compatible with other training code
import os
import skimage.io as io
import matplotlib.pyplot as plt

def test_output():
    # Test whether the converted data format is compatible
    from pycocotools.coco import COCO

    dataDir = os.path.expanduser('~/data/coco/')
    dataType = 'val2017'

    # Show image
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    imgIds = coco.getImgIds(imgIds = [324158])
    img = coco.loadImgs(imgIds[0])[0]
    print(img['coco_url'])
    I = io.imread(img['coco_url'])
    plt.imshow(I)

    # Show human keypoint
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
    coco_kps = COCO(annFile)
    annIds = coco_kps.getAnnIds(imgIds=img['id'])
    anns = coco_kps.loadAnns(annIds)
    print(anns)
    coco_kps.showAnns(anns)
    plt.show()



if __name__ == '__main__':
    test_output()
