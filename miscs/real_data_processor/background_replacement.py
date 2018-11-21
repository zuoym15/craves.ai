import cv2
import numpy as np
import os
import json

class background_replace(object):
    def __init__(self, img_folder):
        print('start going through folder...')
        self.img_folder = img_folder
        self.img_list = self.find_img(img_folder)
        self.num_bg_img = len(self.img_list)
        print('finish going through folder, total images: ' + str(self.num_bg_img))

    def find_img(self, file_dir):   
        L = []
        for root, dirs, files in os.walk(file_dir):  
            for file in files:  
                if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':
                    L.append(os.path.join(root, file)) 
        return L

    def read_random_bg(self):
        id = np.random.choice(self.num_bg_img, 1)
        return cv2.imread(self.img_list[id[0]])

    def create_mask(self, img, color):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        if color == 'green':
            threshold = [(20, 0, 128), (235, 128, 255)]
        elif color == 'white':
            threshold = [(100, 110, 110), (200, 140, 140)]

        else:
            raise Exception('Color undefined')
        
        mask = cv2.inRange(img, threshold[0], threshold[1])
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        # mask =  cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask =  cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = mask > 0

        # img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)

        # thres_img = np.zeros_like(img, np.uint8)
        # thres_img[mask] = img[mask]

        binary_img = np.zeros((img.shape[0],img.shape[1]), np.uint8)
        binary_img[mask] = 255

        # cv2.imshow('img', binary_img)
        # cv2.waitKey(0)
        # exit(0)

        return mask

    def mask2bb(self, mask):
        ''' Convert binary seg mask of object to bouding box, (x0, y0, x1, y1) format '''
        y, x = np.where(mask == False)
        bb = [[int(x.min()), int(y.min())], [int(x.max()), int(y.max())]]
        return bb

    def replace(self, img, color = 'green'):
        H, W, _ = img.shape
        mask = self.create_mask(img, color)
        bg = self.read_random_bg()
        bg = cv2.resize(bg, (2*W, 2*H))
        ul = [np.random.randint(0, H), np.random.randint(0, W)]#upperleft corner
        bg = bg[ul[0]:ul[0]+H, ul[1]:ul[1]+W, :]
        img[mask] = bg[mask]
        return img

if __name__ == "__main__":
    obj = background_replace('./sample_bg')
    #data_dir = 'C:\\Users\\Yiming\\Desktop\\arm_miscs\\data\\img\\000'
    data_dir = 'C:\\Users\\Yiming\\Desktop\\arm-pose\\data\\real_20181010'
    L = obj.find_img(data_dir)
    anno = {}
    for i in range(len(L)):
        img_dir = L[i]
        img = cv2.imread(img_dir)
        anno[os.path.basename(img_dir)] = obj.mask2bb(obj.create_mask(img, 'white'))
        #cv2.imshow('img', obj.replace(img, 'white'))
        #cv2.waitKey(0)
        print('processing {}/{}'.format(i, len(L)))

    with open(os.path.join(data_dir, 'pts.json'), 'w') as f:
        json.dump(anno, f)

