import cv2
import os
import numpy as np

def find_mp4(file_dir):   
    L = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.join(root, file)) 
    return L

def video2img(video_dir, save_dir, transform = None):

    if transform is not None:
        M = np.float32([[1,0,transform[0]],[0,1,transform[1]]])
        
    vc = cv2.VideoCapture(video_dir) #读入视频文件
    c=0
    rval=vc.isOpened()
    timeF = 30  #视频帧计数间隔频率
    while rval:   #循环读取视频帧   
        rval, frame = vc.read()
        if rval :
            if c%timeF == 0:

                rows,cols = frame.shape[0:2]
                frame = cv2.warpAffine(frame,M,(cols,rows))

                cv2.imwrite(os.path.join(save_dir, str(c).zfill(8) + '.jpg'), frame) #存储为图像
                cv2.waitKey(1)
            c = c + 1
        else:
            print('fail to read video!')
            break
    vc.release()

def main():
    data_dir = 'C:/Users/Yiming/Desktop/arm_miscs/data/validate'
    img_dir = os.path.join(data_dir, 'frame')
    video_dir = os.path.join(data_dir, 'video')
    video_list = find_mp4(video_dir)
    for i in range(len(video_list)):
        print('start processing video {}/{}'.format(i+1, len(video_list)))
        cur_dir = os.path.join(img_dir, 'batch_' + str(i).zfill(3))
        if not os.path.isdir(cur_dir):
            os.mkdir(cur_dir)
        video2img(video_list[i], cur_dir, [-23, 0])

if __name__ == "__main__":
    main()



