# Weichao Qiu @ 2018
from diva.ground_truth import SeqGT, SeqViewer, FrameCamGT
import os, argparse


def main(data_root):
    seq_gt = SeqGT(data_root) 
    frames = seq_gt.get_frames()
    cams = seq_gt.get_cams()
    cam_name = cams[0]
    # print('Frames of this video')
    # print(frames)

    frame_gt = FrameCamGT(data_root, frames[0], cam_name)
    im = frame_gt.get_lit()
    
    print('DB stat of %s' % data_root)
    print('# of frames %d: [%d, %d]' % (len(frames), frames[0], frames[-1]))
    print('Image size:', im.shape)
    print(cams)
    stat = seq_gt.stat()
    print(stat)
    # frame_range = [0,500]
    # seq_gt.clip(frame_range)
    # seq_viewer = SeqViewer()
    # seq_gt.vis_seq()
    # seq_viewer.vis_activity(seq_gt.get_activity())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')

    args = parser.parse_args()
    data_root = args.data_root
    main(data_root)