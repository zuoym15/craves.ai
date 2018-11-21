# Weichao Qiu @ 2018
# Visualize 2D segmentation mask of an object
import human_ground_truth as gt
import matplotlib.pyplot as plt

def main():
    data_root = r'D:\temp\DIVA\car_detection_training'
    cam_name = 'FusionCameraActor_1'

    seq_gt = gt.SeqGT(data_root)
    frames = seq_gt.get_frames()
    frame_id = frames[0]
    
    frame_world_gt = gt.FrameWorldGT(data_root, frame_id)
    frame_cam_gt = gt.FrameCamGT(data_root, frame_id, cam_name)
    print('Number of frames %d' % len(seq_gt.get_frames()))
    print('Number of objs in the world %d' % len(frame_world_gt.get_obj_list()))
    print('Number of objs in the view %d' % len(frame_cam_gt.get_obj_list_in_view()))

    obj_list = frame_cam_gt.world.get_obj_list()
    lit_im = frame_cam_gt.get_lit()
    obj_list_in_view = frame_cam_gt.get_obj_list_in_view()
    big_objs = sorted([v for v in obj_list_in_view if v.category == 'car'], key = lambda obj: obj.get_area(), reverse=True)

    for obj_id in range(10):
        big_obj = big_objs[obj_id]

        fig = plt.figure()
        ax1 = plt.subplot(121); ax2 = plt.subplot(122)
        frame_viewer1 = gt.FrameViewer(fig, ax1, title='Bounding box')
        frame_viewer1.plot_im(lit_im)
        frame_viewer1.plot_bb(big_obj.get_bb(), big_obj.get_name() + '_seg_bb')
        mask = big_obj.get_mask()
        frame_viewer2 = gt.FrameViewer(fig, ax2, title='Seg mask')
        frame_viewer2.plot_im(mask)

        # frame_viewer.plot_bb(big_obj.get_bb_from_kps(), big_obj.get_name() + '_kp_bb')
        frame_viewer1.show()

if __name__ == '__main__':
    main()