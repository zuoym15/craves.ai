# Weichao Qiu @ 2018
# Video related utility functions

# import skvideo.io
import cv2
class VideoWriter:
    def __init__(self, filename):
        ffmpeg_paras = {'-r': 1}
        self.writer = skvideo.io.FFmpegWriter(filename)

    def write_frame(self, frame_data):
        self.writer.writeFrame(frame_data)

    def __del__(self):
        self.writer.close()

class OpencvVideoWriter:
    def __init__(self, filename):
        self.video = None 
        # Postpone the video initialization until the first frame
        self.filename = filename
        # Example from here: https://stackoverflow.com/questions/14440400/creating-a-video-using-opencv-2-4-0-in-python

    def add_image_file(self, filename):
        frame_data = cv2.imread(filename)
        self.add_image_data(frame_data)

    def add_image_data(self, frame_data):
        height, width, c = frame_data.shape
        if self.video is None:
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            # https: // docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
            # fourcc = cv2.cv.CV_FOURCC(*'XVID')
            # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
            # fourcc = -1
            self.video = cv2.VideoWriter(self.filename, fourcc, 1, (width, height))
        self.video.write(frame_data)

    def __del__(self):
        self.video.release()

def compress_video(video_filename, images):
    ''' Compress image files into a video
    video_filename : output filename
    images : a list of image filename '''
    video_writer = OpencvVideoWriter(video_filename)
    for image_filename in images:
        video_writer.add_image_file(image_filename)
