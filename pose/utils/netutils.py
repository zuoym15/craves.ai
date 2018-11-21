import imageio
import matplotlib.pyplot as plt
import cv2
# import Image
from PIL import Image
# from Pillow import Image
import threading
# from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from http.server import BaseHTTPRequestHandler,HTTPServer
# from SocketServer import ThreadingMixIn
from socketserver import ThreadingMixIn

# import StringIO
import io
import time

from urllib import request



def cam_client(host = '127.0.0.1', imshow = False):
    img_dir = 'http://' + host + ':8080/cam.png'
    im = imageio.imread(img_dir)

    if imshow:
        plt.imshow(im)
        plt.show()

    return im

def send_command(command, host = '127.0.0.1'):
    command_url = 'http://' + host + ':8080/' + command
    print(command_url)
    request.urlopen(command_url)

capture=None

class CamHandler(BaseHTTPRequestHandler):
    # def send_frame_PIL(self):
    #     jpg = Image.fromarray(imgRGB)
    #     # tmpFile = StringIO.StringIO()
    #     # tmpFile = io.StringIO()
    #     tmpFile = io.BytesIO()
    #     # jpg.save(tmpFile,'JPEG')
    #     jpg.save(tmpFile,'PNG')
    #     # print(str(tmpFile.getbuffer().nbytes))
    #     # self.wfile.write(b"--jpgboundary")
    #     # self.send_header('Content-type','image/jpeg')
    #     # self.send_header('Content-length',str(len(tmpFile)))
    #     # self.send_header('Content-length',str(len(tmpFile).len))
    #     # self.send_header('Content-length', str(len(bytes)))
    #     # jpg.save(self.wfile,'JPEG') # Probably because wfile is closed?
    #     self.wfile.write(tmpFile.getbuffer())

    def send_frame(self):
        self.send_frame_cv2()

    def send_frame_cv2(self):
        rc,img = capture.read()
        if not rc:
            print('Fail to capture image')
            return 

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        bytes = cv2.imencode('.png', imgRGB)[1]
        print(len(bytes))

        self.send_header('Content-type','image/png')
        self.end_headers()

        self.wfile.write(cv2.imencode('.png', img)[1])

    def do_GET(self):
        if self.path.endswith('.png'):
            self.send_response(200)
            # self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            # self.end_headers()
            # while True:
            self.send_frame()

        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            # self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
            self.wfile.write('<img src="http://127.0.0.1:8080/cam.png"/>')
            self.wfile.write('</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def main():
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #capture.set(cv2.CAP_PROP_SATURATION,0.2)
    # capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320); 
    # capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240);
    # capture.set(cv2.cv.CV_CAP_PROP_SATURATION, 0.2);

    global img
    server = ThreadedHTTPServer(('0.0.0.0', 8080), CamHandler)
    print("server started")
    server.serve_forever()
    # except KeyboardInterrupt:
    #     capture.release()
    #     server.socket.close()

if __name__ == '__main__':
    main()