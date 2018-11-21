# Weichao Qiu @ 2018
import PyQt5.QtWidgets as qw # TODO: modify it to be PyQt4
import PyQt5.QtGui as qui
import sys, threading, time

from unrealcv import client
from unrealcv.util import read_png, read_npy

class FPSCounter:
	''' Compute FPS for per second, no moving average '''
	def __init__(self):
		self.count = 0
		self.start = 0
		self.elapsed = 0
		self.FPS = 0

	def tick(self):
		self.count += 1
		now = time.time()
		if self.start == 0:
			self.start = now

		elapsed = now - self.start
		if elapsed > 1: # Per second stat
			# Make print optional
			self.FPS = self.count

			# reset counter
			self.start = now
			self.count = 0

	def __str__(self):
		return str(self.FPS)

class FusionSensor:
	def __init__(self, sensors):
		self.sensors = sensors

	def capture_frame(self):
		''' Return a list for fusion sensor '''
		return [s.capture_frame() for s in self.sensors]

class UE4Sensor:
	def __init__(self, cmd):
		client.connect()
		self.cmd = cmd

	def capture_frame(self):
		res = client.request(self.cmd)
		if self.cmd.endswith('png') or self.cmd.endswith('bmp'):
			frame = read_png(res)
		elif self.cmd.endswith('npy'):
			frame = read_npy(res)
			frame[frame>5000] = 0 # remove too far away data
			frame = frame / frame.max() * 255.0 # For visualization only
			frame = frame.astype('uint8')
		else:
			assert False, 'Unknown format'

		return frame

class SensorStreamer:
	def __init__(self, sensor):
		self.on_data_ready = None
		self.sensor = sensor

	def _capture_func(self):
		while self.running:
			frame = self.sensor.capture_frame()
			if self.on_data_ready != None:
				self.on_data_ready(frame)

	def start(self):
		self.running = True

		self.thread = threading.Thread(target = self._capture_func)
		self.thread.daemon = True
		self.thread.start()

	def stop(self):
		self.running = False

	def __del__(self):
		self.running = False

def draw_frame(panel, frame):
	if frame is None:
		return

	if len(frame.shape) == 2:
		height, width = frame.shape; channel = 1
	else:
		height, width, channel = frame.shape

	bytes_per_line = channel * width
	data_ptr = frame.tobytes()

	if channel == 4:
		qImg = qui.QImage(data_ptr, width, height, bytes_per_line, qui.QImage.Format_RGBA8888)
		# Check format here http://doc.qt.io/qt-5/qimage.html#Format-enum
	elif channel == 3:
		qImg = qui.QImage(data_ptr, width, height, bytes_per_line, qui.QImage.Format_RGB888)
	elif channel == 1:
		qImg = qui.QImage(data_ptr, width, height, bytes_per_line, qui.QImage.Format_Grayscale8)
	else:
		assert(False)

	pixmap = qui.QPixmap(qImg)
	panel.setPixmap(pixmap)
	panel.resize(width, height)

def main():
	app = qw.QApplication([])
	root = qw.QWidget()
	root.resize(640, 480)

	grid_layout = qw.QGridLayout()
	root.setLayout(grid_layout)

	fps_lbl = qw.QLabel(root)
	fps_counter = FPSCounter()

	sensors = [
		UE4Sensor('vget /sensor/0/lit png'),
		UE4Sensor('vget /sensor/0/object_mask png'),
		# UE4Sensor('vget /sensor/0/depth npy'),
	]
	streamer = SensorStreamer(FusionSensor(sensors))

	col = 2
	panels = []
	for i in range(len(sensors)):
		panel = qw.QLabel(root)
		panels.append(panel)
		grid_layout.addWidget(panel, i / col, i % col)

	streamer.on_data_ready = lambda frames: (
		fps_counter.tick(),
		fps_lbl.setText(str(fps_counter)),
		[draw_frame(p, v) for (p, v) in zip(panels, frames)]
	)

	streamer.start()
	root.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()
