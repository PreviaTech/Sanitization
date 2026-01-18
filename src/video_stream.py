import cv2

class VideoStream:

	def __init__(self, source = "vid3.mp4"):
		self.cap = cv2.VideoCapture(source)
		if not self.cap.isOpened():
			raise RuntimeError("Cannot open video source")


	def read(self):
		return self.cap.read()

	def release(self):
		self.cap.release()