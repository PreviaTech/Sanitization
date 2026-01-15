import cv2 
import time
from video_stream import VideoStream

stream = VideoStream(0)

while True:
	ret, frame = stream.read()
	if not ret:
		print("Frame read failed")
		break

	cv2.imshow("Live feed:", frame)

	if cv2.waitKey(1) & 0xFF == 27: 
		break

stream.release()
cv2.destroyAllWindows()