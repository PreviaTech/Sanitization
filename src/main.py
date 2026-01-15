import cv2 
import time
from video_stream import VideoStream

stream = VideoStream(0)
prev_time = time.time()

while True:
	ret, frame = stream.read()
	if not ret:
		break

	curr_time = time.time()
	fps = 1 / (curr_time - prev_time)
	prev_time = curr_time

	cv2.putText(
		frame,
		f"FPS: {int(fps)}",
		(20, 40),
		cv2.FONT_HERSHEY_SIMPLEX,
		1,
		(0, 255, 0),
		2
	)

	cv2.imshow("Live Feed", frame)

	if cv2.waitKey(1) & 0xFF == 27:
		break

stream.release()
cv2.destroyAllWindows()