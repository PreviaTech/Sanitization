import cv2 
import time
from video_stream import VideoStream
from fake_detector import get_fake_detections
from visualizer import Visualizer

stream = VideoStream(0)
# prev_time = time.time()
visualizer = Visualizer()

while True:
	ret, frame = stream.read()
	if not ret:
		break

	detections = get_fake_detections(frame.shape)
	requires_cleaning = True
	frame = visualizer.draw_banner(frame, requires_cleaning)

	# for det in detections:
	# 	x1, y1, x2, y2 = det["bbox"]
	# 	label = f'{det["class"]} | {det["confidence"]:.2f}'

	# 	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
	# 	color = (0, 255, 255) if det["class"] == "litter_single" else (0, 0, 255)
	# 	cv2.putText(
	# 		frame,
	# 		label,
	# 		(x1, y1 - 10),
	# 		cv2.FONT_HERSHEY_SIMPLEX,
	# 		0.6,
	# 		(0, 255, 0),
	# 		2
	# 	)

	cv2.imshow("Live Feed", frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break

stream.release()
cv2.destroyAllWindows()