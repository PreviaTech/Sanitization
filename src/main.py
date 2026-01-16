import cv2 
import time
from video_stream import VideoStream
from fake_detector import get_fake_detections
from tracker import Tracker
from visualizer import Visualizer

stream = VideoStream(0)
# prev_time = time.time()
visualizer = Visualizer()
tracker = Tracker()

while True:
	ret, frame = stream.read()
	if not ret:
		break

	detections = get_fake_detections(frame.shape)
	stable_tracks = tracker.update(detections)

	stable_detections = [
		{
			"bbox": t.bbox,
			"class": t.cls,
			"confidence": t.conf
		}
		for t in stable_tracks
	]

	frame = visualizer.draw_detections(frame, stable_detections)
	frame = visualizer.draw_banner(frame, requires_cleaning = True)

	cv2.imshow("Live Feed", frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break

stream.release()
cv2.destroyAllWindows()