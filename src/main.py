import cv2 
import time
from video_stream import VideoStream
from fake_detector import get_fake_detections
from segment_analyzer import SegmentAnalyzer
from filters import apply_physical_filters
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

	segment_analyzer = SegmentAnalyzer(
		window_size = 50,
		threshold = 0.20
	)

	stable_detections = [
		{
			"bbox": t.bbox,
			"class": t.cls,
			"confidence": t.conf
		}
		for t in stable_tracks
	]

	filtered_detections = apply_physical_filters(
		stable_detections,
		frame.shape
	)

	requires_cleaning, segment_score = segment_analyzer.update(
		filtered_detections,
		frame.shape
	)

	frame = visualizer.draw_detections(frame, filtered_detections)
	frame = visualizer.draw_banner(frame, requires_cleaning)
	frame = visualizer.draw_metrics(frame, segment_score)

	cv2.imshow("Live Feed", frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break

stream.release()
cv2.destroyAllWindows()