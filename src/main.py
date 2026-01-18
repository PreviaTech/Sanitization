import cv2
import time 

from video_stream import VideoStream 
from yolo_detector import YOLODetector 
from tracker import Tracker
from filters import DetectionFilter
from segment_analyzer import SegmentAnalyzer
from surface_analyzer import SurfaceAnalyzer
from roi_debug_visualizer import ROIDebugVisualizer
from visualizer import Visualizer


def main():
	video_stream = VideoStream()
	detector = YOLODetector(
		model_path = "yolov8n.pt",
		img_size = 640,
		conf_threshold = 0.25
	)

	tracker = Tracker()
	detection_filter = DetectionFilter()
	segment_analyzer = SegmentAnalyzer()
	surface_analyzer = SurfaceAnalyzer()
	visualizer = Visualizer()
	debug_viz = ROIDebugVisualizer()

	prev_time = time.time()

	while True:
		ret, frame = video_stream.read()
		if not ret or frame is None:
			break

		current_time = time.time()
		fps = 1.0 / max(current_time - prev_time, 1e-6)
		prev_time = current_time
		
		detections = detector.detect(frame)
		detections = detection_filter.apply(detections, frame.shape)
		tracks = tracker.update(detections)
		surface_score = surface_analyzer.update(frame)
		segment_state = segment_analyzer.update(
			tracks,
			frame.shape,
			surface_score,
			fps
		)

		current_time = time.time()
		fps = 1.0 / (current_time - prev_time)
		prev_time = current_time

		debug_frame = debug_viz.visualize(
			frame,
			surface_analyzer,
			segment_state["surface_score"]
		)

		cv2.imshow("Demo", debug_frame)

		if cv2.waitKey(1) & 0xFF == 27:
			break

	video_stream.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()