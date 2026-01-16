from collections import deque

class SegmentAnalyzer:
	def __init__(self, window_size = 50, threshold = 0.20):
		"""
		window_size: number of recent frames to consider
		threshold: score above which cleaning is required
		"""

		self.window = deque(maxlen = window_size)
		self.threshold = threshold

	def update(self, detections, frame_shape):
		h, w, _ = frame_shape
		frame_area = w * h

		frame_score = 0.0

		for det in detections:
			x1, y1, x2, y2 = det["bbox"]
			area = max(0, (x2 - x1) * (y2 - y1)) / frame_area

			frame_score += det["confidence"] * area

		self.window.append(frame_score)

		avg_score = sum(self.window) / len(self.window)

		requires_cleaning = avg_score > self.threshold

		return requires_cleaning, avg_score