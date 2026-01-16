import cv2 

class Visualizer:
	def __init__(self):
		self.color = {
			"litter_single": (0, 255, 255),
			"litter_cluster": (0, 0, 255)
		}

	def draw_detections(self, frame, detections):
		for det in detections:
			x1, y1, x2, y2 = det["bbox"]
			cls = det["class"]
			conf = det["confidence"]

			color = self.color.get(cls, (255, 255, 255))
			label = f"{cls} | {conf:.2f}"

			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			cv2.putText(
					frame,
					label, 
					(x1, max(y1 - 10, 20)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					color,
					2
				)

		return frame

	def draw_banner(self, frame, requires_cleaning):
		h, w, _ = frame.shape

		if requires_cleaning:
			text = "Area requires cleaning"
			color = (0, 0, 255)

		else:
			text = "Road segment clean"
			color = (0, 255, 0)

		cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
		cv2.putText(
				frame, 
				text,
				(int(w * 0.05), 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.2,
				color,
				3
		)

		return frame

	def draw_metrics(self, frame, score):
		text = f"Cleanliness score: {score:.3f}"
		cv2.putText(
			frame,
			text,
			(20, frame.shape[0] - 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2
		)

		return frame