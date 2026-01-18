import torch 
import numpy as np
from ultralytics import YOLO


class YOLODetector:
	"""
	- Running YOLO inference on a frame
	- Converting YOLO outputs into locked detection format
	- Performing deterministic class mapping only
	"""

	def __init__(
		self,
		model_path: str = "yolov8n.pt",
		device: str = None,
		img_size: int = 640,
		conf_threshold: float = 0.25,
		iou_threshold: float = 0.45,
		max_detections: int = 50
	):

		"""
		model_path: 
			Pretrained YOLO model path (YOLO8n.pt used for stability and thermals)

		device:
			"cuda" | "cpu" | None (auto)

		conf_threshold:
			Inference resolution cap (controls thermals)

		iou_threshold:
			NMS IoU threshold

		max_detections:
			Hard cap to prevent pathological overload 
		"""

		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"

		self.device = device
		self.img_size = img_size
		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold
		self.max_detections = max_detections
		self.model = YOLO(model_path)
		self.model.to(self.device)

		self.warmup()

	def warmup(self):
		"""
		single warmup inference to stabilize first-frame latency 
		"""

		dummy = np.zeros((self.img_size, self.img_size, 3), dtype = np.uint8)
		_ = self.model(
			dummy,
			imgsz = self.img_size,
			conf = self.conf_threshold,
			iou = self.iou_threshold,
			verbose = False
		)

	def detect(self, frame):
		"""
		Args:
			frame: BGR image (numpy array)

		Returns:
			List of detections in locked format:
			{
				"bbox": [x1, y1, x2, y2],
				"class": "litter_single" | "litter_cluster",
				"confidence": float
			}
		"""

		height, width = frame.shape[:2]
		detections = []

		results = self.model(
			frame,
			imgsz = self.img_size,
			conf = self.conf_threshold,
			iou = self.iou_threshold,
			max_det = self.max_detections,
			device = self.device,
			verbose = False
		)

		if not results:
			return detections

		boxes = results[0].boxes
		if boxes is None or len(boxes) == 0:
			return detections 

		for box in boxes:
			xyxy = box.xyxy[0].cpu().numpy()
			conf = float(box.conf[0].cpu().numpy())

			x1, y1, x2, y2 = xyxy.tolist()

			x1 = max(0, min(x1, width - 1))
			y1 = max(0, min(y1, height - 1))
			x2 = max(0, min(x2, width - 1))
			y2 = max(0, min(y2, height - 1))

			box_w = x2 - x1
			box_h = y2 - y1
			box_area = box_w * box_h
			frame_area = width * height 

			if box_area <= 0:
				continue 

			area_ratio = box_area / frame_area

			# This is where dust heavy regions get implicitly treated as clusters without introducing a dust class

			if area_ratio >= 0.02:
				mapped_class = "litter_cluster"
			else:
				mapped_class = "litter_single"


			detections.append(
				{
					"bbox": [int(x1), int(y1), int(x2), int(y2)],
					"class": mapped_class,
					"confidence": conf
				}
			)

		return detections 












