class DetectionFilter:
	"""
	Post YOLO detection filtering 
	"""

	def __init__(self):
		# aspect ratio bounds (width/height)
		self.min_aspect_ratio = 0.2
		self.max_aspect_ratio = 5.0

		# Area ratio bounds 
		self.min_area_ratio = 0.001

		# vertical postioon zones 
		self.upper_zone_threshold = 0.3 # Above this = far from camera
		self.upper_zone_penalty = 0.6

		# Horizontal edge zones
		self.edge_zone_width = 0.1 # 10 % from edge
		self.edge_zone_penalty = 0.7 

	def _compute_position_penalty(self, bbox, frame_shape):
		"""
		Compute confidence penalty based on bounding box position

			- upper regions mroe likely to be distant objects or sky 
			- edge regions more likely to be out of context objects
		"""

		h, w, _ = frame_shape
		x1, y1, x2, y2 = bbox

		# vertical positions
		center_y = (y1 + y2) / 2.0
		norm_y = center_y / h 

		if norm_y < self.upper_zone_threshold:
			vertical_penalty = self.upper_zone_penalty
		else:
			vertical_penalty = 1.0

		#horizontal position
		center_x = (x1 + x2) / 2.0
		norm_x = center_x / w

		if norm_x < self.edge_zone_width or norm_x > (1.0 - self.edge_zone_width):
			horizontal_penalty = self.edge_zone_penalty
		else:
			horizontal_penalty = 1.0 

		return vertical_penalty * horizontal_penalty

	def apply(self, detections, frame_shape):
		"""
		Args:
			detections: list of detection dicts:
						- "bbox": [x1, y1, x2, y2]
						- "class": str
						- "confidence": float
			frame_shape = (H, W, C)

		Returns:
			filtered list of detections

		"""

		h, w, _ = frame_shape
		filtered = []

		frame_area = w * h 

		for det in detections:
			x1, y1, x2, y2 = det["bbox"]
			cls = det["class"]
			conf = det["confidence"]

			x1 = max(0, min(x1, w - 1))
			x2 = max(0, min(x2, w - 1))
			y1 = max(0, min(y1, h - 1))
			y2 = max(0, min(y2, h - 1))

			box_w = max(1, x2 - x1)
			box_h = max(1, y2 - y1)
			box_area = box_w * box_h

			area_ratio = box_area / frame_area
			aspect_ratio = box_w / box_h

			if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
				continue

			if area_ratio < self.min_area_ratio:
				continue

			if cls == "litter_cluster":
				if area_ratio < 0.01 or area_ratio > 0.4:
					continue

			position_penalty = self._compute_position_penalty(
				[x1, y1, x2, y2],
				frame_shape
			)

			adjusted_conf = conf * position_penalty

			det["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
			det["confidence"] = adjusted_conf

			filtered.append(det)

		return filtered