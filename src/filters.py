def apply_physical_filters(detections, frame_shape):
	h, w, _ = frame_shape
	filtered = []

	for det in detections:
		x1, y1, x2, y2 = det["bbox"]
		cls = det["class"]

		box_w = max(1, x2 - x1)
		box_h = max(1, y2 - y1)

		area_ratio = (box_w * box_h) / (w * h)
		aspect_ratio = box_w / box_h

		if aspect_ratio < 0.2 or aspect_ratio > 5.0:
			continue

		if cls == "litter_single":
			if area_ratio < 0.0005 or area_ratio > 0.02:
				continue

		if cls == "litter_cluster":
			if area_ratio < 0.01 or area_ratio > 0.4:
				continue

		center_y = (y1 + y2) / 2
		norm_y = center_y / h

		if norm_y < 0.3:
			det["confidence"] *= 0.6

		filtered.append(det)

	return filtered