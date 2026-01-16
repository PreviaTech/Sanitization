def iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	inter_w = max(0, xB - xA)
	inter_h = max(0, yB - yA)
	inter_area = inter_w * inter_h

	areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	union = areaA + areaB - inter_area + 1e-6

	return inter_area / union


class Track:
	def __init__(self, det):
		self.bbox = det["bbox"]
		self.cls = det["class"]
		self.conf = det["confidence"]

		self.age = 1
		self.missed = 0


class Tracker:
	def __init__(self, iou_thresh = 0.4, max_missed = 8, min_age = 5):
		self.tracks = []
		self.iou_thresh = iou_thresh
		self.max_missed = max_missed
		self.min_age = min_age

	def update(self, detections):
		for tr in self.tracks:
			tr.missed += 1

		for det in detections:
			matched = False
			for tr in self.tracks:
				if tr.cls == det["class"] and iou(tr.bbox, det["bbox"]) > self.iou_thresh:
					tr.conf = 0.3 * det["confidence"] + 0.7 * tr.conf
					tr.bbox = det["bbox"]
					tr.age += 1
					matched = True
					break

			if not matched:
				self.tracks.append(Track(det))


		self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

		return [t for t in self.tracks if t.age >= self.min_age]