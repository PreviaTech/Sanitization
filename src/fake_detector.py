import math 

frame_count = 0

def get_fake_detections(frame_shape):
	global frame_count
	frame_count += 1

	h, w, _ = frame_shape

	dx = int(40 * math.sin(frame_count * 0.05))

	return [
		{
			"bbox": [
				int(0.3*w + dx),
				int(0.6*h),
				int(0.4*w + dx),
				int(0.75*h)
			],
			"class": "litter_single",
			"confidence": 0.65
		},
		{
			"bbox": [
				int(0.55*w),
				int(0.55*h),
				int(0.85*w),
				int(0.85*h)
			],
			"class": "litter_cluster",
			"confidence": 0.82
		}
	]