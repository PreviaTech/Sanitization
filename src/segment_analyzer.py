from collections import deque


class SegmentAnalyzer:

    """
    temporal stability option 
    """
    # Hysteresis threshold 
    CLEANING_ON_THRESHOLD = 0.50    # Need to exceed thjis to trigger cleaning
    CLEANING_OFF_THRESHOLD = 0.30   # Need to drop below this to clear cleaning 

    # Surface thresholds
    SURFACE_ACCUMULATION_THRESHOLD = 0.025
    SURFACE_CLEAN_THRESHOLD = 0.018
    SURFACE_HOLD_THRESHOLD = 0.022

    # Distance based trigger
    DIRTY_DISTANCE_TRIGGER = 12.0   # cleaning after 12 meters
    DIRTY_DISTANCE_SOFT_CAP = 30.0  # compression accumulation starts
    DIRTY_DISTANCE_HARD_CAP = 50.0  # ABSOLUTE MAX

    #Weighthing
    SURFACE_WEIGHT = 0.6    # surface texture

    # Decay parameters
    BASE_DECAY_RATE = 2.0  
    SUSTAINED_CLEAN_FRAMES = 10     #frames needed for sustained clean
    SUSTAINED_CLEAN_MULTIPLIER = 2.5    #Extra decay 

    def __init__(self, window_size=20, assumed_speed_kmph=10.0):
        """
        Args:
            window_size = temporal smoothing window (note: for faster response reduce)
            assumed_speed_kmph = speed 
        """

        self.window = deque(maxlen = window_size)
        self.requires_cleaning = False
        self.clean_frame_count = 0
        self.dirty_frame_count = 0 

        self.assumed_speed_mps = assumed_speed_kmph / 3.6
        self.dirty_distance_m = 0.0

        # Diagnosis
        self.last_accumulation_reason = "none"
        self.last_decay_reason = "none"

    def compute_object_score(self, tracks, frame_shape):
        """
        computing object detection score with distance based penalties

        Penalties:
            - Detections in upper half of the frame (far from camera)
            - Very small detections (noise)
            - Low confidence detections
        """

        h, w, _ = frame_shape
        frame_area = w * h 
        object_score = 0.0

        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            center_y = (y1 + y2) / 2.0
            norm_y = center_y / h 

            if norm_y < 0.4:
                # Upper 40% of the frame (sky)
                position_weight = 0.2
            elif norm_y < 0.6:
                # Middle
                position_weight = 0.7
            else:
                # bottom 40% 
                position_weight = 1.0

            area = max(0, (x2 - x1) * (y2 - y1)) / frame_area

            conf = getattr(track, "confidence", 0.3)

            if area < 0.005:
                area *= 0.5

            object_score += conf * area * position_weight

        return object_score

    def _update_dirty_distance(self, surface_score, meters_per_frame):
        """
        updating dirty_distance_m based on surface condition

        Logic:
            - Accumulate when surface is dirty and confidence is high
            - Decay aggressively when surface is clean
            - Hold or slow-decay in neutral zone
        """

        # accumulation logic
        if surface_score > self.SURFACE_ACCUMULATION_THRESHOLD:
            self.clean_frame_count = 0
            self.dirty_frame_count += 1

            # Confidence 0.0 at threshold and 1.0 at 0.05+
            effective_surface = surface_score - self.SURFACE_ACCUMULATION_THRESHOLD
            accumulation_confidence = min(effective_surface / 0.025, 1.0)

            # Penalize accu,ulation if noise (single frame spike)
            if self.dirty_frame_count < 3:
                accumulation_confidence *= 0.5

            # Gain
            accumulation_gain = accumulation_confidence * meters_per_frame

            # SOFT CAP - compress accu,ulation above threshold 
            if self.dirty_distance_m > self.DIRTY_DISTANCE_SOFT_CAP:
                overshoot = self.dirty_distance_m - self.DIRTY_DISTANCE_SOFT_CAP
                compression = 1.0 / (1.0 + overshoot / 10.0)
                accumulation_gain *= compression

            self.dirty_distance_m += accumulation_gain
            self.last_accumulation_reason = f"surface={surface_score:.3f}, conf={accumulation_confidence:.2f}"

        # Decay logic
        elif surface_score < self.SURFACE_CLEAN_THRESHOLD:
            self.dirty_frame_count = 0
            self.clean_frame_count += 1

            # clean confidence at 0.0 - 1.0 and 0.0 at threshold 
            clean_confidence = (self.SURFACE_CLEAN_THRESHOLD - surface_score) / self.SURFACE_CLEAN_THRESHOLD
            clean_confidence = max(0.0, min(1.0, clean_confidence))

            # BAse decay
            base_decay = self.BASE_DECAY_RATE * meters_per_frame

            # confidence multiplier
            confidence_multiplier = 1.0 + 3.0 * clean_confidence

            # clean multiplier
            if self.clean_frame_count > self.SUSTAINED_CLEAN_FRAMES:
                sustained_multiplier = self.SUSTAINED_CLEAN_MULTIPLIER
            else:
                sustained_multiplier = 1.0

            total_decay = base_decay * confidence_multiplier * sustained_multiplier
            self.dirty_distance_m = max(0.0, self.dirty_distance_m - total_decay)
            self.last_decay_reason = f"surface={surface_score:.3f}, frames={self.clean_frame_count}"

        # neutral zone
        else:
            self.dirty_frame_count = 0
            self.clean_frame_count += 1

            slow_decay = 0.5 * meters_per_frame
            self.dirty_distance_m = max(0.0, self.dirty_distance_m - slow_decay)
            self.last_decay_reason = f"neutral zone slow decay"

        # hard cap
        self.dirty_distance_m = min(self.dirty_distance_m, self.DIRTY_DISTANCE_HARD_CAP)

    def update(self, tracks, frame_shape, surface_score, fps):
        """
        Main loop
        
        Args:
            - tracks: list of track objects
            - frame_shape: (H, W, C)
            - surface_score: float from SurfaceAnalyzer 
            - fps: current fps

        Returns:
            - dict with keys: requires_cleaning, avg_score, dirty_distance_m, surface_score
        """

        h, w, _ = frame_shape

        # Object based score
        object_score = self.compute_object_score(tracks, frame_shape)

        frame_score = (1.0 - self.SURFACE_WEIGHT) * object_score + self.SURFACE_WEIGHT * surface_score
        self.window.append(frame_score)

        # temporal soothing
        avg_score = sum(self.window) / len(self.window)

        # distance calc
        meters_per_frame = self.assumed_speed_mps / max(fps, 1e-3)

        self._update_dirty_distance(surface_score, meters_per_frame)

        # State transition hysteresis 
        # preventing flickering between clean and dirty states

        if not self.requires_cleaning:
            # clean - check trigger for cleaning
            if avg_score > self.CLEANING_ON_THRESHOLD or self.dirty_distance_m > self.DIRTY_DISTANCE_TRIGGER:
                self.requires_cleaning = True
        else:
            # dirty - check for clearing cleaning
            if avg_score < self.CLEANING_OFF_THRESHOLD and self.dirty_distance_m < (self.DIRTY_DISTANCE_TRIGGER * 0.5):
                self.requires_cleaning = False


        return {
            "requires_cleaning": self.requires_cleaning,
            "avg_score": float(avg_score),
            "dirty_distance_m": float(self.dirty_distance_m),
            "surface_score": float(surface_score),
            "object_score": float(object_score),
            "clean_frame_count": self.clean_frame_count,
            "dirty_frame_count": self.dirty_frame_count,
        }
