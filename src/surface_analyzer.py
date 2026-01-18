import cv2
import numpy as np
from collections import deque


class SurfaceAnalyzer:
    """
    Surface texture analysis with viewpoint robustness.
    
    CRITICAL DEPLOYMENT NOTE:
    - Current ROI tuned for BIKE-MOUNTED LEFT-SIDE CAMERA
    - MUST be recalibrated for van top-down deployment
    - Flag: BIKE_CAMERA_MODE vs VAN_CAMERA_MODE
    """
    
    # === ROI Configuration ===
    BIKE_CAMERA_MODE = True  # False for van
    
    # Sky detection threshold
    SKY_INTENSITY_THRESHOLD = 200  # Mean intensity above this = sky
    SKY_SATURATION_THRESHOLD = 30  # Low saturation + high intensity = overexposed
    
    # Weighting for texture components
    VARIANCE_WEIGHT = 0.6
    EDGE_WEIGHT = 0.4

    def __init__(self, window_size=15):
        """
        Args:
            window_size: Temporal smoothing window (reduced from 30 for faster response)
        """
        self.window = deque(maxlen=window_size)
        
        # ROI configuration - will be set based on camera mode
        self.roi_config = self._get_roi_config()
        
        # Diagnostic counters
        self.sky_suppression_count = 0
        self.total_frames = 0

    def _get_roi_config(self):
        """
        Returns ROI configuration based on camera mounting.
        """
        if self.BIKE_CAMERA_MODE:
            # Bike camera: side-mounted, angled left
            # Road is in center-left and lower portion
            return {
                "x_start_ratio": 0.25,  # Start 25% from left
                "x_end_ratio": 0.65,    # End 65% from left
                "y_start_ratio": 0.50,  # Start 50% from top
                "y_end_ratio": 1.0      # Full height
            }
        else:
            # Van camera: top-mounted, angled down+forward
            # Road is in center and lower portion
            return {
                "x_start_ratio": 0.30,
                "x_end_ratio": 0.70,
                "y_start_ratio": 0.60,
                "y_end_ratio": 1.0
            }

    def _extract_roi(self, frame):
        """
        Extract region of interest based on camera configuration.
        """
        h, w, _ = frame.shape
        
        x1 = int(w * self.roi_config["x_start_ratio"])
        x2 = int(w * self.roi_config["x_end_ratio"])
        y1 = int(h * self.roi_config["y_start_ratio"])
        y2 = int(h * self.roi_config["y_end_ratio"])
        
        roi = frame[y1:y2, x1:x2]
        return roi

    def _is_sky_or_overexposed(self, roi):
        """
        Detect if ROI contains sky or bright overexposure artifacts.
        
        Returns:
            bool: True if ROI should be suppressed
        """
        # Convert to HSV for better sky detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Extract channels
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # Mean intensity
        mean_intensity = np.mean(v_channel)
        
        # Mean saturation
        mean_saturation = np.mean(s_channel)
        
        # Sky characteristics:
        # - High intensity (bright)
        # - Low saturation (not colorful)
        is_sky = (mean_intensity > self.SKY_INTENSITY_THRESHOLD and 
                  mean_saturation < self.SKY_SATURATION_THRESHOLD)
        
        # Also check for overexposure (very bright pixels)
        bright_pixel_ratio = np.mean(v_channel > 240)
        is_overexposed = bright_pixel_ratio > 0.4
        
        return is_sky or is_overexposed

    def _texture_variance(self, gray):
        """
        Normalized local variance as texture disorder proxy.
        
        Higher variance = more texture variation = potentially dirtier
        """
        return np.var(gray) / (255.0 ** 2)

    def _edge_density(self, gray):
        """
        Fraction of pixels that are edges.
        
        More edges = more texture detail = potentially dirtier
        """
        edges = cv2.Canny(gray, 50, 150)
        return np.mean(edges > 0)

    def _compute_raw_score(self, gray):
        """
        Compute raw surface score from texture features.
        """
        var_score = self._texture_variance(gray)
        edge_score = self._edge_density(gray)
        
        raw_score = self.VARIANCE_WEIGHT * var_score + self.EDGE_WEIGHT * edge_score
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        
        return raw_score

    def update(self, frame):
        """
        Analyze surface texture and return smoothed score.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            float: Smoothed surface score [0.0, 1.0]
                   Higher = dirtier/more textured
                   0.0 = suppressed (sky/overexposed)
        """
        self.total_frames += 1
        
        # Extract ROI
        roi = self._extract_roi(frame)
        
        # Check for sky/overexposure
        if self._is_sky_or_overexposed(roi):
            self.sky_suppression_count += 1
            self.window.append(0.0)
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Compute raw score
        raw_score = self._compute_raw_score(gray)
        
        # Add to temporal window
        self.window.append(raw_score)
        
        # Temporal smoothing
        smooth_score = sum(self.window) / len(self.window)
        
        return smooth_score

    def get_diagnostics(self):
        """
        Return diagnostic info for tuning and debugging.
        """
        suppression_rate = (self.sky_suppression_count / max(self.total_frames, 1)) * 100
        return {
            "camera_mode": "bike" if self.BIKE_CAMERA_MODE else "van",
            "roi_config": self.roi_config,
            "sky_suppression_rate": f"{suppression_rate:.1f}%",
            "total_frames": self.total_frames
        }