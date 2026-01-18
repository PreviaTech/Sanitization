import cv2
import numpy as np


class ROIDebugVisualizer:
    """
    Debugging tool to visualize
    
    Usage:
        Press 'r' to toggle ROI overlay
        Press 's' to save current frame with annotations
    """
    
    def __init__(self):
        self.show_roi = True
        self.frame_count = 0
    
    def visualize(self, frame, surface_analyzer, surface_score):
        """
        Draw ROI and diagnostic info on frame.
        
        Args:
            frame: Original BGR frame
            surface_analyzer: SurfaceAnalyzer instance
            surface_score: Current surface score
        
        Returns:
            Annotated frame
        """
        self.frame_count += 1
        vis_frame = frame.copy()
        h, w, _ = vis_frame.shape
        
        # Get ROI config
        roi_config = surface_analyzer.roi_config
        
        x1 = int(w * roi_config["x_start_ratio"])
        x2 = int(w * roi_config["x_end_ratio"])
        y1 = int(h * roi_config["y_start_ratio"])
        y2 = int(h * roi_config["y_end_ratio"])
        
        if self.show_roi:
            # Draw ROI rectangle
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Extract ROI for analysis
            roi = frame[y1:y2, x1:x2]
            
            # Show ROI stats
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Color code based on surface score
            if surface_score > 0.035:
                status_color = (0, 0, 255)  # Red = not clean / dirty
                status_text = "NOT CLEAN"
            elif surface_score > 0.025:
                status_color = (0, 165, 255)  # Orange = moderate
                status_text = "MODERATE"
            elif surface_score > 0.018:
                status_color = (0, 255, 255)  # Yellow = neutral
                status_text = "NEUTRAL"
            else:
                status_color = (0, 255, 0)  # Green = clean
                status_text = "CLEAN"
            
            # Draw info panel
            info_y = y1 - 80
            if info_y < 0:
                info_y = y2 + 20
            
            cv2.putText(vis_frame, f"ROI: {x1},{y1} to {x2},{y2}", 
                       (x1, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(vis_frame, f"Surface: {surface_score:.4f} [{status_text}]", 
                       (x1, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            cv2.putText(vis_frame, f"Mean: {mean_intensity:.1f} | Std: {std_intensity:.1f}", 
                       (x1, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show ROI in corner
            roi_small = cv2.resize(roi, (200, 150))
            vis_frame[10:160, w-210:w-10] = roi_small
            cv2.rectangle(vis_frame, (w-210, 10), (w-10, 160), status_color, 2)
        
        return vis_frame
    
    def handle_keypress(self, key, frame):
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            frame: Current frame
        
        Returns:
            bool: True if program should continue, False to exit
        """
        if key == ord('r'):
            self.show_roi = not self.show_roi
            print(f"ROI overlay: {'ON' if self.show_roi else 'OFF'}")
        
        elif key == ord('s'):
            filename = f"debug_frame_{self.frame_count:04d}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        
        elif key == 27:  # ESC
            return False
        
        return True