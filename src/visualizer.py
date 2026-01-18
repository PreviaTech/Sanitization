import cv2


class Visualizer:
    def __init__(self):
        self.color = {
            "litter_single": (0, 255, 255),
            "litter_cluster": (0, 0, 255)
        }

    def draw_detections(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            cls = getattr(track, "class_name", "unknown")
            conf = getattr(track, "confidence", None)

            if conf is None:
                label = f"{cls}"
            else:
                label = f"{cls} | {conf:.2f}"

            color = self.color.get(cls, (255, 255, 255))

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

    def draw_metrics(self, frame, state, fps):
        y = frame.shape[0] - 70

        cv2.putText(
            frame,
            f"Cleanliness score:{state['avg_score']:.3f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Clean frames: {state.get('clean_frame_count', 0)}",
            (20, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Dirt distance:{state['dirty_distance_m']:.1f} m",
            (20, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return frame

    def draw_roi_overlay(self, frame, surface_score, dirt_distance):
        h, w, _ = frame.shape
        surface_score = float(surface_score)
        dirt_distance = float(dirt_distance)

        roi_x1 = int(w * 0.75)
        roi_x2 = w

        overlay = frame.copy()

        # Color logic
        if surface_score > 0.06 or dirt_distance > 10.0:
            color = (0, 0, 255)
            alpha = 0.35
        elif surface_score > 0.03 or dirt_distance > 4.0:
            color = (0, 165, 255)
            alpha = 0.25
        else:
            return frame

        cv2.rectangle(
            overlay,
            (roi_x1, 0),
            (roi_x2, h),
            color,
            -1
        )

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame


    def draw(self, frame, tracks, segment_state, fps):
        requires_cleaning = segment_state["requires_cleaning"]
        avg_score = float(segment_state["avg_score"])
        dirt_distance = float(segment_state["dirty_distance_m"])
        surface_score = float(segment_state["surface_score"])

        frame = self.draw_roi_overlay(frame, surface_score, dirt_distance)
        frame = self.draw_detections(frame, tracks)
        frame = self.draw_banner(frame, requires_cleaning)
        frame = self.draw_metrics(frame, segment_state, fps)

        return frame


    def draw_surface_roi(self, frame, surface_score, requires_cleaning):
        h, w, _ = frame.shape

        roi_x1 = int(w * 0.65)
        roi_y1 = int(h * 0.65)

        overlay = frame.copy()

        if surface_score > 0.03:
            if requires_cleaning:
                color = (0, 0, 255)
                alpha = 0.35
            else:
                color = (0, 255, 255)
                alpha = 0.25

            cv2.rectangle(
                overlay,
                (roi_x1, roi_y1),
                (w, h),
                color,
                -1
            )

            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame