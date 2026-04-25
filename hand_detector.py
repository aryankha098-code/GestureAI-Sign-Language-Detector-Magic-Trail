# hand_detector.py  (MediaPipe 0.10.x + Python 3.13 — no mediapipe.python needed)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # silence oneDNN warnings

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python.core.base_options import BaseOptions

# Hand connection pairs (which landmarks to draw lines between)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm
]

class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5,
                 model_path="hand_landmarker.task"):

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.results  = None
        self.frame_ts = 0

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.frame_ts += 33
        self.results = self.detector.detect_for_video(mp_image, self.frame_ts)

        if draw and self.results.hand_landmarks:
            h, w = frame.shape[:2]
            for hand in self.results.hand_landmarks:
                # Convert to pixel coords
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                # Draw connections
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)

                # Draw landmark dots
                for i, (cx, cy) in enumerate(pts):
                    # Fingertips (4,8,12,16,20) are bigger
                    r = 6 if i in (4, 8, 12, 16, 20) else 4
                    cv2.circle(frame, (cx, cy), r, (0, 255, 150), -1)
                    cv2.circle(frame, (cx, cy), r, (0, 180, 100),  1)

        return frame

    def get_landmarks(self, frame):
        if not self.results or not self.results.hand_landmarks:
            return None

        hand  = self.results.hand_landmarks[0]
        wrist = hand[0]
        lm_list = []
        for lm in hand:
            lm_list.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        return np.array(lm_list, dtype=np.float32)

    def close(self):
        self.detector.close()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    print("Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        frame     = detector.find_hands(frame)
        landmarks = detector.get_landmarks(frame)

        if landmarks is not None:
            text, color = "Hand detected | 63 landmarks OK", (0, 220, 100)
        else:
            text, color = "No hand detected — show your hand!", (0, 80, 255)

        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imshow("Hand Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()