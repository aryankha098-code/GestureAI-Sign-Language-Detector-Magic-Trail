# collect_data.py
import cv2
import csv
import os
import numpy as np
from hand_detector import HandDetector

# ── Configure your gestures here ──────────────────────────────────────────────
GESTURES = ["A", "B", "C", "Hello", "Bye", "Love You", "Cute", "Get Lost"]
SAMPLES_PER_GESTURE = 200   # aim for at least 150–300
OUTPUT_FILE = "data/landmarks.csv"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs("data", exist_ok=True)
detector = HandDetector()
cap = cv2.VideoCapture(0)

# Write CSV header if file doesn't exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"lm_{i}" for i in range(63)] + ["label"]
        writer.writerow(header)

def collect_gesture(gesture_name):
    count = 0
    print(f"\n🖐  Get ready for: '{gesture_name}'  (press SPACE to start)")
    
    # Wait for spacebar
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        detector.find_hands(frame)
        cv2.putText(frame, f"Ready: '{gesture_name}' | SPACE to start",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Collect samples
    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        detector.find_hands(frame)
        landmarks = detector.get_landmarks(frame)

        if landmarks is not None:
            with open(OUTPUT_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(landmarks) + [gesture_name])
            count += 1

        progress = int((count / SAMPLES_PER_GESTURE) * 280)
        cv2.rectangle(frame, (10, 50), (290, 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 50), (10 + progress, 70), (0, 220, 100), -1)
        cv2.putText(frame, f"{gesture_name}: {count}/{SAMPLES_PER_GESTURE}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True

# Main loop over all gestures
for gesture in GESTURES:
    if not collect_gesture(gesture):
        print("Stopped early.")
        break
    print(f"  ✓ Collected {SAMPLES_PER_GESTURE} samples for '{gesture}'")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! Data saved to {OUTPUT_FILE}")