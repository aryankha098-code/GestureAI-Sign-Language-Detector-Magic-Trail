# run_detector.py
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from hand_detector import HandDetector

# ── Load model & labels ────────────────────────────────────────────────────────
model  = tf.keras.models.load_model("models/sign_language_model.h5")
labels = np.load("models/labels.npy", allow_pickle=True)

detector = HandDetector()
cap = cv2.VideoCapture(0)

# Smoothing: only confirm a prediction if it's stable for N frames
STABILITY_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.85

prediction_buffer = deque(maxlen=STABILITY_FRAMES)
word_buffer = []
last_confirmed = None
stable_count = 0

def draw_overlay(frame, prediction, confidence, word):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Prediction + confidence bar
    color = (0, 220, 100) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"{prediction}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    bar_w = int(confidence * 200)
    cv2.rectangle(frame, (120, 20), (320, 38), (60, 60, 60), -1)
    cv2.rectangle(frame, (120, 20), (120 + bar_w, 38), color, -1)
    cv2.putText(frame, f"{confidence:.0%}", (325, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Word builder at bottom
    cv2.rectangle(frame, (0, h - 55), (w, h), (20, 20, 20), -1)
    word_display = " ".join(word) if word else "—"
    cv2.putText(frame, f"Word: {word_display}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    cv2.putText(frame, "SPACE=add  BACKSPACE=del  ENTER=clear",
                (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    return frame

print("Running! SPACE=add letter | BACKSPACE=delete | ENTER=clear | Q=quit")
current_pred, current_conf = "—", 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    detector.find_hands(frame)
    landmarks = detector.get_landmarks(frame)

    if landmarks is not None:
        probs = model.predict(landmarks[np.newaxis], verbose=0)[0]
        idx   = np.argmax(probs)
        conf  = probs[idx]
        pred  = labels[idx]

        prediction_buffer.append(pred)

        # Confirm only when the buffer is full and unanimous
        if (len(prediction_buffer) == STABILITY_FRAMES
                and len(set(prediction_buffer)) == 1
                and conf >= CONFIDENCE_THRESHOLD):
            current_pred, current_conf = pred, conf
        else:
            current_pred = pred
            current_conf = conf

    frame = draw_overlay(frame, current_pred, current_conf, word_buffer)
    cv2.imshow("Sign Language Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and current_pred != "—":   # add letter
        word_buffer.append(current_pred)
    elif key == 8 and word_buffer:                   # backspace
        word_buffer.pop()
    elif key == 13:                                  # enter = clear
        word_buffer.clear()

cap.release()
cv2.destroyAllWindows()