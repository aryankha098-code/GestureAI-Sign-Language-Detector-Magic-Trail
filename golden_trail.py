# golden_trail.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import random
from collections import deque
from hand_detector import HandDetector

# ── Settings ───────────────────────────────────────────────────────────────────
TRAIL_LENGTH  = 180     # number of points stored
MAX_GAP       = 60      # break trail if finger jumps this many pixels
SPARKLE_EVERY = 15      # add a star burst every N new points
FADE_SPEED    = 0.93    # how fast trail fades when not drawing (lower = faster)

# Colors BGR
GOLD_WHITE = (210, 245, 255)   # white-hot core
GOLD_CORE  = (0,   215, 255)   # bright gold
GOLD_MID   = (0,   150, 220)   # mid glow
GOLD_DIM   = (0,    40,  80)   # tail fade
# ──────────────────────────────────────────────────────────────────────────────


def is_two_fingers_up(raw):
    if raw is None:
        return False
    def up(tip, pip): return raw[tip].y < raw[pip].y
    return (up(8,6) and up(12,10)
            and not up(16,14) and not up(20,18))


def get_fingertip_px(raw, shape):
    h, w = shape[:2]
    return (int(raw[8].x * w), int(raw[8].y * h))


def draw_star(canvas, x, y, size, alpha):
    """Draw a 4-point star burst with diagonal rays."""
    bright = tuple(int(c * alpha) for c in GOLD_WHITE)
    mid    = tuple(int(c * alpha * 0.5) for c in GOLD_CORE)

    # Long cross rays
    for dx, dy in [(size,0),(-size,0),(0,size),(0,-size)]:
        cv2.line(canvas, (x,y), (x+dx, y+dy), bright, 1, cv2.LINE_AA)

    # Short diagonal rays
    s2 = int(size * 0.55)
    for dx, dy in [(s2,s2),(-s2,s2),(s2,-s2),(-s2,-s2)]:
        cv2.line(canvas, (x,y), (x+dx, y+dy), mid, 1, cv2.LINE_AA)

    # Bright center dot
    cv2.circle(canvas, (x,y), max(1, size//5), bright, -1, cv2.LINE_AA)


def draw_trail(canvas, points, sparkle_pts):
    """
    Draw the glowing comet trail onto canvas.
    points       = list of (x,y) — None means pen lift
    sparkle_pts  = list of (x, y, age) for persistent star bursts
    """
    n = len(points)
    if n < 2:
        return

    for i in range(1, n):
        if points[i] is None or points[i-1] is None:
            continue

        x1, y1 = points[i-1]
        x2, y2 = points[i]

        # Skip jumps
        if abs(x2-x1) > MAX_GAP or abs(y2-y1) > MAX_GAP:
            continue

        # t: 0 = oldest (tail), 1 = newest (tip)
        t = i / n
        a = t  # opacity

        # ── Outer soft glow (wide, very transparent) ──────────────────────────
        glow_color = tuple(int(c * a * 0.18) for c in GOLD_MID)
        cv2.line(canvas, (x1,y1), (x2,y2), glow_color, 9, cv2.LINE_AA)

        # ── Mid glow ──────────────────────────────────────────────────────────
        mid_color = tuple(int(c * a * 0.45) for c in GOLD_CORE)
        cv2.line(canvas, (x1,y1), (x2,y2), mid_color, 4, cv2.LINE_AA)

        # ── Bright thin core ──────────────────────────────────────────────────
        core_color = tuple(int(c * min(1.0, a * 1.3)) for c in GOLD_CORE)
        cv2.line(canvas, (x1,y1), (x2,y2), core_color, 2, cv2.LINE_AA)

        # ── White-hot center (only near the tip) ──────────────────────────────
        if t > 0.75:
            hot = tuple(int(c * (t - 0.75) * 4) for c in GOLD_WHITE)
            cv2.line(canvas, (x1,y1), (x2,y2), hot, 1, cv2.LINE_AA)

    # ── Draw persistent sparkles ───────────────────────────────────────────────
    for (sx, sy, age) in sparkle_pts:
        alpha = max(0.0, 1.0 - age / 40)   # fades over 40 frames
        size  = int(6 + age * 0.3)
        draw_star(canvas, sx, sy, size, alpha)

    # ── Tip comet head ────────────────────────────────────────────────────────
    live = [p for p in points if p is not None]
    if live:
        tx, ty = live[-1]
        cv2.circle(canvas, (tx,ty), 5, GOLD_WHITE, -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx,ty), 9, GOLD_CORE,   1, cv2.LINE_AA)


def draw_ui(frame, drawing):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,45), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    dot = GOLD_CORE if drawing else (50,50,50)
    cv2.circle(frame, (18,22), 7, dot, -1)
    text = "Drawing  (peace sign)" if drawing else "Raise two fingers to draw"
    cv2.putText(frame, text, (32,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210,210,210), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0,h-28), (w,h), (10,10,10), -1)
    cv2.putText(frame, "C = clear    Q = quit",
                (10,h-9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,100,100), 1)


def main():
    cap      = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=1)

    ret, frame = cap.read()
    h, w = frame.shape[:2]

    trail_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    trail        = deque(maxlen=TRAIL_LENGTH)
    sparkles     = []    # list of [x, y, age]
    point_count  = 0
    drawing      = False
    prev_pt      = None

    print("Golden Trail ready!  ✌ = draw  |  C = clear  |  Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame, draw=False)   # no skeleton clutter

        raw = None
        if detector.results and detector.results.hand_landmarks:
            raw = detector.results.hand_landmarks[0]

        two_up = is_two_fingers_up(raw)

        if two_up and raw is not None:
            tip = get_fingertip_px(raw, frame.shape)

            if prev_pt and (abs(tip[0]-prev_pt[0]) > MAX_GAP or
                            abs(tip[1]-prev_pt[1]) > MAX_GAP):
                trail.append(None)

            trail.append(tip)
            point_count += 1
            drawing  = True
            prev_pt  = tip

            # Spawn a sparkle every N points
            if point_count % SPARKLE_EVERY == 0:
                sparkles.append([tip[0], tip[1], 0])

        else:
            drawing = False
            prev_pt = None
            # Fade canvas when not drawing
            trail_canvas = (trail_canvas * FADE_SPEED).astype(np.uint8)

        # Age sparkles, remove old ones
        sparkles = [[x, y, age+1] for x, y, age in sparkles if age < 40]

        # Redraw trail onto black canvas each frame for clean glow
        trail_canvas[:] = 0
        draw_trail(trail_canvas, list(trail), sparkles)

        # Blend onto webcam frame
        combined = cv2.addWeighted(frame, 0.85, trail_canvas, 1.0, 0)
        draw_ui(combined, drawing)

        cv2.imshow("Golden Trail ✨", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            trail_canvas[:] = 0
            trail.clear()
            sparkles.clear()
            point_count = 0

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()