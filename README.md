# 🤲 GestureAI — Sign Language Detector & Magic Trail

> *Where your hands become the controller.*

GestureAI is a real-time hand gesture intelligence system that bridges the gap between human expression and machine understanding. Point your camera, raise your hand, and watch it come alive — recognize sign language gestures on the fly, or switch into creative mode and paint glowing golden light trails through the air with just two fingers.

Built entirely from scratch using computer vision and deep learning — no cloud APIs, no internet required. Everything runs locally on your webcam in real time.

---

## ✨ Highlights

- 🧠 **AI-powered** — custom trained neural network, not rule-based
- ⚡ **Real-time** — runs smoothly on CPU, no GPU needed
- 🎨 **Creative mode** — draw glowing golden trails in the air
- 🔧 **Fully customizable** — add any gesture you want in minutes
- 🪶 **Lightweight** — 63 landmark values as input, no raw image processing

---

## 📸 Projects Overview

This repo contains **two projects** in one:

| Project | Description |
|---|---|
| 🤟 Sign Language Detector | Detects 8 custom gestures in real-time using a trained neural network |
| ✨ Golden Trail | Draw glowing golden light trails in the air using a peace sign gesture |

---

## 🤟 Project 1 — Sign Language Detector

Recognizes the following gestures live from your webcam:

- **Letters:** A, B, C
- **Phrases:** Hello, Bye, Love You, Cute, Get Lost

### How it works

1. **MediaPipe** detects your hand and extracts 21 landmark points (63 values total)
2. Landmarks are normalized relative to the wrist so position doesn't matter
3. A **Keras neural network** classifies the gesture in real-time
4. Prediction is confirmed only when stable for 15 consecutive frames (no flickering)

---

## ✨ Project 2 — Golden Trail

Draw glowing golden light trails in mid-air using just your hand.

- ✌️ **Peace sign** → drawing mode ON, golden trail follows your fingertip
- 🖐️ **Any other gesture** → drawing pauses, trail slowly fades
- Star sparkles appear along the path
- White-hot core with soft outer glow — like writing with light

---

## 🛠️ Tech Stack

- **Python 3.13**
- **OpenCV** — webcam capture and drawing
- **MediaPipe 0.10.x** — hand landmark detection (Tasks API)
- **TensorFlow / Keras** — gesture classification model
- **NumPy / scikit-learn** — data processing and evaluation

---

## 📁 Project Structure

```
Sign Language Detector/
├── hand_detector.py        # Core hand detection module (reused by both projects)
├── collect_data.py         # Collect landmark training data from webcam
├── train_model.py          # Train the Keras classifier
├── run_detector.py         # Live sign language detection
├── golden_trail.py         # Golden light trail drawing mode
├── hand_landmarker.task    # MediaPipe hand model file
├── data/
│   └── landmarks.csv       # Collected training data
└── models/
    ├── sign_language_model.h5   # Trained model
    ├── labels.npy               # Gesture label names
    └── training_curves.png      # Accuracy/loss graph
```

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/sign-language-detector.git
cd sign-language-detector
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe==0.10.33 tensorflow numpy scikit-learn matplotlib
```

### 3. Download the MediaPipe hand model

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task'); print('Downloaded!')"
```

---

## 🚀 Usage

### Run the Sign Language Detector

```bash
python run_detector.py
```

| Key | Action |
|---|---|
| `SPACE` | Add current letter to word |
| `BACKSPACE` | Delete last letter |
| `ENTER` | Clear word |
| `Q` | Quit |

---

### Run the Golden Trail

```bash
python golden_trail.py
```

| Gesture / Key | Action |
|---|---|
| ✌️ Peace sign | Draw golden trail |
| Any other gesture | Pause drawing |
| `C` | Clear canvas |
| `Q` | Quit |

---

## 🏋️ Train Your Own Gestures

### Step 1 — Collect data

Edit the `GESTURES` list in `collect_data.py` then run:

```bash
python collect_data.py
```

Press `SPACE` to start recording each gesture, hold your sign steady while the progress bar fills (200 samples per gesture).

### Step 2 — Train the model

```bash
python train_model.py
```

Expected accuracy: **92–97%** for clean, consistent gestures.

### Step 3 — Run live detection

```bash
python run_detector.py
```

---

## 🧠 Model Architecture

```
Input (63 values — 21 landmarks × x,y,z)
    ↓
Dense(256) + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64)  + Dropout(0.2)
    ↓
Dense(num_classes) — Softmax
```

Training uses **EarlyStopping** so it automatically stops when accuracy plateaus.

---

## ⚠️ Known Issues & Notes

- Requires **Python 3.10–3.13** — MediaPipe 0.10.x does not support older versions
- The `mediapipe.python.solutions` API is removed in 0.10.x — this project uses the newer Tasks API
- oneDNN warnings from TensorFlow are harmless — suppressed automatically
- Works best with good lighting and a plain background

---

## 🙌 Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe) by Google for hand landmark detection
- [TensorFlow / Keras](https://www.tensorflow.org/) for the classification model
- [OpenCV](https://opencv.org/) for real-time video processing

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with Python 🐍 | Portfolio Project*



https://github.com/user-attachments/assets/49781670-834a-478f-848e-2e60b15f3bf1



https://github.com/user-attachments/assets/46c59556-1ce6-435b-b0f4-32f0f8b9fb30



