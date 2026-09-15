# 💵 Indian Currency Live Detector

Real-time Indian currency note detection and classification — straight from your webcam, with voice announcements and running-total tracking.

---

## ✨ What is this?

Point your webcam at an Indian currency note, and this project will:

- 🔍 **Detect** the note in the live video feed — even angled, partially covered by fingers, or against a cluttered background
- 🧠 **Classify** the denomination using a custom-trained YOLOv8 object detector
- 🗳️ **Filter** results over multiple frames, ensuring a single blurry frame doesn't trigger a false announcement
- 🔊 **Announce** the detected denomination out loud using offline text-to-speech
- 💰 **Track** a running total of everything shown so far, reading it back on request
- 🎙️ **Respond** to spoken commands ("total", "reset") using fully offline speech recognition

Run it live with an on-screen video window, or in a fully **headless** (no display) mode for embedded or wearable setups.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🎥 **Live Detection** | YOLOv8 object detector trained on real-world note photos (varied backgrounds, angles, partial occlusion) |
| 🙅 **False-Positive Rejection** | Trained with hard-negative "background only" images so empty scenes don't trigger false detections |
| 🗳️ **Stability Filtering** | Requires a detection to hold steady across several frames before announcing — filtering out one-off misreads |
| 🔊 **Voice Announcements** | Speaks the detected denomination via offline TTS; borderline detections are announced tentatively |
| 💰 **Running Total** | Tracks and accumulates the total value of notes seen in the current session |
| 🎙️ **Voice Commands** | Say "total" to hear the running sum, or "reset" to clear it — powered by offline Vosk speech recognition |
| ⌨️ **Keyboard Fallback** | Press `T` / `R` as alternatives to voice commands for noisy environments |
| 🖥️ **Headless Mode** | Run without a video window (just camera + voice) for screen-free/embedded applications |

---

## 🧠 How it Works

```
Webcam Frame
     │
     ▼
YOLOv8 Object Detection
     │
     ▼
Confidence Threshold Filter
     │
     ▼
Stability Check (held across several frames)
     │
     ▼
Speak Result + Add to Running Total


Background Audio Stream
     │
     ▼
Vosk Voice Command Listener
     │
     ▼
"total" → Speak Running Total   |   "reset" → Clear Total
```

---

## 🛠️ Tech Stack

- **Ultralytics YOLOv8** — real-time object detection & classification
- **PyTorch** — GPU-accelerated training and inference
- **OpenCV** — video capture and on-screen display
- **pyttsx3** — offline text-to-speech engine
- **Vosk** — fully offline speech recognition for voice commands
- **PyAudio** — microphone input streaming

---

## 📁 Project Structure

```
indian-currency-live-detector/
├── data/                       # Raw training images (not committed)
├── yolo_data/                  # YOLO-format dataset (images + labels)
├── runs/                       # YOLO training runs + trained weights
├── live_camera_yolo.py         # Main live webcam app (with video UI)
├── live_camera_headless.py     # Headless version (no display window)
├── auto_label.py                # Auto-generates approx YOLO bounding-box labels
├── clean_unlabeled.py           # Keeps only successfully auto-labeled images
├── add_negatives.py             # Adds background-only images to reduce false positives
├── make_val_split.py            # Splits labeled data into train/validation sets
├── data.yaml                    # YOLO dataset config
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Sujoy2005/indian-currency-live-detector.git
cd indian-currency-live-detector
```

### 2. Set up a virtual environment (Python 3.11 recommended)

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install ultralytics opencv-python pyttsx3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Enable Voice Commands *(optional)*

Download a [Vosk speech model](https://alphacephei.com/vosk/models) and extract it to the project root. If not found, voice commands are safely disabled — everything else still works.

### 5. Run the Detector!

**With video interface:**
```bash
python live_camera_yolo.py
```

**Headless (no display):**
```bash
python live_camera_headless.py
```

---

## 💡 Tips for Best Accuracy

- **Steady hands** — hold the note reasonably steady for a second so the stability filter can confirm the class
- **Lighting** — good, even lighting helps; avoid heavy glare or deep, harsh shadows across the note
- **Microphone** — for voice commands, a headset/earphone mic works far better than a laptop's built-in mic
- **Iterative training** — if a specific denomination is consistently misread, add more real-world training photos of it and retrain

---

## 🗺️ Roadmap Ideas

- [ ] Expand training data for weaker classes
- [ ] Add multi-language voice announcements (e.g., Hindi)
- [ ] Implement audio position guidance
- [ ] Package as a standalone executable (`.exe`)

---

## 🤝 Contributing & License

Issues and pull requests are warmly welcome! This project is open source and available under the **MIT License**.
