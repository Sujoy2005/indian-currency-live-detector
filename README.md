💵 Indian Currency Live Detector
Real-time Indian currency note detection and classification — straight from your webcam, with voice announcements and running-total tracking.

Python PyTorch Ultralytics YOLO OpenCV License

✨ What is this?
Point your webcam at an Indian currency note, and this project will:

🔍 Detect the note in the live video feed — even angled, partially covered by fingers, or against a cluttered background
🧠 Classify the denomination using a custom-trained YOLOv8 object detector
🗳️ Smooth results over multiple frames, so a single blurry/noisy frame doesn't trigger a wrong announcement
🔊 Announce the detected denomination out loud (offline text-to-speech)
💰 Keep a running total of everything shown so far, and read it back on request
🎙️ Respond to spoken commands ("total", "reset") using fully offline speech recognition
All running live, with either an on-screen video window or a fully headless (no display) mode.

🎯 Features
Feature	Description
🎥 Live detection	YOLOv8 object detector trained on real-world note photos (varied backgrounds, angles, partial occlusion)
🙅 False-positive rejection	Trained with hard-negative "background only" images so empty scenes don't trigger a false detection
🗳️ Stability filtering	Requires a detection to hold steady across several frames before announcing — filters out one-off misreads
🔊 Voice announcements	Speaks the detected denomination via offline TTS; low-confidence detections are announced tentatively ("maybe X rupees, not sure")
💰 Running total	Tracks the total value of notes seen so far
🎙️ Voice commands	Say "total" to hear the running total, or "reset" to clear it — powered by offline Vosk speech recognition
⌨️ Keyboard fallback	T / R keys do the same thing as the voice commands, for when mic conditions are noisy
🖥️ Headless mode	Run with no video window at all — just camera + voice — for embedded/no-screen setups (e.g. a wearable device)
🧠 How it works
Webcam Frame
     │
     ▼
YOLOv8 Object Detection (denomination + bounding box, single pass)
     │
     ▼
Confidence Threshold Filter
     │
     ▼
Stability Check (same class seen across several consecutive frames)
     │
     ▼
Speak Result (tentative if confidence is borderline) + Add to Running Total
     │
     ▼
Voice Command Listener (background thread) ──► "total" / "reset" ──► Speak Running Total
🛠️ Tech Stack
Ultralytics YOLOv8 — real-time object detection & classification, trained on a custom dataset
PyTorch (CUDA) — GPU-accelerated training and inference
OpenCV — video capture and (optional) on-screen display
pyttsx3 — offline text-to-speech
Vosk — fully offline speech recognition for voice commands
PyAudio — microphone input stream
📁 Project Structure
indian-currency-live-detector/
├── data/                          # Training images, organized by denomination folder (not committed — see .gitignore)
│   ├── 10/  20/  50/  100/  200/  500/  Background/
├── yolo_data/                      # YOLO-format dataset (images + labels, train/val split) — not committed
├── runs/                            # Training runs + trained weights (not committed)
│   └── detect/train-*/weights/best.pt
├── live_camera_yolo.py             # Main live webcam app — detection, voice, running total (with video window)
├── live_camera_headless.py         # Same as above, but with no display window (for no-screen setups)
├── auto_label.py                   # Auto-generates approximate YOLO bounding-box labels from denomination folders
├── clean_unlabeled.py              # Keeps only successfully auto-labeled images for training
├── add_negatives.py                # Adds background-only images as hard negatives (reduces false positives)
├── make_val_split.py               # Splits labeled data into train/validation sets
├── data.yaml                       # YOLO dataset config (class names + paths)
├── requirements.txt
└── README.md
🚀 Getting Started
1. Clone the repo
git clone https://github.com/Sujoy2005/indian-currency-live-detector.git
cd indian-currency-live-detector
2. Set up a virtual environment (Python 3.11 recommended)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
3. Install dependencies
pip install ultralytics opencv-python pyttsx3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # for NVIDIA GPU training/inference

# Optional — for voice commands ("total" / "reset"):
pip install vosk pyaudio
4. (Optional) Enable voice commands
Download a Vosk speech model (fully offline after this one-time download):

Small/fast model: vosk-model-small-en-us-0.15
More accurate model (recommended if your mic picks up background noise): vosk-model-en-us-0.22
Extract it into the project root, and make sure VOSK_MODEL_PATH in live_camera_yolo.py matches the folder name. If the model folder isn't found, voice commands are simply disabled — everything else still works, and T / R keyboard shortcuts always work as a fallback.

5. Train the model (if training from scratch)
# 1. Auto-label your per-denomination image folders (data/10, data/20, ...)
python auto_label.py

# 2. Keep only successfully auto-labeled images
python clean_unlabeled.py

# 3. Add background-only images as hard negatives (reduces false positives)
python add_negatives.py

# 4. Split into train/validation sets
python make_val_split.py

# 5. Train (GPU strongly recommended — device=0 uses your first CUDA GPU)
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 device=0
After training, update MODEL_PATH in live_camera_yolo.py (and live_camera_headless.py) to point at the new runs/detect/train-*/weights/best.pt.

6. Run it!
python live_camera_yolo.py
Hold a note in front of your webcam. Press q to quit, T to hear the running total, R to reset it — or just say "total" / "reset" out loud if voice commands are set up.

For a no-display setup:

python live_camera_headless.py
Stop it with Ctrl+C in the terminal.

💡 Tips for Best Accuracy
Hold the note reasonably steady for a second so the stability filter can confirm it
Good, even lighting helps — avoid heavy glare or deep shadows across the note
For voice commands, a headset/earphone mic works far better than a laptop's built-in mic, especially in noisy rooms
If a denomination is consistently misread, add more real-world training photos for that denomination to data/<amount>/ and retrain
🗺️ Roadmap Ideas
 Expand training data further for weaker classes (denominations with fewer real-world photos)
 Multi-language voice announcements (e.g. Hindi)
 Position guidance ("move closer" / "move back") for a screen-free experience
 Package as a standalone executable
🤝 Contributing
Issues and pull requests are welcome! If you add more training data or improve detection accuracy, feel free to open a PR.

📜 License
This project is open source under the MIT License.