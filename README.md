# 💵 Indian Currency Live Detector

> Real-time Indian currency note detection and classification — straight from your webcam, with OCR verification and voice announcements.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Contrib-green.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange.svg)](https://scikit-learn.org/)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-yellow.svg)](https://github.com/tesseract-ocr/tesseract)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

---

## ✨ What is this?

Point your webcam at an Indian currency note, and this project will:

1. 🔍 **Detect** the note in the live video feed (even while you're holding it)
2. 🧠 **Classify** the denomination using a trained Random Forest model built on SIFT + Bag-of-Visual-Words features
3. 🔤 **Verify** the classification by reading the printed number directly off the note with OCR
4. 🗳️ **Smooth** results over multiple frames with majority voting, so a single bad-lighting frame doesn't throw off the answer
5. 🔊 **Announce** the detected denomination out loud

All running live, in a single OpenCV window.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🎥 **Live detection** | Background-thread detection pipeline keeps the video feed smooth while classification runs asynchronously |
| 🧩 **Smart note isolation** | Canny-edge + contour shape/solidity filtering to isolate the note from cluttered backgrounds |
| 🙅 **False-positive rejection** | Skin-tone filtering + a trained "Background" class so faces/hands/tables aren't misread as currency |
| 🔤 **OCR cross-check** | Tesseract OCR reads the printed denomination number to catch cases where color/lighting confuses the classifier |
| 🗳️ **Temporal voting** | Rolling majority vote across recent frames for a stable, confident final answer |
| 🔊 **Voice announcements** | Speaks the detected denomination once per note using offline text-to-speech |
| 💰 **Running total** | Tracks and displays the total value of notes currently in view |

---

## 🧠 How it works

```
Webcam Frame
     │
     ▼
Downscale + Canny Edge Detection
     │
     ▼
Contour Filtering (area, aspect ratio, solidity)
     │
     ▼
Skin-Tone Rejection ──► reject if hand/face
     │
     ▼
Feature Extraction (Hu Moments + Haralick + Color Histogram + SIFT-BoVW)
     │
     ▼
Random Forest Classification ──► "Background"? ──► ignore, reset voting
     │
     ▼
OCR Verification (Tesseract) ──► overrides classifier if a confident digit match is found
     │
     ▼
Majority Vote (last N frames)
     │
     ▼
Draw Box + Label + Speak Result
```

---

## 🛠️ Tech Stack

- **OpenCV (contrib)** — video capture, Canny edge detection, contour analysis, SIFT, Bag-of-Visual-Words
- **scikit-learn** — Random Forest classifier, `RandomizedSearchCV` for hyperparameter tuning
- **mahotas** — Haralick texture features
- **Tesseract OCR (pytesseract)** — printed denomination verification
- **pyttsx3** — offline text-to-speech
- **NumPy** — feature vector handling

---

## 📁 Project Structure

```
indian-currency-live-detector/
├── data/                      # Training images, organized by denomination folder (not committed — see .gitignore)
│   ├── 10/  20/  50/  100/  200/  500/  Background/
├── model/                     # Trained model + BoVW codebook + feature arrays
│   ├── rfclassifier_600.sav
│   ├── bovw_codebook_600.pickle
│   ├── data_600.npy
│   └── label_600.npy
├── bovw.py                    # Builds the SIFT Bag-of-Visual-Words vocabulary + feature dataset
├── train.py                   # Trains the Random Forest classifier
├── hyper_train.py             # Randomized hyperparameter search + retrains the best model
├── live_camera.py             # Main live webcam detection + OCR + voice app
├── predict.py                 # Single-image prediction utility
├── currency.py                # Shared feature-extraction helpers
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Sujoy2005/indian-currency-live-detector.git
cd indian-currency-live-detector
```

### 2. Set up a virtual environment (Python 3.11 recommended)

```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install numpy opencv-contrib-python mahotas scikit-learn joblib pytesseract pyttsx3
```

### 4. Install Tesseract OCR (required for OCR verification)

Download and install from [UB-Mannheim's Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki), then confirm the path in `live_camera.py` matches your install location:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 5. Build the model (if training from scratch)

```bash
python bovw.py data
python train.py model/rfclassifier_600.sav
```

Optional — search for better hyperparameters:

```bash
python hyper_train.py
```

### 6. Run it!

```bash
python live_camera.py
```

Hold a note in front of your webcam and watch it get detected, classified, verified, and announced. Press **`q`** to quit.

---

## 💡 Tips for Best Accuracy

- Use a **plain, dark, contrasting background** (a black cloth or mat works great)
- Keep the note **flat** and **well-lit** — avoid shadows and glare
- Hold the note **close enough** that the printed number is crisp and readable
- Keep it steady for a second or two to let the majority vote stabilize

---

## 🗺️ Roadmap Ideas

- [ ] Swap classical CV detection for a lightweight trained object detector for background-agnostic detection
- [ ] Expand dataset with more lighting/angle/wear diversity per denomination
- [ ] Add a simple GUI/dashboard for scan history and totals
- [ ] Package as a standalone executable

---

## 🤝 Contributing

Issues and pull requests are welcome! If you add more training data or improve detection accuracy, feel free to open a PR.

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

<p align="center">Made with 🧠 + ☕ by <a href="https://github.com/Sujoy2005">Sujoy</a></p>
