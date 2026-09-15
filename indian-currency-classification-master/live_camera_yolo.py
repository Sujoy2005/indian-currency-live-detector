import platform
import shutil
import os
import time
import json
import threading
import queue
import cv2
import pyttsx3
from ultralytics import YOLO

try:
    import vosk
    import pyaudio
    VOICE_COMMANDS_AVAILABLE = True
except ImportError:
    VOICE_COMMANDS_AVAILABLE = False

OS_NAME = platform.system()

# Path to your trained YOLO weights (from `yolo detect train ...`)
MODEL_PATH = "runs/detect/train-10/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"YOLO model not found at {MODEL_PATH}. Update MODEL_PATH to point "
        f"to your trained weights (runs/detect/train-*/weights/best.pt)."
    )

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # e.g. {0: '10', 1: '20', ...} from data.yaml

CONF_THRESHOLD = 0.65  # only trust detections the model is at least this sure about

# --- Voice announcements (same pattern as the classic pipeline: run on its
# own thread so speaking never blocks the camera loop) ---
speech_queue = queue.Queue()
last_announced = None
stop_flag = False
tts_speaking = threading.Event()  # set while text-to-speech is actively talking

NUMBER_WORDS = {"10": "ten", "20": "twenty", "50": "fifty", "100": "one hundred",
                 "200": "two hundred", "500": "five hundred"}

# --- Running total ---
running_total = 0
total_lock = threading.Lock()


def add_to_total(class_label):
    global running_total
    try:
        value = int(class_label)
    except ValueError:
        return
    with total_lock:
        running_total += value


def speak_total():
    with total_lock:
        total = running_total
    speech_queue.put(f"total so far is {total} rupees")


def reset_total():
    global running_total
    with total_lock:
        running_total = 0
    speech_queue.put("total reset to zero")


def speak_denomination(class_label, confidence=1.0):
    spoken = NUMBER_WORDS.get(class_label, class_label)
    if confidence < 0.65:
        # Low-confidence match -- say it tentatively rather than stating it
        # as fact, so the person knows to double-check (e.g. try a
        # steadier/closer shot) instead of trusting a possibly-wrong read.
        speech_queue.put(f"maybe {spoken} rupees, not sure")
    else:
        speech_queue.put(f"{spoken} rupees")
        add_to_total(class_label)


def speech_worker():
    pythoncom = None
    if OS_NAME == "Windows":
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pythoncom = None

    while not stop_flag:
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if text is None:
            break

        # A fresh engine per utterance instead of one reused instance --
        # reusing the same pyttsx3 engine across multiple say()/runAndWait()
        # calls reliably stops working after the first utterance on many
        # Windows/SAPI5 setups (known pyttsx3 issue). Slightly slower to
        # set up each time, but this is what actually keeps speaking on
        # every detection instead of just the first one.
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 130)
            engine.setProperty('volume', 1.0)
            try:
                voices = engine.getProperty('voices')
                for v in voices:
                    if 'english' in v.name.lower() or 'en' in (v.id or '').lower():
                        engine.setProperty('voice', v.id)
                        break
            except Exception:
                pass
            tts_speaking.set()  # tell the voice listener to ignore mic input
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
            time.sleep(0.3)  # let any speaker echo tail off before listening again
        except Exception as e:
            print(f"WARNING: text-to-speech failed for this utterance ({e}).")
        finally:
            tts_speaking.clear()

    if pythoncom is not None:
        pythoncom.CoUninitialize()


VOSK_MODEL_PATH = "vosk-model-en-us-0.22"  # bigger, more accurate model (small one struggles with background noise)


def voice_command_listener():
    """Listens to the microphone in the background for spoken commands:
    - saying "total" (e.g. "what's the total", "tell me the total") announces
      the running total so far.
    - saying "reset" clears the running total back to zero.
    Runs entirely offline once the Vosk model folder is downloaded -- no
    internet needed at runtime."""
    if not VOICE_COMMANDS_AVAILABLE:
        print("Voice commands disabled: install with `pip install vosk pyaudio`")
        return
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"Voice commands disabled: Vosk model folder '{VOSK_MODEL_PATH}' not found.")
        print("Download from https://alphacephei.com/vosk/models "
              "(vosk-model-small-en-us-0.15) and place it in this project folder.")
        return

    vosk.SetLogLevel(-1)  # silence Vosk's own console spam
    vosk_model = vosk.Model(VOSK_MODEL_PATH)
    recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                      input=True, frames_per_buffer=2000)  # smaller buffer = faster reaction to speech
    stream.start_stream()

    print("Voice commands ready -- say 'total' to hear the running total, "
          "or 'reset' to clear it.")

    while not stop_flag:
        try:
            data = stream.read(2000, exception_on_overflow=False)
        except Exception:
            continue

        if tts_speaking.is_set():
            # Discard audio captured while the system itself is talking,
            # so it doesn't hear its own voice through the speakers and
            # misread it as a new command (feedback loop).
            recognizer.Reset()
            continue

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").lower()
            if text:
                print(f"[voice] heard: '{text}'")
            if not text:
                continue
            if "reset" in text or "clear" in text:
                reset_total()
            elif "total" in text:
                speak_total()

    stream.stop_stream()
    stream.close()
    pa.terminate()


def open_camera(index=0):
    if OS_NAME == "Windows":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif OS_NAME == "Linux":
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        return cv2.VideoCapture(index)


CAM_INDEX = 0
cap = open_camera(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Try changing CAM_INDEX.")

speaker = threading.Thread(target=speech_worker, daemon=True)
speaker.start()

voice_listener = threading.Thread(target=voice_command_listener, daemon=True)
voice_listener.start()

print(f"Webcam started on {OS_NAME}. Hold a currency note in front of the camera.")
print("Press 'q' to quit.")

# Simple stability check: require the same class to be seen for a few
# consecutive frames before announcing, so a single noisy/blurry frame
# doesn't trigger a wrong announcement.
STABLE_FRAMES_NEEDED = 4  # lowered now that background false-positives are fixed via negative training samples
MAX_MISSES_BEFORE_CLEAR = 8  # tolerate several no-detection frames before hiding the box/resetting
last_seen_class = None
stable_count = 0
miss_streak = 0
last_box = None  # keep drawing the last known box during brief misses, for a steady display
last_seen_conf = 0.0  # confidence of the most recent detection, used when announcing

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        current_class = None
        if len(results.boxes) > 0:
            # Take the highest-confidence detection this frame
            best = max(results.boxes, key=lambda b: float(b.conf))
            cls_id = int(best.cls)
            current_class = CLASS_NAMES[cls_id]
            conf = float(best.conf)
            last_seen_conf = conf
            x1, y1, x2, y2 = map(int, best.xyxy[0])
            last_box = (x1, y1, x2, y2, current_class, conf)
            miss_streak = 0
        else:
            miss_streak += 1
            if miss_streak >= MAX_MISSES_BEFORE_CLEAR:
                last_box = None  # only clear the display after several consecutive misses

        # Draw the last known box even on a brief miss, so it doesn't flicker
        if last_box is not None:
            x1, y1, x2, y2, box_label, box_conf = last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{box_label} ({box_conf:.0%})", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if current_class is not None and current_class == last_seen_class:
            stable_count += 1
        elif current_class is not None:
            stable_count = 1
            last_seen_class = current_class
        # (a brief single-frame miss does NOT reset stable_count/last_seen_class
        # here -- only a sustained miss_streak, handled below, does)

        if current_class is not None and stable_count >= STABLE_FRAMES_NEEDED:
            if current_class != last_announced:
                speak_denomination(current_class, last_seen_conf)
                last_announced = current_class
        elif miss_streak >= MAX_MISSES_BEFORE_CLEAR:
            # Only reset once the note has been gone for a while, so the
            # same note isn't re-announced on every tiny flicker.
            last_announced = None
            last_seen_class = None
            stable_count = 0

        cv2.imshow("Live Currency Detection (YOLO) - press q to quit, T=total, R=reset", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            speak_total()
        elif key == ord('r'):
            reset_total()
finally:
    print("[shutdown] step 1: stopping threads...")
    t0 = time.time()
    stop_flag = True
    speech_queue.put(None)
    speaker.join(timeout=2)
    print(f"[shutdown] step 1 done ({time.time()-t0:.1f}s)")

    print("[shutdown] step 2: releasing camera...")
    t0 = time.time()
    cap.release()
    print(f"[shutdown] step 2 done ({time.time()-t0:.1f}s)")

    print("[shutdown] step 3: closing windows...")
    t0 = time.time()
    cv2.destroyAllWindows()
    print(f"[shutdown] step 3 done ({time.time()-t0:.1f}s)")

    print("Shutting down...")
    os._exit(0)

'''
Sample run: python live_camera_yolo.py
Requires: pip install ultralytics
Uses the YOLO model trained via:
    yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 device=0
'''