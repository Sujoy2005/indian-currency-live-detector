"""
Adds background-only images (no currency note) into the YOLO training set
as "negative" examples -- images with NO label file. YOLO treats an image
with no matching .txt as "this image has zero objects", which teaches the
model to stop firing false-positive boxes on empty/background scenes.

Run this AFTER your normal auto_label.py / clean_unlabeled.py /
make_val_split.py steps, then retrain.

Usage:
    python add_negatives.py
"""
import shutil
from pathlib import Path

BACKGROUND_DIR = Path("data/Background")
TRAIN_IMAGES = Path("yolo_data/images/train")

if not BACKGROUND_DIR.exists():
    raise SystemExit(f"{BACKGROUND_DIR} not found -- nothing to add.")

added = 0
for img_path in BACKGROUND_DIR.glob("*.*"):
    dst = TRAIN_IMAGES / img_path.name
    if dst.exists():
        continue
    shutil.copy(str(img_path), str(dst))
    # Deliberately do NOT create a matching .txt label file -- an image
    # with no label file is how YOLO represents "no objects in this image".
    added += 1

print(f"Added {added} background-only images as negative examples.")
print("No label files were created for them (this is intentional).")
print("\nNow retrain:")
print("  yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 device=0")
