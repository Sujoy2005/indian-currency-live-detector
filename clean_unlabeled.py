"""
Removes only the UNLABELED images from the training folder (images
auto_label.py couldn't find a confident note-box for). Keeps every image
that already has an auto-generated label -- no trimming, no manual review
needed to get started.

Run this AFTER auto_label.py.

Usage:
    python clean_unlabeled.py
"""
import shutil
from pathlib import Path

IMAGES_DIR = Path("yolo_data/images/train")
LABELS_DIR = Path("yolo_data/labels/train")
LEFTOVER_IMAGES = Path("yolo_data/images/_unlabeled")
LEFTOVER_IMAGES.mkdir(parents=True, exist_ok=True)

kept = 0
moved = 0

for img_path in list(IMAGES_DIR.glob("*.*")):
    label_path = LABELS_DIR / f"{img_path.stem}.txt"
    if label_path.exists():
        kept += 1
    else:
        shutil.move(str(img_path), str(LEFTOVER_IMAGES / img_path.name))
        moved += 1

print(f"Kept for training (all auto-labeled): {kept} images")
print(f"Moved out (had no auto-label): {moved} images")
print("\nReady to train with all auto-labeled images -- no manual")
print("labeling needed for round 1.")
