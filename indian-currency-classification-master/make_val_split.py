"""
Moves a random ~15% of the labeled training images (and their matching
labels) into the validation set, so YOLO has something to evaluate on
during training.

Run this AFTER clean_unlabeled.py.

Usage:
    python make_val_split.py
"""
import random
import shutil
from pathlib import Path

TRAIN_IMAGES = Path("yolo_data/images/train")
TRAIN_LABELS = Path("yolo_data/labels/train")
VAL_IMAGES = Path("yolo_data/images/val")
VAL_LABELS = Path("yolo_data/labels/val")
VAL_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.15
random.seed(42)

stems = [p.stem for p in TRAIN_IMAGES.glob("*.*")]
random.shuffle(stems)
val_count = max(1, int(len(stems) * VAL_FRACTION))
val_stems = set(stems[:val_count])

moved = 0
for stem in val_stems:
    img_src = None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        candidate = TRAIN_IMAGES / f"{stem}{ext}"
        if candidate.exists():
            img_src = candidate
            break
    label_src = TRAIN_LABELS / f"{stem}.txt"

    if img_src is not None:
        shutil.move(str(img_src), str(VAL_IMAGES / img_src.name))
    if label_src.exists():
        shutil.move(str(label_src), str(VAL_LABELS / label_src.name))
    moved += 1

print(f"Moved {moved} images (+labels) to validation set.")
print(f"Remaining in train: {len(list(TRAIN_IMAGES.glob('*.*')))} images")
