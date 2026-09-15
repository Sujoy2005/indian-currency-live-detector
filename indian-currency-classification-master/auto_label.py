"""
Auto-labeling helper for YOLO training data.

Instead of manually drawing a box on every single image in LabelImg, this
script finds the note-shaped region in each image automatically (using the
same contour/shape logic as the live detector) and writes a YOLO-format
.txt label file for it.

You still need to REVIEW the results in LabelImg afterward (some boxes
will be wrong or missing), but reviewing/correcting an existing box is
much faster than drawing every box from scratch.

Usage:
    python auto_label.py

Expects:
    data/10/*.jpg, data/20/*.jpg, ... (your existing per-denomination folders)

Writes:
    yolo_data/images/train/<image>          (copied)
    yolo_data/labels/train/<image>.txt       (YOLO box, if one was found)

A summary at the end tells you how many images got an automatic box vs.
how many were skipped (no confident note shape found) and need manual
labeling in LabelImg.
"""
import cv2
import numpy as np
import math
import os
import shutil
from pathlib import Path

DATA_DIR = "data"
OUT_IMAGES = Path("yolo_data/images/train")
OUT_LABELS = Path("yolo_data/labels/train")
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

# Folder name -> YOLO class index. Must match data.yaml exactly.
CLASS_MAP = {"10": 0, "20": 1, "50": 2, "100": 3, "200": 4, "500": 5}


def find_note_box(img):
    """Same shape-based approach as detect_notes() in live_camera.py:
    find the largest note-shaped contour and return its (x, y, w, h) in
    pixel coordinates, or None if nothing confident was found."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 10, 60)

    area = img.shape[0] * img.shape[1]
    ks = max(int(math.log(area, 7)), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = area * 0.05
    max_area = area * 0.5
    best = None
    best_area = 0

    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area <= min_area or cnt_area >= max_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or cnt_area / hull_area < 0.4:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect < 1.1 or aspect > 3.3:
            continue

        if cnt_area > best_area:
            best_area = cnt_area
            best = (x, y, w, h)

    return best


def write_yolo_label(path, class_id, box, img_w, img_h):
    x, y, w, h = box
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    with open(path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def main():
    labeled = 0
    skipped = 0

    for folder_name, class_id in CLASS_MAP.items():
        src_dir = Path(DATA_DIR) / folder_name
        if not src_dir.exists():
            print(f"  (skip) {src_dir} not found")
            continue

        for img_path in src_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            dst_img = OUT_IMAGES / img_path.name
            if not dst_img.exists():
                shutil.copy(str(img_path), str(dst_img))

            box = find_note_box(img)
            label_path = OUT_LABELS / (img_path.stem + ".txt")

            if box is not None:
                h, w = img.shape[:2]
                write_yolo_label(label_path, class_id, box, w, h)
                labeled += 1
            else:
                skipped += 1
                # No confident box found -- leave no label file, so
                # LabelImg will show this image with nothing drawn and
                # you know it needs a manual box.

        print(f"  {folder_name}: done")

    print(f"\nAuto-labeled: {labeled} images")
    print(f"Needs manual labeling in LabelImg: {skipped} images")
    print("\nOpen yolo_data/images/train in LabelImg (YOLO format) to review.")
    print("Images with an existing box will show it automatically -- just")
    print("check it's correctly placed and move on (or fix/redraw if not).")


if __name__ == "__main__":
    main()
