"""Auto-annotate keyboard images using contour detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import IMAGE_EXTENSIONS, LABELS_DIR, SELECTED_DIR, RAW_DIR, ensure_dir  # type: ignore[reportMissingImports]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-annotate keyboard images.")
    parser.add_argument("--source", type=str, default="data/raw", help="Source image folder.")
    parser.add_argument("--visualize", action="store_true", help="Show detection results.")
    return parser.parse_args()


def classify_key(bbox, row_idx, col_idx, rows, cols_in_row):
    """Map a key position to class_id based on standard row/col."""
    # QWERTY layout rows with key counts
    row_key_counts = [12, 12, 10, 9, 5]
    if row_idx >= len(row_key_counts):
        row_idx = len(row_key_counts) - 1
    offset = sum(row_key_counts[:row_idx])
    if col_idx < row_key_counts[row_idx]:
        return offset + col_idx
    return offset + min(col_idx, row_key_counts[row_idx] - 1)


def detect_keys(image_path):
    """Detect key regions in a keyboard image."""
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [!] Cannot read {image_path}")
        return []

    h, w = img.shape[:2]

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold for varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 8
    )

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter to find key-like rectangles
    boxes = []
    min_area = (h * w) * 0.001
    max_area = (h * w) * 0.08

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if aspect < 0.4 or aspect > 4.0:
            continue
        boxes.append({
            "x1": x / w, "y1": y / h,
            "x2": (x + bw) / w, "y2": (y + bh) / h,
            "cx": (x + bw / 2) / w,
            "cy": (y + bh / 2) / h,
            "bw": bw / w, "bh": bh / h,
        })

    # Fallback with different params if too few boxes
    if len(boxes) < 10:
        thresh2 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 55, 5
        )
        closed2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
        contours2, _ = cv2.findContours(closed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours2:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if aspect < 0.4 or aspect > 4.0:
                continue
            boxes.append({
                "x1": x / w, "y1": y / h,
                "x2": (x + bw) / w, "y2": (y + bh) / h,
                "cx": (x + bw / 2) / w,
                "cy": (y + bh / 2) / h,
                "bw": bw / w, "bh": bh / h,
            })

    if len(boxes) < 10:
        print(f"  [!] Only detected {len(boxes)} regions, skipping.")
        return []

    # Group into rows by y-coordinate
    boxes.sort(key=lambda b: b["cy"])
    bh_values = [b["bh"] for b in boxes]
    median_bh = np.median(bh_values) if bh_values else 0.06
    row_threshold = median_bh * 1.2

    rows = []
    current_row = []
    current_y = 0.0
    for box in boxes:
        if not current_row:
            current_row.append(box)
            current_y = box["cy"]
        elif abs(box["cy"] - current_y) < row_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            current_y = box["cy"]
    if current_row:
        rows.append(current_row)

    # Sort each row by x
    for row in rows:
        row.sort(key=lambda b: b["cx"])

    # Generate labels
    results = []
    for row_idx, row in enumerate(rows):
        for col_idx, box in enumerate(row):
            cls_id = classify_key(box, row_idx, col_idx, len(rows), len(row))
            results.append({
                "class_id": cls_id,
                "x_center": f"{box['cx']:.6f}",
                "y_center": f"{box['cy']:.6f}",
                "width": f"{box['bw']:.6f}",
                "height": f"{box['bh']:.6f}",
            })
    return results


def main():
    args = parse_args()
    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        return

    image_files = sorted(p for p in source_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not image_files:
        print(f"[ERROR] No images found in {source_dir}")
        return

    ensure_dir(LABELS_DIR)
    total_ok = total_fail = 0

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        labels = detect_keys(img_path)
        if not labels:
            total_fail += 1
            continue
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for lbl in labels:
                f.write(f"{lbl['class_id']} {lbl['x_center']} {lbl['y_center']} "
                        f"{lbl['width']} {lbl['height']}\n")
        total_ok += 1
        print(f"  [OK] Detected {len(labels)} keys -> {label_path.name}")

    print(f"\n=== Summary: {total_ok} OK, {total_fail} failed ===")


if __name__ == "__main__":
    main()
