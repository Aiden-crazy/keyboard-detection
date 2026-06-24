"""Run YOLO prediction on keyboard images and save visualized results."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import MODELS_DIR, OUTPUTS_DIR, RAW_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict keyboard keys/characters with a trained YOLO model.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best.pt", help="Trained model path.")
    parser.add_argument("--source", type=Path, default=RAW_DIR, help="Image file or directory.")
    parser.add_argument("--imgsz", type=int, default=960, help="Prediction image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "predict", help="Prediction output directory.")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    ensure_dir(args.out)
    model = YOLO(str(args.model))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        project=str(args.out.parent),
        name=args.out.name,
        exist_ok=True,
        save=True,
        save_txt=True,
    )
    print(f"prediction results saved to: {args.out}")


if __name__ == "__main__":
    main()
