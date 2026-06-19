"""Export a trained YOLO model to ONNX for possible HALCON deployment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import MODELS_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best.pt", help="Trained pt model path.")
    parser.add_argument("--imgsz", type=int, default=960, help="Export image size.")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "best.onnx", help="Output ONNX path.")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    ensure_dir(args.out.parent)
    model = YOLO(str(args.model))
    exported_path = Path(model.export(format="onnx", imgsz=args.imgsz))

    if exported_path.resolve() != args.out.resolve():
        shutil.copy2(exported_path, args.out)
    print(f"onnx model saved to: {args.out}")


if __name__ == "__main__":
    main()
