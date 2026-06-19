"""Train a YOLO model for keyboard key/character detection."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from common import DATASET_DIR, DATASET_YAML, MODELS_DIR, PROJECT_ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for keyboard detection.")
    parser.add_argument("--data", type=Path, default=DATASET_YAML, help="YOLO dataset yaml path.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model, for example yolo11n.pt or yolo11s.pt.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device, for example 0 or cpu.")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "outputs" / "train", help="Training output directory.")
    parser.add_argument("--name", default="keyboard_yolo", help="Experiment name.")
    return parser.parse_args()


def fix_dataset_yaml(yaml_path: Path, dataset_dir: Path) -> str:
    """Rewrite `path` to absolute so ultralytics resolves it. Returns the original value."""
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    original = data.get("path", "")
    data["path"] = str(dataset_dir).replace("\\", "/")
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return original


def restore_yaml_path(yaml_path: Path, original: str) -> None:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = original
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def resolve_model(name: str) -> Path:
    """Resolve a model name/path, checking local models/ directory first."""
    path = Path(name)
    if path.exists():
        return path.resolve()
    local = MODELS_DIR / path.name
    if local.exists():
        return local
    raise FileNotFoundError(f"Model not found: {name} (checked {local} as well)")


def main() -> None:
    from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

    args = parse_args()
    ensure_dir(args.project)
    ensure_dir(MODELS_DIR)

    original_path = fix_dataset_yaml(args.data, DATASET_DIR)
    try:
        model = YOLO(str(resolve_model(args.model)))
        train_kwargs = {
            "data": str(args.data),
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "project": str(args.project),
            "name": args.name,
            "exist_ok": True,
        }
        if args.device is not None:
            train_kwargs["device"] = args.device

        results = model.train(**train_kwargs)
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"

        if best_pt.exists():
            target = MODELS_DIR / "best.pt"
            shutil.copy2(best_pt, target)
            print(f"best model copied to: {target}")
        else:
            print(f"training finished, but best.pt not found under: {save_dir}")
    finally:
        restore_yaml_path(args.data, original_path)


if __name__ == "__main__":
    main()
