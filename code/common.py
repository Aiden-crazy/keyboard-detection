"""Common paths and helpers for the keyboard detection project."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SELECTED_DIR = PROJECT_ROOT / "data" / "selected"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
DATASET_DIR = PROJECT_ROOT / "datasets" / "keyboard"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
