"""Split selected images and YOLO labels into train/val/test folders."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from common import DATASET_DIR, LABELS_DIR, SELECTED_DIR, ensure_dir, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split keyboard detection images into YOLO dataset folders.")
    parser.add_argument("--images", type=Path, default=SELECTED_DIR, help="Directory containing selected images.")
    parser.add_argument("--labels", type=Path, default=LABELS_DIR, help="Directory containing YOLO txt labels.")
    parser.add_argument("--out", type=Path, default=DATASET_DIR, help="Output dataset directory.")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio.")
    parser.add_argument("--val", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--copy-empty-labels", action="store_true", help="Create empty txt if an image has no label file.")
    return parser.parse_args()


def clean_subset_dirs(out_dir: Path) -> None:
    for kind in ("images", "labels"):
        for subset in ("train", "val", "test"):
            subset_dir = ensure_dir(out_dir / kind / subset)
            for item in subset_dir.iterdir():
                if item.is_file():
                    item.unlink()


def copy_pair(image_path: Path, label_dir: Path, out_dir: Path, subset: str, copy_empty_labels: bool) -> bool:
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists() and not copy_empty_labels:
        print(f"skip without label: {image_path.name}")
        return False

    image_out = out_dir / "images" / subset / image_path.name
    label_out = out_dir / "labels" / subset / f"{image_path.stem}.txt"
    shutil.copy2(image_path, image_out)

    if label_path.exists():
        shutil.copy2(label_path, label_out)
    else:
        label_out.write_text("", encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train + val + test must be 1.0")

    all_images = list_images(args.images)
    if not all_images:
        raise FileNotFoundError(f"No images found in {args.images}")

    # Filter to only images that have corresponding labels (unless copy_empty_labels)
    if args.copy_empty_labels:
        images = all_images
    else:
        images = [img for img in all_images if (args.labels / f"{img.stem}.txt").exists()]
        skipped = len(all_images) - len(images)
        if skipped > 0:
            print(f"跳过 {skipped} 张未标注图片（仅划分有标签的图片）")

    if not images:
        raise FileNotFoundError("No labeled images found — cannot split. Please annotate some images first.")

    random.seed(args.seed)
    random.shuffle(images)

    n = len(images)
    n_train = max(1, int(n * args.train)) if n >= 1 else 0
    n_val = max(1, int(n * args.val)) if n >= 3 else (1 if n >= 2 else 0)

    if n_train + n_val > n:
        n_val = n - n_train

    subsets = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    # YOLO requires at least one image in val — duplicate from train if needed
    if not subsets["val"] and subsets["train"]:
        subsets["val"] = [subsets["train"][0]]

    clean_subset_dirs(args.out)

    copied = {"train": 0, "val": 0, "test": 0}
    for subset, subset_images in subsets.items():
        for image_path in subset_images:
            if copy_pair(image_path, args.labels, args.out, subset, args.copy_empty_labels):
                copied[subset] += 1

    print("dataset split finished")
    print(f"output: {args.out}")
    print(f"train: {copied['train']}, val: {copied['val']}, test: {copied['test']}")


if __name__ == "__main__":
    main()
