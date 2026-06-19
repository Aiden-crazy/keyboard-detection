"""Validate image and YOLO label files before training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from common import DATASET_YAML, LABELS_DIR, SELECTED_DIR, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check selected images and YOLO label files.")
    parser.add_argument("--images", type=Path, default=SELECTED_DIR, help="Directory containing selected images.")
    parser.add_argument("--labels", type=Path, default=LABELS_DIR, help="Directory containing YOLO txt labels.")
    parser.add_argument("--data", type=Path, default=DATASET_YAML, help="Dataset yaml path.")
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> dict[int, str]:
    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    names = data.get("names", {})
    return {int(class_id): str(name) for class_id, name in names.items()}


def check_label_line(line: str, line_no: int, label_path: Path, class_count: int) -> tuple[int | None, str | None]:
    parts = line.split()
    if len(parts) != 5:
        return None, f"{label_path.name}:{line_no} 格式错误，应为 5 列，实际为 {len(parts)} 列"

    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None, f"{label_path.name}:{line_no} 存在非数字内容"

    if class_id < 0 or class_id >= class_count:
        return None, f"{label_path.name}:{line_no} 类别编号 {class_id} 超出范围 0-{class_count - 1}"

    if any(value < 0 or value > 1 for value in values):
        return None, f"{label_path.name}:{line_no} 坐标值必须在 0 到 1 之间"

    width, height = values[2], values[3]
    if width <= 0 or height <= 0:
        return None, f"{label_path.name}:{line_no} 标注框宽高必须大于 0"

    return class_id, None


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.data)
    images = list_images(args.images)

    if not images:
        raise FileNotFoundError(f"没有在 {args.images} 中找到训练图片")

    missing_labels: list[str] = []
    empty_labels: list[str] = []
    errors: list[str] = []
    class_counter: Counter[int] = Counter()
    total_boxes = 0

    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            empty_labels.append(label_path.name)
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            class_id, error = check_label_line(line.strip(), line_no, label_path, len(class_names))
            if error:
                errors.append(error)
            elif class_id is not None:
                class_counter[class_id] += 1
                total_boxes += 1

    print("数据检查完成")
    print(f"图片数量: {len(images)}")
    print(f"标注目标总数: {total_boxes}")
    print(f"缺少标签文件: {len(missing_labels)}")
    print(f"空标签文件: {len(empty_labels)}")
    print(f"标注格式错误: {len(errors)}")

    if missing_labels:
        print("\n缺少标签的图片:")
        for name in missing_labels[:50]:
            print(f"  {name}")
        if len(missing_labels) > 50:
            print(f"  ... 还有 {len(missing_labels) - 50} 个")

    if empty_labels:
        print("\n空标签文件:")
        for name in empty_labels[:50]:
            print(f"  {name}")
        if len(empty_labels) > 50:
            print(f"  ... 还有 {len(empty_labels) - 50} 个")

    if errors:
        print("\n标注错误:")
        for error in errors[:80]:
            print(f"  {error}")
        if len(errors) > 80:
            print(f"  ... 还有 {len(errors) - 80} 个")

    print("\n类别目标数量:")
    for class_id, name in class_names.items():
        count = class_counter[class_id]
        if count > 0:
            print(f"  {class_id:>2} {name:<10} {count}")

    if missing_labels:
        print("\n提示：以上图片缺少标注，划分数据集时会自动跳过。请到标注页面补充标注。")
    if errors:
        raise SystemExit("标注格式存在问题，请修正后再训练。")


if __name__ == "__main__":
    main()
