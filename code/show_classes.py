"""Print or export class names from dataset.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common import DATASET_YAML, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show keyboard detection class names.")
    parser.add_argument("--data", type=Path, default=DATASET_YAML, help="Dataset yaml path.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "classes.txt", help="Output class list path.")
    parser.add_argument("--export", action="store_true", help="Export class names to a text file.")
    return parser.parse_args()


def load_names(data_yaml: Path) -> dict[int, str]:
    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    names = data.get("names", {})
    return {int(class_id): str(name) for class_id, name in names.items()}


def main() -> None:
    args = parse_args()
    names = load_names(args.data)

    lines = [f"{class_id}: {name}" for class_id, name in names.items()]
    print("类别列表:")
    for line in lines:
        print(line)

    if args.export:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"类别列表已导出到: {args.out}")


if __name__ == "__main__":
    main()
