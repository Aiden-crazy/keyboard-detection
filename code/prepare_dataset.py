"""Run data check and dataset split in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare keyboard dataset before training.")
    parser.add_argument("--skip-check", action="store_true", help="Skip label validation.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n运行命令:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not args.skip_check:
        run_command([python, str(PROJECT_ROOT / "code" / "check_dataset.py")])
    run_command([python, str(PROJECT_ROOT / "code" / "split_dataset.py")])
    print("\n数据准备完成，可以开始训练。")
    print("推荐训练命令:")
    print("python vision/code/train_yolo.py --model yolov8n.pt --imgsz 640 --epochs 80 --batch 4")


if __name__ == "__main__":
    main()
