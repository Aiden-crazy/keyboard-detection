"""Remove Python cache files generated during development."""

from __future__ import annotations

from pathlib import Path

from common import PROJECT_ROOT


def main() -> None:
    removed_files = 0
    removed_dirs = 0

    for pycache_dir in PROJECT_ROOT.rglob("__pycache__"):
        for item in pycache_dir.iterdir():
            if item.is_file():
                item.unlink()
                removed_files += 1
        pycache_dir.rmdir()
        removed_dirs += 1

    print(f"已清理缓存目录: {removed_dirs}")
    print(f"已删除缓存文件: {removed_files}")


if __name__ == "__main__":
    main()
