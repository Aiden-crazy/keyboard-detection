"""Download EasyOCR model files for keyboard key detection.

Downloads the detection and recognition models from GitHub with automatic
fallback to ghproxy mirror (useful for users in China).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "easyocr"

FILES = [
    {
        "name": "craft_mlt_25k",
        "url_github": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "filename": "craft_mlt_25k.pth",
        "min_size_mb": 70,
    },
    {
        "name": "english_g2",
        "url_github": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
        "filename": "english_g2.pth",
        "min_size_mb": 10,
    },
]


def _mirror_url(github_url: str) -> str:
    return "https://ghproxy.net/" + github_url


def _download(url: str, dest: Path) -> bool:
    try:
        print(f"  from {url[:80]}...")
        last_pct = [0]

        def _report(block_num: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            pct = min(block_num * block_size * 100 // total_size, 100)
            if pct - last_pct[0] >= 10:
                print(f"  {pct}%", end="", flush=True)
                last_pct[0] = pct

        urlretrieve(url, dest, reporthook=_report)
        if last_pct[0] > 0:
            print("  done")
        return True
    except Exception as exc:
        print(f"\n  failed: {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def ensure_models() -> bool:
    """Download EasyOCR models if missing. Returns True on success."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for item in FILES:
        target = MODEL_DIR / item["filename"]
        if target.exists() and target.stat().st_size > item["min_size_mb"] * 1024 * 1024:
            print(f"[OK] {item['filename']} already exists")
            continue

        print(f"Downloading {item['name']} ({item['min_size_mb']}+ MB)...")
        zip_path = MODEL_DIR / f"{item['name']}.zip"
        success = _download(item["url_github"], zip_path) or _download(_mirror_url(item["url_github"]), zip_path)

        if not success:
            print(f"[FAIL] Could not download {item['name']}.")
            print("       Please manually download the model files to:")
            print(f"       {MODEL_DIR}")
            all_ok = False
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extract(item["filename"], MODEL_DIR)
            print(f"[OK] {item['filename']} extracted")
        except Exception as exc:
            print(f"[FAIL] Extract error: {exc}")
            all_ok = False
        finally:
            zip_path.unlink(missing_ok=True)

    return all_ok


if __name__ == "__main__":
    ok = ensure_models()
    if ok:
        print("\nAll EasyOCR models ready.")
    else:
        print("\nSome models could not be downloaded. See messages above.")
    sys.exit(0 if ok else 1)
