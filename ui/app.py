"""Gradio UI for keyboard YOLO training and detection."""

from __future__ import annotations

import ctypes
import os
import signal
import sys

# ── Suppress Intel MKL / OpenBLAS thread pools & console hooks ──────────
for _k, _v in [
    ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("OMP_WAIT_POLICY", "passive"),
    ("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1"),
]:
    os.environ.setdefault(_k, _v)


# ── Windows console close → immediate exit (bypasses MKL cleanup hang) ──
if sys.platform == "win32":
    _CTRL_CLOSE_EVENT = 2
    _CTRL_LOGOFF_EVENT = 5
    _CTRL_SHUTDOWN_EVENT = 6

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)  # type: ignore[misc]
    def _win_console_handler(event_type: int) -> int:
        if event_type in (_CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT):
            os._exit(0)
        return 0  # pass other events to next handler

    try:
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_win_console_handler, True)  # type: ignore[attr-defined]
    except Exception:
        pass


def _signal_handler(sig: int, frame: object) -> None:
    print("\nShutting down...")
    os._exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import cv2  # type: ignore[reportMissingImports]
import gradio as gr  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SELECTED_DIR = DATA_DIR / "selected"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
TRAIN_DIR = OUTPUTS_DIR / "train"
DEFAULT_MODEL = MODELS_DIR / "best.pt"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "ui"
TRAIN_SCRIPT = ROOT / "code" / "train_yolo.py"
PREPARE_SCRIPT = ROOT / "code" / "prepare_dataset.py"
DATASET_DIR = ROOT / "datasets" / "keyboard"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEMP_DIR = DATA_DIR / ".temp"
RESULT_IMAGES = ["results.png", "confusion_matrix.png", "P_curve.png", "R_curve.png", "PR_curve.png", "F1_curve.png", "labels.jpg"]
_MODEL_CACHE: dict[str, Any] = {}

_CANVAS_JS = """<script>
(function() {
  var pollTimer, imgCache = {};

  function initCanvas(canvas) {
    var imgDataEl = document.getElementById('anno-img-data');
    var classesEl = document.getElementById('anno-classes');
    if (!imgDataEl || !classesEl) return;
    try {
      var imgInfo = JSON.parse(imgDataEl.value);
      var classes = JSON.parse(classesEl.value);
    } catch(e) { return; }

    canvas.dataset.annoReady = '1';
    var IMG_W = imgInfo.w, IMG_H = imgInfo.h;
    var CLASSES = classes;
    var boxes = [];
    var selectedIndex = -1;
    var hoveredIndex = -1;
    var drawing = false, hasMoved = false;
    var startX = 0, startY = 0, currentX = 0, currentY = 0;
    var ctx = canvas.getContext('2d');

    // Load image, then init everything after it's ready
    var img = imgCache[imgInfo.b64] || new Image();
    imgCache[imgInfo.b64] = img;
    img.src = img.src || 'data:image/png;base64,' + imgInfo.b64;

    function afterImageReady() {
      redraw();
      // Load pre-filled boxes (OCR/model auto-annotations)
      var preFilledEl = document.getElementById('anno-pre-filled');
      if (preFilledEl && preFilledEl.value) {
        try {
          var preBoxes = JSON.parse(preFilledEl.value);
          for (var pb = 0; pb < preBoxes.length; pb++) {
            boxes.push(preBoxes[pb]);
          }
          redraw();
          updateTable();
        } catch(e) {}
      }
    }

    if (img.complete && img.naturalWidth) {
      afterImageReady();
    } else {
      img.onload = afterImageReady;
    }

    function getCoords(e) {
      var rect = canvas.getBoundingClientRect();
      return {
        x: (e.clientX - rect.left) * (IMG_W / rect.width),
        y: (e.clientY - rect.top) * (IMG_H / rect.height)
      };
    }

    function hitTest(x, y, tolerance) {
      tolerance = tolerance || 0;
      for (var i = boxes.length - 1; i >= 0; i--) {
        if (x >= boxes[i].x1 - tolerance && x <= boxes[i].x2 + tolerance &&
            y >= boxes[i].y1 - tolerance && y <= boxes[i].y2 + tolerance) {
          return i;
        }
      }
      return -1;
    }

    function redraw() {
      ctx.clearRect(0, 0, IMG_W, IMG_H);
      ctx.drawImage(img, 0, 0);
      for (var i = 0; i < boxes.length; i++) {
        var b = boxes[i], isSel = i === selectedIndex, isHov = i === hoveredIndex;
        ctx.strokeStyle = isSel ? '#FFD700' : '#00FF00';
        ctx.lineWidth = isSel ? 3 : 2;
        ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
        if (isHov && !isSel) {
          ctx.fillStyle = 'rgba(0,255,0,0.15)';
          ctx.fillRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
        }
        var found = CLASSES.find(function(c) { return c.id === b.class_id; });
        var text = found ? found.str : '?';
        ctx.font = '16px sans-serif';
        var tm = ctx.measureText(text);
        ctx.fillStyle = isSel ? '#FFD700' : '#00FF00';
        ctx.fillRect(b.x1, b.y1 - 18, tm.width + 6, 18);
        ctx.fillStyle = '#000';
        ctx.fillText(text, b.x1 + 3, b.y1 - 4);
      }
      if (drawing) {
        ctx.setLineDash([6, 3]);
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
        ctx.setLineDash([]);
      }
    }

    function updateTable() {
      var tbody = document.getElementById('box-tbody');
      var rows = '';
      for (var i = 0; i < boxes.length; i++) {
        var b = boxes[i];
        var found = CLASSES.find(function(c) { return c.id === b.class_id; });
        var name = found ? found.str : '?';
        rows += '<tr class="' + (i === selectedIndex ? 'selected' : '') + '" onclick="window.__selectBox(' + i + ')">' +
          '<td>' + (i + 1) + '</td><td>' + b.class_id + ':' + name + '</td>' +
          '<td>' + Math.round(b.x1) + '</td><td>' + Math.round(b.y1) + '</td>' +
          '<td>' + Math.round(b.x2) + '</td><td>' + Math.round(b.y2) + '</td>' +
          '<td><button class="del-btn" onclick="event.stopPropagation();window.__deleteBox(' + i + ');">x</button></td></tr>';
      }
      tbody.innerHTML = rows;
      var countEl = document.getElementById('box-count');
      if (countEl) countEl.textContent = boxes.length;
      // Scroll to selected row
      if (selectedIndex >= 0) {
        setTimeout(function() {
          var selRow = tbody.querySelector('tr.selected');
          if (selRow) selRow.scrollIntoView({ block: 'nearest' });
        }, 50);
      }
    }

    // Exposed global functions
    window.__selectBox = function(i) {
      if (i >= 0 && i < boxes.length) {
        selectedIndex = i;
        window.__setClass(boxes[i].class_id, document.querySelector('#anno-keyboard .kb-key[data-cls="' + boxes[i].class_id + '"]'));
      }
      redraw();
      updateTable();
    };

    window.__deleteBox = function(i) {
      boxes.splice(i, 1);
      selectedIndex = -1;
      redraw();
      updateTable();
    };

    window.__deleteSelected = function() {
      if (selectedIndex >= 0) {
        boxes.splice(selectedIndex, 1);
        selectedIndex = -1;
        redraw();
        updateTable();
      }
    };

    window.__getAnnotations = function() {
      return JSON.stringify(boxes.map(function(b) {
        return {
          class_id: b.class_id,
          x_center: Math.round((b.x1 + b.x2) / 2 / IMG_W * 1e6) / 1e6,
          y_center: Math.round((b.y1 + b.y2) / 2 / IMG_H * 1e6) / 1e6,
          width: Math.round((b.x2 - b.x1) / IMG_W * 1e6) / 1e6,
          height: Math.round((b.y2 - b.y1) / IMG_H * 1e6) / 1e6
        };
      }));
    };

    // Canvas event handlers
    canvas.onmousedown = function(e) {
      var coords = getCoords(e);
      startX = coords.x; startY = coords.y;
      currentX = coords.x; currentY = coords.y;
      drawing = true; hasMoved = false;
      e.preventDefault();
    };

    canvas.onmousemove = function(e) {
      var coords = getCoords(e);
      var ci = document.getElementById('coord-info');
      if (ci) ci.textContent = 'X: ' + Math.round(coords.x) + '  Y: ' + Math.round(coords.y);
      if (drawing) {
        currentX = coords.x; currentY = coords.y;
        if (Math.abs(currentX - startX) > 2 || Math.abs(currentY - startY) > 2) hasMoved = true;
        redraw();
      } else {
        // Hover detection
        var prevHover = hoveredIndex;
        hoveredIndex = hitTest(coords.x, coords.y, 2);
        canvas.style.cursor = hoveredIndex >= 0 ? 'pointer' : 'crosshair';
        if (prevHover !== hoveredIndex) redraw();
      }
    };

    canvas.onmouseup = function(e) {
      if (!drawing) return;
      drawing = false;
      if (!hasMoved) {
        // Click (not drag) — try to select a box
        var found = hitTest(currentX, currentY, 3);
        if (found >= 0) {
          selectedIndex = found;
          window.__setClass(boxes[found].class_id, document.querySelector('#anno-keyboard .kb-key[data-cls="' + boxes[found].class_id + '"]'));
        } else {
          selectedIndex = -1;
        }
        redraw(); updateTable();
      } else {
        var x1 = Math.min(startX, currentX), y1 = Math.min(startY, currentY);
        var x2 = Math.max(startX, currentX), y2 = Math.max(startY, currentY);
        if (x2 - x1 > 3 && y2 - y1 > 3) {
          boxes.push({ class_id: currentClassId, x1: x1, y1: y1, x2: x2, y2: y2 });
          redraw(); updateTable();
        }
      }
    };

    canvas.onmouseleave = function() {
      if (drawing) { drawing = false; redraw(); }
      if (hoveredIndex >= 0) { hoveredIndex = -1; redraw(); }
    };

    // Clear button
    var clearBtn = document.getElementById('btn-clear');
    if (clearBtn && !clearBtn.__bound) {
      clearBtn.__bound = true;
      clearBtn.onclick = function() {
        boxes = []; selectedIndex = -1; redraw(); updateTable();
      };
    }

    // Delete-selected button
    var delSelBtn = document.getElementById('btn-del-selected');
    if (delSelBtn && !delSelBtn.__bound) {
      delSelBtn.__bound = true;
      delSelBtn.onclick = function() {
        if (selectedIndex >= 0) {
          boxes.splice(selectedIndex, 1);
          selectedIndex = -1;
          redraw(); updateTable();
        }
      };
    }

    // Keyboard key click: set current class
    var currentClassId = 0;
    window.__setClass = function(clsId, btn) {
      currentClassId = clsId;
      var keys = document.querySelectorAll('#anno-keyboard .kb-key');
      for (var k = 0; k < keys.length; k++) keys[k].classList.remove('active');
      if (btn) btn.classList.add('active');
      var lbl = document.getElementById('current-class-label');
      if (lbl) {
        var found = CLASSES.find(function(c) { return c.id === clsId; });
        lbl.textContent = clsId + ': ' + (found ? found.str : '?');
      }
    };
    // Init first key as active
    var firstKey = document.querySelector('#anno-keyboard .kb-key');
    if (firstKey) currentClassId = parseInt(firstKey.getAttribute('data-cls'));

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIndex >= 0) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        boxes.splice(selectedIndex, 1);
        selectedIndex = -1;
        redraw(); updateTable();
      }
    });
  }

  function tryInit() {
    var canvas = document.getElementById('anno-canvas');
    if (canvas && canvas.dataset.annoReady !== '1') initCanvas(canvas);
  }

  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(tryInit, 400);
})();
</script>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch keyboard training/detection UI.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO model path.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_path(file_item: Any) -> Path:
    if isinstance(file_item, (str, Path)):
        return Path(file_item)
    return Path(str(file_item.name))


def count_files(path: Path, exts: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in exts)


def dataset_status() -> str:
    return "\n".join(
        [
            "当前数据状态：",
            f"原始图片 raw：{count_files(RAW_DIR, IMAGE_EXTS)} 张",
            f"训练图片 selected：{count_files(SELECTED_DIR, IMAGE_EXTS)} 张",
            f"标注 labels：{count_files(LABELS_DIR, {'.txt'})} 个 txt 文件",
            "已划分 train/val/test："
            f"{count_files(DATASET_DIR / 'images' / 'train', IMAGE_EXTS)} / "
            f"{count_files(DATASET_DIR / 'images' / 'val', IMAGE_EXTS)} / "
            f"{count_files(DATASET_DIR / 'images' / 'test', IMAGE_EXTS)} 张",
            f"默认模型：{DEFAULT_MODEL} {'已存在' if DEFAULT_MODEL.exists() else '不存在'}",
        ]
    )


def next_available_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index:03d}{suffix}"
        index += 1
    return candidate


def safe_stem(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in text.strip())
    return cleaned.strip("_") or "keyboard"


def build_label_map(label_files: list[Any] | None) -> dict[str, Path]:
    label_map: dict[str, Path] = {}
    for file_item in label_files or []:
        source = as_path(file_item)
        if source.suffix.lower() == ".txt":
            label_map[source.stem] = source
    return label_map


def copy_files(files: list[Any] | None, target_dir: Path, exts: set[str]) -> list[Path]:
    ensure_dir(target_dir)
    copied: list[Path] = []
    for file_item in files or []:
        source = as_path(file_item)
        if source.suffix.lower() not in exts:
            continue
        target = target_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    return copied


def import_dataset(
    image_files: list[Any] | None,
    label_files: list[Any] | None,
    copy_raw: bool,
    target_folder: str,
    rename_mode: str,
    name_prefix: str,
    start_index: int,
) -> tuple[str, str]:
    folder_map = {
        "训练图片 selected": SELECTED_DIR,
        "原始图片 raw": RAW_DIR,
    }
    image_target_dir = folder_map.get(target_folder, SELECTED_DIR)
    ensure_dir(image_target_dir)
    ensure_dir(LABELS_DIR)

    label_map = build_label_map(label_files)
    copied_images: list[Path] = []
    copied_labels: list[Path] = []
    renamed_pairs: list[str] = []
    prefix = safe_stem(name_prefix)
    number = int(start_index)

    for file_item in image_files or []:
        source = as_path(file_item)
        if source.suffix.lower() not in IMAGE_EXTS:
            continue

        if rename_mode == "按前缀自动重命名":
            stem = f"{prefix}_{number:04d}"
            number += 1
        else:
            stem = safe_stem(source.stem)

        image_target = next_available_path(image_target_dir, stem, source.suffix.lower())
        shutil.copy2(source, image_target)
        copied_images.append(image_target)

        label_source = label_map.get(source.stem)
        if label_source is not None:
            label_target = next_available_path(LABELS_DIR, image_target.stem, ".txt")
            shutil.copy2(label_source, label_target)
            copied_labels.append(label_target)
            renamed_pairs.append(f"{source.name} + {label_source.name} -> {image_target.name} + {label_target.name}")
        else:
            renamed_pairs.append(f"{source.name} -> {image_target.name}，未找到同名标注")

        if copy_raw and image_target_dir != RAW_DIR:
            ensure_dir(RAW_DIR)
            shutil.copy2(image_target, RAW_DIR / image_target.name)

    unused_labels = [label for stem, label in label_map.items() if not any(as_path(img).stem == stem for img in image_files or [])]
    for label in unused_labels:
        target = next_available_path(LABELS_DIR, safe_stem(label.stem), ".txt")
        shutil.copy2(label, target)
        copied_labels.append(target)

    tips = [
        f"图片保存位置：{image_target_dir}",
        f"已导入图片：{len(copied_images)} 张",
        f"已导入/匹配标注：{len(copied_labels)} 个 -> {LABELS_DIR}",
    ]
    if copy_raw and image_target_dir != RAW_DIR:
        tips.append(f"已同步备份图片到：{RAW_DIR}")
    if renamed_pairs:
        tips.append("\n命名结果：")
        tips.extend(renamed_pairs[:80])
    if copied_images and not copied_labels:
        tips.append("\n注意：YOLO 训练需要同名 .txt 标注文件，例如 keyboard_0001.jpg 对应 keyboard_0001.txt。")
    return "\n".join(tips), dataset_status()


def training_images(exp_name: str) -> list[str]:
    exp_dir = TRAIN_DIR / (exp_name.strip() or "keyboard_yolo")
    if not exp_dir.exists():
        return []
    paths = [exp_dir / name for name in RESULT_IMAGES]
    paths += sorted(exp_dir.glob("train_batch*.jpg"))[:4]
    paths += sorted(exp_dir.glob("val_batch*.jpg"))[:4]
    return [str(path) for path in paths if path.exists()]


def stream_command(command: list[str]) -> Iterator[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        yield line
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def train_model(base: str, imgsz: int, epochs: int, batch: int, device: str, exp_name: str, prepare: bool) -> Iterator[tuple[str, list[str], str, str]]:
    exp_name = exp_name.strip() or "keyboard_yolo"
    logs: list[str] = []

    def update(text: str) -> tuple[str, list[str], str, str]:
        logs.append(text)
        if len(logs) > 500:
            logs[: len(logs) - 500] = []
        return "".join(logs), training_images(exp_name), str(DEFAULT_MODEL), dataset_status()

    try:
        yield update("开始训练任务。\n" + dataset_status() + "\n\n")
        if prepare:
            cmd = [sys.executable, str(PREPARE_SCRIPT)]
            yield update("正在检查标注并划分数据集...\n" + " ".join(cmd) + "\n")
            for line in stream_command(cmd):
                yield update(line)
            yield update("\n数据准备完成。\n\n")

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--model",
            base.strip() or "models/yolo11n.pt",
            "--imgsz",
            str(imgsz),
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--name",
            exp_name,
        ]
        if device.strip():
            cmd += ["--device", device.strip()]
        yield update("开始 YOLO 训练，日志会实时刷新。\n" + " ".join(cmd) + "\n\n")
        for line in stream_command(cmd):
            yield update(line)
        _MODEL_CACHE.clear()
        yield update(f"\n训练完成。最佳模型路径：{DEFAULT_MODEL}\n")
    except subprocess.CalledProcessError as exc:
        yield update(f"\n训练失败，退出码：{exc.returncode}。请检查上方日志。\n")
    except Exception as exc:  # noqa: BLE001
        yield update(f"\n训练异常：{exc}\n")


def load_model(model_path: str) -> Any:
    from ultralytics import YOLO  # type: ignore[reportMissingImports]

    resolved = Path(model_path).resolve()
    key = str(resolved)
    if key not in _MODEL_CACHE:
        if not resolved.exists():
            raise FileNotFoundError(f"模型文件不存在：{resolved}")
        _MODEL_CACHE[key] = YOLO(str(resolved))
    return _MODEL_CACHE[key]


def normalize_image(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        raise ValueError("请先上传一张键盘图片。")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def load_class_list() -> list[dict[str, object]]:
    """Read class names from dataset.yaml and return a list of {id, str} dicts."""
    yaml_path = ROOT / "dataset.yaml"
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    return [{"id": int(k), "str": str(v)} for k, v in names.items()]


def numpy_to_base64(image: np.ndarray) -> tuple[str, int, int]:
    """Convert a numpy RGB image to a base64 JPEG string. Returns (b64, disp_w, disp_h)."""
    h, w = image.shape[:2]
    max_dim = 1920
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    disp_h, disp_w = image.shape[:2]
    img = Image.fromarray(image)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), disp_w, disp_h


def build_annotation_html(b64: str, img_w: int, img_h: int, classes: list[dict[str, object]], auto_annotations: list[dict[str, object]] | None = None) -> str:
    """Return self-contained HTML for the annotation canvas (no <script>, options pre-built)."""
    classes_json = json.dumps(classes, ensure_ascii=False)
    img_info_json = json.dumps({"w": img_w, "h": img_h, "b64": b64}, ensure_ascii=False)
    auto_json = json.dumps(auto_annotations or [], ensure_ascii=False)

    # 68-key visual keyboard layout: (class_id, label, width_units)
    keyboard_rows = [
        [(36, "Esc", 1.0), (1, "1", 1.0), (2, "2", 1.0), (3, "3", 1.0), (4, "4", 1.0), (5, "5", 1.0), (6, "6", 1.0), (7, "7", 1.0), (8, "8", 1.0), (9, "9", 1.0), (0, "0", 1.0), (43, "Bksp", 1.7)],
        [(37, "Tab", 1.3), (26, "Q", 1.0), (32, "W", 1.0), (14, "E", 1.0), (27, "R", 1.0), (29, "T", 1.0), (34, "Y", 1.0), (30, "U", 1.0), (18, "I", 1.0), (24, "O", 1.0), (25, "P", 1.0), (42, "Enter", 1.7)],
        [(38, "Caps", 1.6), (10, "A", 1.0), (28, "S", 1.0), (13, "D", 1.0), (15, "F", 1.0), (16, "G", 1.0), (17, "H", 1.0), (19, "J", 1.0), (20, "K", 1.0), (21, "L", 1.0)],
        [(39, "Shift", 2.0), (35, "Z", 1.0), (33, "X", 1.0), (12, "C", 1.0), (31, "V", 1.0), (11, "B", 1.0), (23, "N", 1.0), (22, "M", 1.0), (39, "Shift", 2.0)],
        [(40, "Ctrl", 1.25), (41, "Alt", 1.25), (44, "Space", 6.0), (41, "Alt", 1.25), (40, "Ctrl", 1.25)],
    ]
    unit_px = 48
    keys_html = '<div class="keyboard" id="anno-keyboard">'
    for row in keyboard_rows:
        keys_html += '<div class="kb-row">'
        for cls_id, label, width in row:
            w = int(unit_px * width)
            keys_html += f'<button class="kb-key" data-cls="{cls_id}" style="width:{w}px;" onclick="window.__setClass({cls_id},this)">{label}</button>'
        keys_html += '</div>'
    keys_html += '</div>'

    return f"""<div id="annotation-root">
<style>
#annotation-root * {{ box-sizing: border-box; }}
#annotation-root .toolbar {{ display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 10px; padding: 12px 16px; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; }}
#annotation-root .toolbar .hint {{ font-size: 15px; color: #666; padding-top: 4px; }}
#annotation-root .toolbar button.danger {{ background: #fee2e2; border-color: #fca5a5; color: #991b1b; font-weight: 600; padding: 8px 18px; font-size: 17px; border-radius: 5px; cursor: pointer; }}
#annotation-root .toolbar button.danger:hover {{ background: #fecaca; }}
#annotation-root .keyboard {{ background: #e2e6ea; padding: 10px 8px; border-radius: 8px; display: inline-block; }}
#annotation-root .kb-row {{ display: flex; gap: 3px; margin-bottom: 3px; justify-content: flex-start; }}
#annotation-root .kb-key {{ height: 42px; flex-shrink: 0; border: 1px solid #999; border-radius: 5px; background: #fff; cursor: pointer; font-size: 15px; font-weight: 500; color: #333; box-shadow: 0 1px 2px rgba(0,0,0,0.15); transition: all 0.1s; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 0 4px; }}
#annotation-root .kb-key:hover {{ background: #dbeafe; border-color: #4a90d9; }}
#annotation-root .kb-key.active {{ background: #2563eb; color: #fff; border-color: #1d4ed8; box-shadow: 0 0 0 3px rgba(37,99,235,0.3); }}
#annotation-root .canvas-wrapper {{ position: relative; display: inline-block; max-width: 100%; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
#annotation-root #anno-canvas {{ display: block; max-width: 100%; max-height: 68vh; cursor: crosshair; }}
#annotation-root #coord-info {{ font-size: 14px; color: #666; margin-top: 4px; }}
#annotation-root .section-title {{ font-size: 17px; font-weight: 600; margin: 10px 0 4px; }}
#annotation-root .box-table {{ width: 100%; border-collapse: collapse; font-size: 16px; margin-top: 8px; }}
#annotation-root .box-table th, #annotation-root .box-table td {{ padding: 8px 12px; border: 1px solid #e0e0e0; text-align: center; }}
#annotation-root .box-table th {{ background: #f2f2f2; font-weight: 600; }}
#annotation-root .box-table tr {{ cursor: pointer; }}
#annotation-root .box-table tr:hover {{ background: #f0f7ff; }}
#annotation-root .box-table tr.selected {{ background: #dbeafe; }}
#annotation-root .del-btn {{ background: none; border: none; color: #dc2626; cursor: pointer; font-size: 18px; padding: 2px 6px; }}
#annotation-root .del-btn:hover {{ color: #991b1b; }}
</style>
<div class="toolbar">
  <div style="flex:1;min-width:500px;">
    <div style="margin-bottom:6px;font-size:17px;font-weight:600;">当前类别：<span id="current-class-label" style="color:#2563eb;">0: 0</span></div>
    {keys_html}
  </div>
  <div style="display:flex;flex-direction:column;gap:8px;align-items:flex-end;">
    <span class="hint">点击键盘按键选择类别<br>在图片上拖拽绘制标注框<br>点击已有框 → Delete 键删除</span>
    <button id="btn-del-selected" style="padding:8px 16px;background:#fff3cd;border:1px solid #ffc107;color:#856404;font-weight:600;font-size:15px;border-radius:5px;cursor:pointer;">删除选中框</button>
    <button id="btn-clear" class="danger">清除全部</button>
  </div>
</div>
<div class="canvas-wrapper">
  <canvas id="anno-canvas" width="{img_w}" height="{img_h}"></canvas>
</div>
<div id="coord-info">X: -  Y: -</div>
<div class="section-title">标注列表（共 <span id="box-count">0</span> 个框）</div>
<div style="max-height:280px;overflow-y:auto;">
  <table class="box-table">
    <thead><tr><th>#</th><th>类别</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th><th>操作</th></tr></thead>
    <tbody id="box-tbody"></tbody>
  </table>
</div>
<textarea id="anno-img-data" style="display:none;">{img_info_json}</textarea>
<textarea id="anno-classes" style="display:none;">{classes_json}</textarea>
<textarea id="anno-pre-filled" style="display:none;">{auto_json}</textarea>
</div>"""

def _build_annotation_result(image: np.ndarray, auto_annotations: list[dict[str, object]] | None = None) -> tuple[str, str, str]:
    """Normalize image, save temp, encode base64, build annotation HTML. Returns (html, temp_path, b64)."""
    image = normalize_image(image)
    ensure_dir(TEMP_DIR)
    temp_path = TEMP_DIR / f"annotation_{uuid.uuid4().hex[:8]}.jpg"
    Image.fromarray(image).save(str(temp_path), "JPEG", quality=95)
    b64, disp_w, disp_h = numpy_to_base64(image)
    classes = load_class_list()
    html = build_annotation_html(b64, disp_w, disp_h, classes, auto_annotations)
    return html, str(temp_path), b64


def on_image_upload(image: np.ndarray | None) -> tuple[str, str, str]:
    """Handle image upload — just display the canvas, no auto-annotation."""
    if image is None:
        return (
            "<div style='padding:40px;text-align:center;color:#888;'><h3>请先上传一张键盘图片</h3></div>",
            "",
            "",
        )
    return _build_annotation_result(image)


def trigger_ocr(image: np.ndarray | None, conf: float = 0.10) -> tuple[str, str, str]:
    """Run EasyOCR on the uploaded image and pre-fill annotation boxes."""
    if image is None:
        return (
            "<div style='padding:40px;text-align:center;color:#888;'><h3>请先上传一张键盘图片</h3></div>",
            "",
            "",
        )
    image = normalize_image(image)
    _, disp_w, disp_h = numpy_to_base64(image)
    try:
        auto_annotations = ocr_auto_annotate(image, disp_w, disp_h, conf_threshold=float(conf))
    except Exception:
        auto_annotations = None
    return _build_annotation_result(image, auto_annotations)


def trigger_model(image: np.ndarray | None, model_path: str = "", conf: float = 0.10) -> tuple[str, str, str]:
    """Run YOLO model on the uploaded image and pre-fill annotation boxes."""
    if image is None:
        return (
            "<div style='padding:40px;text-align:center;color:#888;'><h3>请先上传一张键盘图片</h3></div>",
            "",
            "",
        )
    image = normalize_image(image)
    original_h, original_w = image.shape[:2]
    _, disp_w, disp_h = numpy_to_base64(image)
    auto_annotations = None
    if model_path and Path(model_path).exists():
        try:
            model = load_model(model_path)
            result = model.predict(source=image, conf=float(conf), imgsz=640, verbose=False)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                scale_x = disp_w / original_w
                scale_y = disp_h / original_h
                auto_annotations = []
                for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                    x1, y1, x2, y2 = box.tolist()
                    auto_annotations.append({
                        "class_id": int(cls_id),
                        "x1": round(x1 * scale_x),
                        "y1": round(y1 * scale_y),
                        "x2": round(x2 * scale_x),
                        "y2": round(y2 * scale_y),
                    })
        except Exception:
            pass
    return _build_annotation_result(image, auto_annotations)


OCR_TEXT_TO_CLASS: dict[str, int] = {
    **{str(d): d for d in range(10)},
    **{chr(ord("a") + i): 10 + i for i in range(26)},
    **{chr(ord("A") + i): 10 + i for i in range(26)},
    # Punctuation on number keys (OCR may read the symbol instead of the digit)
    "!": 1, "@": 2, "#": 3, "$": 4, "%": 5, "^": 6, "&": 7, "*": 8, "(": 9, ")": 0,
    # Special keys
    "esc": 36, "escape": 36,
    "tab": 37, "tah": 37,
    "caps": 38, "capslock": 38, "caps lock": 38, "caplock": 38,
    "shift": 39, "shft": 39, "shif": 39, "shlft": 39,
    "ctrl": 40, "control": 40, "ctr": 40, "ctl": 40, "ctri": 40,
    "alt": 41, "ait": 41,
    "enter": 42, "return": 42, "ent": 42, "enterkey": 42,
    "backspace": 43, "back": 43, "bksp": 43, "bkspace": 43, "backspc": 43, "bk Space": 43,
    "space": 44, "spc": 44, "spacebar": 44, "space bar": 44,
}
_OCR_READER: object = None


def _get_ocr_reader() -> object:
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr  # type: ignore[reportMissingImports]
        model_dir = MODELS_DIR / "easyocr"
        ensure_dir(model_dir)
        _OCR_READER = easyocr.Reader(["en"], gpu=False, model_storage_directory=str(model_dir), verbose=False)
    return _OCR_READER


def ocr_auto_annotate(image: np.ndarray, disp_w: int, disp_h: int, conf_threshold: float = 0.10) -> list[dict[str, object]]:
    """Use EasyOCR to detect text regions on keyboard and map to class IDs."""
    reader = _get_ocr_reader()
    original_h, original_w = image.shape[:2]
    scale_x = disp_w / original_w
    scale_y = disp_h / original_h

    results = reader.readtext(
        image,
        detail=1,
        text_threshold=0.35,   # lower = more text detected (default 0.7)
        low_text=0.15,         # lower bound for low-confidence text (default 0.4)
        mag_ratio=2.0,         # 2.0: ~36% faster than 2.5, still catches small keys
    )
    annotations: list[dict[str, object]] = []
    for bbox, text, conf in results:
        if conf < conf_threshold:
            continue
        cleaned = re.sub(r"\s+", " ", text).strip().lower()

        # Skip known non-class keys
        if cleaned in {"win", "windows", "fn", "prtsc", "prtscreen", "print", "scroll",
                       "pause", "insert", "ins", "home", "end", "pgup", "pgdn",
                       "num", "numlk", "numlock", "delete", "del",
                       "up", "down", "left", "right"}:
            continue
        # Skip function keys F1-F24
        if re.match(r"^f\d{1,2}$", cleaned):
            continue

        cls_id = OCR_TEXT_TO_CLASS.get(cleaned)
        if cls_id is None:
            stripped = re.sub(r"[^a-z0-9]", "", cleaned)
            cls_id = OCR_TEXT_TO_CLASS.get(stripped)
        if cls_id is None and len(cleaned) == 1:
            cls_id = OCR_TEXT_TO_CLASS.get(cleaned)
        if cls_id is None:
            continue
        x1 = min(p[0] for p in bbox)
        y1 = min(p[1] for p in bbox)
        x2 = max(p[0] for p in bbox)
        y2 = max(p[1] for p in bbox)
        annotations.append({
            "class_id": cls_id,
            "x1": round(x1 * scale_x),
            "y1": round(y1 * scale_y),
            "x2": round(x2 * scale_x),
            "y2": round(y2 * scale_y),
            "_rel_y": (y1 + y2) / 2 / original_h,  # relative Y for position heuristic
        })

    # Post-processing: fix common OCR confusions
    for ann in annotations:
        if ann["class_id"] == 2:  # OCR read "2" — might be Z
            if ann["_rel_y"] > 0.55:  # in bottom 45% = letter zone
                has_number_row_2 = any(
                    a["class_id"] == 2 and a.get("_rel_y", 0.5) < 0.35
                    for a in annotations
                )
                if not has_number_row_2:
                    ann["class_id"] = 35  # remap to Z
        elif ann["class_id"] == 0:  # OCR read "0" — might be O
            if ann["_rel_y"] > 0.55:
                has_number_row_0 = any(
                    a["class_id"] == 0 and a.get("_rel_y", 0.5) < 0.35
                    for a in annotations
                )
                if not has_number_row_0:
                    ann["class_id"] = 24  # remap to O

    # Strip internal keys before returning
    for ann in annotations:
        ann.pop("_rel_y", None)
    return annotations


def dhash(path: Path, hash_size: int = 16) -> str:
    """Compute difference hash of an image — tolerant to JPEG recompression."""
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("L").resize((hash_size + 1, hash_size), PILImage.LANCZOS)
    pixels = list(img.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return hex(int("".join(bits), 2))


def have_same_image(path_a: Path, path_b: Path, threshold: int = 8) -> bool:
    """Check if two images are perceptually identical using dHash."""
    try:
        h1, h2 = dhash(path_a), dhash(path_b)
        return bin(int(h1, 16) ^ int(h2, 16)).count("1") <= threshold
    except Exception:
        return False


def save_annotations(temp_path_str: str, json_data: str, b64_fallback: str) -> tuple[str, str]:
    """Save annotations: copy image to selected/, write YOLO .txt to labels/."""
    if not temp_path_str and not b64_fallback:
        return "请先上传一张图片再保存。", dataset_status()

    temp_path = Path(temp_path_str) if temp_path_str else None
    if not temp_path or not temp_path.exists():
        # Temp file missing (e.g. .temp was cleaned) — recreate from base64 fallback
        if not b64_fallback:
            return "临时图片文件已丢失，请重新上传图片。", dataset_status()
        ensure_dir(TEMP_DIR)
        temp_path = TEMP_DIR / f"annotation_{uuid.uuid4().hex[:8]}.jpg"
        temp_path.write_bytes(base64.b64decode(b64_fallback))

    try:
        boxes = json.loads(json_data or "[]")
    except json.JSONDecodeError:
        return "标注数据解析失败，请重试。", dataset_status()

    # Check if this image matches an existing selected image by perceptual hash
    matched_stem = None
    for p in sorted(SELECTED_DIR.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if have_same_image(temp_path, p):
                matched_stem = p.stem
                break

    if matched_stem:
        image_target = SELECTED_DIR / f"{matched_stem}{temp_path.suffix.lower()}"
    else:
        existing = count_files(SELECTED_DIR, IMAGE_EXTS)
        stem = f"keyboard_{existing + 1:04d}"
        image_target = next_available_path(SELECTED_DIR, stem, temp_path.suffix.lower())

    shutil.copy2(str(temp_path), str(image_target))

    label_path = LABELS_DIR / f"{image_target.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        for box in boxes:
            cls_id = int(box["class_id"])
            xc = float(box["x_center"])
            yc = float(box["y_center"])
            bw = float(box["width"])
            bh = float(box["height"])
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    temp_path.unlink(missing_ok=True)

    return (
        f"已保存：\n图片 -> vision/data/selected/{image_target.name}\n标注 -> vision/data/labels/{label_path.name}\n（{len(boxes)} 个框）",
        dataset_status(),
    )


def build_result_rows(result: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return rows
    for index, (box, cls_id, score) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()), start=1):
        x1, y1, x2, y2 = box.tolist()
        rows.append([index, result.names.get(int(cls_id), str(cls_id)), round(float(score), 4), int(x1), int(y1), int(x2), int(y2)])
    return rows


def detect(image: np.ndarray | None, model_path: str, conf: float, imgsz: int) -> tuple[np.ndarray, list[list[Any]], str]:
    model = load_model(model_path)
    result = model.predict(source=normalize_image(image), conf=conf, imgsz=imgsz, verbose=False)[0]
    plotted_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    rows = build_result_rows(result)
    ensure_dir(DEFAULT_OUTPUT_DIR)
    output_path = DEFAULT_OUTPUT_DIR / "latest_result.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(plotted_rgb, cv2.COLOR_RGB2BGR))
    return plotted_rgb, rows, f"检测完成：共检测到 {len(rows)} 个目标。结果图已保存到：{output_path}"


def create_demo(default_model: Path) -> gr.Blocks:
    with gr.Blocks(title="键盘字符检测与训练系统") as demo:
        gr.Markdown("# 键盘字符检测与训练系统\n拖入图片和 YOLO 标注即可导入数据，点击训练后可实时查看日志和训练曲线。")
        with gr.Tabs():
            with gr.Tab("数据上传 / 训练可视化"):
                with gr.Row():
                    with gr.Column():
                        image_files = gr.File(label="拖入图片", file_count="multiple", file_types=["image"], type="filepath")
                        label_files = gr.File(label="可选：拖入同名 YOLO 标注 .txt", file_count="multiple", file_types=[".txt"], type="filepath")
                        target_folder = gr.Radio(
                            label="图片放入哪个文件夹",
                            choices=["训练图片 selected", "原始图片 raw"],
                            value="训练图片 selected",
                        )
                        rename_mode = gr.Radio(
                            label="命名方式",
                            choices=["保留原文件名", "按前缀自动重命名"],
                            value="按前缀自动重命名",
                        )
                        name_prefix = gr.Textbox(label="图片命名前缀", value="keyboard")
                        start_index = gr.Number(label="起始编号", value=1, precision=0)
                        copy_raw = gr.Checkbox(label="图片放入 selected 时，同时备份到 vision/data/raw", value=True)
                        import_button = gr.Button("按设置命名并放入文件夹")
                    with gr.Column():
                        import_info = gr.Textbox(label="导入和命名结果", lines=10)
                        status = gr.Textbox(label="数据状态", value=dataset_status(), lines=8)
                        refresh = gr.Button("刷新状态")
                with gr.Row():
                    base = gr.Textbox(label="基础模型", value="models/yolo11n.pt")
                    exp_name = gr.Textbox(label="实验名称", value="keyboard_yolo")
                    device = gr.Textbox(label="设备：留空自动，CPU 填 cpu，显卡填 0", value="")
                with gr.Row():
                    train_imgsz = gr.Slider(label="训练图片尺寸", minimum=320, maximum=1280, value=640, step=32)
                    epochs = gr.Slider(label="训练轮数", minimum=1, maximum=300, value=80, step=1)
                    batch = gr.Slider(label="批大小", minimum=1, maximum=32, value=4, step=1)
                prepare = gr.Checkbox(label="训练前自动检查标注并划分 train/val/test", value=True)
                train_button = gr.Button("开始训练并可视化", variant="primary")
                with gr.Row():
                    train_log = gr.Textbox(label="训练日志", lines=24, max_lines=30)
                    gallery = gr.Gallery(label="训练曲线 / 评估图", columns=2, height=520)
                trained_path = gr.Textbox(label="训练完成模型路径", value=str(DEFAULT_MODEL))
                import_button.click(
                    import_dataset,
                    [image_files, label_files, copy_raw, target_folder, rename_mode, name_prefix, start_index],
                    [import_info, status],
                )
                refresh.click(dataset_status, None, status)
                train_button.click(train_model, [base, train_imgsz, epochs, batch, device, exp_name, prepare], [train_log, gallery, trained_path, status])

            with gr.Tab("图片检测"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="输入键盘图片", type="numpy")
                        model_input = gr.Textbox(label="模型路径", value=str(default_model))
                        conf_slider = gr.Slider(label="置信度阈值", minimum=0.05, maximum=0.95, value=0.25, step=0.05)
                        imgsz_slider = gr.Slider(label="推理图片尺寸", minimum=320, maximum=1280, value=640, step=32)
                        detect_button = gr.Button("开始检测", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(label="检测结果", type="numpy")
                        summary_output = gr.Textbox(label="运行信息")
                table = gr.Dataframe(headers=["序号", "类别", "置信度", "x1", "y1", "x2", "y2"], label="检测结果表")
                detect_button.click(detect, [image_input, model_input, conf_slider, imgsz_slider], [image_output, table, summary_output])

            with gr.Tab("图片标注"):
                gr.Markdown("### 在线上传图片并绘制 YOLO 标注框")
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        annotation_image = gr.Image(label="1. 上传图片", type="numpy", height=320)
                        auto_conf = gr.Slider(label="OCR 置信度阈值（越低框越多、误标也越多）", minimum=0.05, maximum=0.5, value=0.20, step=0.05)
                        ocr_button = gr.Button("OCR 自动标注（推荐）", variant="primary")
                        gr.Markdown("<span style='font-size:12px;color:#888;'>使用 EasyOCR 识别按键文字，<br>字母数字都能检测，与训练无关。</span>")
                        gr.Markdown("---")
                        auto_model = gr.Textbox(label="YOLO 模型路径（需训练 100+ 张图才好用）", value=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else "models/yolo11n.pt")
                        model_button = gr.Button("模型自动标注（数据少时效果差）", variant="secondary", size="sm")
                    with gr.Column(scale=1, min_width=180):
                        save_button = gr.Button("3. 保存标注到数据集", variant="primary")
                        save_status = gr.Textbox(label="保存状态", lines=2, interactive=False)
                    with gr.Column(scale=6):
                        gr.Markdown("**2. 推荐先用 OCR 自动标注 → 手动修正 → 攒够 100+ 张图再训练模型**")
                annotation_html = gr.HTML(label="标注画布")
                temp_image_path = gr.State("")
                image_b64_state = gr.State("")
                hidden_annotations = gr.Textbox(visible=False, value="[]")

                annotation_image.change(
                    fn=on_image_upload,
                    inputs=[annotation_image],
                    outputs=[annotation_html, temp_image_path, image_b64_state],
                )
                ocr_button.click(
                    fn=trigger_ocr,
                    inputs=[annotation_image, auto_conf],
                    outputs=[annotation_html, temp_image_path, image_b64_state],
                )
                model_button.click(
                    fn=trigger_model,
                    inputs=[annotation_image, auto_model, auto_conf],
                    outputs=[annotation_html, temp_image_path, image_b64_state],
                )
                save_button.click(
                    fn=save_annotations,
                    inputs=[temp_image_path, hidden_annotations, image_b64_state],
                    outputs=[save_status, status],
                    js="""(path, data, b64) => {
                        const json = (window.__getAnnotations && window.__getAnnotations()) || '[]';
                        return [path, json, b64];
                    }""",
                )

        gr.Markdown("提示：训练图片保存到 `vision/data/selected/`，标注保存到 `vision/data/labels/`；最佳模型会复制到 `vision/models/best.pt`。")
    return demo


def main() -> None:
    args = parse_args()
    os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR))
    # Auto-create essential directories
    for d in [DATA_DIR, RAW_DIR, SELECTED_DIR, LABELS_DIR, TEMP_DIR, MODELS_DIR, OUTPUTS_DIR, DEFAULT_OUTPUT_DIR]:
        ensure_dir(d)
    demo = create_demo(args.model)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        head=_CANVAS_JS,
        theme=gr.themes.Soft(text_size="lg"),
    )


if __name__ == "__main__":
    main()
