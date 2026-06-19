# 代码使用说明

本文件夹用于保存键盘字符检测项目的 Python 实现代码。

## 1. 安装依赖

在项目根目录运行：

```bash
pip install -r vision/code/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖中包含 YOLO、OpenCV、PyYAML 和 Gradio。Gradio 用于运行 `vision/ui/app.py` 可视化界面。

## 2. 脚本总览

| 文件 | 作用 | 常用阶段 |
| --- | --- | --- |
| `common.py` | 公共路径和工具函数 | 所有脚本内部使用 |
| `show_classes.py` | 查看或导出类别编号表 | 标注前 |
| `check_dataset.py` | 检查图片和标注是否匹配、格式是否正确 | 训练前 |
| `split_dataset.py` | 将图片和标签划分为训练集、验证集、测试集 | 训练前 |
| `prepare_dataset.py` | 一键执行数据检查和数据划分 | 训练前 |
| `train_yolo.py` | 训练 YOLO 模型 | 训练阶段 |
| `predict_yolo.py` | 使用模型预测图片 | 测试阶段 |
| `export_onnx.py` | 导出 ONNX 模型 | 扩展部署 |
| `clean_cache.py` | 清理 Python 缓存文件 | 维护阶段 |

## 3. 数据准备

原始图片放入：

```text
vision/data/raw/
```

筛选后用于训练的图片放入：

```text
vision/data/selected/
```

YOLO 标注文件放入：

```text
vision/data/labels/
```

图片和标签必须同名，例如：

```text
20260425_kb01_front_0001.jpg
20260425_kb01_front_0001.txt
```

## 4. 查看类别编号

标注前建议先查看类别编号：

```bash
python vision/code/show_classes.py
```

如果想导出类别列表：

```bash
python vision/code/show_classes.py --export
```

导出文件：

```text
vision/docs/classes.txt
```

## 5. 检查数据

标注完成后，训练前先运行：

```bash
python vision/code/check_dataset.py
```

它会检查：

1. `selected` 中是否有图片；
2. 每张图片是否有同名 `.txt` 标签；
3. 标签格式是否为 YOLO 格式；
4. 类别编号是否超出范围；
5. 坐标是否在 0 到 1 之间；
6. 每个类别出现了多少目标。

## 6. 划分数据集

单独划分数据集：

```bash
python vision/code/split_dataset.py
```

执行后会生成：

```text
vision/datasets/keyboard/images/train
vision/datasets/keyboard/images/val
vision/datasets/keyboard/images/test
vision/datasets/keyboard/labels/train
vision/datasets/keyboard/labels/val
vision/datasets/keyboard/labels/test
```

## 7. 一键准备数据

推荐使用这个命令，它会先检查数据，再划分数据集：

```bash
python vision/code/prepare_dataset.py
```

如果你只是临时测试，想跳过检查：

```bash
python vision/code/prepare_dataset.py --skip-check
```

## 8. 训练模型

推荐先使用轻量模型：

```bash
python vision/code/train_yolo.py --model yolov8n.pt --imgsz 640 --epochs 80 --batch 4
```

如果 GPU 可用：

```bash
python vision/code/train_yolo.py --model yolov8n.pt --imgsz 640 --epochs 80 --batch 4 --device 0
```

训练完成后，最佳模型会复制到：

```text
vision/models/best.pt
```

## 9. 预测图片

```bash
python vision/code/predict_yolo.py --model vision/models/best.pt --source vision/data/raw
```

结果保存到：

```text
vision/outputs/predict/
```

## 10. 启动 UI

训练得到 `vision/models/best.pt` 后，可以启动 Web 界面：

```bash
python vision/ui/app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

## 11. 导出 ONNX

如果后续要尝试其他部署方式，可以导出 ONNX：

```bash
python vision/code/export_onnx.py --model vision/models/best.pt --imgsz 640
```

导出文件：

```text
vision/models/best.onnx
```

## 12. 清理缓存

如果项目里出现 `__pycache__`，可以运行：

```bash
python vision/code/clean_cache.py
```

## 13. 推荐运行顺序

1. 采集直视键盘图片；
2. 筛选图片到 `vision/data/selected/`；
3. 运行 `python vision/code/show_classes.py` 查看类别编号；
4. 标注图片，标签放到 `vision/data/labels/`；
5. 运行 `python vision/code/prepare_dataset.py`；
6. 运行 `python vision/code/train_yolo.py --model yolov8n.pt --imgsz 640 --epochs 80 --batch 4`；
7. 运行 `python vision/code/predict_yolo.py --model vision/models/best.pt --source vision/data/raw`；
8. 运行 `python vision/ui/app.py` 打开界面演示。
