# 项目实施流程

## 阶段 1：明确识别范围

先确定本项目要识别哪些按键。

建议最小版本：

1. 字母：`A-Z`；
2. 数字：`0-9`；
3. 功能键：`Esc`、`Tab`、`Shift`、`Ctrl`、`Alt`、`Enter`、`Backspace`、`Space`。

如果时间不够，可以先只做 `A-Z` 和 `0-9`。

## 阶段 2：采集原始图片

按照 `docs/acquisition_guide.md` 采集图片。

保存位置：

```text
vision/data/raw/
```

第一批建议 80 到 120 张左右。

本项目采集方式以直视拍照为主，拍摄距离基本固定，不需要大量斜视角度图片。主要变化因素放在光照、键盘区域和键盘型号上。

## 阶段 3：筛选训练图片

从 `data/raw/` 中选择清晰、有代表性的图片，复制到：

```text
vision/data/selected/
```

筛选原则：

1. 删除严重模糊图片；
2. 删除字符完全看不清的图片；
3. 保留不同光照和不同键盘区域；
4. 拍摄角度和距离尽量保持一致；
5. 不要只保留最好看的图片，也要保留少量轻微反光、轻微阴影样本。

## 阶段 4：标注数据

推荐使用 LabelImg、Roboflow 或 CVAT。

标注方式：

> 框住整个按键，类别填写该按键的主要字符或功能名。

标注结果保存为 YOLO 格式。

原始图片建议放在：

```text
vision/data/selected/
```

标注文件建议放在：

```text
vision/data/labels/
```

图片和标注文件要同名，例如：

```text
20260425_kb01_front_0001.jpg
20260425_kb01_front_0001.txt
```

## 阶段 5：整理 YOLO 数据集

推荐整理成以下结构：

```text
vision/datasets/keyboard/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

可以使用 `vision/code/split_dataset.py` 自动从 `data/selected/` 和 `data/labels/` 划分数据集。

## 阶段 6：训练 YOLO 模型

推荐使用 YOLOv8 或更新版本的 YOLO。

推荐模型：

1. `yolov8n`：速度快，适合快速验证；
2. `yolov8s`：精度更好，训练时间稍长；
3. 直视拍摄场景下，优先从 `yolov8n` 开始。

初始训练建议：

```bash
python vision/code/train_yolo.py --model yolov8n.pt --imgsz 960 --epochs 100 --batch 8
```

如果显卡较弱，可以使用：

```bash
python vision/code/train_yolo.py --model yolov8n.pt --imgsz 640 --epochs 80 --batch 4
```

## 阶段 7：评估模型

重点查看：

1. precision：检测结果中有多少是正确的；
2. recall：实际目标中有多少被检测出来；
3. mAP50：整体检测效果；
4. 混淆类别，例如 `O` 和 `0`、`I` 和 `1`。

如果效果不好，优先补充直视图片中的光照变化和容易混淆类别，而不是盲目调参数。

## 阶段 8：预测和可视化

训练完成后，用测试图片预测：

```bash
python vision/code/predict_yolo.py --model vision/models/best.pt --source vision/data/raw
```

检测结果保存到：

```text
vision/outputs/predict/
```

## 阶段 9：与 HALCON 结合

HALCON 可承担以下角色：

1. 相机直视采集键盘图片；
2. 图像预处理，例如裁剪键盘区域、灰度化、增强对比度；
3. 展示检测结果；
4. 如果版本支持，可以加载 ONNX 模型进行推理。

推荐结合方式：

1. YOLO 在 Python 环境中训练；
2. 训练完成后导出 ONNX：

```bash
python vision/code/export_onnx.py --model vision/models/best.pt --imgsz 960
```

3. 在 HALCON 中读取图片并调用深度学习/ONNX 模型；
4. 在 HALCON 窗口中绘制检测框和类别。

如果 HALCON 版本不方便直接调用 YOLO，也可以：

1. HALCON 负责采图并保存到 `data/halcon/`；
2. Python YOLO 负责识别；
3. 把识别结果保存为 `.txt` 或 `.json`；
4. HALCON 或 Python 负责显示结果。

## 阶段 10：答辩材料准备

建议准备以下内容：

1. 项目背景：键盘字符检测的应用场景；
2. 技术选型：为什么选择 YOLO + HALCON；
3. 数据采集：展示直视拍摄环境和不同光照样本；
4. 数据标注：展示标注截图；
5. 模型训练：展示训练参数和曲线；
6. 检测效果：展示成功案例和失败案例；
7. 总结改进：增加数据、改进光照、提高分辨率、增加 OCR 后处理。
