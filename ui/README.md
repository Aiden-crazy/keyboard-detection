# 键盘字符检测 UI 使用说明

本文件夹提供一个不依赖 HALCON 的可视化界面，用于课程展示和调试。

UI 基于 Gradio 实现，运行后会在浏览器中打开网页。你可以上传键盘图片，系统会调用训练好的 YOLO 模型进行检测，并显示检测框、类别、置信度和坐标。

## 1. 什么时候使用 UI

UI 主要用于模型训练完成之后的演示阶段。

也就是说，通常要先得到模型：

```text
vision/models/best.pt
```

然后再运行 UI。

如果还没有训练模型，UI 可以启动，但检测时会提示找不到模型文件。

## 2. 安装依赖

在已经激活的 Anaconda 环境中运行：

```bash
pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果你已经运行过项目依赖安装，也仍然需要额外安装 Gradio。

## 3. 启动 UI

在项目根目录运行：

```bash
python vision/ui/app.py
```

启动后终端会显示一个地址，通常是：

```text
http://127.0.0.1:7860
```

用浏览器打开这个地址即可。

## 4. 指定模型路径

默认模型路径是：

```text
vision/models/best.pt
```

如果你的模型放在其他位置，可以启动时指定：

```bash
python vision/ui/app.py --model vision/models/best.pt
```

也可以在网页界面的“模型路径”输入框中修改。

## 5. UI 中的参数

### 5.1 置信度阈值

默认值：`0.25`

含义：检测结果的最低可信度。

- 检测框太少：可以调低，例如 `0.15`；
- 误检太多：可以调高，例如 `0.4`。

### 5.2 推理图片尺寸

默认值：`640`

含义：YOLO 推理时的图像尺寸。

- 速度优先：使用 `640`；
- 字符较小：可以尝试 `960`；
- 显存或速度不够：降低到 `640` 或更低。

## 6. 检测结果保存位置

每次检测后，最新结果图会保存到：

```text
vision/outputs/ui/latest_result.jpg
```

网页中也会直接显示检测后的图片。

## 7. 推荐演示流程

答辩展示时可以按下面流程：

1. 打开项目目录，说明数据采集和训练流程；
2. 展示几张原始键盘图片；
3. 展示 YOLO 训练结果和 `best.pt`；
4. 启动 UI：

```bash
python vision/ui/app.py
```

5. 浏览器上传一张键盘图片；
6. 点击“开始检测”；
7. 展示检测框、类别和置信度表格。

## 8. 常见问题

### 8.1 找不到模型文件

如果出现模型不存在，请先确认：

```text
vision/models/best.pt
```

是否存在。

如果还没有这个文件，需要先训练：

```bash
python vision/code/train_yolo.py --model yolov8n.pt --imgsz 640 --epochs 80 --batch 4
```

### 8.2 浏览器打不开

确认终端是否显示：

```text
Running on local URL: http://127.0.0.1:7860
```

如果 7860 端口被占用，可以换端口：

```bash
python vision/ui/app.py --port 7861
```

### 8.3 检测结果不准确

优先检查：

1. 模型是否已经训练完成；
2. 标注是否准确；
3. 图片是否清晰；
4. 类别顺序是否和 `dataset.yaml` 一致；
5. 置信度阈值是否过高或过低。
