# 项目文件放置说明

本文档说明 `vision` 项目中每个文件夹、每个关键文件应该放什么内容，以及后续操作时应该怎么放。

## 1. 总体目录结构

当前项目推荐结构如下：

```text
vision/
  README.md
  dataset.yaml
  code/
  ui/
  docs/
  data/
    raw/
    selected/
    labels/
    halcon/
  datasets/
    keyboard/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
  models/
  outputs/
  scripts/
```

如果某些目录暂时是空的，这是正常的。比如 `models/` 只有训练完成后才会出现模型文件。

## 2. `vision/README.md`

### 放什么

项目总说明。

### 用来干什么

用于说明：

1. 项目目标；
2. 技术路线；
3. YOLO 和 HALCON 的选择；
4. 项目目录结构；
5. 基本采集和训练流程。

### 是否需要手动修改

一般不需要频繁修改。只有项目方案变化时再改。

## 3. `vision/dataset.yaml`

### 放什么

YOLO 数据集配置文件。

当前内容包括：

1. 数据集路径；
2. 训练集路径；
3. 验证集路径；
4. 测试集路径；
5. 类别编号和类别名称。

### 重要性

非常重要。YOLO 训练时会读取这个文件。

### 当前类别

当前已经包含：

1. 数字：`0-9`；
2. 字母：`A-Z`；
3. 功能键：`Esc`、`Tab`、`CapsLock`、`Shift`、`Ctrl`、`Alt`、`Enter`、`Backspace`、`Space`。

### 注意事项

1. 标注工具里的类别顺序必须和这里一致；
2. 不要随意调整类别顺序；
3. 如果调整了类别顺序，已经标注好的数据可能会类别错乱；
4. 类别名尽量使用英文或 ASCII 字符，不要使用中文和空格。

## 4. `vision/code/`

### 放什么

所有 Python 代码脚本。

### 当前文件说明

| 文件 | 应该放什么/用来做什么 |
| --- | --- |
| `common.py` | 公共路径配置和通用函数，不需要手动运行 |
| `requirements.txt` | Python 依赖列表 |
| `show_classes.py` | 查看类别编号和类别名称 |
| `check_dataset.py` | 检查图片和标签文件是否正确 |
| `split_dataset.py` | 划分训练集、验证集、测试集 |
| `prepare_dataset.py` | 一键执行数据检查和数据划分 |
| `train_yolo.py` | 训练 YOLO 模型 |
| `predict_yolo.py` | 使用训练好的模型预测图片 |
| `export_onnx.py` | 导出 ONNX 模型 |
| `clean_cache.py` | 清理 `__pycache__` 缓存目录 |
| `README.md` | 代码使用说明 |

### 不应该放什么

不要把图片、模型、标注文件放在 `code/` 里。

## 5. `vision/ui/`

### 放什么

Web 可视化界面代码。

### 当前文件说明

| 文件 | 用途 |
| --- | --- |
| `app.py` | Gradio Web UI 主程序 |
| `README.md` | UI 使用说明 |
| `start_ui.bat` | Windows 一键启动脚本，如果存在可双击或命令行运行 |

### 怎么运行

在项目根目录运行：

```bash
python vision/ui/app.py
```

或者如果当前在 `vision/` 目录下，运行：

```bash
python ui/app.py
```

浏览器访问：

```text
http://127.0.0.1:7860
```

### 注意事项

UI 默认读取模型：

```text
vision/models/best.pt
```

如果还没有训练出模型，UI 可以打开，但点击检测会提示模型不存在。

## 6. `vision/docs/`

### 放什么

项目文档、说明、规范。

### 当前文件说明

| 文件 | 内容 |
| --- | --- |
| `acquisition_guide.md` | 原始图片采集规范 |
| `annotation_guide.md` | 标注规范 |
| `workflow.md` | 项目完整实施流程 |
| `next_steps_guide.md` | 后续操作完整说明 |
| `file_placement_guide.md` | 本文件，说明每个文件夹放什么 |
| `classes.txt` | 可选文件，由 `show_classes.py --export` 生成 |

### 不应该放什么

不要把大量图片、模型文件放到 `docs/`。文档目录只放文字说明、表格、流程说明等。

## 7. `vision/data/`

### 放什么

项目原始数据、中间数据和标注文件。

它下面主要有几个子目录：

```text
vision/data/raw/
vision/data/selected/
vision/data/labels/
vision/data/halcon/
```

## 8. `vision/data/raw/`

### 放什么

放所有拍摄得到的原始图片。

例如：

```text
vision/data/raw/20260425_kb01_full_normal_0001.jpg
vision/data/raw/20260425_kb01_full_normal_0002.jpg
vision/data/raw/20260425_kb01_left_dark_0003.jpg
```

### 作用

这个目录是原始备份，尽量不要修改和覆盖。

### 注意事项

1. 拍到的原图先全部放这里；
2. 不要直接在这里删来删去；
3. 不要把标注文件放这里；
4. 不要把训练生成的结果图放这里。

## 9. `vision/data/selected/`

### 放什么

放筛选后准备用于训练和标注的图片。

例如：

```text
vision/data/selected/20260425_kb01_full_normal_0001.jpg
vision/data/selected/20260425_kb01_full_normal_0002.jpg
```

### 这些图片从哪里来

从 `vision/data/raw/` 复制过来。

### 筛选标准

放进 `selected/` 的图片应该满足：

1. 字符清晰可见；
2. 键盘区域完整或局部明确；
3. 图片方向正确；
4. 没有严重模糊；
5. 没有严重反光；
6. 文件名规范。

### 不建议放入的图片

1. 严重模糊；
2. 字符看不清；
3. 过曝严重；
4. 反光严重；
5. 键盘只占很小一部分；
6. 文件名混乱。

## 10. `vision/data/labels/`

### 放什么

放 YOLO 格式标注文件，后缀为 `.txt`。

例如：

```text
vision/data/labels/20260425_kb01_full_normal_0001.txt
vision/data/labels/20260425_kb01_full_normal_0002.txt
```

### 必须和图片同名

如果图片是：

```text
vision/data/selected/20260425_kb01_full_normal_0001.jpg
```

标签必须是：

```text
vision/data/labels/20260425_kb01_full_normal_0001.txt
```

只有后缀不同，前面的文件名必须完全一致。

### 标签内容格式

每一行代表一个目标：

```text
class_id x_center y_center width height
```

示例：

```text
10 0.5123 0.3812 0.0421 0.0578
11 0.5610 0.3820 0.0415 0.0582
```

### 注意事项

1. 一张图片对应一个 `.txt`；
2. 如果图片里有很多按键，`.txt` 里就会有很多行；
3. 类别编号必须和 `dataset.yaml` 一致；
4. 坐标必须是 0 到 1 之间；
5. 不要手动乱改标签文件，除非你确认格式正确。

## 11. `vision/data/halcon/`

### 放什么

如果后续使用 HALCON 采集图片或做预处理，可以把 HALCON 输出文件放这里。

例如：

```text
vision/data/halcon/halcon_capture_0001.jpg
vision/data/halcon/halcon_processed_0001.jpg
```

### 当前是否必须使用

不是必须。

如果项目不使用 HALCON，这个文件夹可以保持为空。

## 12. `vision/datasets/keyboard/`

### 放什么

这是 YOLO 训练真正读取的数据集目录。

它由脚本自动生成：

```bash
python vision/code/split_dataset.py
```

或者：

```bash
python vision/code/prepare_dataset.py
```

### 目录结构

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

### 各目录说明

| 目录 | 放什么 |
| --- | --- |
| `images/train/` | 训练集图片 |
| `images/val/` | 验证集图片 |
| `images/test/` | 测试集图片 |
| `labels/train/` | 训练集标签 |
| `labels/val/` | 验证集标签 |
| `labels/test/` | 测试集标签 |

### 是否手动放文件

一般不建议手动放。

正确流程是：

1. 图片放到 `data/selected/`；
2. 标签放到 `data/labels/`；
3. 运行 `split_dataset.py` 自动复制到 `datasets/keyboard/`。

## 13. `vision/models/`

### 放什么

放训练后的模型文件。

常见文件：

```text
vision/models/best.pt
vision/models/best.onnx
vision/models/yolov8n.pt
```

### 文件说明

| 文件 | 说明 |
| --- | --- |
| `best.pt` | 训练完成后效果最好的 YOLO 模型，是最重要的模型文件 |
| `best.onnx` | 从 `best.pt` 导出的 ONNX 模型 |
| `yolov8n.pt` | 可选，YOLO 官方预训练模型，如果手动下载可以放这里 |

### 注意事项

1. `best.pt` 是 UI 默认使用的模型；
2. 不要误删 `best.pt`；
3. 如果重新训练，`best.pt` 可能会被覆盖；
4. 可以备份不同版本，例如 `best_20260425.pt`。

## 14. `vision/outputs/`

### 放什么

放训练结果、预测结果、UI 输出结果。

可能包含：

```text
vision/outputs/train/
vision/outputs/predict/
vision/outputs/ui/
```

### 各目录说明

| 目录 | 内容 |
| --- | --- |
| `train/` | YOLO 训练过程输出，如曲线图、混淆矩阵、权重 |
| `predict/` | 使用 `predict_yolo.py` 生成的预测结果 |
| `ui/` | UI 最新检测结果图 |

### 注意事项

这个目录可以定期清理，但如果需要写报告或答辩，建议保留：

1. `results.png`；
2. `confusion_matrix.png`；
3. 检测结果图片；
4. 训练日志。

## 15. `vision/scripts/`

### 放什么

备用脚本目录。

当前项目主要脚本已经放在 `code/`，所以 `scripts/` 可以暂时为空。

如果后续有临时转换脚本、数据处理脚本，也可以放这里，但建议核心脚本仍放在 `code/`。

## 16. 图片命名规范

推荐命名：

```text
日期_键盘编号_拍摄区域_场景_序号.jpg
```

示例：

```text
20260425_kb01_full_normal_0001.jpg
20260425_kb01_letter_normal_0002.jpg
20260425_kb01_left_dark_0003.jpg
20260425_kb01_num_shadow_0004.jpg
```

### 字段说明

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| 日期 | `20260425` | 拍摄日期 |
| 键盘编号 | `kb01` | 第几个键盘 |
| 拍摄区域 | `full`、`letter`、`left`、`right`、`num`、`func` | 拍摄区域 |
| 场景 | `normal`、`dark`、`bright`、`reflect`、`shadow`、`blur` | 拍摄场景 |
| 序号 | `0001` | 图片编号 |

### 命名注意事项

1. 不要使用中文；
2. 不要使用空格；
3. 不要使用特殊符号；
4. 文件名尽量短但有意义；
5. 图片和标签必须同名。

## 17. 标注规则

本项目推荐：

> 框住整个按键，类别为按键上的主要字符或功能名。

不要有的图片框字符笔画，有的图片框整个键帽。标注规则必须统一。

例如：

| 按键 | 标注方式 | 类别 |
| --- | --- | --- |
| A 键 | 框整个 A 键键帽 | `A` |
| 1 键 | 框整个 1 键键帽 | `1` |
| Enter 键 | 框整个 Enter 键键帽 | `Enter` |
| 空格键 | 框整个空格键 | `Space` |

## 18. 完整操作流程

### 第一步：拍照

拍摄图片，全部放入：

```text
vision/data/raw/
```

### 第二步：筛选

挑选清晰图片，复制到：

```text
vision/data/selected/
```

### 第三步：查看类别

```bash
python vision/code/show_classes.py
```

### 第四步：标注

使用 LabelImg、Roboflow 或 CVAT 标注。

图片来自：

```text
vision/data/selected/
```

标签输出到：

```text
vision/data/labels/
```

### 第五步：检查并划分数据

```bash
python vision/code/prepare_dataset.py
```

### 第六步：训练

```bash
python vision/code/train_yolo.py --model yolov8n.pt --imgsz 640 --epochs 80 --batch 4
```

### 第七步：预测

```bash
python vision/code/predict_yolo.py --model vision/models/best.pt --source vision/data/raw
```

### 第八步：打开 UI

```bash
python vision/ui/app.py
```

## 19. 训练前最终检查

训练前确认：

1. `vision/data/selected/` 有图片；
2. `vision/data/labels/` 有标签；
3. 图片和标签同名；
4. 标签类别编号正确；
5. 已运行 `prepare_dataset.py`；
6. `vision/datasets/keyboard/images/train/` 不为空；
7. `vision/datasets/keyboard/labels/train/` 不为空。

## 20. 简单记忆版

你只需要记住：

```text
raw 放原图
selected 放训练图片
labels 放标注 txt
datasets 放脚本划分后的 YOLO 数据集
models 放训练好的模型
outputs 放训练和预测结果
code 放 Python 脚本
ui 放界面代码
docs 放说明文档
```
