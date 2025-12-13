# 道路病害检测系统 - 模块化训练框架

## 🛣️ 项目概述

这是一个基于 YOLOv11 的道路病害检测系统，专门为 RTX 5060 Ti 优化配置。项目采用模块化架构设计，将训练流程拆分为独立的功能模块，提高了代码的可维护性和扩展性。

## 🏗️ 架构设计

### 模块结构

```
road_detection/
├── core/                    # 核心功能模块
│   ├── config.py           # 训练配置管理
│   ├── device.py           # 设备管理
│   └── model.py            # 模型管理
├── training/               # 训练相关模块
│   ├── trainer.py          # 主训练器
│   └── config_factory.py   # 配置工厂
└── utils/                  # 工具模块
    ├── dataset.py          # 数据集分析
    ├── validation.py       # 模型验证
    ├── plotting.py         # 绘图工具
    └── logger.py           # 日志管理
```

### 核心特性

1. **模块化设计**: 功能分离，易于维护和扩展
2. **配置工厂**: 支持多种预设配置（标准、优化、平衡、快速）
3. **智能设备管理**: 自动检测和优化 GPU/CPU 配置
4. **数据集分析**: 自动分析类别分布和图像统计
5. **训练恢复**: 智能检测和恢复中断的训练
6. **性能验证**: 完整的模型验证和性能评估
7. **中文支持**: 完整的中文日志和报告输出

## 🚀 快速开始

### 环境要求

- Python 3.12
- PyTorch 2.6.0+ (支持 RTX 50 系列)
- CUDA 12.8

### 安装依赖

```bash
# 使用uv包管理器
uv sync

# 或者手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics>=8.3.234
```

### 基础训练

#### 1. 标准训练

```bash
# 使用默认配置进行训练
uv run python train.py --config-type standard

# 指定数据路径
uv run python train.py --config-type standard --data datasets/yolo_format/road.yaml
```

#### 2. 优化训练（推荐）

```bash
# 针对RDD2022数据集的优化配置
uv run python train.py --config-type optimized

# 自定义参数
uv run python train.py --config-type optimized --epochs 200 --model-size m
```

#### 3. 快速训练

```bash
# 用于快速实验和测试
uv run python train.py --config-type fast --epochs 50
```

#### 4. 恢复训练

```bash
# 自动检测恢复点
uv run python train.py --resume

# 指定恢复路径
uv run python train.py --resume --resume-path runs/detect/train/weights/last.pt
```

### 高级用法

#### 自定义配置

```bash
# 完全自定义训练参数
uv run python train.py --config-type custom \
    --epochs 150 \
    --model-size s \
    --optimizer AdamW \
    --lr0 0.002 \
    --mosaic 0.8 \
    --mixup 0.5
```

#### 数据集分析

```bash
# 启用详细的数据集分析
uv run python train.py --config-type optimized --analyze-dataset

# 生成训练报告
uv run python train.py --config-type optimized --generate-report --save-dir results/
```

## 📊 配置选项

### 预设配置类型

| 配置类型    | 适用场景       | 特点                  |
| ----------- | -------------- | --------------------- |
| `standard`  | 通用训练       | 平衡的速度和精度      |
| `optimized` | RDD2022 数据集 | 针对 4 类别不平衡优化 |
| `balanced`  | 中等数据集     | 速度和精度的最佳平衡  |
| `fast`      | 快速实验       | 最小训练时间          |

### 模型大小选择

| 模型 | 参数量 | 速度 | 精度 | 适用场景           |
| ---- | ------ | ---- | ---- | ------------------ |
| `n`  | 最小   | 最快 | 较低 | 实时检测、边缘设备 |
| `s`  | 小     | 快   | 中等 | 平衡性能和速度     |
| `m`  | 中等   | 中等 | 高   | 高精度要求         |
| `l`  | 大     | 慢   | 很高 | 最高精度要求       |
| `x`  | 最大   | 最慢 | 最高 | 研究和竞赛         |

## 🔧 核心功能

### 1. 数据集分析

```python
from road_detection.utils import DatasetAnalyzer

analyzer = DatasetAnalyzer('datasets/yolo_format/road.yaml')
class_distribution = analyzer.analyze_class_distribution()
class_weights = analyzer.calculate_class_weights()
report = analyzer.generate_analysis_report()
```

### 2. 训练配置管理

```python
from road_detection.training import create_training_config

# 创建优化配置
config = create_training_config('optimized', epochs=200, model_size='s')

# 自定义配置
config = create_training_config('custom', lr0=0.001, mosaic=0.8)
```

### 3. 训练器使用

```python
from road_detection.training import RoadDamageTrainer

trainer = RoadDamageTrainer(config)
results = trainer.run_full_pipeline(resume=False, export_format='onnx')
```

### 4. 模型验证

```python
from road_detection.utils import ModelValidator

validator = ModelValidator('path/to/model.pt', 'datasets/yolo_format/road.yaml')
results = validator.validate(save_dir='validation_results/')
validator.check_performance_target(target_map50=0.80)
```

## 📈 训练监控

### 实时日志

训练过程中会输出详细的中文日志，包括：

- 数据集统计信息
- 训练进度和损失变化
- 验证结果和性能指标
- 每个类别的检测性能

### 训练报告

训练完成后可生成详细的训练报告，包含：

- 数据集分析结果
- 训练配置参数
- 验证性能指标
- 改进建议和后续步骤

## 🎯 性能目标

默认的性能目标是mAP@0.5 ≥ 80%，可以通过以下参数调整：

```bash
uv run python train.py --target-map50 0.85  # 设置更高的目标
```

## 🔍 故障排除

### 常见问题

1. **GPU 内存不足**

   ```bash
   # 减小批次大小和图像尺寸
   uv run python train.py --batch-size 8 --img-size 512
   ```

2. **训练中断**

   ```bash
   # 自动恢复训练
   uv run python train.py --resume
   ```

3. **类别不平衡**

   ```bash
   # 使用优化配置，自动处理类别不平衡
   uv run python train.py --config-type optimized
   ```

### 调试模式

```bash
# 启用详细的数据集分析
uv run python train.py --analyze-dataset --generate-report
```

## 📁 项目结构

```
road-detection-project/
├── road_detection/          # 核心模块包
│   ├── core/               # 核心功能
│   ├── training/           # 训练相关
│   └── utils/              # 工具函数
├── train.py                # 主训练脚本（重构版）
├── train_RDD2022.py        # 原始训练脚本（保留）
├── train.py                # 原始训练脚本（保留）
├── convert.py              # 数据转换工具
├── detect.py               # 检测脚本
├── analyze_training_results.py  # 训练结果分析
├── model_optimization.py   # 模型优化工具
├── split_validation.py     # 验证集分割工具
├── configs/                # 配置文件
├── datasets/               # 数据集目录
│   ├── yolo_format/        # YOLO格式数据
│   └── RDD2022/            # 原始RDD2022数据
└── runs/                   # 训练结果输出
```

## 🤝 贡献指南

1. 遵循模块化设计原则
2. 添加完整的中文注释
3. 编写单元测试
4. 更新相关文档

## 🙏 致谢

- YOLOv11 团队提供的优秀检测框架
- RDD2022 数据集提供者
- 开源社区的支持

---

**注意**: 本项目专门针对 RTX 5060 Ti 和 CUDA 12.8 进行了优化配置，确保在新一代 GPU 上获得最佳性能。
