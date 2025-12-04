# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

道路病害检测系统 (Road Damage Detection System) - RTX 5060 Ti 优化配置的 YOLOv11 实现

## Non-Obvious Project-Specific Information

### Critical Training Patterns

- **训练恢复机制**: [`train.py`](train.py:172-198) 实现了智能训练恢复，会检查多个可能的路径 (`runs/detect/train/weights/last.pt`, `train2`, `train3`) 来找到可恢复的权重文件
- **数据集统计验证**: [`get_dataset_stats()`](train.py:13-142) 函数会验证图片-标签匹配率，低于 100%时会警告并列出缺失标签的具体文件
- **增强配置特殊性**: 训练配置中使用了特定的增强参数组合，如 `mosaic: 0.8`, `mixup: 0.5`, `copy_paste: 0.3`，这些是针对道路病害检测优化的非标准值

### Data Conversion Gotchas

- **类别映射双重标准**: [`convert.py`](convert.py:69-72) 根据文件路径是否包含"RDD2020"自动选择不同的类别映射，这种隐式逻辑容易出错
- **XML 解析容错处理**: [`parse_xml()`](convert.py:35-113) 对 XML 文件的各种缺失情况都做了容错处理，但警告信息是中文，需要注意编码问题
- **路径硬编码假设**: 转换脚本假设标签路径可以通过简单替换 `images` -> `labels` 获得，这种假设在 [`convert.py`](convert.py:59) 和 [`train_test.py`](train_test.py:67) 中重复使用

### Dataset Management Patterns

- **验证集划分种子**: [`split_val.py`](split_val.py:23) 使用固定随机种子 42 确保可复现性，这个种子值在项目中没有文档说明
- **最小数据集保护**: [`train_test.py`](train_test.py:57) 创建子集时强制要求至少 10 张训练图片和 5 张验证图片，防止测试失败
- **YOLO 格式路径约定**: 所有脚本都假设 YOLO 格式数据在 `datasets/yolo_format/` 目录下，但这个路径在多个文件中硬编码

### CUDA/硬件特定配置

- **RTX 50 系列支持**: [`pyproject.toml`](pyproject.toml:34-41) 配置了特殊的 CUDA 12.8 索引源和 PyTorch 版本要求，专门为 RTX 5060 Ti 优化
- **清华镜像源**: 默认使用 `https://pypi.tuna.tsinghua.edu.cn/simple` 作为包索引源，这会影响依赖安装速度
- **混合精度训练**: 所有训练脚本都默认启用 `amp: True`，这对 RTX 50 系列的性能至关重要

### 模型导出和验证

- **ONNX 导出**: [`train.py`](train.py:353) 训练完成后自动导出 ONNX 格式，但导出路径依赖于训练结果的 `save_dir` 属性
- **验证指标缓存**: 模型验证结果会缓存在 `runs/` 目录下，但清理机制不明确，可能导致磁盘空间问题

### 错误处理和日志

- **中文日志输出**: 所有用户可见的日志信息都使用中文和 emoji，但错误追踪栈仍然是英文
- **静默失败风险**: XML 解析失败时只是打印警告并继续，不会中断整个转换流程
- **GPU 检测详细输出**: [`setup_training()`](train.py:144-156) 会输出详细的 GPU 信息，包括 CUDA 版本和设备名称
