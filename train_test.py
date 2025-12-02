#!/usr/bin/env python3
"""
道路病害检测模型快速训练测试脚本
在小规模数据集上基于YOLO11n进行快速训练测试
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import os
import argparse
import random
import shutil
from sklearn.model_selection import train_test_split


def create_subset_dataset(original_data_yaml: str, subset_ratio: float = 0.1, output_dir: str = "datasets/test_subset"):
    """
    创建原始数据集的小规模子集用于快速训练测试

    Args:
        original_data_yaml: 原始数据集YAML配置文件路径
        subset_ratio: 子集占原始数据的比例
        output_dir: 输出目录

    Returns:
        str: 新的子集数据集YAML配置文件路径
    """
    print(f"📦 创建小规模数据集子集 (比例: {subset_ratio})...")

    # 读取原始数据配置
    with open(original_data_yaml, 'r', encoding='utf-8') as f:
        original_config = yaml.safe_load(f)

    # 创建输出目录结构
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_train_dir = images_dir / "train"
    images_val_dir = images_dir / "val"
    labels_train_dir = labels_dir / "train"
    labels_val_dir = labels_dir / "val"

    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 处理训练集
    original_train_path = Path(original_config['path']) / original_config['train']
    train_images = list(original_train_path.rglob("*.jpg")) + \
                   list(original_train_path.rglob("*.png")) + \
                   list(original_train_path.rglob("*.jpeg"))

    # 根据比例选取子集
    subset_train_size = max(10, int(len(train_images) * subset_ratio))  # 至少10张图片
    subset_train_images = random.sample(train_images, min(subset_train_size, len(train_images)))

    # 复制训练图片和标签
    for img_path in subset_train_images:
        # 复制图片
        dst_img_path = images_train_dir / img_path.name
        shutil.copy(img_path, dst_img_path)

        # 查找并复制对应的标签文件
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if label_path.exists():
            dst_label_path = labels_train_dir / label_path.name
            shutil.copy(label_path, dst_label_path)

    print(f"✅ 训练集: 复制了 {len(subset_train_images)} 张图片")

    # 处理验证集
    original_val_path = Path(original_config['path']) / original_config['val']
    val_images = list(original_val_path.rglob("*.jpg")) + \
                 list(original_val_path.rglob("*.png")) + \
                 list(original_val_path.rglob("*.jpeg"))

    # 根据比例选取子集
    subset_val_size = max(5, int(len(val_images) * subset_ratio))  # 至少5张图片
    subset_val_images = random.sample(val_images, min(subset_val_size, len(val_images)))

    # 复制验证图片和标签
    for img_path in subset_val_images:
        # 复制图片
        dst_img_path = images_val_dir / img_path.name
        shutil.copy(img_path, dst_img_path)

        # 查找并复制对应的标签文件
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if label_path.exists():
            dst_label_path = labels_val_dir / label_path.name
            shutil.copy(label_path, dst_label_path)

    print(f"✅ 验证集: 复制了 {len(subset_val_images)} 张图片")

    # 创建新的YAML配置文件
    subset_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': original_config['nc'],
        'names': original_config['names']
    }

    subset_yaml_path = output_path / "test_dataset.yaml"
    with open(subset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(subset_config, f, allow_unicode=True)

    print(f"✅ 子集数据集创建完成: {subset_yaml_path}")
    return str(subset_yaml_path)


def setup_training():
    """配置训练环境和参数"""
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"🔥 CUDA版本: {torch.version.cuda}")
        device = 'cuda'
    else:
        print("⚠️  GPU不可用，使用CPU训练")
        device = 'cpu'

    return device


def train_model_fast(data_yaml_path: str, epochs: int = 10, img_size: int = 640):
    """
    快速训练YOLO模型用于测试

    Args:
        data_yaml_path: 数据配置文件路径
        epochs: 训练轮数（默认10轮）
        img_size: 输入图像尺寸
    """
    device = setup_training()

    # 加载预训练模型
    model_name = "yolo11n.pt"
    print(f"📦 加载预训练模型: {model_name}")
    model = YOLO(model_name)

    # 快速训练配置
    training_config = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': 4,              # 较小的批次大小
        'device': device,
        'optimizer': 'SGD',      # 使用简单优化器
        'lr0': 0.01,             # 较高的初始学习率
        'lrf': 0.01,             # 最终学习率倍数
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 1,      # 减少预热轮数
        'warmup_momentum': 0.8,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.3,           # 减少mosaic概率
        'mixup': 0.0,
        'amp': True,             # 混合精度加速
        'cache': False,
        'project': 'runs/test_train',  # 分离的项目目录
        'name': 'exp'            # 实验名称
    }

    print("🚀 开始快速训练模型...")
    print(f"📊 训练轮数: {epochs}")
    print(f"📐 图像尺寸: {img_size}")
    print(f"🔧 设备: {device}")

    # 开始训练
    results = model.train(**training_config)

    print("✅ 快速训练完成!")
    if results and hasattr(results, 'save_dir'):
        print(f"📁 模型保存在: {Path(results.save_dir).resolve()}")
    else:
        print("📁 模型训练完成")

    return model, results


def validate_model(model, data_yaml_path: str):
    """验证模型性能"""
    print("🔍 验证模型性能...")

    # 在验证集上评估
    metrics = model.val(data=data_yaml_path)

    print("📊 验证结果:")
    print(f"   mAP@0.5: {metrics.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"   精确率: {metrics.box.p:.3f}")
    print(f"   召回率: {metrics.box.r:.3f}")

    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='道路病害检测模型快速训练测试')
    parser.add_argument('--data', type=str, default='datasets/yolo_format/road.yaml',
                       help='原始数据配置文件路径')
    parser.add_argument('--subset-ratio', type=float, default=0.05,
                       help='子集占原始数据的比例 (默认: 0.05)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数 (默认: 10)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸 (默认: 640)')

    args = parser.parse_args()

    print("🛣️  道路病害检测模型快速训练测试")
    print("=" * 50)
    print(f"📊 配置: 轮数={args.epochs}, 尺寸={args.img_size}, 子集比例={args.subset_ratio}")

    try:
        # 创建小规模数据集
        subset_yaml_path = create_subset_dataset(
            original_data_yaml=args.data,
            subset_ratio=args.subset_ratio
        )

        # 快速训练模型
        model, training_results = train_model_fast(
            data_yaml_path=subset_yaml_path,
            epochs=args.epochs,
            img_size=args.img_size
        )

        # 验证模型
        metrics = validate_model(model, subset_yaml_path)

        print("\n🎉 快速训练测试完成!")
        print(f"📁 测试数据集: {subset_yaml_path}")
        print(f"📊 最佳mAP@0.5: {metrics.box.map50:.3f}")

    except Exception as e:
        print(f"❌ 训练测试过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()