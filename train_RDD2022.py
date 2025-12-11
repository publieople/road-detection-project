#!/usr/bin/env python3
"""
道路病害检测模型训练脚本
专门针对4类别不平衡问题和高准确率需求
"""

from matplotlib.pylab import f
import torch
from ultralytics import YOLO # pyright: ignore[reportPrivateImportUsage]
from pathlib import Path
import yaml
import os
import numpy as np
from collections import Counter

def analyze_dataset_labels(data_yaml_path: str) -> dict:
    """
    分析数据集中的标签分布，计算类别权重
    """
    print(f"📊 分析数据集标签分布: {data_yaml_path}")

    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    base_path = Path(data_yaml_path).parent
    train_path = base_path / data_config['train']
    label_path = base_path / 'labels' / 'train'

    # 统计每个类别的实例数量
    class_counts = Counter()
    total_instances = 0

    if label_path.exists():
        for txt_file in label_path.glob("*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
                        total_instances += 1

    # 计算类别权重（用于平衡损失函数）
    num_classes = data_config.get('nc', 4)
    class_weights = {}

    print("类别分布统计:")
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        class_name = data_config['names'][class_id]
        print(f"  {class_name} (类别{class_id}): {count} 个实例 ({percentage:.1f}%)")

        # 计算权重（实例数越少，权重越高）
        if count > 0:
            class_weights[class_id] = total_instances / (num_classes * count)
        else:
            class_weights[class_id] = 1.0

    print(f"\n类别权重 (用于损失函数平衡):")
    for class_id, weight in class_weights.items():
        class_name = data_config['names'][class_id]
        print(f"  {class_name}: {weight:.3f}")

    return {
        'class_counts': dict(class_counts),
        'class_weights': class_weights,
        'total_instances': total_instances
    }

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

def train_optimized_model(data_yaml_path: str, model_size: str = 's', epochs: int = 150,
                         img_size: int = 640, resume: bool = False):
    """
    训练YOLO模型，专门针对道路病害检测
    """
    device = setup_training()

    # 分析数据集标签分布
    dataset_stats = analyze_dataset_labels(data_yaml_path)
    class_weights = dataset_stats['class_weights']

    if resume:
        print("🔄 检测是否存在中断的训练，尝试恢复...")
        possible_weights = [
            'runs/detect/train/weights/last.pt',
            'runs/detect/train2/weights/last.pt',
            'runs/detect/train3/weights/last.pt',
        ]

        resume_path = None
        for weight_path in possible_weights:
            if Path(weight_path).exists():
                resume_path = weight_path
                break

        if resume_path:
            print(f"✅ 找到中断的训练权重: {resume_path}")
            model = YOLO(resume_path)
            print("🚀 从上次中断处继续训练...")
        else:
            print("⚠️  未找到可恢复的权重文件，开始新的训练...")
            model_name = f'yolo11{model_size}.pt'
            print(f"📦 加载预训练模型: {model_name}")
            model = YOLO(model_name)
    else:
        model_name = f'yolo11{model_size}.pt'
        print(f"📦 加载预训练模型: {model_name}")
        model = YOLO(model_name)

    # 训练配置
    training_config = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': 16,  # 批次大小
        'workers': 4,  # 数据加载线程数
        'cache': False, # Windows 下需禁用
        'device': device,

        # 优化器配置
        'optimizer': 'AdamW',  # 使用AdamW优化器
        'lr0': 0.001,  # 初始学习率
        'lrf': 0.01,  # 最终学习率倍数
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 学习率调度
        'warmup_epochs': 10,  # 增加预热轮数
        'warmup_momentum': 0.8,
        'cos_lr': True,  # 使用余弦退火学习率

        # 损失函数配置（针对类别不平衡）
        'box': 7.5,  # box loss增益
        'cls': 1.5,  # 增加cls loss增益（原0.5）
        'dfl': 1.5,  # dfl loss增益

        # 数据增强（针对道路病害优化）
        'hsv_h': 0.015,  # HSV色调增强
        'hsv_s': 0.7,  # HSV饱和度增强
        'hsv_v': 0.4,  # HSV明度增强

        # 几何增强
        'degrees': 15.0,  # 旋转增强
        'translate': 0.3,  # 平移增强
        'scale': 0.7,  # 缩放增强
        'shear': 5.0,  # 剪切增强
        'perspective': 0.001,  # 透视增强
        'fliplr': 0.8,  # 左右翻转
        'flipud': 0.2,  # 上下翻转

        # 高级增强
        'mosaic': 0.8,  # Mosaic增强
        'mixup': 0.3,  # MixUp增强
        'copy_paste': 0.2,  # 复制粘贴增强
        'auto_augment': 'rand-m9-mstd0.5-inc1',  # 自动增强
        'erasing': 0.4,  # 随机擦除

        # 训练策略
        'close_mosaic': 20,  # 后期关闭Mosaic
        'patience': 30,  # 早停耐心值
        'single_cls': False,  # 多类别检测

        # 性能优化
        'amp': True,  # 混合精度训练
        'compile': False,  # 模型编译（可选）

        # 验证和评估
        'val': True,
        'split': 'val',
        'save': True,
        'save_period': 10,  # 每10轮保存一次
        'plots': True,  # 生成图表
    }

    print("🚀 开始训练...")
    print(f"📊 训练轮数: {epochs}")
    print(f"📐 图像尺寸: {img_size}")
    print(f"🔧 设备: {device}")
    print(f"📦 模型大小: {model_size}")

    # 开始训练
    if resume and resume_path:
        training_config['resume'] = True
        results = model.train(**training_config)
    else:
        results = model.train(**training_config)

    print("✅ 训练完成!")
    return model, results, dataset_stats

def validate_model(model, data_yaml_path: str):
    """验证模型性能"""
    print("🔍 验证模型性能...")

    # 在验证集上评估
    metrics = model.val(data=data_yaml_path)

    print("📊 验证结果:")
    print(f"   mAP@0.5: {metrics.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"   平均精确率: {metrics.box.mp:.3f}")
    print(f"   平均召回率: {metrics.box.mr:.3f}")

    # 打印每个类别的性能
    if hasattr(metrics.box, 'ap50'):
        print("\n各类别AP@0.5:")
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config['names']

        for i, ap50 in enumerate(metrics.box.ap50):
            if i < len(class_names):
                print(f"   {class_names[i]}: {ap50:.3f}")

    return metrics

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='道路病害检测模型训练')
    parser.add_argument('--resume', action='store_true', help='从上次中断处恢复训练')
    parser.add_argument('--data', type=str, default='datasets/yolo_format/road.yaml', help='数据配置文件路径')
    parser.add_argument('--model-size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')

    args = parser.parse_args()

    # 数据配置文件路径
    data_yaml = args.data

    # 检查数据文件是否存在
    if not Path(data_yaml).exists():
        print(f"❌ 数据配置文件不存在: {data_yaml}")
        return

    print("🛣️  道路病害检测模型训练")
    print("=" * 60)

    if args.resume:
        print("🔄 已启用训练恢复模式")

    print(f"📊 配置: 模型={args.model_size}, 轮数={args.epochs}, 尺寸={args.img_size}")

    try:
        # 训练模型
        model, training_results, dataset_stats = train_optimized_model(
            data_yaml_path=data_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            img_size=args.img_size,
            resume=args.resume
        )

        # 验证模型
        metrics = validate_model(model, data_yaml)

        print("\n🎉 训练完成!")
        print("📋 训练总结:")
        print(f"   - 训练图片: {dataset_stats['total_instances']} 个实例")
        print(f"   - 病害类别: 4类")
        print(f"   - 类别分布: {dataset_stats['class_counts']}")
        print(f"   - 最佳mAP@0.5: {metrics.box.map50:.3f}")

        # 检查是否达到目标准确率
        if metrics.box.map50 >= 0.80:
            print("🎯 目标达成！模型准确率 ≥ 80%")
        else:
            print(f"⚠️  未达目标。当前准确率: {metrics.box.map50:.1%}, 目标: 80%")
            print("💡 建议: 增加训练轮数、调整超参数或收集更多数据")

        # 导出模型
        print("\n💾 导出训练好的模型...")
        model.export(format='onnx', simplify=True)

        # 保存训练报告
        try:
            save_dir = getattr(training_results, 'save_dir', None)
            if save_dir:
                report_path = Path(save_dir) / "training_report.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("道路病害检测模型训练报告\n")
                    f.write("=" * 50 + "\n")
                    training_time = getattr(training_results, 'time', '未知')
                    f.write(f"训练时间: {training_time}\n")
                    f.write(f"最佳mAP@0.5: {metrics.box.map50:.3f}\n")
                    f.write(f"最终mAP@0.5:0.95: {metrics.box.map:.3f}\n")
                    f.write(f"类别分布: {dataset_stats['class_counts']}\n")
                    f.write(f"类别权重: {dataset_stats['class_weights']}\n")
                    f.write("=" * 50 + "\n")
                    f.write("模型配置和超参数详见 args.yaml 文件\n")

                print(f"📄 训练报告已保存: {report_path}")
            else:
                print("⚠️  无法保存训练报告，缺少保存路径信息")
        except Exception as e:
            print(f"⚠️  保存训练报告时出错: {e}")

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()