#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11-ConvNeXt 示例训练脚本
演示如何使用ConvNeXt-Tiny Backbone训练道路病害检测模型
"""

import sys
import argparse
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from road_detection.models.yolo11_convnext_trainer import YOLO11ConvNeXtTrainer
from road_detection.training.config_factory import create_training_config
from road_detection.utils import setup_chinese_fonts
import torch


def main():
    """主函数"""

    # 设置中文字体
    setup_chinese_fonts()

    # 参数解析
    parser = argparse.ArgumentParser(
        description='YOLO11-ConvNeXt 道路病害检测模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 基础训练 (ConvNeXt-Tiny, yolo11n)
  python train_convnext.py --data datasets/yolo_format/road.yaml

  # 使用更大的模型
  python train_convnext.py --model-size m --backbone-size small

  # 自定义参数
  python train_convnext.py \\
    --data datasets/yolo_format/road.yaml \\
    --epochs 200 \\
    --batch 16 \\
    --imgsz 1280 \\
    --optimizer adamw \\
    --lr0 0.001

  # 恢复训练
  python train_convnext.py --resume

  # 仅验证
  python train_convnext.py --val-only --weights runs/detect/train/weights/best.pt
        """
    )

    # 模型参数
    parser.add_argument(
        '--model-size',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO模型大小 (默认: n)'
    )

    parser.add_argument(
        '--backbone-type',
        type=str,
        default='tiny',
        choices=['tiny', 'small'],
        help='ConvNeXt Backbone类型 (默认: tiny)'
    )

    # 数据参数
    parser.add_argument(
        '--data',
        type=str,
        default='datasets/yolo_format/road.yaml',
        help='数据配置文件路径'
    )

    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮数 (默认: 100)'
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=8,
        help='批次大小 (默认: 8)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=1280,
        help='输入图像大小 (默认: 1280)'
    )

    # 优化参数
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=['adamw', 'sgd'],
        help='优化器 (默认: adamw)'
    )

    parser.add_argument(
        '--lr0',
        type=float,
        default=0.001,
        help='初始学习率 (默认: 0.001)'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.05,
        help='权重衰减 (默认: 0.05)'
    )

    # 增强参数
    parser.add_argument(
        '--mosaic',
        type=float,
        default=0.8,
        help='Mosaic增强概率 (默认: 0.8)'
    )

    parser.add_argument(
        '--mixup',
        type=float,
        default=0.5,
        help='Mixup增强概率 (默认: 0.5)'
    )

    # 设备参数
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA设备ID或cpu (默认: 0)'
    )

    # 其他参数
    parser.add_argument(
        '--resume',
        action='store_true',
        help='恢复上次训练'
    )

    parser.add_argument(
        '--val-only',
        action='store_true',
        help='仅执行验证'
    )

    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='加载权重文件'
    )

    parser.add_argument(
        '--export',
        type=str,
        default=None,
        choices=['onnx', 'tflite', 'pb', 'torchscript'],
        help='导出模型格式'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )

    args = parser.parse_args()

    # 显示配置信息
    print("\n" + "="*70)
    print("🚀 YOLO11-ConvNeXt 道路病害检测模型训练")
    print("="*70)

    print("\n📋 训练配置:")
    print(f"  模型大小: yolo11{args.model_size}")
    print(f"  Backbone: ConvNeXt-{args.backbone_type.capitalize()}")
    print(f"  数据文件: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  图像大小: {args.imgsz}")
    print(f"  优化器: {args.optimizer}")
    print(f"  初始学习率: {args.lr0}")
    print(f"  Mosaic: {args.mosaic}")
    print(f"  Mixup: {args.mixup}")
    print(f"  设备: {args.device}")

    # 检查GPU
    if args.device != 'cpu':
        if torch.cuda.is_available():
            device_id = int(args.device) if args.device.isdigit() else 0
            print(f"\n✅ GPU可用: {torch.cuda.get_device_name(device_id)}")
            print(f"   显存: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
        else:
            print("\n⚠️  GPU不可用，将使用CPU")
            args.device = 'cpu'
    else:
        print(f"\n⚠️  使用CPU进行训练，速度会很慢")

    # 创建训练器
    print(f"\n📦 初始化训练器...")
    trainer = YOLO11ConvNeXtTrainer(
        model_size=args.model_size,
        backbone_type=args.backbone_type,
        device=f'cuda:{args.device}' if args.device != 'cpu' else 'cpu'
    )

    # 仅验证模式
    if args.val_only:
        print("\n🔍 执行验证...")
        metrics = trainer.validate(data=args.data)
        print(f"\n✅ 验证完成")
        print(f"  精度: {metrics.box.map:.4f}")
        return

    # 仅导出模式
    if args.export:
        print(f"\n💾 导出模型为 {args.export}...")
        export_path = trainer.export_model(format=args.export)
        print(f"✅ 模型已导出: {export_path}")
        return

    # 训练模式
    print(f"\n🚀 开始训练...")

    try:
        results = trainer.train(
            data=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=f'cuda:{args.device}' if args.device != 'cpu' else 'cpu',
            resume=args.resume,
            optimizer=args.optimizer,
            lr0=args.lr0,
            weight_decay=args.weight_decay,
            mosaic=args.mosaic,
            mixup=args.mixup,
            verbose=args.verbose,
            seed=42,
        )

        print("\n✅ 训练完成!")

        # 验证
        print("\n🔍 执行验证...")
        metrics = trainer.validate(data=args.data)

        print(f"\n📊 最终结果:")
        print(f"  精度: {metrics.box.map:.4f}")
        if hasattr(metrics, 'speed'):
            print(f"  推理速度: {metrics.speed:.2f} ms")

        print("\n✨ 训练流程完成!")

    except KeyboardInterrupt:
        print("\n⚠️  训练被中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
