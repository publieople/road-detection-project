#!/usr/bin/env python3
"""
道路病害检测模型训练脚本
基于yolo11训练专门的道路病害检测模型
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import os

def get_dataset_stats(data_yaml_path: str) -> dict:
    """从数据配置文件中获取统计信息"""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        # 获取类别信息
        nc = data_config.get('nc', 0)
        names = data_config.get('names', [])

        # 计算训练和验证图片数量
        def count_images_in_path(path_pattern):
            """计算指定路径模式下的图片数量"""
            if isinstance(path_pattern, list):
                # 如果是列表，处理多个路径
                total_count = 0
                for pattern in path_pattern:
                    path = Path(pattern)
                    if path.exists():
                        if path.is_dir():
                            total_count += len(list(path.rglob('*.jpg')) + list(path.rglob('*.png')) +
                                             list(path.rglob('*.jpeg')) + list(path.rglob('*.JPG')))
                        else:
                            # 如果是文件，可能是包含图片路径的txt文件
                            with open(path, 'r') as f:
                                total_count += len([line for line in f.readlines() if line.strip()])
                return total_count
            else:
                # 单个路径
                path = Path(path_pattern)
                if path.exists():
                    if path.is_dir():
                        return len(list(path.rglob('*.jpg')) + list(path.rglob('*.png')) +
                                 list(path.rglob('*.jpeg')) + list(path.rglob('*.JPG')))
                    else:
                        # 如果是文件，可能是包含图片路径的txt文件
                        with open(path, 'r') as f:
                            return len([line for line in f.readlines() if line.strip()])
                return 0

        train_count = count_images_in_path(data_config.get('train', ''))
        val_count = count_images_in_path(data_config.get('val', ''))

        return {
            'train_count': train_count,
            'val_count': val_count,
            'num_classes': nc,
            'class_names': names
        }
    except Exception as e:
        print(f"⚠️  获取数据集统计信息失败: {e}")
        return {
            'train_count': 0,
            'val_count': 0,
            'num_classes': 0,
            'class_names': []
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

def train_model(data_yaml_path: str, model_size: str = 'n', epochs: int = 100, img_size: int = 1280, resume: bool = False):
    """
    训练YOLO模型

    Args:
        data_yaml_path: 数据配置文件路径
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        epochs: 训练轮数
        img_size: 输入图像尺寸
        resume: 是否从上次中断处恢复训练
    """

    device = setup_training()

    if resume:
        # 恢复训练模式
        print("🔄 检测是否存在中断的训练，尝试恢复...")

        # 检查是否存在上次训练的权重文件
        possible_weights = [
            'runs/detect/train/weights/last.pt',  # 默认训练路径
            'runs/detect/train2/weights/last.pt', # 第二次训练路径
            'runs/detect/train3/weights/last.pt', # 第三次训练路径
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
            # 选择预训练模型
            model_name = f'yolo11{model_size}.pt'
            print(f"📦 加载预训练模型: {model_name}")
            model = YOLO(model_name)
    else:
        # 正常训练模式
        # 选择预训练模型
        model_name = f'yolo11{model_size}.pt'
        print(f"📦 加载预训练模型: {model_name}")
        model = YOLO(model_name)

    # 训练配置
    training_config = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': 16,                              # 批次大小，可根据GPU内存调整
        'workers': 4,                            # 数据加载线程数
        'cache': False,                          # 是否缓存数据集

        'device': device,
        'optimizer': 'AdamW',
        'lr0': 0.001,                            # 初始学习率
        'lrf': 0.01,                             # 最终学习率倍数
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 10,                     # 预热轮数
        'warmup_momentum': 0.8,
        'box': 7.5,                              # box loss增益
        'cls': 0.5,                              # cls loss增益
        'dfl': 1.5,                              # dfl loss增益

        'hsv_h': 0.015,                          # HSV色调增强
        'hsv_s': 0.7,                            # HSV饱和度增强
        'hsv_v': 0.4,                            # HSV明度增强
        'degrees': 5.0,                          # 旋转增强
        'scale': 0.7,                            # 缩放增强
        'shear': 0.0,                            # 剪切增强
        'perspective': 0.0005,                   # 透视增强
        'fliplr': 0.8,                           # 左右翻转
        'flipud': 0.3,                           # 上下翻转
        'translate': 0.2,                        # 平移增强
        'mosaic': 0.8,                           # mosaic增强
        'mixup': 0.3,                            # mixup增强
        'copy_paste': 0.2,                       # 复制粘贴增强
        'auto_augment': 'rand-m9-mstd0.5-inc1',  # 自动增强策略
        'erasing': 0.6,                          # 随机擦除

        'crop_fraction': 1.0,                    # 裁剪比例
        "amp": True,                             # 混合精度加速

        'close_mosaic': 10,                      # 关闭mosaic的最后轮数
        'overlap_mask': False,                   # 是否使用重叠掩码
        'single_cls': False,                     # 是否为单类别检测
        'patience': 50,                          # 早停耐心值
        'cos_lr': True,                          # 使用余弦退火学习率
    }

    print("🚀 开始训练模型...")
    print(f"📊 训练轮数: {epochs}")
    print(f"📐 图像尺寸: {img_size}")
    print(f"🔧 设备: {device}")

    # 开始训练
    if resume and resume_path:
        # 恢复训练模式
        training_config['resume'] = True
        results = model.train(**training_config)
    else:
        # 正常训练模式
        results = model.train(**training_config)

    # 保存训练结果
    print("✅ 训练完成!")
    if results and hasattr(results, 'save_dir'):
        print(f"📁 模型保存在: {Path(results.save_dir).resolve()}")
    else:
        print("📁 模型训练完成，保存路径信息不可用")

    return model, results

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

    return metrics

def main():
    """主函数"""
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='道路病害检测模型训练')
    parser.add_argument('--resume', action='store_true', help='从上次中断处恢复训练')
    parser.add_argument('--data', type=str, default='datasets/yolo_format/road.yaml', help='数据配置文件路径')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')

    args = parser.parse_args()

    # 数据配置文件路径
    data_yaml = args.data

    # 检查数据文件是否存在
    if not Path(data_yaml).exists():
        print(f"❌ 数据配置文件不存在: {data_yaml}")
        return

    print("🛣️  道路病害检测模型训练")
    print("=" * 50)

    if args.resume:
        print("🔄 已启用训练恢复模式")

    print(f"📊 配置: 模型={args.model_size}, 轮数={args.epochs}, 尺寸={args.img_size}")

    try:
        # 训练模型
        model, training_results = train_model(
            data_yaml_path=data_yaml,
            model_size=args.model_size,
            epochs=args.epochs,          # 训练50轮
            img_size=args.img_size,      # 输入图像尺寸
            resume=args.resume           # 是否恢复训练
        )

        # 验证模型
        metrics = validate_model(model, data_yaml)

        # 获取数据集统计信息
        dataset_stats = get_dataset_stats(data_yaml)

        print("\n🎉 训练流程完成!")
        print("📋 总结:")
        print(f"   - 训练图片: {dataset_stats['train_count']}张")
        print(f"   - 验证图片: {dataset_stats['val_count']}张")
        print(f"   - 病害类别: {dataset_stats['num_classes']}类")
        if dataset_stats['class_names']:
            print(f"   - 类别名称: {', '.join(dataset_stats['class_names'])}")
        print(f"   - 最佳mAP@0.5: {metrics.box.map50:.3f}")

        # 导出模型
        print("\n💾 导出训练好的模型...")
        model.export(format='onnx', simplify=True)

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()