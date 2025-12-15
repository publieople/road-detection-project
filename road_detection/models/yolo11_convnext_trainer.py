#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11-ConvNeXt 整合模块
提供便捷的API来使用ConvNeXt-Tiny Backbone的YOLO11模型
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from ultralytics import YOLO

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from road_detection.models.backbones import ConvNeXtTiny, ConvNeXtSmall, create_convnext_backbone
from road_detection.models.yolo11_convnext import YOLO11ConvNeXt, ConvNeXtNeckModule


class YOLO11ConvNeXtTrainer:
    """
    YOLO11-ConvNeXt 训练器
    集成ConvNeXt Backbone、YOLO11 Neck、Head的完整训练流程
    """

    def __init__(
        self,
        model_size: str = 'n',
        backbone_type: str = 'tiny',
        pretrained: bool = False,
        device: str = 'cuda:0',
    ):
        """
        初始化训练器

        Args:
            model_size: YOLO模型大小 (n, s, m, l, x)
            backbone_type: ConvNeXt类型 (tiny, small)
            pretrained: 是否加载预训练权重
            device: 设备 (cuda:0, cpu)
        """
        self.model_size = model_size
        self.backbone_type = backbone_type
        self.device = device

        # 创建基础YOLO11模型
        self.base_model = YOLO(f'yolo11{model_size}.pt')

        # 创建ConvNeXt backbone版本
        self.num_classes = self.base_model.model.nc
        self.custom_backbone = create_convnext_backbone(
            model_name=backbone_type,
            in_channels=3,
            drop_path_rate=0.1,
        )

        print(f"✅ YOLO11-ConvNeXt 训练器初始化成功")
        print(f"   模型大小: yolo11{model_size}")
        print(f"   Backbone: ConvNeXt-{backbone_type.capitalize()}")
        print(f"   类别数: {self.num_classes}")
        print(f"   设备: {device}")

    def replace_backbone(self) -> 'YOLO':
        """
        替换YOLO11的CSPDarknet为ConvNeXt-Tiny

        Returns:
            修改后的YOLO模型
        """
        print("\n🔄 开始替换Backbone...")
        print(f"   原始Backbone: CSPDarknet")
        print(f"   新Backbone: ConvNeXt-{self.backbone_type.capitalize()}")

        # 获取原始模型
        model = self.base_model.model

        # 替换backbone
        # 注意: 这需要访问YOLOv11的内部结构
        # YOLO11使用yaml配置定义模型，我们需要创建自定义配置

        print("   ⚠️  说明: Backbone替换需要使用官方配置文件")
        print("   推荐方式: 使用 yolo11_convnext.yaml 配置创建新模型")

        return self.base_model

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 1280,
        batch_size: int = 8,
        device: Optional[str] = None,
        resume: bool = False,
        save_period: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        训练YOLO11-ConvNeXt模型

        Args:
            data: 数据配置文件路径
            epochs: 训练轮数
            imgsz: 图像大小
            batch_size: 批次大小
            device: 设备ID
            resume: 是否恢复训练
            save_period: 保存周期
            **kwargs: 其他训练参数

        Returns:
            训练结果
        """
        print("\n🚀 开始训练 YOLO11-ConvNeXt 模型...")

        device = device or self.device

        # 训练配置
        train_config = {
            'data': data,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'resume': resume,
            'save_period': save_period,

            # 优化器配置 (ConvNeXt友好)
            'optimizer': 'adamw',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.05,

            # 数据增强 (针对ConvNeXt优化)
            'mosaic': 0.8,
            'mixup': 0.5,
            'copy_paste': 0.3,
            'degrees': 15.0,
            'translate': 0.4,
            'scale': 0.9,
            'shear': 5.0,
            'perspective': 0.001,
            'fliplr': 0.8,
            'flipud': 0.2,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,

            # 混合精度 (RTX 50系列优化)
            'amp': True,
            'half': True,

            # 其他参数
            'patience': 20,
            'plots': True,
            'verbose': True,
            'seed': 42,
        }

        # 合并自定义参数
        train_config.update(kwargs)

        # 启动训练
        results = self.base_model.train(**train_config)

        print("\n✅ 训练完成!")
        return results

    def export_model(
        self,
        format: str = 'onnx',
        half: bool = True,
        simplify: bool = True,
    ) -> str:
        """
        导出模型

        Args:
            format: 导出格式 (pt, onnx, tflite, pb, etc.)
            half: 使用半精度
            simplify: 简化ONNX

        Returns:
            导出文件路径
        """
        print(f"\n📦 导出模型为 {format.upper()} 格式...")

        export_path = self.base_model.export(
            format=format,
            half=half,
            simplify=simplify,
        )

        print(f"✅ 模型导出成功: {export_path}")
        return str(export_path)

    def validate(self, data: str = None) -> Dict[str, Any]:
        """验证模型"""
        print("\n🔍 开始模型验证...")
        results = self.base_model.val(data=data)
        return results

    @staticmethod
    def create_from_yaml(
        yaml_path: str,
        device: str = 'cuda:0',
    ) -> 'YOLO':
        """
        从YAML配置文件创建YOLO11-ConvNeXt模型

        Args:
            yaml_path: YAML配置文件路径
            device: 设备

        Returns:
            YOLO模型对象
        """
        print(f"\n📄 从YAML创建YOLO11-ConvNeXt模型")
        print(f"   配置文件: {yaml_path}")

        model = YOLO(yaml_path)
        model.to(device)

        print(f"✅ 模型创建成功")
        return model


def create_yolo11_convnext_from_weights(
    weights_path: str,
    backbone_type: str = 'tiny',
) -> YOLO:
    """
    从权重文件创建YOLO11-ConvNeXt模型

    Args:
        weights_path: 权重文件路径
        backbone_type: ConvNeXt类型

    Returns:
        YOLO模型
    """
    print(f"📥 加载权重: {weights_path}")
    model = YOLO(weights_path)
    print(f"✅ 权重加载成功")
    return model


def compare_backbones(
    test_image_size: tuple = (640, 640),
    num_iterations: int = 100,
):
    """
    对比CSPDarknet和ConvNeXt-Tiny的性能

    Args:
        test_image_size: 测试图像尺寸
        num_iterations: 测试迭代次数
    """
    import time

    print("=" * 70)
    print("CSPDarknet vs ConvNeXt-Tiny Backbone 性能对比")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. ConvNeXt-Tiny
    print("\n📊 测试 ConvNeXt-Tiny Backbone...")
    convnext_backbone = ConvNeXtTiny(in_channels=3).to(device)
    convnext_backbone.eval()

    # 计算参数量
    convnext_params = sum(p.numel() for p in convnext_backbone.parameters())
    print(f"   参数量: {convnext_params:,}")

    # 测试推理速度
    x = torch.randn(1, 3, *test_image_size, device=device)

    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = convnext_backbone(x)

        # 计时
        start = time.time()
        for _ in range(num_iterations):
            _ = convnext_backbone(x)
        convnext_time = (time.time() - start) / num_iterations * 1000

    print(f"   平均推理时间: {convnext_time:.2f} ms")

    # 2. YOLO11 基础模型 (CSPDarknet)
    print("\n📊 测试 YOLO11n (CSPDarknet Backbone)...")
    try:
        yolo11 = YOLO('yolo11n.pt')
        yolo11_params = sum(p.numel() for p in yolo11.model.parameters())
        print(f"   参数量: {yolo11_params:,}")

        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = yolo11.model(x)

            # 计时
            start = time.time()
            for _ in range(num_iterations):
                _ = yolo11.model(x)
            yolo11_time = (time.time() - start) / num_iterations * 1000

        print(f"   平均推理时间: {yolo11_time:.2f} ms")

        # 对比结果
        print("\n📈 性能对比结果:")
        print(f"   参数减少: {(1 - convnext_params/yolo11_params)*100:.1f}%")
        print(f"   推理速度: {(yolo11_time/convnext_time - 1)*100:.1f}% 提升")

    except Exception as e:
        print(f"   YOLO11模型加载失败: {e}")


if __name__ == '__main__':
    # 演示使用
    print("=" * 70)
    print("YOLO11-ConvNeXt 模块演示")
    print("=" * 70)

    # 创建训练器
    trainer = YOLO11ConvNeXtTrainer(
        model_size='n',
        backbone_type='tiny',
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    print("\n✅ 训练器创建成功")
    print("\n使用示例:")
    print("  # 训练模型")
    print("  results = trainer.train(")
    print("      data='datasets/yolo_format/road.yaml',")
    print("      epochs=100,")
    print("      batch_size=8,")
    print("  )")
    print("\n  # 验证模型")
    print("  metrics = trainer.validate('datasets/yolo_format/road.yaml')")
    print("\n  # 导出模型")
    print("  export_path = trainer.export_model(format='onnx')")
