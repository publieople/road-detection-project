#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理模块
Model management module
支持标准YOLO11和ConvNeXt-Tiny Backbone版本
"""

from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import torch
import os

# 修复 Windows 上 torch._dynamo 的路径问题
os.environ['TORCH_DISABLE_DYNAMO'] = '1'

class RoadDamageModel:
    """道路病害检测模型管理类

    支持两种模式:
    1. 标准YOLO11: 使用CSPDarknet Backbone
    2. ConvNeXt版本: 使用ConvNeXt-Tiny Backbone
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = 'n',
        backbone_type: Literal['csp', 'convnext'] = 'csp',
    ):
        """
        初始化模型

        Args:
            model_path: 模型路径（如果提供则加载现有模型）
            model_size: 模型大小 (n, s, m, l, x)
            backbone_type: Backbone类型 ('csp' for CSPDarknet, 'convnext' for ConvNeXt-Tiny)
        """
        self.model_size = model_size
        self.backbone_type = backbone_type
        self.model = None

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.create_model(model_size, backbone_type)

    def create_model(
        self,
        model_size: str = 'n',
        backbone_type: Literal['csp', 'convnext'] = 'csp',
    ):
        """
        创建新的YOLO模型

        Args:
            model_size: 模型大小 (n, s, m, l, x)
            backbone_type: Backbone类型
        """
        if backbone_type == 'convnext':
            # 使用ConvNeXt配置
            self._create_convnext_model(model_size)
        else:
            # 使用标准YOLO11 (CSPDarknet)
            self._create_standard_model(model_size)

    def _create_standard_model(self, model_size: str = 'n'):
        """创建标准YOLO11模型 (CSPDarknet Backbone)"""
        model_name = f'yolo11{model_size}.pt'
        print(f"📦 创建新模型: {model_name} (CSPDarknet)")

        try:
            self.model = YOLO(model_name)
            self.model_size = model_size
            self.backbone_type = 'csp'
            print(f"✅ 模型创建成功: {model_name}")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            raise

    def _create_convnext_model(self, model_size: str = 'n'):
        """创建ConvNeXt-Tiny Backbone YOLO11模型"""
        print(f"📦 创建新模型: yolo11{model_size}-convnext (ConvNeXt-Tiny)")

        try:
            # 使用ConvNeXt配置YAML文件
            yaml_path = Path(__file__).parent.parent.parent / 'datasets' / 'yolo_format' / 'yolo11_convnext.yaml'

            if not yaml_path.exists():
                print(f"⚠️  警告: ConvNeXt配置文件不存在: {yaml_path}")
                print(f"   将使用标准YOLO11配置")
                self._create_standard_model(model_size)
                return

            self.model = YOLO(str(yaml_path))
            self.model_size = model_size
            self.backbone_type = 'convnext'
            print(f"✅ 模型创建成功: yolo11{model_size}-convnext")
        except Exception as e:
            print(f"❌ ConvNeXt模型创建失败: {e}")
            print(f"   将使用标准YOLO11配置")
            self._create_standard_model(model_size)

    def load_model(self, model_path: str):
        """
        加载现有模型

        Args:
            model_path: 模型文件路径
        """
        print(f"📂 加载模型: {model_path}")

        try:
            self.model = YOLO(model_path)

            # 尝试推断模型大小
            model_name = Path(model_path).name
            for size in ['n', 's', 'm', 'l', 'x']:
                if f'yolo11{size}' in model_name or f'yolov8{size}' in model_name:
                    self.model_size = size
                    break

            print(f"✅ 模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def train(self, config: Dict[str, Any], resume: bool = False) -> Any:
        """
        训练模型

        Args:
            config: 训练配置字典
            resume: 是否恢复训练

        Returns:
            训练结果
        """
        if not self.model:
            raise ValueError("模型未初始化")

        print("🚀 开始训练...")

        if resume:
            config['resume'] = True
            print("🔄 恢复训练模式")

        try:
            results = self.model.train(**config)
            print("✅ 训练完成!")
            return results
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            raise

    def validate(self, data_yaml_path: str) -> Any:
        """
        验证模型

        Args:
            data_yaml_path: 数据配置文件路径

        Returns:
            验证结果
        """
        if not self.model:
            raise ValueError("模型未初始化")

        print("🔍 验证模型性能...")

        try:
            metrics = self.model.val(data=data_yaml_path)
            print("✅ 验证完成!")
            return metrics
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            raise

    def predict(self, image_path: str, conf: float = 0.5, iou: float = 0.7) -> Any:
        """
        预测单张图片

        Args:
            image_path: 图片路径
            conf: 置信度阈值
            iou: IOU阈值

        Returns:
            预测结果
        """
        if not self.model:
            raise ValueError("模型未初始化")

        try:
            results = self.model(image_path, conf=conf, iou=iou)
            return results
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            raise

    def export_model(self, format: str = 'onnx', simplify: bool = True, **kwargs) -> str:
        """
        导出模型

        Args:
            format: 导出格式
            simplify: 是否简化模型
            **kwargs: 其他导出参数

        Returns:
            导出文件路径
        """
        if not self.model:
            raise ValueError("模型未初始化")

        print(f"💾 导出模型为 {format} 格式...")

        try:
            export_path = self.model.export(format=format, simplify=simplify, **kwargs)
            print(f"✅ 模型导出成功: {export_path}")
            return export_path
        except Exception as e:
            print(f"❌ 模型导出失败: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        if not self.model:
            return {}

        info = {
            'model_size': self.model_size,
            'model_type': type(self.model).__name__,
            'tasks': getattr(self.model, 'tasks', []),
            'names': getattr(self.model, 'names', []),
        }

        # 获取模型参数信息
        if hasattr(self.model, 'model'):
            try:
                param_count = sum(p.numel() for p in self.model.model.parameters())
                info['parameter_count'] = param_count
            except:
                pass

        return info

    def save_model(self, save_path: str):
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        if not self.model:
            raise ValueError("模型未初始化")

        try:
            self.model.save(save_path)
            print(f"💾 模型已保存: {save_path}")
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
            raise

def find_resume_weights() -> Optional[str]:
    """
    查找可恢复的权重文件

    Returns:
        权重文件路径或None
    """
    possible_weights = [
        'runs/detect/train/weights/last.pt',   # 默认训练路径
        'runs/detect/train2/weights/last.pt',  # 第二次训练路径
        'runs/detect/train3/weights/last.pt',  # 第三次训练路径
    ]

    for weight_path in possible_weights:
        if Path(weight_path).exists():
            return weight_path

    return None