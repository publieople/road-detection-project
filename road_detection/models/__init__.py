#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型模块
包含自定义backbone、YOLO集成和训练器
"""

from .backbones import (
    ConvNeXtTiny,
    ConvNeXtSmall,
    create_convnext_backbone,
)

from .yolo11_convnext import (
    YOLO11ConvNeXt,
    ConvNeXtNeckModule,
)

from .yolo11_convnext_trainer import (
    YOLO11ConvNeXtTrainer,
)

__all__ = [
    'ConvNeXtTiny',
    'ConvNeXtSmall',
    'create_convnext_backbone',
    'YOLO11ConvNeXt',
    'ConvNeXtNeckModule',
    'YOLO11ConvNeXtTrainer',
]
