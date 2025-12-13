#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置管理模块
Training configuration management module
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 基础配置
    data_yaml_path: str = "datasets/yolo_format/road.yaml"
    model_size: str = "n"  # n, s, m, l, x
    epochs: int = 100
    img_size: int = 640
    batch_size: int = 16
    workers: int = 4
    
    # 设备配置
    device: str = "auto"  # auto, cuda, cpu
    amp: bool = True  # 混合精度训练
    
    # 优化器配置
    optimizer: str = "SGD"  # SGD, AdamW
    lr0: float = 0.001  # 初始学习率
    lrf: float = 0.1  # 最终学习率倍数
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    
    # 损失函数配置
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    
    # 数据增强配置
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.2
    scale: float = 0.7
    shear: float = 0.0
    perspective: float = 0.0005
    fliplr: float = 0.8
    flipud: float = 0.3
    mosaic: float = 0.5
    mixup: float = 0.3
    copy_paste: float = 0.2
    auto_augment: str = "rand-m9-mstd0.5-inc1"
    erasing: float = 0.6
    
    # 训练策略
    close_mosaic: int = 10
    patience: int = 50
    cos_lr: bool = True
    
    # 保存配置
    project: str = "./runs"
    name: str = "detect"
    save_period: int = 10
    
    # 高级配置
    single_cls: bool = False
    overlap_mask: bool = False
    cache: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'data': self.data_yaml_path,
            'epochs': self.epochs,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'workers': self.workers,
            'device': self.device,
            'optimizer': self.optimizer,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'box': self.box_gain,
            'cls': self.cls_gain,
            'dfl': self.dfl_gain,
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            'fliplr': self.fliplr,
            'flipud': self.flipud,
            'mosaic': self.mosaic,
            'mixup': self.mixup,
            'copy_paste': self.copy_paste,
            'auto_augment': self.auto_augment,
            'erasing': self.erasing,
            'close_mosaic': self.close_mosaic,
            'overlap_mask': self.overlap_mask,
            'single_cls': self.single_cls,
            'patience': self.patience,
            'cos_lr': self.cos_lr,
            'project': self.project,
            'name': self.name,
            'save_period': self.save_period,
            'amp': self.amp,
            'cache': self.cache,
            'val': True,
            'save': True,
            'plots': True
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def save_to_yaml(self, yaml_path: str):
        """保存为YAML文件"""
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """从YAML文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

class OptimizedTrainingConfig(TrainingConfig):
    """优化的训练配置（针对RDD2022数据集）"""
    
    def __init__(self):
        super().__init__()
        # 覆盖为优化配置
        self.model_size = "s"
        self.epochs = 150
        self.optimizer = "AdamW"
        self.lr0 = 0.002
        self.lrf = 0.05
        self.momentum = 0.95
        self.weight_decay = 0.0001
        self.warmup_epochs = 3
        self.patience = 15
        
        # 增强的损失函数配置
        self.box_gain = 9.0
        self.cls_gain = 3.0
        self.dfl_gain = 2.5
        
        # 更强的数据增强
        self.degrees = 15.0
        self.translate = 0.4
        self.scale = 0.9
        self.shear = 5.0
        self.perspective = 0.001
        self.flipud = 0.2
        self.mosaic = 0.8
        self.mixup = 0.3
        self.copy_paste = 0.4
        self.erasing = 0.4
        self.close_mosaic = 20