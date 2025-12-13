#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置工厂模块
Training configuration factory module
"""

from ..core.config import TrainingConfig, OptimizedTrainingConfig
from typing import Dict, Any, Optional

def create_training_config(config_type: str = "standard", **kwargs) -> TrainingConfig:
    """
    创建训练配置
    
    Args:
        config_type: 配置类型 ('standard', 'optimized', 'balanced', 'fast')
        **kwargs: 额外的配置参数
        
    Returns:
        训练配置对象
    """
    if config_type == "standard":
        config = TrainingConfig()
    elif config_type == "optimized":
        config = OptimizedTrainingConfig()
    elif config_type == "balanced":
        config = create_balanced_config()
    elif config_type == "fast":
        config = create_fast_config()
    else:
        raise ValueError(f"未知的配置类型: {config_type}")
    
    # 应用额外的配置参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def create_balanced_config() -> TrainingConfig:
    """
    创建平衡配置（在速度和精度之间平衡）
    
    Returns:
        平衡训练配置
    """
    config = TrainingConfig()
    
    # 平衡的训练参数
    config.epochs = 100
    config.model_size = 's'
    config.optimizer = 'AdamW'
    config.lr0 = 0.001
    config.lrf = 0.1
    config.batch_size = 16
    
    # 平衡的增强参数
    config.mosaic = 0.7
    config.mixup = 0.4
    config.copy_paste = 0.3
    config.degrees = 12.0
    config.translate = 0.3
    config.scale = 0.8
    
    # 平衡的优化参数
    config.box_gain = 8.0
    config.cls_gain = 1.5
    config.dfl_gain = 2.0
    config.patience = 30
    
    return config

def create_fast_config() -> TrainingConfig:
    """
    创建快速训练配置（用于快速实验）
    
    Returns:
        快速训练配置
    """
    config = TrainingConfig()
    
    # 快速的训练参数
    config.epochs = 50
    config.model_size = 'n'
    config.optimizer = 'SGD'
    config.lr0 = 0.01
    config.lrf = 0.2
    config.batch_size = 32
    
    # 简化的增强参数
    config.mosaic = 0.5
    config.mixup = 0.2
    config.copy_paste = 0.1
    config.degrees = 5.0
    config.translate = 0.1
    config.scale = 0.5
    
    # 简化的优化参数
    config.box_gain = 5.0
    config.cls_gain = 0.3
    config.dfl_gain = 1.0
    config.patience = 10
    
    return config

def create_config_for_dataset(dataset_type: str, **kwargs) -> TrainingConfig:
    """
    根据数据集类型创建配置
    
    Args:
        dataset_type: 数据集类型 ('rdd2022', 'rdd2020', 'custom')
        **kwargs: 额外的配置参数
        
    Returns:
        训练配置对象
    """
    if dataset_type == "rdd2022":
        # RDD2022数据集专用配置
        config = OptimizedTrainingConfig()
        config.data_yaml_path = kwargs.get('data_yaml_path', 'datasets/yolo_format/road.yaml')
        
        # RDD2022特定的增强参数
        config.mosaic = 0.8
        config.mixup = 0.5
        config.copy_paste = 0.3
        config.degrees = 15.0
        config.translate = 0.4
        config.scale = 0.9
        
        # RDD2022特定的优化参数
        config.box_gain = 9.0
        config.cls_gain = 3.0
        config.dfl_gain = 2.5
        
    elif dataset_type == "rdd2020":
        # RDD2020数据集专用配置
        config = TrainingConfig()
        config.data_yaml_path = kwargs.get('data_yaml_path', 'datasets/rdd2020/road.yaml')
        
        # RDD2020特定的参数
        config.epochs = 120
        config.model_size = 's'
        config.mosaic = 0.6
        config.mixup = 0.4
        
    elif dataset_type == "custom":
        # 自定义数据集配置
        config = TrainingConfig()
        config.data_yaml_path = kwargs.get('data_yaml_path', 'datasets/custom/data.yaml')
        
        # 使用标准参数，可以通过kwargs覆盖
        config.epochs = kwargs.get('epochs', 100)
        config.model_size = kwargs.get('model_size', 'n')
        
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    # 应用额外的配置参数
    for key, value in kwargs.items():
        if hasattr(config, key) and key != 'data_yaml_path':
            setattr(config, key, value)
    
    return config

def get_config_recommendations(dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据数据集统计信息获取配置建议
    
    Args:
        dataset_stats: 数据集统计信息
        
    Returns:
        配置建议字典
    """
    recommendations = {}
    
    # 基于数据集大小的建议
    train_count = dataset_stats.get('train_count', 0)
    if train_count < 100:
        recommendations['epochs'] = 50
        recommendations['model_size'] = 'n'
        recommendations['mosaic'] = 0.3  # 小数据集减少增强
        recommendations['mixup'] = 0.1
    elif train_count < 1000:
        recommendations['epochs'] = 100
        recommendations['model_size'] = 's'
        recommendations['mosaic'] = 0.5
        recommendations['mixup'] = 0.3
    else:
        recommendations['epochs'] = 150
        recommendations['model_size'] = 'm'
        recommendations['mosaic'] = 0.8
        recommendations['mixup'] = 0.5
    
    # 基于类别数量的建议
    num_classes = dataset_stats.get('num_classes', 0)
    if num_classes <= 3:
        recommendations['cls_gain'] = 0.5
    elif num_classes <= 5:
        recommendations['cls_gain'] = 1.0
    else:
        recommendations['cls_gain'] = 2.0
    
    # 基于类别分布的建议
    # 这里可以添加更复杂的类别平衡分析
    
    return recommendations