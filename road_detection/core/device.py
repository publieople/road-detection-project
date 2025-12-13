#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备管理模块
Device management module
"""

import torch
import platform
from typing import Optional

def setup_device(prefer_gpu: bool = True) -> str:
    """
    配置训练设备
    
    Args:
        prefer_gpu: 是否优先使用GPU
        
    Returns:
        设备名称 ('cuda' 或 'cpu')
    """
    if prefer_gpu and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"🚀 GPU可用: {device_name}")
        print(f"🔧 CUDA版本: {cuda_version}")
        
        # 检查GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
        
        # 设置GPU优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return 'cuda'
    else:
        print("💻 使用CPU训练")
        cpu_count = torch.get_num_threads()
        print(f"🔧 CPU线程数: {cpu_count}")
        
        # 设置CPU优化
        if platform.system() == "Windows":
            torch.set_num_threads(min(cpu_count, 8))  # Windows下限制线程数
        else:
            torch.set_num_threads(cpu_count)
        
        return 'cpu'

def get_device_info() -> dict:
    """
    获取设备信息
    
    Returns:
        设备信息字典
    """
    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__
    }
    
    if torch.cuda.is_available():
        info.update({
            'device': 'cuda',
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'gpu_count': torch.cuda.device_count()
        })
    else:
        info.update({
            'device': 'cpu',
            'cpu_count': torch.get_num_threads()
        })
    
    return info

def check_memory_requirements(batch_size: int, img_size: int, model_size: str = 'n') -> bool:
    """
    检查内存需求
    
    Args:
        batch_size: 批次大小
        img_size: 图像尺寸
        model_size: 模型大小 (n, s, m, l, x)
        
    Returns:
        是否满足内存要求
    """
    if not torch.cuda.is_available():
        return True  # CPU训练不检查内存
    
    # 估算GPU内存需求 (GB)
    base_memory = {
        'n': 2.0,  # YOLOv11-n
        's': 3.5,  # YOLOv11-s
        'm': 5.0,  # YOLOv11-m
        'l': 8.0,  # YOLOv11-l
        'x': 12.0  # YOLOv11-x
    }
    
    # 计算内存需求
    model_base = base_memory.get(model_size, 2.0)
    img_memory = (img_size ** 2) / (640 ** 2)  # 相对640x640的倍数
    batch_memory = batch_size / 16  # 相对批次16的倍数
    
    required_memory = model_base * img_memory * batch_memory
    
    # 获取可用GPU内存
    available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # 预留20%的安全边际
    safe_memory = available_memory * 0.8
    
    if required_memory > safe_memory:
        print(f"⚠️  内存需求警告:")
        print(f"   需要内存: {required_memory:.1f} GB")
        print(f"   可用内存: {available_memory:.1f} GB")
        print(f"   建议降低批次大小或图像尺寸")
        return False
    
    return True

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU缓存已清理")

def set_random_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    
    # Python随机种子
    random.seed(seed)
    
    # Numpy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 随机种子已设置为: {seed}")