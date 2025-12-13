#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块
Logger utility module
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class TrainingLogger:
    """训练日志器"""
    
    def __init__(self, name: str = "RoadDamageTraining", log_dir: Optional[str] = None):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_dir: 日志保存目录
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_dir:
            log_path = Path(log_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = str(log_path)
        else:
            self.log_file = None
    
    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def log_training_start(self, config: dict):
        """记录训练开始"""
        self.info("=" * 60)
        self.info("🚀 训练开始")
        self.info("=" * 60)
        self.info(f"数据配置: {config.get('data', 'unknown')}")
        self.info(f"模型大小: {config.get('model_size', 'unknown')}")
        self.info(f"训练轮数: {config.get('epochs', 'unknown')}")
        self.info(f"图像尺寸: {config.get('img_size', 'unknown')}")
        self.info(f"批次大小: {config.get('batch_size', 'unknown')}")
        self.info(f"优化器: {config.get('optimizer', 'unknown')}")
        self.info(f"初始学习率: {config.get('lr0', 'unknown')}")
    
    def log_training_end(self, results: dict):
        """记录训练结束"""
        self.info("=" * 60)
        self.info("✅ 训练完成")
        self.info("=" * 60)
        
        if 'mAP50' in results:
            self.info(f"mAP@0.5: {results['mAP50']:.3f}")
        if 'mAP5095' in results:
            self.info(f"mAP@0.5:0.95: {results['mAP5095']:.3f}")
        if 'precision' in results:
            self.info(f"精确率: {results['precision']:.3f}")
        if 'recall' in results:
            self.info(f"召回率: {results['recall']:.3f}")
    
    def log_epoch_progress(self, epoch: int, total_epochs: int, metrics: dict):
        """记录训练进度"""
        progress = (epoch / total_epochs) * 100
        self.info(f"📊 训练进度: {epoch}/{total_epochs} ({progress:.1f}%)")
        
        if 'train_loss' in metrics:
            self.info(f"   训练损失: {metrics['train_loss']:.4f}")
        if 'val_loss' in metrics:
            self.info(f"   验证损失: {metrics['val_loss']:.4f}")
        if 'mAP50' in metrics:
            self.info(f"   mAP@0.5: {metrics['mAP50']:.3f}")
    
    def log_validation_results(self, results: dict):
        """记录验证结果"""
        self.info("🔍 验证结果:")
        
        if 'mAP50' in results:
            self.info(f"   mAP@0.5: {results['mAP50']:.3f}")
        if 'mAP5095' in results:
            self.info(f"   mAP@0.5:0.95: {results['mAP5095']:.3f}")
        if 'precision' in results:
            self.info(f"   精确率: {results['precision']:.3f}")
        if 'recall' in results:
            self.info(f"   召回率: {results['recall']:.3f}")
        
        # 记录每个类别的性能
        if 'class_ap50' in results:
            self.info("   各类别AP@0.5:")
            for class_result in results['class_ap50']:
                self.info(f"     {class_result['class_name']}: {class_result['ap50']:.3f}")
    
    def log_dataset_stats(self, stats: dict):
        """记录数据集统计"""
        self.info("📊 数据集统计:")
        self.info(f"   训练图片: {stats.get('train_count', 0)} 张")
        self.info(f"   验证图片: {stats.get('val_count', 0)} 张")
        self.info(f"   类别数量: {stats.get('num_classes', 0)}")
        
        if 'class_names' in stats:
            self.info(f"   类别名称: {', '.join(stats['class_names'])}")
    
    def log_device_info(self, device_info: dict):
        """记录设备信息"""
        self.info("🔧 设备信息:")
        self.info(f"   平台: {device_info.get('platform', 'unknown')}")
        self.info(f"   PyTorch版本: {device_info.get('torch_version', 'unknown')}")
        
        if device_info.get('device') == 'cuda':
            self.info(f"   GPU: {device_info.get('device_name', 'unknown')}")
            self.info(f"   CUDA版本: {device_info.get('cuda_version', 'unknown')}")
            self.info(f"   GPU内存: {device_info.get('gpu_memory_gb', 0):.1f} GB")
        else:
            self.info(f"   CPU线程数: {device_info.get('cpu_count', 'unknown')}")
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误信息"""
        self.error(f"❌ 错误发生: {context}")
        self.error(f"错误类型: {type(error).__name__}")
        self.error(f"错误信息: {str(error)}")
        
        # 记录堆栈跟踪
        import traceback
        self.error("堆栈跟踪:")
        for line in traceback.format_exc().splitlines():
            self.error(line)
    
    def log_warning_with_context(self, message: str, context: str = ""):
        """记录带上下文的警告"""
        if context:
            self.warning(f"⚠️  {context}: {message}")
        else:
            self.warning(f"⚠️  {message}")
    
    def log_info_with_emoji(self, message: str, emoji: str = "📋"):
        """记录带emoji的信息"""
        self.info(f"{emoji} {message}")
    
    def get_log_file_path(self) -> Optional[str]:
        """获取日志文件路径"""
        return self.log_file

def create_logger(name: str = "RoadDamageTraining", log_dir: Optional[str] = None) -> TrainingLogger:
    """
    创建训练日志器
    
    Args:
        name: 日志器名称
        log_dir: 日志保存目录
        
    Returns:
        训练日志器
    """
    return TrainingLogger(name, log_dir)

# 全局日志器实例
_global_logger = None

def get_global_logger() -> TrainingLogger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = create_logger()
    return _global_logger

def set_global_logger(logger: TrainingLogger):
    """设置全局日志器"""
    global _global_logger
    _global_logger = logger