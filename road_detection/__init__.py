"""
道路病害检测系统 - 模块化训练框架
Road Damage Detection System - Modular Training Framework
"""
__version__ = "1.0.0"
__author__ = "Road Detection Team"

from .core import *
from .utils import *
from .training import *

__all__ = [
    "RoadDamageTrainer",
    "DatasetAnalyzer", 
    "ModelValidator",
    "TrainingConfig",
    "setup_device",
    "get_dataset_stats",
    "validate_model"
]