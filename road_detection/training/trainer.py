#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器模块
Trainer module
"""

from ..core.model import RoadDamageModel, find_resume_weights
from ..core.device import setup_device, check_memory_requirements, clear_gpu_cache, set_random_seed
from ..core.config import TrainingConfig
from ..utils.dataset import DatasetAnalyzer, get_dataset_stats
from ..utils.validation import ModelValidator
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml

class RoadDamageTrainer:
    """道路病害检测训练器"""
    
    def __init__(self, config: TrainingConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.model = None
        self.dataset_stats = None
        self.training_results = None
        self.validation_results = None
        
        # 设置随机种子
        set_random_seed(42)
        
        # 设置设备
        self.device = setup_device()
        self.config.device = self.device
        
        print("🚀 道路病害检测训练器初始化完成")
    
    def prepare_training(self) -> Dict[str, Any]:
        """
        准备训练
        
        Returns:
            准备结果
        """
        print("\n" + "=" * 60)
        print("🛠️  训练准备阶段")
        print("=" * 60)
        
        # 1. 检查数据文件
        if not Path(self.config.data_yaml_path).exists():
            raise FileNotFoundError(f"数据配置文件不存在: {self.config.data_yaml_path}")
        
        # 2. 分析数据集
        print("📊 分析数据集...")
        self.dataset_stats = get_dataset_stats(self.config.data_yaml_path)
        
        # 3. 详细分析（优化配置）
        if self.config.optimizer == "AdamW" and self.config.cls_gain > 1.0:
            # 对于优化配置，进行详细数据集分析
            analyzer = DatasetAnalyzer(self.config.data_yaml_path)
            class_weights = analyzer.calculate_class_weights()
            
            # 将类别权重应用到配置中
            if class_weights:
                # 这里可以根据类别权重调整损失函数参数
                print(f"⚖️  应用类别权重: {class_weights}")
        
        # 4. 检查内存需求
        if not check_memory_requirements(
            self.config.batch_size, 
            self.config.img_size, 
            self.config.model_size
        ):
            print("⚠️  内存需求警告，建议调整配置")
        
        # 5. 清理GPU缓存
        clear_gpu_cache()
        
        print("✅ 训练准备完成!")
        return self.dataset_stats
    
    def create_or_load_model(self, resume_path: Optional[str] = None) -> RoadDamageModel:
        """
        创建或加载模型
        
        Args:
            resume_path: 恢复训练的路径
            
        Returns:
            模型对象
        """
        print("\n" + "=" * 60)
        print("📦 模型准备阶段")
        print("=" * 60)
        
        if resume_path and Path(resume_path).exists():
            # 恢复训练
            print(f"🔄 从指定路径恢复训练: {resume_path}")
            self.model = RoadDamageModel(resume_path)
        else:
            # 查找可恢复的权重
            resume_weights = find_resume_weights()
            if resume_weights:
                print(f"🔄 找到可恢复的权重: {resume_weights}")
                self.model = RoadDamageModel(resume_weights)
            else:
                # 创建新模型
                print(f"📦 创建新模型: yolo11{self.config.model_size}")
                self.model = RoadDamageModel(model_size=self.config.model_size)
        
        print("✅ 模型准备完成!")
        return self.model
    
    def train(self, resume: bool = False) -> Tuple[RoadDamageModel, Any]:
        """
        执行训练
        
        Args:
            resume: 是否恢复训练
            
        Returns:
            (模型对象, 训练结果)
        """
        print("\n" + "=" * 60)
        print("🚀 开始训练")
        print("=" * 60)
        
        if not self.model:
            raise ValueError("模型未初始化，请先调用 create_or_load_model()")
        
        # 打印训练配置
        self._print_training_config()
        
        # 获取训练配置
        training_config = self.config.to_dict()
        
        # 开始训练
        try:
            self.training_results = self.model.train(training_config, resume=resume)
            print("✅ 训练完成!")
            
            return self.model, self.training_results
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            raise
    
    def validate(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        验证模型
        
        Args:
            save_dir: 结果保存目录
            
        Returns:
            验证结果
        """
        print("\n" + "=" * 60)
        print("🔍 模型验证阶段")
        print("=" * 60)
        
        if not self.model:
            raise ValueError("模型未初始化，请先训练模型")
        
        try:
            # 创建验证器
            validator = ModelValidator(
                model_path=self.model.model.ckpt_path,  # 获取当前模型路径
                data_yaml_path=self.config.data_yaml_path
            )
            
            # 执行验证
            self.validation_results = validator.validate(save_dir=save_dir)
            
            # 检查性能目标（优化配置默认目标0.80）
            target_map50 = 0.85 if self.config.optimizer == "AdamW" and self.config.cls_gain > 1.0 else 0.80
            validator.check_performance_target(target_map50)
            
            print("✅ 模型验证完成!")
            return self.validation_results
            
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            raise
    
    def export_model(self, format: str = 'onnx', **kwargs) -> str:
        """
        导出模型
        
        Args:
            format: 导出格式
            **kwargs: 其他导出参数
            
        Returns:
            导出文件路径
        """
        print("\n" + "=" * 60)
        print("💾 模型导出阶段")
        print("=" * 60)
        
        if not self.model:
            raise ValueError("模型未初始化，请先训练模型")
        
        try:
            export_path = self.model.export_model(format=format, **kwargs)
            print("✅ 模型导出完成!")
            return export_path
            
        except Exception as e:
            print(f"❌ 模型导出失败: {e}")
            raise
    
    def generate_training_report(self) -> str:
        """
        生成训练报告
        
        Returns:
            训练报告
        """
        if not self.dataset_stats or not self.validation_results:
            return "训练报告: 缺少必要数据"
        
        report = []
        report.append("=" * 60)
        report.append("道路病害检测模型训练报告")
        report.append("=" * 60)
        report.append("")
        
        # 数据集信息
        report.append("📊 数据集信息:")
        report.append(f"  训练图片: {self.dataset_stats['train_count']} 张")
        report.append(f"  验证图片: {self.dataset_stats['val_count']} 张")
        report.append(f"  类别数量: {self.dataset_stats['num_classes']}")
        report.append(f"  类别名称: {', '.join(self.dataset_stats['class_names'])}")
        report.append("")
        
        # 训练配置
        report.append("⚙️ 训练配置:")
        report.append(f"  模型大小: {self.config.model_size}")
        report.append(f"  训练轮数: {self.config.epochs}")
        report.append(f"  图像尺寸: {self.config.img_size}")
        report.append(f"  批次大小: {self.config.batch_size}")
        report.append(f"  优化器: {self.config.optimizer}")
        report.append(f"  初始学习率: {self.config.lr0}")
        report.append("")
        
        # 验证结果
        report.append("🎯 验证结果:")
        report.append(f"  mAP@0.5: {self.validation_results['mAP50']:.3f}")
        report.append(f"  mAP@0.5:0.95: {self.validation_results['mAP5095']:.3f}")
        report.append(f"  平均精确率: {self.validation_results['precision']:.3f}")
        report.append(f"  平均召回率: {self.validation_results['recall']:.3f}")
        
        # 各类别性能
        if self.validation_results.get('class_ap50'):
            report.append("\n📈 各类别AP@0.5:")
            for class_result in self.validation_results['class_ap50']:
                report.append(f"  {class_result['class_name']}: {class_result['ap50']:.3f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_training_report(self, save_path: str):
        """
        保存训练报告
        
        Args:
            save_path: 保存路径
        """
        report = self.generate_training_report()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 训练报告已保存: {save_path}")
    
    def _print_training_config(self):
        """打印训练配置"""
        print("📋 训练配置:")
        print(f"   模型大小: {self.config.model_size}")
        print(f"   训练轮数: {self.config.epochs}")
        print(f"   图像尺寸: {self.config.img_size}")
        print(f"   批次大小: {self.config.batch_size}")
        print(f"   优化器: {self.config.optimizer}")
        print(f"   初始学习率: {self.config.lr0}")
        print(f"   设备: {self.device}")
        
        # 增强参数
        print("🎨 数据增强:")
        print(f"   Mosaic: {self.config.mosaic}")
        print(f"   Mixup: {self.config.mixup}")
        print(f"   旋转: {self.config.degrees}°")
        print(f"   缩放: {self.config.scale}")
        
        # 损失函数
        print("📉 损失函数:")
        print(f"   Box增益: {self.config.box_gain}")
        print(f"   Class增益: {self.config.cls_gain}")
        print(f"   DFL增益: {self.config.dfl_gain}")
    
    def run_full_pipeline(self, resume: bool = False, export_format: str = 'onnx') -> Dict[str, Any]:
        """
        运行完整的训练流程
        
        Args:
            resume: 是否恢复训练
            export_format: 导出格式
            
        Returns:
            完整结果字典
        """
        print("\n" + "=" * 60)
        print("🎯 开始完整训练流程")
        print("=" * 60)
        
        try:
            # 1. 准备训练
            dataset_stats = self.prepare_training()
            
            # 2. 创建或加载模型
            model = self.create_or_load_model()
            
            # 3. 训练
            model, training_results = self.train(resume=resume)
            
            # 4. 验证
            validation_results = self.validate()
            
            # 5. 导出模型
            export_path = self.export_model(format=export_format)
            
            # 6. 生成报告
            report = self.generate_training_report()
            
            # 构建完整结果
            full_results = {
                'dataset_stats': dataset_stats,
                'training_results': training_results,
                'validation_results': validation_results,
                'export_path': export_path,
                'report': report,
                'config': self.config.to_dict()
            }
            
            print("\n" + "=" * 60)
            print("🎉 完整训练流程完成!")
            print("=" * 60)
            
            return full_results
            
        except Exception as e:
            print(f"❌ 训练流程失败: {e}")
            raise