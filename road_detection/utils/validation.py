#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证工具模块
Model validation utility module
"""

from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

def validate_model(model_path: str, data_yaml_path: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    验证模型性能

    Args:
        model_path: 模型文件路径
        data_yaml_path: 数据配置文件路径
        save_dir: 结果保存目录

    Returns:
        验证结果字典
    """
    print("🔍 开始模型验证...")

    try:
        # 加载模型
        model = YOLO(model_path)

        # 执行验证
        metrics = model.val(data=data_yaml_path, workers=0)

        # 获取类别信息
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])

        # 提取关键指标
        results = {
            'mAP50': metrics.box.map50,
            'mAP5095': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'class_names': class_names,
            'class_ap50': []
        }

        # 获取每个类别的AP@0.5
        if hasattr(metrics.box, 'ap50'):
            for i, ap50 in enumerate(metrics.box.ap50):
                if i < len(class_names):
                    results['class_ap50'].append({
                        'class_id': i,
                        'class_name': class_names[i],
                        'ap50': float(ap50)
                    })

        # 打印验证结果
        print("📊 验证结果:")
        print(f"   mAP@0.5: {results['mAP50']:.3f}")
        print(f"   mAP@0.5:0.95: {results['mAP5095']:.3f}")
        print(f"   平均精确率: {results['precision']:.3f}")
        print(f"   平均召回率: {results['recall']:.3f}")

        # 打印每个类别的性能
        if results['class_ap50']:
            print("\n📈 各类别AP@0.5:")
            for class_result in results['class_ap50']:
                print(f"   {class_result['class_name']}: {class_result['ap50']:.3f}")

        # 保存验证结果
        if save_dir:
            save_validation_results(results, save_dir)

        print("✅ 模型验证完成!")
        return results

    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        raise

def save_validation_results(results: Dict[str, Any], save_dir: str):
    """
    保存验证结果

    Args:
        results: 验证结果字典
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存为YAML文件
    yaml_path = save_path / "validation_results.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, allow_unicode=True, sort_keys=False)

    # 保存为文本报告
    report_path = save_path / "validation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("道路病害检测模型验证报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"mAP@0.5: {results['mAP50']:.3f}\n")
        f.write(f"mAP@0.5:0.95: {results['mAP5095']:.3f}\n")
        f.write(f"平均精确率: {results['precision']:.3f}\n")
        f.write(f"平均召回率: {results['recall']:.3f}\n\n")

        if results['class_ap50']:
            f.write("各类别AP@0.5:\n")
            for class_result in results['class_ap50']:
                f.write(f"  {class_result['class_name']}: {class_result['ap50']:.3f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"验证时间: {Path(save_dir).name}\n")

    print(f"📁 验证结果已保存到: {save_dir}")

class ModelValidator:
    """模型验证器类"""

    def __init__(self, model_path: str, data_yaml_path: str):
        """
        初始化模型验证器

        Args:
            model_path: 模型文件路径
            data_yaml_path: 数据配置文件路径
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.model = None
        self.validation_results = None

        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def validate(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行验证

        Args:
            save_dir: 结果保存目录

        Returns:
            验证结果
        """
        print("🔍 开始模型验证...")

        try:
            # 执行验证（Windows上禁用多进程workers避免崩溃）
            metrics = self.model.val(data=self.data_yaml_path, workers=0)

            # 获取类别信息
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get('names', [])

            # 构建结果
            self.validation_results = {
                'mAP50': metrics.box.map50,
                'mAP5095': metrics.box.map,
                'precision': metrics.box.mp,
                'recall': metrics.box.mr,
                'class_names': class_names,
                'class_ap50': []
            }

            # 获取每个类别的AP@0.5
            if hasattr(metrics.box, 'ap50'):
                for i, ap50 in enumerate(metrics.box.ap50):
                    if i < len(class_names):
                        self.validation_results['class_ap50'].append({
                            'class_id': i,
                            'class_name': class_names[i],
                            'ap50': float(ap50)
                        })

            # 打印结果
            self._print_results()

            # 保存结果
            if save_dir:
                self.save_results(save_dir)

            print("✅ 模型验证完成!")
            return self.validation_results

        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            raise

    def _print_results(self):
        """打印验证结果"""
        if not self.validation_results:
            return

        print("📊 验证结果:")
        print(f"   mAP@0.5: {self.validation_results['mAP50']:.3f}")
        print(f"   mAP@0.5:0.95: {self.validation_results['mAP5095']:.3f}")
        print(f"   平均精确率: {self.validation_results['precision']:.3f}")
        print(f"   平均召回率: {self.validation_results['recall']:.3f}")

        # 打印每个类别的性能
        if self.validation_results['class_ap50']:
            print("\n📈 各类别AP@0.5:")
            for class_result in self.validation_results['class_ap50']:
                print(f"   {class_result['class_name']}: {class_result['ap50']:.3f}")

    def save_results(self, save_dir: str):
        """
        保存验证结果

        Args:
            save_dir: 保存目录
        """
        if not self.validation_results:
            print("⚠️  没有验证结果可保存")
            return

        save_validation_results(self.validation_results, save_dir)

    def check_performance_target(self, target_map50: float = 0.80) -> bool:
        """
        检查是否达到性能目标

        Args:
            target_map50: 目标mAP@0.5

        Returns:
            是否达到目标
        """
        if not self.validation_results:
            print("⚠️  请先执行验证")
            return False

        current_map50 = self.validation_results['mAP50']

        if current_map50 >= target_map50:
            print(f"🎯 目标达成！模型准确率 ≥ {target_map50:.0%}")
            print(f"   当前mAP@0.5: {current_map50:.3f}")
            return True
        else:
            print(f"⚠️  未达目标。当前准确率: {current_map50:.1%}, 目标: {target_map50:.0%}")
            print("💡 建议: 增加训练轮数、调整超参数或收集更多数据")
            return False

    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        与基线模型比较

        Args:
            baseline_results: 基线模型结果

        Returns:
            比较结果
        """
        if not self.validation_results or not baseline_results:
            print("⚠️  缺少比较数据")
            return {}

        comparison = {
            'mAP50_improvement': self.validation_results['mAP50'] - baseline_results['mAP50'],
            'mAP5095_improvement': self.validation_results['mAP5095'] - baseline_results['mAP5095'],
            'precision_improvement': self.validation_results['precision'] - baseline_results['precision'],
            'recall_improvement': self.validation_results['recall'] - baseline_results['recall']
        }

        print("📊 与基线模型比较:")
        print(f"   mAP@0.5 改进: {comparison['mAP50_improvement']:+.3f}")
        print(f"   mAP@0.5:0.95 改进: {comparison['mAP5095_improvement']:+.3f}")
        print(f"   精确率 改进: {comparison['precision_improvement']:+.3f}")
        print(f"   召回率 改进: {comparison['recall_improvement']:+.3f}")

        return comparison