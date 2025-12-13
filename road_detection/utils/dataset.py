#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分析工具模块
Dataset analysis utility module
"""

import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

def get_dataset_stats(data_yaml_path: str) -> Dict[str, any]:
    """
    从数据配置文件中获取统计信息
    
    Args:
        data_yaml_path: 数据配置文件路径
        
    Returns:
        数据集统计信息字典
    """
    try:
        print(f"📊 正在分析数据集配置: {data_yaml_path}")
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 获取类别信息
        nc = data_config.get('nc', 0)
        names = data_config.get('names', [])
        
        print(f"🎯 类别数量: {nc}")
        print(f"🏷️  类别名称: {names}")
        
        # 获取基础路径
        yaml_dir = Path(data_yaml_path).parent
        base_path = yaml_dir
        
        print(f"📂 YAML文件所在目录: {base_path}")
        
        # 计算训练和验证图片数量
        def count_images_and_labels(train_val_path):
            """计算指定路径下的图片和标签数量"""
            if not train_val_path:
                return 0, 0
            
            # 构建完整的图片路径
            image_path = base_path / train_val_path
            
            print(f"\n🔍 检查路径: {image_path}")
            
            if not image_path.exists():
                print(f"❌ 路径不存在: {image_path}")
                return 0, 0
            
            # 统计图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            image_files = []
            
            # 获取所有文件，然后按扩展名过滤
            for file_path in image_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_files.append(file_path)
            
            # 去重（按文件名）
            unique_files = list(set(image_files))
            total_images = len(unique_files)
            print(f"📸 找到图片文件: {total_images} 张")
            
            # 检查对应的标签路径
            label_path = Path(str(image_path).replace('images', 'labels'))
            print(f"🏷️  标签路径: {label_path}")
            
            if label_path.exists():
                # 统计标签文件
                label_files = list(label_path.rglob('*.txt'))
                total_labels = len(label_files)
                print(f"📝 找到标签文件: {total_labels} 个")
                
                # 检查匹配情况
                if total_images > 0:
                    match_ratio = (total_labels / total_images) * 100
                    print(f"✅ 图片-标签匹配率: {match_ratio:.1f}%")
                    
                    if match_ratio < 100:
                        print(f"⚠️  警告: {total_images - total_labels} 张图片缺少标签文件")
                        
                        # 列出前10个没有标签的图片
                        missing_labels = []
                        for img_file in image_files[:10]:  # 只检查前10个
                            expected_label = label_path / (img_file.stem + '.txt')
                            if not expected_label.exists():
                                missing_labels.append(img_file.name)
                        
                        if missing_labels:
                            print(f"   缺失标签的图片示例: {missing_labels[:5]}")
            else:
                print(f"❌ 标签目录不存在: {label_path}")
                total_labels = 0
            
            return total_images, total_labels
        
        # 统计训练集
        train_path = data_config.get('train', 'images/train')
        train_images, train_labels = count_images_and_labels(train_path)
        
        # 统计验证集
        val_path = data_config.get('val', 'images/val')
        val_images, val_labels = count_images_and_labels(val_path)
        
        # 总计
        total_images = train_images + val_images
        total_labels = train_labels + val_labels
        
        print(f"\n数据集统计总结:")
        print("=" * 60)
        print(f"训练集: {train_images} 张图片, {train_labels} 个标签")
        print(f"验证集: {val_images} 张图片, {val_labels} 个标签")
        print(f"总计: {total_images} 张图片, {total_labels} 个标签")
        
        # YOLO训练时的实际使用数量（有标签的图片）
        usable_train = min(train_images, train_labels)
        usable_val = min(val_images, val_labels)
        usable_total = usable_train + usable_val
        
        print(f"\nYOLO训练实际可用:")
        print(f"   训练集: {usable_train} 张图片")
        print(f"   验证集: {usable_val} 张图片")
        print(f"   总计: {usable_total} 张图片")
        
        if usable_total < total_images:
            print(f"警告: 由于缺少标签文件，YOLO将只使用 {usable_total}/{total_images} 张图片")
        
        return {
            'train_count': train_labels,  # 实际有标签的训练图片数量
            'val_count': val_labels,      # 实际有标签的验证图片数量
            'total_images': total_images, # 总图片数量
            'total_labels': total_labels, # 总标签数量
            'num_classes': nc,
            'class_names': names
        }
        
    except Exception as e:
        print(f"⚠️  获取数据集统计信息失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'train_count': 0,
            'val_count': 0,
            'total_images': 0,
            'total_labels': 0,
            'num_classes': 0,
            'class_names': []
        }

class DatasetAnalyzer:
    """数据集分析器类"""
    
    def __init__(self, data_yaml_path: str):
        """
        初始化数据集分析器
        
        Args:
            data_yaml_path: 数据配置文件路径
        """
        self.data_yaml_path = data_yaml_path
        self.data_config = None
        self.base_path = None
        self.class_distribution = None
        
        self._load_config()
    
    def _load_config(self):
        """加载数据配置"""
        try:
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                self.data_config = yaml.safe_load(f)
            
            self.base_path = Path(self.data_yaml_path).parent
            print(f"✅ 数据配置加载成功: {self.data_yaml_path}")
            
        except Exception as e:
            print(f"❌ 数据配置加载失败: {e}")
            raise
    
    def analyze_class_distribution(self) -> Dict[int, int]:
        """
        分析类别分布
        
        Returns:
            类别分布字典 {class_id: count}
        """
        print("📊 分析类别分布...")
        
        class_counts = Counter()
        total_instances = 0
        
        # 获取类别数量
        num_classes = self.data_config.get('nc', 0)
        
        # 分析训练集
        train_label_dir = self.base_path / 'labels' / 'train'
        if train_label_dir.exists():
            for txt_file in train_label_dir.glob("*.txt"):
                with open(txt_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            if class_id < num_classes:
                                class_counts[class_id] += 1
                                total_instances += 1
        
        # 分析验证集
        val_label_dir = self.base_path / 'labels' / 'val'
        if val_label_dir.exists():
            for txt_file in val_label_dir.glob("*.txt"):
                with open(txt_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            if class_id < num_classes:
                                class_counts[class_id] += 1
                                total_instances += 1
        
        # 打印分布统计
        class_names = self.data_config.get('names', [])
        print("类别分布统计:")
        for class_id in range(num_classes):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            class_name = class_names[class_id] if class_id < len(class_names) else f"类别{class_id}"
            print(f"  {class_name} (类别{class_id}): {count} 个实例 ({percentage:.1f}%)")
        
        self.class_distribution = dict(class_counts)
        return self.class_distribution
    
    def calculate_class_weights(self) -> Dict[int, float]:
        """
        计算类别权重（用于平衡损失函数）
        
        Returns:
            类别权重字典 {class_id: weight}
        """
        if self.class_distribution is None:
            self.analyze_class_distribution()
        
        num_classes = self.data_config.get('nc', 0)
        total_instances = sum(self.class_distribution.values())
        
        class_weights = {}
        
        print("\n类别权重计算:")
        for class_id in range(num_classes):
            count = self.class_distribution.get(class_id, 0)
            
            # 计算权重（实例数越少，权重越高）
            if count > 0:
                class_weights[class_id] = total_instances / (num_classes * count)
            else:
                class_weights[class_id] = 1.0
            
            class_name = self.data_config['names'][class_id] if class_id < len(self.data_config['names']) else f"类别{class_id}"
            print(f"  {class_name}: {class_weights[class_id]:.3f}")
        
        return class_weights
    
    def analyze_image_sizes(self) -> Dict[str, any]:
        """
        分析图片尺寸分布
        
        Returns:
            图片尺寸统计信息
        """
        print("📐 分析图片尺寸分布...")
        
        image_sizes = []
        
        # 分析训练集图片
        train_img_dir = self.base_path / 'images' / 'train'
        if train_img_dir.exists():
            for img_file in train_img_dir.glob("*.jpg"):
                try:
                    import cv2
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        image_sizes.append((w, h))
                except:
                    continue
        
        # 分析验证集图片
        val_img_dir = self.base_path / 'images' / 'val'
        if val_img_dir.exists():
            for img_file in val_img_dir.glob("*.jpg"):
                try:
                    import cv2
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        image_sizes.append((w, h))
                except:
                    continue
        
        if image_sizes:
            widths, heights = zip(*image_sizes)
            
            stats = {
                'total_images': len(image_sizes),
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'size_distribution': image_sizes
            }
            
            print(f"图片尺寸统计:")
            print(f"  总图片数: {stats['total_images']}")
            print(f"  平均尺寸: {stats['avg_width']:.0f} x {stats['avg_height']:.0f}")
            print(f"  尺寸范围: {stats['min_width']}x{stats['min_height']} - {stats['max_width']}x{stats['max_height']}")
            
            return stats
        else:
            print("⚠️  未找到有效的图片文件")
            return {'total_images': 0}
    
    def generate_analysis_report(self) -> str:
        """
        生成分析报告
        
        Returns:
            分析报告字符串
        """
        print("📝 生成数据集分析报告...")
        
        # 基础统计
        basic_stats = get_dataset_stats(self.data_yaml_path)
        
        # 类别分布
        if self.class_distribution is None:
            self.analyze_class_distribution()
        
        # 类别权重
        class_weights = self.calculate_class_weights()
        
        # 图片尺寸
        image_stats = self.analyze_image_sizes()
        
        # 生成报告
        report = []
        report.append("=" * 60)
        report.append("道路病害检测数据集分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 基础统计
        report.append("📊 基础统计:")
        report.append(f"  训练图片: {basic_stats['train_count']} 张")
        report.append(f"  验证图片: {basic_stats['val_count']} 张")
        report.append(f"  类别数量: {basic_stats['num_classes']}")
        report.append(f"  类别名称: {', '.join(basic_stats['class_names'])}")
        report.append("")
        
        # 类别分布
        report.append("📈 类别分布:")
        total_instances = sum(self.class_distribution.values())
        for class_id, count in self.class_distribution.items():
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            class_name = self.data_config['names'][class_id] if class_id < len(self.data_config['names']) else f"类别{class_id}"
            report.append(f"  {class_name}: {count} 个实例 ({percentage:.1f}%)")
        report.append("")
        
        # 类别权重
        report.append("⚖️ 类别权重:")
        for class_id, weight in class_weights.items():
            class_name = self.data_config['names'][class_id] if class_id < len(self.data_config['names']) else f"类别{class_id}"
            report.append(f"  {class_name}: {weight:.3f}")
        report.append("")
        
        # 图片尺寸
        if image_stats['total_images'] > 0:
            report.append("📐 图片尺寸:")
            report.append(f"  总图片数: {image_stats['total_images']}")
            report.append(f"  平均尺寸: {image_stats['avg_width']:.0f} x {image_stats['avg_height']:.0f}")
            report.append(f"  尺寸范围: {image_stats['min_width']}x{image_stats['min_height']} - {image_stats['max_width']}x{image_stats['max_height']}")
            report.append("")
        
        # 训练建议
        report.append("💡 训练建议:")
        max_count = max(self.class_distribution.values())
        min_count = min(self.class_distribution.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            report.append(f"  ⚠️  检测到类别不平衡 (比例: {imbalance_ratio:.1f}:1)")
            report.append("     建议: 使用类别加权损失函数或过采样策略")
        else:
            report.append("  ✅ 类别分布相对均衡")
        
        if image_stats['total_images'] > 0:
            avg_size = (image_stats['avg_width'] + image_stats['avg_height']) / 2
            if avg_size > 1000:
                report.append("  📏 图片尺寸较大，建议使用较大的输入尺寸")
            elif avg_size < 500:
                report.append("  📏 图片尺寸较小，可以使用较小的输入尺寸")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)