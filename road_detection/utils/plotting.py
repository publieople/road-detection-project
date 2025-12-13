#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图工具模块
Plotting utility module
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional

def setup_chinese_fonts():
    """
    配置matplotlib中文字体支持
    """
    try:
        # Windows系统常见中文字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        ]
        
        # 查找可用的中文字体
        available_fonts = []
        for font_path in font_paths:
            if Path(font_path).exists():
                available_fonts.append(font_path)
        
        if available_fonts:
            primary_font = available_fonts[0]
            font_name = Path(primary_font).stem
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            fm.fontManager.addfont(primary_font)
            print(f"✅ 中文字体配置成功: {font_name}")
        else:
            print("⚠️  未找到中文字体，图表中的中文可能显示为方块")
    
    except Exception as e:
        print(f"⚠️  中文字体配置失败: {e}")

def plot_training_curves(results_csv: str, save_path: Optional[str] = None, 
                        show_plot: bool = False) -> Optional[plt.Figure]:
    """
    绘制训练曲线
    
    Args:
        results_csv: 训练结果CSV文件路径
        save_path: 保存路径
        show_plot: 是否显示图表
        
    Returns:
        图表对象
    """
    try:
        import pandas as pd
        
        # 读取CSV文件
        df = pd.read_csv(results_csv)
        
        if df.empty:
            print("⚠️  训练结果文件为空")
            return None
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程分析', fontsize=16)
        
        # 1. 损失函数曲线
        ax1 = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/box_loss'], label='训练Box损失', color='blue', alpha=0.7)
        if 'val/box_loss' in df.columns:
            ax1.plot(df['epoch'], df['val/box_loss'], label='验证Box损失', color='red', alpha=0.7)
        if 'train/cls_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/cls_loss'], label='训练分类损失', color='green', alpha=0.7)
        if 'val/cls_loss' in df.columns:
            ax1.plot(df['epoch'], df['val/cls_loss'], label='验证分类损失', color='orange', alpha=0.7)
        
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失值')
        ax1.set_title('损失函数曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. mAP曲线
        ax2 = axes[0, 1]
        if 'metrics/mAP50(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='blue', linewidth=2)
        if 'metrics/mAP50-95(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='red', linewidth=2)
        
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('mAP值')
        ax2.set_title('mAP性能曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 精确率和召回率
        ax3 = axes[1, 0]
        if 'metrics/precision(B)' in df.columns:
            ax3.plot(df['epoch'], df['metrics/precision(B)'], label='精确率', color='blue', alpha=0.7)
        if 'metrics/recall(B)' in df.columns:
            ax3.plot(df['epoch'], df['metrics/recall(B)'], label='召回率', color='red', alpha=0.7)
        
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('指标值')
        ax3.set_title('精确率与召回率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 学习率曲线
        ax4 = axes[1, 1]
        if 'lr/pg0' in df.columns:
            ax4.plot(df['epoch'], df['lr/pg0'], label='学习率', color='purple', alpha=0.7)
        
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('学习率')
        ax4.set_title('学习率变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        print(f"❌ 绘制训练曲线失败: {e}")
        return None

def plot_class_distribution(class_counts: Dict[int, int], class_names: List[str], 
                          save_path: Optional[str] = None, show_plot: bool = False) -> Optional[plt.Figure]:
    """
    绘制类别分布图
    
    Args:
        class_counts: 类别统计 {class_id: count}
        class_names: 类别名称列表
        save_path: 保存路径
        show_plot: 是否显示图表
        
    Returns:
        图表对象
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('类别分布分析', fontsize=16)
        
        # 准备数据
        class_ids = list(class_counts.keys())
        counts = list(class_counts.values())
        total_instances = sum(counts)
        
        # 类别标签
        labels = []
        for class_id in class_ids:
            if class_id < len(class_names):
                labels.append(f"{class_names[class_id]}({class_id})")
            else:
                labels.append(f"类别{class_id}")
        
        # 1. 柱状图
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_ids)))
        bars = ax1.bar(labels, counts, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        ax1.set_xlabel('类别')
        ax1.set_ylabel('实例数量')
        ax1.set_title('各类别实例数量分布')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 饼图
        percentages = [count/total_instances*100 for count in counts]
        
        # 只显示占比大于1%的类别
        filtered_labels = []
        filtered_percentages = []
        other_percentage = 0
        
        for label, percentage in zip(labels, percentages):
            if percentage > 1:
                filtered_labels.append(label)
                filtered_percentages.append(percentage)
            else:
                other_percentage += percentage
        
        if other_percentage > 0:
            filtered_labels.append('其他')
            filtered_percentages.append(other_percentage)
        
        ax2.pie(filtered_percentages, labels=filtered_labels, autopct='%1.1f%%',
                startangle=90, colors=colors[:len(filtered_labels)])
        ax2.set_title('类别占比分布')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 类别分布图已保存: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        print(f"❌ 绘制类别分布图失败: {e}")
        return None

def plot_validation_comparison(results_list: List[Dict[str, Any]], model_names: List[str],
                             save_path: Optional[str] = None, show_plot: bool = False) -> Optional[plt.Figure]:
    """
    绘制多个模型的验证结果对比
    
    Args:
        results_list: 验证结果列表
        model_names: 模型名称列表
        save_path: 保存路径
        show_plot: 是否显示图表
        
    Returns:
        图表对象
    """
    try:
        if len(results_list) != len(model_names):
            raise ValueError("结果列表和模型名称列表长度必须相同")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('模型性能对比分析', fontsize=16)
        
        # 提取指标
        map50_values = [r['mAP50'] for r in results_list]
        map5095_values = [r['mAP5095'] for r in results_list]
        precision_values = [r['precision'] for r in results_list]
        recall_values = [r['recall'] for r in results_list]
        
        # 1. mAP对比
        ax1 = axes[0, 0]
        x_pos = np.arange(len(model_names))
        
        ax1.bar(x_pos - 0.2, map50_values, 0.4, label='mAP@0.5', alpha=0.8)
        ax1.bar(x_pos + 0.2, map5095_values, 0.4, label='mAP@0.5:0.95', alpha=0.8)
        
        ax1.set_xlabel('模型')
        ax1.set_ylabel('mAP值')
        ax1.set_title('mAP性能对比')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (map50, map5095) in enumerate(zip(map50_values, map5095_values)):
            ax1.text(i - 0.2, map50 + 0.01, f'{map50:.3f}', ha='center', va='bottom')
            ax1.text(i + 0.2, map5095 + 0.01, f'{map5095:.3f}', ha='center', va='bottom')
        
        # 2. 精确率和召回率对比
        ax2 = axes[0, 1]
        
        ax2.bar(x_pos - 0.2, precision_values, 0.4, label='精确率', alpha=0.8)
        ax2.bar(x_pos + 0.2, recall_values, 0.4, label='召回率', alpha=0.8)
        
        ax2.set_xlabel('模型')
        ax2.set_ylabel('指标值')
        ax2.set_title('精确率与召回率对比')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (precision, recall) in enumerate(zip(precision_values, recall_values)):
            ax2.text(i - 0.2, precision + 0.01, f'{precision:.3f}', ha='center', va='bottom')
            ax2.text(i + 0.2, recall + 0.01, f'{recall:.3f}', ha='center', va='bottom')
        
        # 3. 综合性能雷达图
        ax3 = axes[1, 0]
        
        # 归一化到0-1范围
        all_values = map50_values + map5095_values + precision_values + recall_values
        max_val = max(all_values) if all_values else 1
        
        normalized_map50 = [v/max_val for v in map50_values]
        normalized_map5095 = [v/max_val for v in map5095_values]
        normalized_precision = [v/max_val for v in precision_values]
        normalized_recall = [v/max_val for v in recall_values]
        
        # 绘制每个模型的雷达图
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for i, name in enumerate(model_names):
            values = [normalized_map50[i], normalized_map5095[i], 
                     normalized_precision[i], normalized_recall[i]]
            
            # 闭合图形
            values += values[:1]
            
            # 角度
            angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
            angles += angles[:1]
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
            ax3.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(['mAP@0.5', 'mAP@0.5:0.95', '精确率', '召回率'])
        ax3.set_ylim(0, 1)
        ax3.set_title('综合性能雷达图')
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax3.grid(True)
        
        # 4. 性能提升分析
        ax4 = axes[1, 1]
        
        if len(results_list) > 1:
            # 计算相对于第一个模型的改进
            baseline_map50 = map50_values[0]
            baseline_map5095 = map5095_values[0]
            
            map50_improvements = [(v - baseline_map50) / baseline_map50 * 100 for v in map50_values[1:]]
            map5095_improvements = [(v - baseline_map5095) / baseline_map5095 * 100 for v in map5095_values[1:]]
            
            x_pos_improve = np.arange(1, len(model_names))
            
            ax4.bar(x_pos_improve - 0.2, map50_improvements, 0.4, 
                   label='mAP@0.5改进(%)', alpha=0.8)
            ax4.bar(x_pos_improve + 0.2, map5095_improvements, 0.4, 
                   label='mAP@0.5:0.95改进(%)', alpha=0.8)
            
            ax4.set_xlabel('模型')
            ax4.set_ylabel('改进百分比 (%)')
            ax4.set_title('相对于基线模型的改进')
            ax4.set_xticks(x_pos_improve)
            ax4.set_xticklabels(model_names[1:], rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 模型对比图已保存: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        
        return fig
        
    except Exception as e:
        print(f"❌ 绘制模型对比图失败: {e}")
        return None

def save_plot(fig: plt.Figure, save_path: str, dpi: int = 300):
    """
    保存图表
    
    Args:
        fig: 图表对象
        save_path: 保存路径
        dpi: 分辨率
    """
    try:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"📊 图表已保存: {save_path}")
    except Exception as e:
        print(f"❌ 保存图表失败: {e}")

def close_plots():
    """关闭所有图表"""
    plt.close('all')