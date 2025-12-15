#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证集失败样本处理脚本 - 根据模型推理失败结果处理验证失败的样本
Failed Validation Filter - Remove samples where model inference fails
"""

import argparse
import sys
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

def load_dataset_config(yaml_path: str) -> Dict:
    """加载YOLO格式数据集配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_yolo_labels(label_path: str) -> List[np.ndarray]:
    """
    加载YOLO格式标签 (cx, cy, w, h, class_id)
    返回: [(class_id, cx, cy, w, h), ...]
    """
    boxes = []
    if not Path(label_path).exists():
        return boxes

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                if len(parts) >= 5:
                    boxes.append(parts[:5])  # class_id, cx, cy, w, h
    except Exception as e:
        print(f"⚠️  读取标签失败 {label_path}: {e}")

    return boxes

def has_class_in_whitelist(label_path: str, class_whitelist: Optional[List[int]] = None) -> bool:
    """
    检查样本是否包含白名单中的类别

    Args:
        label_path: 标签文件路径
        class_whitelist: 类别白名单 [0, 1, 2, ...] 或 None 表示不做过滤

    Returns:
        如果白名单为空/None返回True（不过滤）
        如果样本中有白名单内的类别返回True
        否则返回False
    """
    # 如果没有设置白名单，默认接受所有样本
    if class_whitelist is None or len(class_whitelist) == 0:
        return True

    # 转换为集合以加快查找
    whitelist_set = set(class_whitelist)

    # 加载标签并检查
    boxes = load_yolo_labels(label_path)
    for box in boxes:
        class_id = int(box[0])
        if class_id in whitelist_set:
            return True

    return False

def normalize_box(box: np.ndarray, img_width: int, img_height: int) -> Tuple[int, int, int, int, int]:
    """
    将YOLO格式归一化坐标转换为像素坐标
    YOLO格式: (cx, cy, w, h) - 都是相对于图像尺寸的比例 [0, 1]
    输出: (class_id, x1, y1, x2, y2) - 像素坐标
    """
    class_id, cx, cy, w, h = box

    x1 = int((cx - w/2) * img_width)
    y1 = int((cy - h/2) * img_height)
    x2 = int((cx + w/2) * img_width)
    y2 = int((cy + h/2) * img_height)

    return int(class_id), x1, y1, x2, y2

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个(x1,y1,x2,y2)格式的框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # 计算交集
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # 计算并集
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def match_predictions_to_labels(preds: List[np.ndarray], labels: List[np.ndarray],
                                iou_threshold: float = 0.5) -> Tuple[List[float], List[int]]:
    """
    匹配预测框和标签框，计算IoU

    Args:
        preds: 预测框列表 [(x1,y1,x2,y2,conf), ...]
        labels: 标签框列表 [(class_id,x1,y1,x2,y2), ...]
        iou_threshold: IoU阈值

    Returns:
        (匹配IoU列表, 未匹配标签索引列表)
    """
    matched_ious = []
    unmatched_label_indices = list(range(len(labels)))

    # 对每个预测框，找最佳匹配的标签框
    for pred in preds:
        best_iou = 0.0
        best_label_idx = -1

        for label_idx, label in enumerate(labels):
            if label_idx not in unmatched_label_indices:
                continue

            iou = calculate_iou(pred, label)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = label_idx

        if best_label_idx >= 0 and best_iou >= iou_threshold:
            matched_ious.append(best_iou)
            unmatched_label_indices.remove(best_label_idx)

    return matched_ious, unmatched_label_indices

def infer_with_yolo(model, image_path: str, conf_threshold: float = 0.25) -> List[np.ndarray]:
    """
    使用YOLO模型推理图片

    Args:
        model: YOLO模型
        image_path: 图片路径
        conf_threshold: 置信度阈值

    Returns:
        预测框列表 [(x1,y1,x2,y2,conf,class_id), ...]
    """
    try:
        results = model(image_path, conf=conf_threshold, verbose=False)

        preds = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    preds.append(np.array([x1, y1, x2, y2, conf, class_id]))

        return preds
    except Exception as e:
        print(f"⚠️  推理失败 {image_path}: {e}")
        return []

def calculate_sample_difficulty(image_path: str, label_path: str, model,
                               iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    计算样本的难度指标

    Args:
        image_path: 图片路径
        label_path: 标签路径
        model: YOLO模型
        iou_threshold: IoU阈值

    Returns:
        难度指标字典
    """
    try:
        # 获取图片尺寸
        img = Image.open(image_path)
        img_width, img_height = img.size

        # 加载标签和推理
        gt_boxes_yolo = load_yolo_labels(label_path)
        preds = infer_with_yolo(model, image_path)

        # 转换坐标格式
        gt_boxes_pixel = []
        for box in gt_boxes_yolo:
            class_id, x1, y1, x2, y2 = normalize_box(box, img_width, img_height)
            gt_boxes_pixel.append(np.array([x1, y1, x2, y2, float(class_id)]))

        # 如果没有标签，难度设为中等
        if len(gt_boxes_pixel) == 0:
            return {
                'has_objects': False,
                'num_gt': 0,
                'num_pred': len(preds),
                'mean_iou': 0.5,
                'min_iou': 0.5,
                'max_conf': float(max([p[4] for p in preds])) if preds else 0.0,
                'mean_conf': float(np.mean([p[4] for p in preds])) if preds else 0.0,
                'unmatched_count': 0,
                'false_positive_count': len(preds)
            }

        # 如果有标签但无预测，难度很高
        if len(preds) == 0:
            return {
                'has_objects': True,
                'num_gt': len(gt_boxes_pixel),
                'num_pred': 0,
                'mean_iou': 0.0,
                'min_iou': 0.0,
                'max_conf': 0.0,
                'mean_conf': 0.0,
                'unmatched_count': len(gt_boxes_pixel),
                'false_positive_count': 0
            }

        # 匹配预测和标签
        matched_ious, unmatched_indices = match_predictions_to_labels(
            preds, gt_boxes_pixel, iou_threshold
        )

        # 计算指标
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        min_iou = float(np.min(matched_ious)) if matched_ious else 0.0
        max_conf = float(max([p[4] for p in preds]))
        mean_conf = float(np.mean([p[4] for p in preds]))
        unmatched_count = len(unmatched_indices)
        false_positive_count = len(preds) - len(matched_ious)

        return {
            'has_objects': True,
            'num_gt': len(gt_boxes_pixel),
            'num_pred': len(preds),
            'mean_iou': mean_iou,
            'min_iou': min_iou,
            'max_conf': max_conf,
            'mean_conf': mean_conf,
            'unmatched_count': unmatched_count,
            'false_positive_count': false_positive_count
        }

    except Exception as e:
        print(f"⚠️  计算难度失败 {image_path}: {e}")
        return {
            'has_objects': False,
            'num_gt': 0,
            'num_pred': 0,
            'mean_iou': 0.5,
            'min_iou': 0.5,
            'max_conf': 0.0,
            'mean_conf': 0.0,
            'unmatched_count': 0,
            'false_positive_count': 0,
            'error': str(e)
        }

def check_inference_pass(metrics: Dict[str, float],
                        iou_threshold: float = 0.5) -> Tuple[bool, str]:
    """
    判断样本的推理验证是否通过

    验证失败的情况：
    1. 有真实标签但模型未检测到 (num_pred == 0)
    2. 检测到但IoU过低 (mean_iou < iou_threshold)
    3. 漏检率过高 (未匹配的真实框过多)

    Args:
        metrics: 难度指标字典
        iou_threshold: IoU通过阈值

    Returns:
        (是否通过, 失败原因)
    """
    if 'error' in metrics:
        return False, "处理错误"

    # 情况1: 有标签但无预测 = 验证失败
    if metrics['has_objects'] and metrics['num_pred'] == 0:
        return False, "漏检:无预测"

    # 情况2: 无标签 = 验证通过
    if not metrics['has_objects'] and metrics['num_pred'] == 0:
        return True, "正确:无目标无预测"

    # 情况3: 有标签的情况下，IoU过低 = 验证失败
    if metrics['has_objects'] and metrics['mean_iou'] < iou_threshold:
        return False, f"IoU过低:{metrics['mean_iou']:.3f}"

    # 情况4: 漏检率过高 = 验证失败
    if metrics['unmatched_count'] > 0:
        miss_rate = metrics['unmatched_count'] / max(metrics['num_gt'], 1)
        if miss_rate > 0.3:  # 漏检超过30%
            return False, f"漏检率高:{miss_rate:.1%}"

    # 情况5: 误检过多 = 验证失败
    if metrics['false_positive_count'] > metrics['num_gt']:
        return False, f"误检过多:{metrics['false_positive_count']}"

    return True, "通过验证"



def scan_training_samples(config: Dict, yaml_dir: Path, model,
                         iou_threshold: float = 0.5) -> Dict[str, Dict[str, Any]]:
    """
    扫描训练集中所有样本的验证情况

    Args:
        config: 数据配置字典
        yaml_dir: 配置文件所在目录
        model: YOLO模型
        iou_threshold: IoU阈值

    Returns:
        {image_path: {'metrics': ..., 'is_pass': ..., 'reason': ...}, ...}
    """
    train_images_rel = config.get('train', 'images/train')
    train_images_path = yaml_dir / train_images_rel

    if not train_images_path.exists():
        print(f"⚠️  训练集路径不存在: {train_images_path}")
        return {}

    # 获取标签路径映射
    def get_label_path(image_path: Path) -> Path:
        rel_path = image_path.relative_to(train_images_path)
        labels_dir = yaml_dir / 'labels' / 'train'
        label_path = labels_dir / rel_path.with_suffix('.txt')
        return label_path

    # 列出所有训练图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in train_images_path.rglob('*')
                   if f.suffix.lower() in image_extensions and f.is_file()]

    print(f"\n📊 扫描训练集样本 ({len(image_files)}张图片)...")

    train_samples = {}
    passed_count = 0

    for idx, image_path in enumerate(image_files):
        if (idx + 1) % max(1, len(image_files) // 10) == 0:
            print(f"   进度: {idx + 1}/{len(image_files)}")

        label_path = get_label_path(image_path)
        metrics = calculate_sample_difficulty(image_path, str(label_path), model, iou_threshold)
        is_pass, reason = check_inference_pass(metrics, iou_threshold)

        train_samples[str(image_path)] = {
            'metrics': metrics,
            'is_pass': is_pass,
            'reason': reason
        }

        if is_pass:
            passed_count += 1

    print(f"   训练集中通过验证的样本: {passed_count}/{len(image_files)}")

    return train_samples


def filter_failed_validations(data_yaml_path: str, model_path: str,
                              action_prob: float = 1.0,
                              iou_threshold: float = 0.5,
                              action: str = 'move',
                              output_dir: Optional[str] = None,
                              backup: bool = True,
                              enable_replacement: bool = False,
                              include_classes: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    处理验证集中推理验证失败的样本

    Args:
        data_yaml_path: 数据配置文件路径
        model_path: YOLO模型路径
        action_prob: 操作概率 [0, 1]，验证失败的样本以此概率执行action操作
        iou_threshold: IoU阈值 [0, 1]，判断验证是否通过的标准
        action: 操作类型，'move'（默认）将失败样本移到训练集，'copy'复制到训练集，'delete'删除失败样本
        output_dir: 输出目录（保存失败样本信息）
        backup: 是否备份原数据
        enable_replacement: 是否启用替换功能（每移动一个失败样本，从训练集中移动一个通过验证的样本到验证集）
        include_classes: 类别白名单 [0, 1, 2, ...]，只处理包含这些类别的样本，None 或 [] 表示处理所有样本

    Returns:
        处理统计信息
    """
    from ultralytics import YOLO

    print("\n" + "=" * 80)
    action_text = "移动到训练集" if action == 'move' else "删除"
    print(f"🎯 验证集失败样本处理 - {action_text}")
    if enable_replacement:
        print(f"🔄 已启用替换功能 - 保持数据集比例")
    print("=" * 80)

    # 加载配置
    config = load_dataset_config(data_yaml_path)
    yaml_dir = Path(data_yaml_path).parent

    val_images_rel = config.get('val', 'images/val')
    val_images_path = yaml_dir / val_images_rel

    print(f"📂 验证集路径: {val_images_path}")
    print(f"🎲 操作概率: {action_prob:.1%}")
    print(f"⚙️  IoU阈值: {iou_threshold:.2f}")
    if include_classes:
        print(f"🏷️  类别白名单: {include_classes}")
    if enable_replacement:
        print(f"🔄 替换模式: 启用")

    if not val_images_path.exists():
        print(f"❌ 验证集路径不存在: {val_images_path}")
        return {}

    # 加载模型
    print(f"\n🚀 加载模型: {model_path}")
    try:
        model = YOLO(model_path)
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return {}

    # 如果启用替换功能，预先扫描训练集
    train_samples = {}
    if enable_replacement and action == 'move':
        train_samples = scan_training_samples(config, yaml_dir, model, iou_threshold)
        # 筛选出通过验证的训练集样本
        replaceable_samples = [
            path for path, info in train_samples.items()
            if info['is_pass']
        ]
        print(f"✅ 可替换的训练集样本: {len(replaceable_samples)}个")
        if not replaceable_samples:
            print("⚠️  没有通过验证的训练集样本，禁用替换功能")
            enable_replacement = False

    # 遍历验证集
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in val_images_path.rglob('*')
                   if f.suffix.lower() in image_extensions and f.is_file()]

    print(f"\n📊 处理{len(image_files)}张图片...")

    # 创建标签路径对应关系
    def get_label_path(image_path: Path) -> Path:
        """根据图片路径获取对应的标签路径"""
        rel_path = image_path.relative_to(val_images_path)

        # 获取标签目录
        if 'train' in val_images_path.parts:
            labels_dir = val_images_path.parent.parent / 'labels' / 'train'
        elif 'val' in val_images_path.parts:
            labels_dir = val_images_path.parent.parent / 'labels' / 'val'
        else:
            labels_dir = val_images_path.parent.parent / 'labels'

        label_path = labels_dir / rel_path.with_suffix('.txt')
        return label_path

    # 检查推理验证
    print("\n🔍 检查推理验证结果...")
    validation_results = {}
    passed_samples = []
    failed_samples = []

    for idx, image_path in enumerate(image_files):
        if (idx + 1) % max(1, len(image_files) // 10) == 0:
            print(f"   进度: {idx + 1}/{len(image_files)}")

        label_path = get_label_path(image_path)

        # 检查是否包含白名单中的类别
        if not has_class_in_whitelist(str(label_path), include_classes):
            # 跳过不包含白名单类别的样本
            validation_results[str(image_path)] = {
                'metrics': {},
                'is_pass': None,
                'reason': '不在类别白名单中，已跳过',
                'skipped_by_filter': True
            }
            continue

        metrics = calculate_sample_difficulty(image_path, str(label_path), model)
        is_pass, reason = check_inference_pass(metrics, iou_threshold)

        validation_results[str(image_path)] = {
            'metrics': metrics,
            'is_pass': is_pass,
            'reason': reason,
            'skipped_by_filter': False
        }

        if is_pass:
            passed_samples.append(str(image_path))
        else:
            failed_samples.append((str(image_path), reason))

    print(f"\n📈 验证统计:")
    print(f"   总样本数: {len(image_files)}")
    filtered_by_class = len([r for r in validation_results.values() if r.get('skipped_by_filter', False)])
    if filtered_by_class > 0:
        print(f"   类别过滤排除: {filtered_by_class}")
    print(f"   通过验证: {len(passed_samples)}")
    print(f"   验证失败: {len(failed_samples)}")

    if failed_samples:
        # 统计失败原因
        failure_reasons = defaultdict(int)
        for _, reason in failed_samples:
            failure_reasons[reason] += 1
        print(f"\n   失败原因分布:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"      {reason}: {count}个")

    # 备份原数据 - 递增备份名称
    if backup:
        # 找到下一个可用的备份编号
        backup_counter = 1
        backup_base = val_images_path.parent / f"val_backup"
        labels_backup_base = val_images_path.parent / "labels_backup_val"

        backup_dir = backup_base.with_name(f"{backup_base.name}_{backup_counter}")
        labels_backup_dir = labels_backup_base.with_name(f"{labels_backup_base.name}_{backup_counter}")

        # 找到第一个不存在的编号
        while backup_dir.exists() or labels_backup_dir.exists():
            backup_counter += 1
            backup_dir = backup_base.with_name(f"{backup_base.name}_{backup_counter}")
            labels_backup_dir = labels_backup_base.with_name(f"{labels_backup_base.name}_{backup_counter}")

        print(f"\n💾 备份验证集...")
        print(f"   图像备份: {backup_dir}")
        print(f"   标签备份: {labels_backup_dir}")

        # 备份图像
        shutil.copytree(val_images_path, backup_dir)

        # 备份标签
        labels_dir = val_images_path.parent.parent / 'labels' / 'val'
        if labels_dir.exists():
            shutil.copytree(labels_dir, labels_backup_dir)
        else:
            print(f"⚠️  标签文件夹不存在: {labels_dir}")


    # 处理验证失败的样本
    action_count = 0
    skipped_count = 0
    processed_samples = []
    replacement_count = 0
    replaced_samples = set()  # 记录已用于替换的训练集样本

    # 获取训练集路径（用于move操作）
    train_images_rel = config.get('train', 'images/train')
    train_images_path = yaml_dir / train_images_rel
    # 构造标签路径：datasets/yolo_format/labels/train (与images同级)
    train_labels_path = yaml_dir / 'labels' / 'train'

    action_verb = "移动" if action == 'move' else ("复制" if action == 'copy' else "删除")
    replacement_verb = ""
    if enable_replacement and action == 'move':
        replacement_verb = " + 替换"
    print(f"\n📋 {action_verb}验证失败的样本{replacement_verb} (概率={action_prob:.1%})...")

    if action in ['move', 'copy']:
        # 确保训练集目录存在
        train_images_path.mkdir(parents=True, exist_ok=True)
        train_labels_path.mkdir(parents=True, exist_ok=True)

    for image_path_str, reason in failed_samples:
        # 以操作概率随机决定是否执行操作
        if random.random() < action_prob:
            try:
                image_path_obj = Path(image_path_str)
                label_path = get_label_path(image_path_obj)

                if action == 'move':
                    # 移动到训练集
                    new_image_path = train_images_path / image_path_obj.name
                    new_label_path = train_labels_path / label_path.name

                    if image_path_obj.exists():
                        shutil.move(str(image_path_obj), str(new_image_path))
                    if label_path.exists():
                        shutil.move(str(label_path), str(new_label_path))

                    action_count += 1

                    # 执行替换操作
                    if enable_replacement and replaceable_samples:
                        # 筛选出未被使用过的可替换样本
                        available_samples = [
                            s for s in replaceable_samples
                            if s not in replaced_samples
                        ]

                        if available_samples:
                            # 随机选择一个训练集样本
                            selected_train_sample = random.choice(available_samples)
                            replaced_samples.add(selected_train_sample)

                            try:
                                selected_path = Path(selected_train_sample)

                                # 获取对应的标签路径
                                def get_train_label_path(img_path: Path) -> Path:
                                    rel_path = img_path.relative_to(train_images_path)
                                    label_path = train_labels_path / rel_path.with_suffix('.txt')
                                    return label_path

                                selected_label_path = get_train_label_path(selected_path)

                                # 移动到验证集
                                val_images_path.mkdir(parents=True, exist_ok=True)
                                val_labels_path = yaml_dir / 'labels' / 'val'
                                val_labels_path.mkdir(parents=True, exist_ok=True)

                                new_val_image_path = val_images_path / selected_path.name
                                new_val_label_path = val_labels_path / selected_label_path.name

                                if selected_path.exists():
                                    shutil.move(str(selected_path), str(new_val_image_path))
                                if selected_label_path.exists():
                                    shutil.move(str(selected_label_path), str(new_val_label_path))

                                replacement_count += 1

                            except Exception as e:
                                print(f"⚠️  替换操作失败 {selected_train_sample}: {e}")

                elif action == 'copy':
                    # 复制到训练集（保留原始验证集副本）
                    new_image_path = train_images_path / image_path_obj.name
                    new_label_path = train_labels_path / label_path.name

                    if image_path_obj.exists():
                        shutil.copy2(str(image_path_obj), str(new_image_path))
                    if label_path.exists():
                        shutil.copy2(str(label_path), str(new_label_path))

                    action_count += 1

                elif action == 'delete':
                    # 删除
                    if image_path_obj.exists():
                        image_path_obj.unlink()
                    if label_path.exists():
                        label_path.unlink()

                    action_count += 1

                processed_samples.append({
                    'image': str(image_path_obj),
                    'action': action,
                    'failure_reason': reason,
                    'metrics': validation_results[image_path_str]['metrics']
                })

            except Exception as e:
                print(f"⚠️  操作失败 {image_path_str}: {e}")
        else:
            skipped_count += 1


    # 统计结果
    remaining_images = len([f for f in val_images_path.rglob('*')
                           if f.suffix.lower() in image_extensions and f.is_file()])

    result = {
        'total_samples': len(image_files),
        'passed_samples': len(passed_samples),
        'failed_samples': len(failed_samples),
        'filtered_by_class': filtered_by_class,
        'action_performed': action_count,
        'action_skipped': skipped_count,
        'replacement_performed': replacement_count,
        'remaining_samples': remaining_images,
        'action_probability': action_prob,
        'action_type': action,
        'iou_threshold': iou_threshold,
        'include_classes': include_classes,
        'enable_replacement': enable_replacement,
        'validation_results': validation_results,
        'processed_sample_details': processed_samples
    }

    # 打印总结
    print(f"\n{'=' * 80}")
    print(f"✅ 处理完成!")
    print(f"{'=' * 80}")
    print(f"📊 统计信息:")
    print(f"   原始样本: {len(image_files)}")
    if filtered_by_class > 0:
        print(f"   类别过滤排除: {filtered_by_class}")
    print(f"   通过验证: {len(passed_samples)}")
    print(f"   验证失败: {len(failed_samples)}")
    action_text = "移动" if action == 'move' else ("复制" if action == 'copy' else "删除")
    print(f"   实际{action_text}: {action_count}")
    print(f"   跳过失败样本: {skipped_count}")
    if enable_replacement:
        print(f"   替换样本: {replacement_count}")
    print(f"   剩余验证集样本: {remaining_images}")
    if len(failed_samples) > 0:
        print(f"   总处理比例: {action_count / len(failed_samples) * 100:.1f}% (基于失败样本)")
    else:
        print(f"   总处理比例: 无失败样本")

    # 保存详细报告
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存JSON报告
        report_file = output_path / "filter_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            # 转换numpy类型以便JSON序列化
            serializable_result = {
                k: v for k, v in result.items()
                if k != 'sample_difficulties'
            }
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        print(f"\n📄 报告已保存: {report_file}")

    return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='验证集失败样本处理 - 根据模型推理结果处理验证失败的样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认行为：移动100%验证失败的样本到训练集（IoU>0.5则通过）
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model runs/detect/train/weights/best.pt

  # 启用替换功能：每移动1个失败样本，从训练集中移动1个通过验证的样本到验证集
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --enable-replacement

  # 只移动50%验证失败的样本（保留一些难例在验证集）
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --action-prob 0.5

  # 启用替换 + 50%概率移动：每次移动都替换，但只移动50%的失败样本
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --enable-replacement --action-prob 0.5

  # 更严格的验证标准（IoU>0.6），启用替换
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --iou-threshold 0.6 --enable-replacement

  # 复制验证失败的样本到训练集（保留原始验证集副本）
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --action copy

  # 复制50%验证失败的样本
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --action copy --action-prob 0.5

  # 删除验证失败的样本
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --action delete

  # 删除50%验证失败的样本
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --action delete --action-prob 0.5

  # 保存详细报告
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --output-dir ./filter_reports

  # 完整示例：启用替换、50%概率移动、严格验证、保存报告
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt \\
    --enable-replacement --action-prob 0.5 --iou-threshold 0.6 --output-dir ./filter_reports

  # 只处理包含类别0（裂缝）的验证失败样本
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --include-classes 0

  # 只处理包含类别0（裂缝）或类别2（坑槽）的验证失败样本
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt --include-classes 0,2

  # 类别过滤 + 替换功能：只处理包含类别1的样本，并启用替换
  python filter_hard_examples.py --data datasets/yolo_format/road.yaml --model best.pt \\
    --include-classes 1 --enable-replacement --action-prob 1.0
        """
    )

    parser.add_argument('--data', type=str, default='datasets/yolo_format/road.yaml',
                       help='数据配置文件路径 (默认: datasets/yolo_format/road.yaml)')
    parser.add_argument('--model', type=str, required=True,
                       help='YOLO模型路径 (如: runs/detect/train/weights/best.pt)')
    parser.add_argument('--action-prob', type=float, default=1.0,
                       help='操作概率 [0-1]，验证失败的样本以此概率被处理 (默认: 1.0=100%%)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU阈值 [0-1]，判断验证是否通过的标准 (默认: 0.5)')
    parser.add_argument('--action', type=str, default='move',
                       choices=['move', 'copy', 'delete'],
                       help='操作类型：move=移动到训练集 (默认), copy=复制到训练集, delete=删除样本')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录，保存处理报告 (可选)')
    parser.add_argument('--no-backup', action='store_true',
                       help='不备份原始数据')
    parser.add_argument('--enable-replacement', action='store_true',
                       help='启用替换功能：每移动一个验证失败样本，从训练集中移动一个通过验证的样本到验证集，保持数据集比例不变')
    parser.add_argument('--include-classes', type=str, default=None,
                       help='类别白名单，逗号分隔的整数列表 (如: 0,2 表示只处理包含类别0或2的样本)。默认为None表示处理所有类别')

    args = parser.parse_args()

    # 检查文件
    if not Path(args.data).exists():
        print(f"❌ 数据配置文件不存在: {args.data}")
        return 1

    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        return 1

    # 验证参数
    if not (0 <= args.action_prob <= 1):
        print(f"❌ 操作概率必须在[0, 1]之间: {args.action_prob}")
        return 1

    if not (0 <= args.iou_threshold <= 1):
        print(f"❌ IoU阈值必须在[0, 1]之间: {args.iou_threshold}")
        return 1

    # 解析类别白名单
    include_classes = None
    if args.include_classes:
        try:
            include_classes = [int(x.strip()) for x in args.include_classes.split(',')]
            print(f"✅ 已设置类别白名单: {include_classes}")
        except ValueError:
            print(f"❌ 类别白名单格式错误，应为整数列表，如: 0,1,2")
            return 1

    try:
        result = filter_failed_validations(
            data_yaml_path=args.data,
            model_path=args.model,
            action_prob=args.action_prob,
            iou_threshold=args.iou_threshold,
            action=args.action,
            output_dir=args.output_dir,
            backup=not args.no_backup,
            enable_replacement=args.enable_replacement,
            include_classes=include_classes
        )

        return 0 if result else 1

    except KeyboardInterrupt:
        print("\n⚠️  操作被用户中断")
        return 1

    except Exception as e:
        print(f"\n❌ 过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
