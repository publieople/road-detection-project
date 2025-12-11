#!/usr/bin/env python3
"""
从训练集中分割出验证集
用于解决RDD2022数据集缺少验证集的问题
"""

import os
import random
import shutil
from pathlib import Path
import yaml

def split_train_val(train_dir: Path, val_dir: Path, dataset_root: Path, split_ratio: float = 0.2, seed: int = 42):
    """
    从训练集中分割出验证集
    
    Args:
        train_dir: 训练集目录路径
        val_dir: 验证集目录路径
        dataset_root: 数据集根目录
        split_ratio: 验证集比例
        seed: 随机种子
    """
    random.seed(seed)

    # 获取所有训练图片
    train_images = list(train_dir.glob("*.jpg"))
    total_images = len(train_images)

    if total_images == 0:
        print(f"❌ 未找到训练图片: {train_dir}")
        return 0

    # 计算验证集数量
    val_count = max(1, int(total_images * split_ratio))

    # 随机选择验证集图片
    val_images = random.sample(train_images, val_count)

    print(f"📊 数据集分割统计:")
    print(f"   总图片数: {total_images}")
    print(f"   验证集数量: {val_count}")
    print(f"   训练集数量: {total_images - val_count}")
    print(f"   验证集比例: {split_ratio:.1%}")

    # 移动验证集图片
    moved_count = 0
    for img_path in val_images:
        # 移动图片
        val_img_path = val_dir / img_path.name
        shutil.move(str(img_path), str(val_img_path))

        # 移动对应的标签文件
        train_label_path = dataset_root / "labels" / "train" / (img_path.stem + ".txt")
        if train_label_path.exists():
            val_label_path = dataset_root / "labels" / "val" / (img_path.stem + ".txt")
            val_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(train_label_path), str(val_label_path))

        moved_count += 1

    return moved_count

def update_yaml_config(yaml_path: Path, train_path: str, val_path: str):
    """
    更新YAML配置文件
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 更新路径
    config['train'] = train_path
    config['val'] = val_path

    # 保存更新后的配置
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"✅ YAML配置已更新: {yaml_path}")

def main():
    """
    主函数：分割验证集
    """
    print("🔀 RDD2022 验证集分割工具")
    print("=" * 50)

    # 数据集路径
    dataset_root = Path("datasets/yolo_format")
    train_img_dir = dataset_root / "images" / "train"
    val_img_dir = dataset_root / "images" / "val"
    yaml_path = dataset_root / "road.yaml"

    # 检查路径
    if not train_img_dir.exists():
        print(f"❌ 训练集路径不存在: {train_img_dir}")
        return

    # 创建验证集目录
    val_img_dir.mkdir(parents=True, exist_ok=True)

    # 分割验证集
    moved_count = split_train_val(train_img_dir, val_img_dir, dataset_root, split_ratio=0.15, seed=42)

    if moved_count > 0:
        print(f"✅ 成功分割 {moved_count} 张图片到验证集")

        # 更新YAML配置
        update_yaml_config(yaml_path, "images/train", "images/val")

        # 验证分割结果
        train_count = len(list(train_img_dir.glob("*.jpg")))
        val_count = len(list(val_img_dir.glob("*.jpg")))

        print(f"\n📊 最终数据集分布:")
        print(f"   训练集: {train_count} 张图片")
        print(f"   验证集: {val_count} 张图片")
        print(f"   总计: {train_count + val_count} 张图片")

        # 验证标签文件
        train_labels = len(list((dataset_root / "labels" / "train").glob("*.txt")))
        val_labels = len(list((dataset_root / "labels" / "val").glob("*.txt")))

        print(f"\n📊 标签文件分布:")
        print(f"   训练集标签: {train_labels} 个")
        print(f"   验证集标签: {val_labels} 个")

        if train_count == train_labels and val_count == val_labels:
            print("✅ 图片和标签文件匹配正确")
        else:
            print("⚠️  图片和标签文件数量不匹配，请检查")

        print("\n🎉 验证集分割完成！可以开始训练模型了。")
    else:
        print("❌ 验证集分割失败")

if __name__ == "__main__":
    main()