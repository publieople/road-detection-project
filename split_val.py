from pathlib import Path
import random
import shutil

# 配置路径
YOLO_ROOT = Path("E:/ToDo/ShandaCup/road-detection-project/datasets/yolo_format")
TRAIN_RATIO = 0.85  # 85%训练，15%验证

def split_dataset():
    """从训练集随机划分15%作为验证集"""

    # 获取所有训练图片
    train_img_dir = YOLO_ROOT / "images" / "train"
    train_label_dir = YOLO_ROOT / "labels" / "train"

    all_images = list(train_img_dir.glob("*.jpg")) + \
                 list(train_img_dir.glob("*.jpeg")) + \
                 list(train_img_dir.glob("*.png"))

    print(f"找到 {len(all_images)} 张训练图片")

    # 随机打乱并划分
    random.seed(42)  # 固定随机种子，保证可复现
    random.shuffle(all_images)

    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    print(f"训练集: {len(train_imgs)} 张")
    print(f"验证集: {len(val_imgs)} 张")

    # 创建验证集目录
    val_img_dir = YOLO_ROOT / "images" / "val"
    val_label_dir = YOLO_ROOT / "labels" / "val"
    val_img_dir.mkdir(exist_ok=True)
    val_label_dir.mkdir(exist_ok=True)

    # 移动验证集文件
    for img_path in val_imgs:
        # 移动图片
        dst_img = val_img_dir / img_path.name
        shutil.move(str(img_path), str(dst_img))

        # 移动对应的标签
        label_path = train_label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            dst_label = val_label_dir / (img_path.stem + ".txt")
            shutil.move(str(label_path), str(dst_label))
        else:
            print(f"警告: {label_path} 不存在")

    print(f"验证集划分完成！")

    # 更新road.yaml
    yaml_path = YOLO_ROOT / "road.yaml"
    with open(yaml_path, 'r') as f:
        content = f.read()

    # 确保val路径正确
    content = content.replace("# val: ", "val: ")
    content = content.replace("val: images/valid", "val: images/val")

    with open(yaml_path, 'w') as f:
        f.write(content)

    print(f"road.yaml已更新！")

if __name__ == '__main__':
    split_dataset()