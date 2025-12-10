import xml.etree.ElementTree as ET
import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional

# ==================== 配置区域（按需修改） ====================
# CRACK500数据集路径
CRACK500_ROOT = Path("./datasets/CRACK500")

# RDD2020数据集路径
RDD2020_ROOT = Path("./datasets/RDD2020")

# Road_diseases_20210513数据集路径
ROAD_DISEASES_ROOT = Path("./datasets/Road_diseases_20210513")

# YOLO格式输出路径
YOLO_ROOT = Path("./datasets/yolo_format")

# RDD2020的类别映射（8类合并版本）
RDD_CLASSES = {
    'D00': 0,  # 纵向裂缝 (Longitudinal Crack)
    'D10': 1,  # 横向裂缝 (Transverse Crack)
    'D20': 2,  # 龟裂 (Alligator Crack)
    'D40': 3,  # 坑槽 (Pothole)
    'D01': 4,  # 轻微纵向裂缝 (Light Longitudinal Crack)
    'D11': 5,  # 轻微横向裂缝 (Light Transverse Crack)
    'D43': 6,  # 其他损坏类型 (合并到other)
    'D44': 6,  # 其他损坏类型 (合并到other)
    'D50': 6   # 其他损坏类型 (合并到other)
}

# CRACK500的类别（通常是裂缝=0）
CRACK_CLASSES = {
    'crack': 0
}

# Road_diseases_20210513的类别映射
ROAD_DISEASES_CLASSES = {
    'Crack': 0,
    'Manhole': 1,
    'Net': 2,
    'Pothole': 3,
    'Patch-Crack': 4,
    'Patch-Net': 5,
    'Patch-Pothole': 6,
    'other': 7,
    'Other': 7
}
# =========================================================

def parse_coco_json(json_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    解析COCO JSON文件，返回{image_id: [(class_id, x_center, y_center, width, height), ...]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建类别映射
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 构建图片信息映射
    images = {img['id']: img for img in data['images']}
    
    # 构建标注映射
    annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations:
            annotations[image_id] = []
        
        # 获取类别名称并映射到ID
        category_name = categories.get(ann['category_id'], 'other')
        class_id = ROAD_DISEASES_CLASSES.get(category_name, 7)  # 默认映射到other类别
        
        # 转换边界框格式 (x, y, width, height) -> (x_center, y_center, width, height)
        x, y, width, height = ann['bbox']
        img_width = images[image_id]['width']
        img_height = images[image_id]['height']
        
        # 转换为归一化的YOLO格式
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        
        # 确保坐标不越界
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_width = max(0.0, min(1.0, norm_width))
        norm_height = max(0.0, min(1.0, norm_height))
        
        annotations[image_id].append((class_id, x_center, y_center, norm_width, norm_height))
    
    return annotations

def parse_xml(xml_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    解析单个XML文件，返回[(class_id, x_center, y_center, width, height), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find('size')
    if size is None:
        print(f"警告：XML {xml_path} 缺少size信息")
        return []

    width_elem = size.find('width')
    height_elem = size.find('height')
    if width_elem is None or height_elem is None or width_elem.text is None or height_elem.text is None:
        print(f"警告：XML {xml_path} 缺少宽高信息")
        return []

    img_width = int(width_elem.text)
    img_height = int(height_elem.text)

    boxes = []

    # 遍历所有目标
    for obj in root.findall('object'):
        # 获取类别名
        name_elem = obj.find('name')
        if name_elem is None or name_elem.text is None:
            print(f"警告：XML {xml_path} 缺少类别信息")
            continue
        class_name = name_elem.text

        # 根据数据集选择类别映射
        if "RDD2020" in str(xml_path):
            class_id = RDD_CLASSES.get(class_name, -1)
        else:
            class_id = CRACK_CLASSES.get(class_name, -1)

        if class_id == -1:
            print(f"警告：XML {xml_path} 包含未知类别 {class_name}")
            continue

        # 获取边界框
        bbox = obj.find('bndbox')
        if bbox is None:
            print(f"警告：XML {xml_path} 缺少边界框信息")
            continue

        xmin_elem = bbox.find('xmin')
        ymin_elem = bbox.find('ymin')
        xmax_elem = bbox.find('xmax')
        ymax_elem = bbox.find('ymax')

        if (xmin_elem is None or ymin_elem is None or xmax_elem is None or ymax_elem is None or
            xmin_elem.text is None or ymin_elem.text is None or xmax_elem.text is None or ymax_elem.text is None):
            print(f"警告：XML {xml_path} 边界框信息不完整")
            continue

        xmin = int(xmin_elem.text)
        ymin = int(ymin_elem.text)
        xmax = int(xmax_elem.text)
        ymax = int(ymax_elem.text)

        # 转换为YOLO格式（中心点+宽高，归一化到0-1）
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 确保坐标不越界
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes

def convert_dataset(xml_dir: Path, img_dir: Path, yolo_dir: Path,
                    split: str = "train", classes: Optional[Dict] = None):
    """
    转换单个数据集
    """
    if classes is None:
        classes = {}
    print(f"\n{'='*50}")
    print(f"开始转换 {split} 集...")
    print(f"XML路径: {xml_dir}")
    print(f"图片路径: {img_dir}")

    # 创建YOLO目录结构
    (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    xml_files = list(xml_dir.glob("*.xml"))
    print(f"找到 {len(xml_files)} 个标注文件")

    converted_count = 0
    for xml_path in xml_files:
        # 获取文件名（不含扩展名）
        filename = xml_path.stem

        # 查找对应的图片文件（支持jpg/jpeg/JPG）
        img_path = None
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            candidate = img_dir / (filename + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"警告：未找到 {filename} 对应的图片文件")
            continue

        # 解析XML获取标注框
        boxes = parse_xml(xml_path)

        if len(boxes) == 0:
            # print(f"跳过 {filename}: 无有效标注")
            continue

        # 复制图片到YOLO目录
        dst_img = yolo_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # 创建YOLO格式的txt标注文件
        txt_path = yolo_dir / "labels" / split / (filename + ".txt")
        with open(txt_path, 'w') as f:
            for box in boxes:
                line = f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n"
                f.write(line)

        converted_count += 1

    print(f"成功转换 {converted_count} 张图片")

def convert_coco_dataset(json_path: Path, img_dir: Path, yolo_dir: Path, split: str = "train"):
    """
    转换COCO JSON格式的数据集
    """
    print(f"\n{'='*50}")
    print(f"开始转换 {split} 集 (COCO格式)...")
    print(f"JSON路径: {json_path}")
    print(f"图片路径: {img_dir}")

    # 创建YOLO目录结构
    (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 解析COCO JSON文件
    annotations = parse_coco_json(json_path)
    print(f"找到 {len(annotations)} 张有标注的图片")

    converted_count = 0
    for image_id, boxes in annotations.items():
        # 获取图片信息
        image_filename = f"{image_id:05d}.jpg"  # 格式化为5位数字文件名
        
        # 查找对应的图片文件
        img_path = img_dir / image_filename
        if not img_path.exists():
            print(f"警告：未找到图片文件 {img_path}")
            continue

        if len(boxes) == 0:
            print(f"跳过 {image_filename}: 无有效标注")
            continue

        # 复制图片到YOLO目录
        dst_img = yolo_dir / "images" / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # 创建YOLO格式的txt标注文件
        txt_path = yolo_dir / "labels" / split / (img_path.stem + ".txt")
        with open(txt_path, 'w') as f:
            for box in boxes:
                line = f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n"
                f.write(line)

        converted_count += 1

    print(f"成功转换 {converted_count} 张图片")
    return converted_count

def main():
    """
    主函数：转换所有数据集
    """
    YOLO_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. 转换CRACK500数据集
    if CRACK500_ROOT.exists():
        print("\n正在转换 CRACK500 数据集...")

        # 转换train集
        convert_dataset(
            xml_dir=CRACK500_ROOT / "Annotations" / "train",
            img_dir=CRACK500_ROOT / "JPEGImages" / "train",
            yolo_dir=YOLO_ROOT,
            split="train",
            classes=CRACK_CLASSES
        )

        # 转换val集
        convert_dataset(
            xml_dir=CRACK500_ROOT / "Annotations" / "val",
            img_dir=CRACK500_ROOT / "JPEGImages" / "val",
            yolo_dir=YOLO_ROOT,
            split="val",
            classes=CRACK_CLASSES
        )
    else:
        print(f"警告：CRACK500路径不存在 {CRACK500_ROOT}")

    # 2. 转换RDD2020数据集
    if RDD2020_ROOT.exists():
        print("\n正在转换 RDD2020 数据集...")

        # 遍历所有国家文件夹
        for country in ["Czech", "India", "Japan"]:
            country_xml_dir = RDD2020_ROOT / "train" / country / "annotations" / "xmls"
            country_img_dir = RDD2020_ROOT / "train" / country / "images"

            if country_xml_dir.exists():
                # 所有国家数据合并到train集
                convert_dataset(
                    xml_dir=country_xml_dir,
                    img_dir=country_img_dir,
                    yolo_dir=YOLO_ROOT,
                    split="train",
                    classes=RDD_CLASSES
                )
    else:
        print(f"警告：RDD2020路径不存在 {RDD2020_ROOT}")

    # 3. 转换Road_diseases_20210513数据集
    if ROAD_DISEASES_ROOT.exists():
        print("\n正在转换 Road_diseases_20210513 数据集...")
        
        # COCO JSON标注文件路径
        json_path = ROAD_DISEASES_ROOT / "train" / "train" / "annotations" / "train.json"
        img_dir = ROAD_DISEASES_ROOT / "train" / "train" / "images"
        
        if json_path.exists() and img_dir.exists():
            convert_coco_dataset(
                json_path=json_path,
                img_dir=img_dir,
                yolo_dir=YOLO_ROOT,
                split="train"
            )
        else:
            print(f"警告：Road_diseases_20210513数据不完整，JSON: {json_path.exists()}, 图片: {img_dir.exists()}")
    else:
        print(f"警告：Road_diseases_20210513路径不存在 {ROAD_DISEASES_ROOT}")

    print(f"\n{'='*50}")
    print(f"转换完成！YOLO格式数据保存在: {YOLO_ROOT}")

    # 打印最终结构
    print("\n数据分布统计:")
    for split in ['train', 'val']:
        img_dir = YOLO_ROOT / "images" / split
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.jpg")))
            print(f"  {split}集: {img_count} 张图片")

    # 生成数据配置文件
    print("\n生成 road.yaml 配置文件...")
    
    # 检查是否有Road_diseases_20210513数据集，如果有则使用新的类别配置
    if ROAD_DISEASES_ROOT.exists() and (ROAD_DISEASES_ROOT / "train" / "train" / "annotations" / "train.json").exists():
        # 使用Road_diseases_20210513的8类别配置
        config_content = f"""
path: {YOLO_ROOT.as_posix()}
train: images/train
val: images/val

nc: 8
names: ['Crack', 'Manhole', 'Net', 'Pothole', 'Patch-Crack', 'Patch-Net', 'Patch-Pothole', 'other']

"""
    else:
        # 使用原有的9类别配置
        config_content = f"""
path: {YOLO_ROOT.as_posix()}
train: images/train
val: images/val

nc: 9
names: ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole', 'light_longitudinal_crack', 'light_transverse_crack', 'other_damage_1', 'other_damage_2', 'other_damage_3']

"""
    
    with open(YOLO_ROOT / "road.yaml", 'w') as f:
        f.write(config_content)

    print(f"配置文件已生成: {YOLO_ROOT / 'road.yaml'}")
    print(f"可直接用于训练的YAML路径: {YOLO_ROOT.resolve() / 'road.yaml'}")

if __name__ == '__main__':
    main()