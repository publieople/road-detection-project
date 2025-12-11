import xml.etree.ElementTree as ET
import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional

# ==================== 配置区域（按需修改） ====================
# RDD2022数据集路径（主要数据集）
RDD2022_ROOT = Path("./datasets/RDD2022")

# YOLO格式输出路径
YOLO_ROOT = Path("./datasets/yolo_format")

# 4类别映射
# 0: 裂缝 (Crack) - 包含纵向裂缝、横向裂缝、轻微裂缝
# 1: 龟裂 (Alligator Crack) - 网状裂缝
# 2: 坑槽 (Pothole) - 坑洞
# 3: 修补 (Repair) - 裂缝修补、坑槽修补
FOUR_CLASSES = {
    # 裂缝类别 - 主要包含裂缝类病害
    'D00': 0,  # 纵向裂缝 (Longitudinal Crack)
    'D10': 0,  # 横向裂缝 (Transverse Crack)
    'D01': 0,  # 轻微纵向裂缝 (Light Longitudinal Crack)
    'D11': 0,  # 轻微横向裂缝 (Light Transverse Crack)
    'D43': 0,  # 其他裂缝类损坏 -> 裂缝

    # 龟裂类别 - 网状裂缝
    'D20': 1,  # 龟裂 (Alligator Crack)

    # 坑槽类别 - 仅包含真正的坑洞，排除车辙等
    'D40': 2,  # 坑槽 (Pothole) - 需要进一步筛选

    # 修补类别 - 各种修补类型
    'Repair': 3,  # 修补（裂缝修补、坑槽修补）

    # 以下类别将被舍弃（不符合4类要求）
    'D44': -1,  # 其他损坏类型 - 舍弃
    'D50': -1,  # 井盖 (Manhole) - 舍弃，不是道路病害
}
# =========================================================


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

        # 使用新的4类别映射
        class_id = FOUR_CLASSES.get(class_name, -1)

        # 舍弃不符合要求的类别（class_id为-1）
        if class_id == -1:
            continue

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

        # 处理浮点数坐标（某些数据集使用浮点数）
        try:
            xmin = int(float(xmin_elem.text))
            ymin = int(float(ymin_elem.text))
            xmax = int(float(xmax_elem.text))
            ymax = int(float(ymax_elem.text))
        except ValueError:
            print(f"警告：XML {xml_path} 边界框坐标格式错误")
            continue

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
                    split: str = "train", classes: Optional[Dict] = None) -> int:
    """
    转换单个数据集
    返回成功转换的图片数量
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
    return converted_count


def convert_rdd2022_dataset():
    """
    转换RDD2022数据集，支持所有国家数据
    """
    if not RDD2022_ROOT.exists():
        print(f"❌ RDD2022路径不存在: {RDD2022_ROOT}")
        return 0

    print(f"\n{'='*60}")
    print("🚀 开始转换 RDD2022 数据集...")
    print(f"数据集路径: {RDD2022_ROOT}")

    # 定义所有国家文件夹
    countries = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]

    total_converted = 0

    for country in countries:
        print(f"\n📍 处理 {country} 数据...")

        # 检查训练集
        train_xml_dir = RDD2022_ROOT / country / "train" / "annotations" / "xmls"
        train_img_dir = RDD2022_ROOT / country / "train" / "images"

        if train_xml_dir.exists() and train_img_dir.exists():
            print(f"  找到训练集: {len(list(train_xml_dir.glob('*.xml')))} 个标注文件")
            converted = convert_dataset(
                xml_dir=train_xml_dir,
                img_dir=train_img_dir,
                yolo_dir=YOLO_ROOT,
                split="train"
            )
            total_converted += converted
            print(f"  ✅ {country} 训练集转换完成: {converted} 张图片")
        else:
            print(f"  ⚠️  {country} 训练集路径不存在")

        # 检查测试集（如果有的话，用作验证集）
        test_xml_dir = RDD2022_ROOT / country / "test" / "annotations" / "xmls"
        test_img_dir = RDD2022_ROOT / country / "test" / "images"

        if test_xml_dir.exists() and test_img_dir.exists():
            print(f"  找到测试集: {len(list(test_xml_dir.glob('*.xml')))} 个标注文件")
            converted = convert_dataset(
                xml_dir=test_xml_dir,
                img_dir=test_img_dir,
                yolo_dir=YOLO_ROOT,
                split="val"
            )
            total_converted += converted
            print(f"  ✅ {country} 测试集转换完成: {converted} 张图片")

    return total_converted

def analyze_class_distribution(yolo_dir: Path):
    """
    分析类别分布情况
    """
    print(f"\n{'='*60}")
    print("📊 分析类别分布...")

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = ['裂缝', '龟裂', '坑槽', '修补']

    # 统计训练集
    train_label_dir = yolo_dir / "labels" / "train"
    if train_label_dir.exists():
        for txt_file in train_label_dir.glob("*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    # 统计验证集
    val_label_dir = yolo_dir / "labels" / "val"
    if val_label_dir.exists():
        for txt_file in val_label_dir.glob("*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    print("类别分布统计:")
    total_instances = sum(class_counts.values())
    for class_id, count in class_counts.items():
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"  {class_names[class_id]} (类别{class_id}): {count} 个实例 ({percentage:.1f}%)")

    return class_counts

def main():
    """
    主函数：转换RDD2022数据集为4类别YOLO格式
    """
    print("🛣️  RDD2022 道路病害数据集转换工具")
    print("=" * 60)
    print("📋 功能: 将RDD2022数据集转换为4类别YOLO格式")
    print("🏷️  类别: 裂缝(0), 龟裂(1), 坑槽(2), 修补(3)")
    print("=" * 60)

    YOLO_ROOT.mkdir(parents=True, exist_ok=True)

    # 转换RDD2022数据集
    total_converted = convert_rdd2022_dataset()

    if total_converted == 0:
        print("❌ 没有成功转换任何数据，请检查数据集路径")
        return

    print(f"\n{'='*60}")
    print(f"✅ 转换完成！总计转换了 {total_converted} 张图片")
    print(f"📁 YOLO格式数据保存在: {YOLO_ROOT}")

    # 分析类别分布
    class_counts = analyze_class_distribution(YOLO_ROOT)

    # 打印最终结构
    print(f"\n{'='*60}")
    print("数据分布统计:")
    for split in ['train', 'val']:
        img_dir = YOLO_ROOT / "images" / split
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.jpg")))
            print(f"  {split}集: {img_count} 张图片")

    # 生成数据配置文件
    print(f"\n{'='*60}")
    print("📝 生成 road.yaml 配置文件...")

    config_content = f"""# RDD2022 道路病害检测数据集配置
# 4类别：裂缝、龟裂、坑槽、修补
path: {YOLO_ROOT.as_posix()}
train: images/train
val: images/val

# 类别数量
nc: 4

# 类别名称
names: ['裂缝', '龟裂', '坑槽', '修补']

# 数据增强配置（针对道路病害优化）
mosaic: 0.8          # Mosaic增强强度
mixup: 0.5           # MixUp增强比例
copy_paste: 0.3      # 复制粘贴增强比例
degrees: 15.0        # 旋转增强角度
translate: 0.3       # 平移增强比例
scale: 0.7           # 缩放增强比例
shear: 5.0           # 剪切增强角度
perspective: 0.001   # 透视增强比例
fliplr: 0.8          # 左右翻转概率
flipud: 0.2          # 上下翻转概率
hsv_h: 0.015         # HSV色调增强
hsv_s: 0.7           # HSV饱和度增强
hsv_v: 0.4           # HSV明度增强
"""

    with open(YOLO_ROOT / "road.yaml", 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"✅ 配置文件已生成: {YOLO_ROOT / 'road.yaml'}")
    print(f"🎯 可直接用于训练的YAML路径: {YOLO_ROOT.resolve() / 'road.yaml'}")

    # 输出类别分布建议
    print(f"\n{'='*60}")
    print("💡 训练建议:")
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 3:
            print(f"⚠️  检测到类别不平衡 (比例: {imbalance_ratio:.1f}:1)")
            print("   建议: 使用类别加权损失函数或过采样策略")
        else:
            print("✅ 类别分布相对均衡")

    print("\n🎉 数据转换完成！可以开始训练模型了。")

if __name__ == '__main__':
    main()