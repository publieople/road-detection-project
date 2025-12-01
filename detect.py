# src/road_detection_project/detect.py
from ultralytics import YOLO
import cv2

def detect_crack(image_path):
    # 加载训练好的模型（先用预训练模型演示）
    model = YOLO('yolov8n.pt')

    # 读取图像
    img = cv2.imread(image_path)

    # 推理
    results = model(img)

    # 显示结果
    annotated = results[0].plot()
    cv2.imshow("道路病害检测", annotated)
    cv2.waitKey(0)

if __name__ == "__main__":
    detect_crack("test_image.jpg")