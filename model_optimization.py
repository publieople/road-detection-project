#!/usr/bin/env python3
"""
模型优化和轻量化脚本
包含模型剪枝、量化和知识蒸馏等功能
"""

import torch
import torch.nn as nn
from ultralytics import YOLO # pyright: ignore[reportPrivateImportUsage]
from pathlib import Path
import numpy as np
import time

class ModelOptimizer:
    def __init__(self, model_path: str):
        """
        初始化模型优化器

        Args:
            model_path: 原始模型路径
        """
        self.model = YOLO(model_path)
        self.original_model_path = model_path

    def prune_model(self, sparsity: float = 0.1, save_path: str = None): # pyright: ignore[reportArgumentType]
        """
        模型剪枝 - 移除不重要的权重

        Args:
            sparsity: 剪枝比例 (0.1 = 移除10%的权重)
            save_path: 保存路径

        Returns:
            剪枝后的模型路径
        """
        print(f"🔪 开始模型剪枝，剪枝比例: {sparsity:.1%}")

        if save_path is None:
            save_path = str(Path(self.original_model_path).parent / f"pruned_{sparsity:.1f}.pt")

        # 获取模型权重
        model_state = self.model.model.state_dict()

        # 计算权重的重要性（基于绝对值）
        importance_scores = {}
        for name, param in model_state.items():
            if 'weight' in name and param.dim() > 1:  # 只剪枝卷积和全连接层的权重
                importance_scores[name] = torch.abs(param)

        # 计算全局阈值
        all_scores = torch.cat([scores.flatten() for scores in importance_scores.values()])
        threshold = torch.quantile(all_scores, sparsity)

        # 应用剪枝
        pruned_state = {}
        for name, param in model_state.items():
            if name in importance_scores:
                mask = importance_scores[name] > threshold
                pruned_state[name] = param * mask
            else:
                pruned_state[name] = param

        # 保存剪枝后的模型
        self.model.model.load_state_dict(pruned_state)
        self.model.save(save_path)

        # 计算剪枝效果
        original_params = sum(p.numel() for p in self.model.model.parameters())
        pruned_params = sum((p != 0).sum() for p in self.model.model.parameters())
        reduction_ratio = 1 - (pruned_params / original_params)

        print(f"✅ 模型剪枝完成")
        print(f"   原始参数数量: {original_params:,}")
        print(f"   剩余参数数量: {pruned_params:,}")
        print(f"   参数减少比例: {reduction_ratio:.1%}")
        print(f"   模型已保存: {save_path}")

        return save_path

    def quantize_model(self, save_path: str = None):
        """
        模型量化 - 将FP32权重转换为INT8

        Args:
            save_path: 保存路径

        Returns:
            量化后的模型路径
        """
        print("📊 开始模型量化 (INT8)")

        if save_path is None:
            save_path = str(Path(self.original_model_path).parent / "quantized_int8.onnx")

        # 导出为ONNX格式（包含量化）
        self.model.export(
            format='onnx',
            simplify=True,
            int8=True
        )

        # 获取导出的ONNX文件路径
        onnx_path = str(Path(self.original_model_path).parent / "best.onnx")

        # 重命名为指定的保存路径
        if onnx_path != save_path:
            import shutil
            shutil.move(onnx_path, save_path)

        print(f"✅ 模型量化完成")
        print(f"   ONNX模型已保存: {save_path}")

        # 计算模型大小
        original_size = Path(self.original_model_path).stat().st_size / (1024 * 1024)  # MB
        quantized_size = Path(save_path).stat().st_size / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1

        print(f"   原始模型大小: {original_size:.1f} MB")
        print(f"   量化模型大小: {quantized_size:.1f} MB")
        print(f"   压缩比例: {compression_ratio:.1f}x")

        return save_path

    def benchmark_model(self, test_images: list, model_path: str | None = None):
        """
        基准测试 - 测试模型性能

        Args:
            test_images: 测试图片路径列表
            model_path: 模型路径（如果不使用当前模型）

        Returns:
            性能指标字典
        """
        print("⚡ 开始模型性能基准测试")

        # 使用指定模型或当前模型
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model

        # 测试推理速度
        inference_times = []
        preprocess_times = []
        postprocess_times = []

        for img_path in test_images[:10]:  # 测试前10张图片
            start_time = time.time()

            # 推理
            results = model(img_path, verbose=False)

            end_time = time.time()
            inference_times.append(end_time - start_time)

        # 计算平均性能
        avg_inference_time = np.mean(inference_times) * 1000  # 转换为ms
        fps = 1000 / avg_inference_time

        # 模型大小
        model_size = Path(model_path or self.original_model_path).stat().st_size / (1024 * 1024)  # MB

        # 参数数量
        param_count = sum(p.numel() for p in model.model.parameters())

        results = {
            'model_path': model_path or self.original_model_path,
            'model_size_mb': model_size,
            'parameter_count': param_count,
            'avg_inference_time_ms': avg_inference_time,
            'fps': fps,
            'test_images_count': len(test_images[:10])
        }

        print("📊 基准测试结果:")
        print(f"   模型大小: {model_size:.1f} MB")
        print(f"   参数数量: {param_count:,}")
        print(f"   平均推理时间: {avg_inference_time:.1f} ms")
        print(f"   FPS: {fps:.1f}")

        return results

def main():
    """
    主函数：模型优化流程
    """
    print("🚀 道路病害检测模型优化工具")
    print("=" * 60)

    # 原始模型路径 - 使用训练结果的实际路径
    original_model = "D:/sd-webui-aki-v4.11.1-cu128/runs/detect/train20/weights/best.pt"

    if not Path(original_model).exists():
        print(f"❌ 模型文件不存在: {original_model}")
        return

    optimizer = ModelOptimizer(original_model)

    # 1. 基准测试原始模型
    print("\n" + "="*60)
    print("1️⃣ 原始模型基准测试")
    test_images = list(Path("datasets/yolo_format/images/val").glob("*.jpg"))[:20]
    if test_images:
        original_results = optimizer.benchmark_model(test_images)
    else:
        print("⚠️  未找到测试图片")
        original_results = None

    # 2. 模型量化
    print("\n" + "="*60)
    print("2️⃣ 模型量化优化")
    try:
        quantized_path = optimizer.quantize_model()
        print("✅ 量化完成")
    except Exception as e:
        print(f"❌ 量化失败: {e}")
        quantized_path = None

    # 3. 基准测试量化模型
    if quantized_path and test_images:
        print("\n" + "="*60)
        print("3️⃣ 量化模型基准测试")
        try:
            quantized_results = optimizer.benchmark_model(test_images, quantized_path)

            # 比较结果
            if original_results and quantized_results:
                print("\n📊 性能对比:")
                print(f"   模型大小: {original_results['model_size_mb']:.1f} MB → {quantized_results['model_size_mb']:.1f} MB")
                print(f"   压缩比例: {original_results['model_size_mb']/quantized_results['model_size_mb']:.1f}x")
                print(f"   推理时间: {original_results['avg_inference_time_ms']:.1f} ms → {quantized_results['avg_inference_time_ms']:.1f} ms")
                print(f"   FPS: {original_results['fps']:.1f} → {quantized_results['fps']:.1f}")
        except Exception as e:
            print(f"❌ 量化模型基准测试失败: {e}")

    # 4. 模型剪枝（可选）
    print("\n" + "="*60)
    print("4️⃣ 模型剪枝优化")
    try:
        # 轻度剪枝
        pruned_path = optimizer.prune_model(sparsity=0.1)
        print("✅ 剪枝完成")

        # 基准测试剪枝模型
        if test_images:
            pruned_results = optimizer.benchmark_model(test_images, pruned_path)
    except Exception as e:
        print(f"❌ 剪枝失败: {e}")

    print("\n" + "="*60)
    print("🎉 模型优化完成！")
    print("💡 建议:")
    print("   - 使用量化模型进行部署，获得更好的压缩效果")
    print("   - 根据实际需求选择剪枝比例")
    print("   - 在目标硬件上测试最终性能")

if __name__ == "__main__":
    main()