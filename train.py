#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
道路病害检测模型训练脚本 - 重构版
基于模块化架构的统一训练入口
"""

import argparse
import sys
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from road_detection.training import RoadDamageTrainer, create_training_config
from road_detection.utils import setup_chinese_fonts

def main():
    """主函数"""
    # 设置中文字体
    setup_chinese_fonts()
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description='道路病害检测模型训练 - 模块化版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 标准训练
  python train.py --config-type standard --data datasets/yolo_format/road.yaml
  
  # 优化训练（针对RDD2022）
  python train.py --config-type optimized --data datasets/yolo_format/road.yaml
  
  # 快速训练
  python train.py --config-type fast --epochs 50 --model-size n
  
  # 恢复训练
  python train.py --resume --resume-path runs/detect/train/weights/last.pt
  
  # 自定义配置
  python train.py --config-type custom --epochs 200 --model-size m --lr0 0.001
        """
    )
    
    # 基础参数
    parser.add_argument('--data', type=str, default='datasets/yolo_format/road.yaml',
                       help='数据配置文件路径 (默认: datasets/yolo_format/road.yaml)')
    parser.add_argument('--config-type', type=str, default='standard',
                       choices=['standard', 'optimized', 'balanced', 'fast', 'custom'],
                       help='配置类型 (默认: standard)')
    parser.add_argument('--dataset-type', type=str, default='rdd2022',
                       choices=['rdd2022', 'rdd2020', 'custom'],
                       help='数据集类型 (默认: rdd2022)')
    
    # 模型参数
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小 (默认: n)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (根据配置类型自动设置)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小 (默认: 16)')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['SGD', 'AdamW'],
                       help='优化器类型')
    parser.add_argument('--lr0', type=float, default=None,
                       help='初始学习率')
    parser.add_argument('--lrf', type=float, default=None,
                       help='最终学习率倍数')
    
    # 训练控制
    parser.add_argument('--resume', action='store_true',
                       help='从上次中断处恢复训练')
    parser.add_argument('--resume-path', type=str, default=None,
                       help='指定恢复训练的权重文件路径')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='训练设备 (默认: auto)')
    
    # 增强参数
    parser.add_argument('--mosaic', type=float, default=None,
                       help='Mosaic增强强度')
    parser.add_argument('--mixup', type=float, default=None,
                       help='MixUp增强比例')
    parser.add_argument('--degrees', type=float, default=None,
                       help='旋转增强角度')
    
    # 性能目标
    parser.add_argument('--target-map50', type=float, default=0.80,
                       help='目标mAP@0.5 (默认: 0.80)')
    
    # 输出控制
    parser.add_argument('--save-dir', type=str, default=None,
                       help='结果保存目录')
    parser.add_argument('--export-format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'tensorrt'],
                       help='模型导出格式 (默认: onnx)')
    parser.add_argument('--no-export', action='store_true',
                       help='跳过模型导出')
    
    # 分析选项
    parser.add_argument('--analyze-dataset', action='store_true',
                       help='详细分析数据集')
    parser.add_argument('--generate-report', action='store_true',
                       help='生成详细训练报告')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("\n" + "=" * 80)
    print("🛣️  道路病害检测模型训练系统 - 模块化版本")
    print("=" * 80)
    print(f"📋 配置类型: {args.config_type}")
    print(f"📊 数据集: {args.data}")
    print(f"🎯 目标mAP@0.5: {args.target_map50}")
    
    # 检查数据文件
    if not Path(args.data).exists():
        print(f"❌ 数据配置文件不存在: {args.data}")
        return 1
    
    try:
        # 创建训练配置
        print(f"\n🔧 创建训练配置...")
        
        if args.config_type == 'custom':
            # 自定义配置
            config_kwargs = {
                'data_yaml_path': args.data,
                'model_size': args.model_size,
                'device': args.device,
                'target_map50': args.target_map50
            }
            
            # 添加非None的参数
            if args.epochs is not None:
                config_kwargs['epochs'] = args.epochs
            if args.img_size is not None:
                config_kwargs['img_size'] = args.img_size
            if args.batch_size is not None:
                config_kwargs['batch_size'] = args.batch_size
            if args.optimizer is not None:
                config_kwargs['optimizer'] = args.optimizer
            if args.lr0 is not None:
                config_kwargs['lr0'] = args.lr0
            if args.lrf is not None:
                config_kwargs['lrf'] = args.lrf
            if args.mosaic is not None:
                config_kwargs['mosaic'] = args.mosaic
            if args.mixup is not None:
                config_kwargs['mixup'] = args.mixup
            if args.degrees is not None:
                config_kwargs['degrees'] = args.degrees
            
            config = create_training_config('standard', **config_kwargs)
        else:
            # 使用预设配置
            config = create_training_config(
                config_type=args.config_type,
                data_yaml_path=args.data,
                model_size=args.model_size,
                device=args.device,
                target_map50=args.target_map50
            )
            
            # 覆盖特定参数
            if args.epochs is not None:
                config.epochs = args.epochs
            if args.img_size is not None:
                config.img_size = args.img_size
            if args.batch_size is not None:
                config.batch_size = args.batch_size
        
        # 分析选项通过优化配置自动启用，不需要额外设置
        # 优化配置（AdamW + 高cls_gain）会自动启用数据集分析
        
        # 创建训练器
        print("\n🏗️  创建训练器...")
        trainer = RoadDamageTrainer(config)
        
        # 执行训练流程
        print("\n🚀 开始训练流程...")
        
        # 准备阶段
        dataset_stats = trainer.prepare_training()
        
        # 创建或加载模型
        model = trainer.create_or_load_model(resume_path=args.resume_path)
        
        # 训练
        model, training_results = trainer.train(resume=args.resume)
        
        # 验证
        validation_results = trainer.validate(save_dir=args.save_dir)
        
        # 导出模型
        if not args.no_export:
            export_path = trainer.export_model(format=args.export_format)
            print(f"💾 模型已导出: {export_path}")
        
        # 生成报告
        if args.generate_report:
            if args.save_dir:
                report_path = Path(args.save_dir) / "training_report.txt"
            else:
                report_path = "training_report.txt"
            trainer.save_training_report(str(report_path))
        
        # 打印总结
        print("\n" + "=" * 80)
        print("🎉 训练流程完成!")
        print("=" * 80)
        print(f"📊 数据集统计:")
        print(f"   训练图片: {dataset_stats['train_count']} 张")
        print(f"   验证图片: {dataset_stats['val_count']} 张")
        print(f"   类别数量: {dataset_stats['num_classes']}")
        print(f"   类别名称: {', '.join(dataset_stats['class_names'])}")
        print(f"\n🎯 验证结果:")
        print(f"   mAP@0.5: {validation_results['mAP50']:.3f}")
        print(f"   mAP@0.5:0.95: {validation_results['mAP5095']:.3f}")
        
        # 检查目标达成
        if validation_results['mAP50'] >= args.target_map50:
            print(f"✅ 目标达成！模型准确率 ≥ {args.target_map50:.0%}")
        else:
            print(f"⚠️  未达目标。当前准确率: {validation_results['mAP50']:.1%}, 目标: {args.target_map50:.0%}")
        
        print("\n💡 提示:")
        print("   - 使用 analyze_training_results.py 分析训练历史")
        print("   - 使用 model_optimization.py 进行模型优化")
        print("   - 使用 detect.py 进行模型推理测试")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return 1
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())