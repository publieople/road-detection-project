import csv
import yaml
import os
from pathlib import Path
import json
from collections import defaultdict, Counter

def read_csv_file(filepath):
    """读取CSV文件"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"读取CSV文件失败 {filepath}: {e}")
        return []
    return data

def analyze_training_results():
    """分析所有训练结果并生成报告"""
    
    runs_dir = Path("runs/detect")
    training_results = []
    
    print("开始分析训练结果...")
    
    # 遍历所有训练目录
    for train_dir in runs_dir.iterdir():
        if train_dir.is_dir() and train_dir.name.startswith('train'):
            try:
                # 提取训练编号
                train_name = train_dir.name
                if train_name == 'train':
                    train_num = 0
                else:
                    # 处理像 'train21' 这样的名称
                    num_part = train_name.replace('train', '')
                    if num_part:
                        train_num = int(num_part)
                    else:
                        continue
                        
                print(f"处理训练目录: {train_name} (编号: {train_num})")
                
                # 检查必要文件
                results_file = train_dir / "results.csv"
                args_file = train_dir / "args.yaml"
                report_file = train_dir / "training_report.txt"
                
                if not (results_file.exists() and args_file.exists()):
                    print(f"  跳过: 缺少必要文件")
                    continue
                
                # 读取训练结果
                results_data = read_csv_file(results_file)
                if not results_data:
                    print(f"  跳过: 无法读取结果文件")
                    continue
                
                # 读取配置参数
                try:
                    with open(args_file, 'r', encoding='utf-8') as f:
                        args = yaml.safe_load(f)
                except Exception as e:
                    print(f"  跳过: 无法读取配置文件 - {e}")
                    continue
                
                # 提取关键指标
                if len(results_data) == 0:
                    continue
                
                # 转换为数值类型
                map50_values = []
                map5095_values = []
                precision_values = []
                recall_values = []
                
                for row in results_data:
                    try:
                        map50_values.append(float(row.get('metrics/mAP50(B)', 0)))
                        map5095_values.append(float(row.get('metrics/mAP50-95(B)', 0)))
                        precision_values.append(float(row.get('metrics/precision(B)', 0)))
                        recall_values.append(float(row.get('metrics/recall(B)', 0)))
                    except (ValueError, KeyError):
                        continue
                
                if not map50_values:
                    print(f"  跳过: 无法提取性能指标")
                    continue
                
                # 找到最佳性能
                best_map50_idx = max(range(len(map50_values)), key=lambda i: map50_values[i])
                final_idx = len(results_data) - 1
                
                # 读取训练报告（如果存在）
                report_data = {}
                if report_file.exists():
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if '最佳mAP@0.5:' in line:
                                    try:
                                        report_data['best_map50_report'] = float(line.split(':')[1].strip())
                                    except:
                                        pass
                                elif '最终mAP@0.5:0.95:' in line:
                                    try:
                                        report_data['final_map5095_report'] = float(line.split(':')[1].strip())
                                    except:
                                        pass
                    except:
                        pass
                
                # 提取训练信息
                training_info = {
                    'train_num': train_num,
                    'model': args.get('model', 'unknown'),
                    'epochs': args.get('epochs', 0),
                    'batch_size': args.get('batch', 0),
                    'optimizer': args.get('optimizer', 'unknown'),
                    'lr0': args.get('lr0', 0.001),
                    'lrf': args.get('lrf', 0.01),
                    
                    # 最佳性能指标
                    'best_map50': map50_values[best_map50_idx],
                    'best_map5095': map5095_values[best_map50_idx] if map5095_values else 0,
                    'best_epoch': best_map50_idx + 1,
                    'best_precision': precision_values[best_map50_idx] if precision_values else 0,
                    'best_recall': recall_values[best_map50_idx] if recall_values else 0,
                    
                    # 最终性能指标
                    'final_map50': map50_values[-1],
                    'final_map5095': map5095_values[-1] if map5095_values else 0,
                    'final_precision': precision_values[-1] if precision_values else 0,
                    'final_recall': recall_values[-1] if recall_values else 0,
                    
                    # 损失函数
                    'final_box_loss': float(results_data[-1].get('train/box_loss', 0)),
                    'final_cls_loss': float(results_data[-1].get('train/cls_loss', 0)),
                    'final_dfl_loss': float(results_data[-1].get('train/dfl_loss', 0)),
                    
                    'final_val_box_loss': float(results_data[-1].get('val/box_loss', 0)),
                    'final_val_cls_loss': float(results_data[-1].get('val/cls_loss', 0)),
                    'final_val_dfl_loss': float(results_data[-1].get('val/dfl_loss', 0)),
                    
                    # 训练时间
                    'total_time': float(results_data[-1].get('time', 0)),
                    
                    # 数据增强参数
                    'mosaic': args.get('mosaic', 0),
                    'mixup': args.get('mixup', 0),
                    'copy_paste': args.get('copy_paste', 0),
                    'degrees': args.get('degrees', 0),
                    'translate': args.get('translate', 0),
                    'scale': args.get('scale', 0),
                    
                    # 报告数据
                    **report_data
                }
                
                training_results.append(training_info)
                print(f"  ✓ 成功提取 {len(results_data)} 个轮次的数据")
                
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                continue
    
    if not training_results:
        print("没有找到有效的训练结果")
        return None
    
    # 按训练编号排序
    training_results.sort(key=lambda x: x['train_num'])
    
    print(f"\n成功分析 {len(training_results)} 个训练结果")
    return training_results

def generate_analysis_report(training_results):
    """生成详细的分析报告"""
    
    if not training_results:
        print("没有训练结果可供分析")
        return
    
    print("\n" + "=" * 80)
    print("道路病害检测系统 - 训练结果分析报告")
    print("=" * 80)
    print()
    
    # 1. 总体概览
    print("📊 总体概览")
    print("-" * 40)
    print(f"总训练次数: {len(training_results)}")
    
    best_map50 = max(training_results, key=lambda x: x['best_map50'])
    best_map5095 = max(training_results, key=lambda x: x['best_map5095'])
    
    print(f"最佳mAP@0.5: {best_map50['best_map50']:.4f} (训练 #{best_map50['train_num']})")
    print(f"最佳mAP@0.5:0.95: {best_map5095['best_map5095']:.4f} (训练 #{best_map5095['train_num']})")
    
    avg_time = sum(r['total_time'] for r in training_results) / len(training_results)
    print(f"平均训练时间: {avg_time/3600:.1f} 小时")
    print()
    
    # 2. 最佳模型分析
    print("🏆 最佳模型分析")
    print("-" * 40)
    
    print(f"最佳模型: 训练 #{best_map50['train_num']}")
    print(f"模型类型: {best_map50['model']}")
    print(f"最佳mAP@0.5: {best_map50['best_map50']:.4f}")
    print(f"最佳mAP@0.5:0.95: {best_map50['best_map5095']:.4f}")
    print(f"最佳精度: {best_map50['best_precision']:.4f}")
    print(f"最佳召回率: {best_map50['best_recall']:.4f}")
    print(f"达到最佳性能的轮次: {best_map50['best_epoch']}")
    print()
    
    # 3. 模型对比分析
    print("🔍 模型对比分析")
    print("-" * 40)
    
    # 按模型类型分组
    model_groups = defaultdict(list)
    for result in training_results:
        model_type = result['model']
        model_groups[model_type].append(result)
    
    for model_type, group in model_groups.items():
        map50_values = [r['best_map50'] for r in group]
        map5095_values = [r['best_map5095'] for r in group]
        
        print(f"\n模型类型: {model_type}")
        print(f"  训练次数: {len(group)}")
        print(f"  平均最佳mAP@0.5: {sum(map50_values)/len(map50_values):.4f}")
        print(f"  最高mAP@0.5: {max(map50_values):.4f}")
        print(f"  平均最佳mAP@0.5:0.95: {sum(map5095_values)/len(map5095_values):.4f}")
        print(f"  最高mAP@0.5:0.95: {max(map5095_values):.4f}")
    
    print()
    
    # 4. 训练配置分析
    print("⚙️ 训练配置分析")
    print("-" * 40)
    
    # 学习率分析
    lr_groups = defaultdict(list)
    for result in training_results:
        lr = result['lr0']
        if lr < 0.001:
            lr_groups['低学习率(<0.001)'].append(result)
        elif lr < 0.01:
            lr_groups['中学习率(0.001-0.01)'].append(result)
        else:
            lr_groups['高学习率(>0.01)'].append(result)
    
    print("学习率影响:")
    for lr_range, group in lr_groups.items():
        avg_map50 = sum(r['best_map50'] for r in group) / len(group)
        print(f"  {lr_range}: 平均mAP@0.5 = {avg_map50:.4f} ({len(group)}次训练)")
    
    # 训练轮次分析
    epoch_groups = defaultdict(list)
    for result in training_results:
        epochs = result['epochs']
        if epochs <= 50:
            epoch_groups['短训练(≤50轮)'].append(result)
        elif epochs <= 100:
            epoch_groups['中训练(51-100轮)'].append(result)
        else:
            epoch_groups['长训练(>100轮)'].append(result)
    
    print("\n训练轮次影响:")
    for epoch_range, group in epoch_groups.items():
        avg_map50 = sum(r['best_map50'] for r in group) / len(group)
        print(f"  {epoch_range}: 平均mAP@0.5 = {avg_map50:.4f} ({len(group)}次训练)")
    
    print()
    
    # 5. 损失函数分析
    print("📉 损失函数分析")
    print("-" * 40)
    
    avg_losses = {
        'Box Loss': sum(r['final_box_loss'] for r in training_results) / len(training_results),
        'Classification Loss': sum(r['final_cls_loss'] for r in training_results) / len(training_results),
        'DFL Loss': sum(r['final_dfl_loss'] for r in training_results) / len(training_results),
        'Val Box Loss': sum(r['final_val_box_loss'] for r in training_results) / len(training_results),
        'Val Classification Loss': sum(r['final_val_cls_loss'] for r in training_results) / len(training_results),
        'Val DFL Loss': sum(r['final_val_dfl_loss'] for r in training_results) / len(training_results)
    }
    
    for loss_name, avg_loss in avg_losses.items():
        print(f"{loss_name}: {avg_loss:.4f}")
    
    print()
    
    # 6. 数据增强效果分析
    print("🎨 数据增强效果分析")
    print("-" * 40)
    
    # Mosaic效果
    high_mosaic = [r for r in training_results if r['mosaic'] > 0.5]
    low_mosaic = [r for r in training_results if r['mosaic'] <= 0.5]
    
    if high_mosaic and low_mosaic:
        high_mosaic_avg = sum(r['best_map50'] for r in high_mosaic) / len(high_mosaic)
        low_mosaic_avg = sum(r['best_map50'] for r in low_mosaic) / len(low_mosaic)
        print(f"Mosaic影响: 高Mosaic({high_mosaic_avg:.4f}) vs 低Mosaic({low_mosaic_avg:.4f})")
    
    # Mixup效果
    high_mixup = [r for r in training_results if r['mixup'] > 0.3]
    low_mixup = [r for r in training_results if r['mixup'] <= 0.3]
    
    if high_mixup and low_mixup:
        high_mixup_avg = sum(r['best_map50'] for r in high_mixup) / len(high_mixup)
        low_mixup_avg = sum(r['best_map50'] for r in low_mixup) / len(low_mixup)
        print(f"Mixup影响: 高Mixup({high_mixup_avg:.4f}) vs 低Mixup({low_mixup_avg:.4f})")
    
    print()
    
    # 7. 推荐配置
    print("💡 推荐配置")
    print("-" * 40)
    
    best_config = best_map50
    
    print("基于最佳性能模型的推荐配置:")
    print(f"  模型: {best_config['model']}")
    print(f"  学习率: {best_config['lr0']}")
    print(f"  学习率衰减: {best_config['lrf']}")
    print(f"  优化器: {best_config['optimizer']}")
    print(f"  Mosaic: {best_config['mosaic']}")
    print(f"  Mixup: {best_config['mixup']}")
    print(f"  Copy Paste: {best_config['copy_paste']}")
    print(f"  数据增强: {best_config['degrees']}°旋转, {best_config['translate']}平移, {best_config['scale']}缩放")
    
    print()
    
    # 8. 训练趋势分析
    print("📈 训练趋势分析")
    print("-" * 40)
    
    # 计算训练编号与性能的相关性
    train_nums = [r['train_num'] for r in training_results]
    map50_values = [r['best_map50'] for r in training_results]
    map5095_values = [r['best_map5095'] for r in training_results]
    
    # 简单的相关性计算
    if len(train_nums) > 2:
        correlation_map50 = calculate_correlation(train_nums, map50_values)
        correlation_map5095 = calculate_correlation(train_nums, map5095_values)
        
        print(f"训练轮次与mAP@0.5相关性: {correlation_map50:.3f}")
        print(f"训练轮次与mAP@0.5:0.95相关性: {correlation_map5095:.3f}")
        
        if correlation_map50 > 0.3:
            print("✅ 模型性能随训练轮次提升明显")
        elif correlation_map50 > 0.1:
            print("📊 模型性能略有提升")
        else:
            print("⚠️ 模型性能提升不明显")
    
    print()
    print("=" * 80)
    
    # 保存详细结果到JSON文件
    save_results_to_json(training_results, 'training_results_analysis.json')
    
    return training_results

def calculate_correlation(x, y):
    """计算两个列表的皮尔逊相关系数"""
    if len(x) != len(y) or len(x) < 2:
        return 0
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] ** 2 for i in range(n))
    sum_y2 = sum(y[i] ** 2 for i in range(n))
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

def save_results_to_json(results, filename):
    """保存结果到JSON文件"""
    try:
        # 转换numpy类型为Python原生类型
        json_results = []
        for result in results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, (int, float, str, bool)):
                    json_result[key] = value
                else:
                    json_result[key] = float(value) if isinstance(value, (int, float)) else str(value)
            json_results.append(json_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 详细分析结果已保存到 {filename}")
        
    except Exception as e:
        print(f"保存JSON文件失败: {e}")

if __name__ == "__main__":
    print("开始分析训练结果...")
    
    # 运行分析
    training_results = analyze_training_results()
    
    if training_results:
        print("\n生成分析报告...")
        generate_analysis_report(training_results)
        print("\n✅ 分析完成！")
    else:
        print("\n❌ 分析失败，没有找到有效的训练结果")