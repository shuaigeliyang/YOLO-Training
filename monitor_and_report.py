"""
YOLO训练监控和报告生成脚本
持续监控训练进度,训练完成后自动生成详细报告
"""

import os
import time
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from datetime import datetime
import pandas as pd

# 配置
TRAIN_DIR = Path("runs/detect/train2")
RESULTS_CSV = TRAIN_DIR / "results.csv"
CHECK_INTERVAL = 60  # 每60秒检查一次
REPORT_PATH = Path("训练报告_train2.md")

def parse_latest_log():
    """解析最新的训练日志"""
    logs_dir = Path("logs")
    log_files = sorted(logs_dir.glob("training_log_*.txt"), key=os.path.getmtime, reverse=True)
    
    if not log_files:
        return None
    
    latest_log = log_files[0]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在读取日志: {latest_log.name}")
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取训练配置
    config = {}
    config_patterns = {
        '数据集': r'数据集配置文件: (.+)',
        '训练轮次': r'训练轮次: (\d+)',
        '批次大小': r'批次大小: (\d+)',
        '学习率': r'学习率: ([\d.]+)',
        '分类损失权重': r'分类损失权重: ([\d.]+)',
        '早停轮数': r'早停轮数: (\d+)',
        'Dropout': r'Dropout: ([\d.]+)',
    }
    
    for key, pattern in config_patterns.items():
        match = re.search(pattern, content)
        if match:
            config[key] = match.group(1)
    
    return config

def check_training_status():
    """检查训练是否完成"""
    if not TRAIN_DIR.exists():
        return False, 0, "训练尚未开始"
    
    # 检查results.csv是否存在
    if not RESULTS_CSV.exists():
        return False, 0, "训练进行中(数据文件未生成)"
    
    try:
        df = pd.read_csv(RESULTS_CSV)
        df.columns = df.columns.str.strip()  # 清理列名空格
        
        current_epoch = len(df)
        
        # 检查weights/best.pt是否存在且最近被修改
        best_pt = TRAIN_DIR / "weights" / "best.pt"
        last_pt = TRAIN_DIR / "weights" / "last.pt"
        
        if best_pt.exists() and last_pt.exists():
            # 检查文件是否在最近5分钟内被修改
            last_modified = os.path.getmtime(last_pt)
            time_diff = time.time() - last_modified
            
            if time_diff > 300:  # 超过5分钟没更新,可能训练完成
                return True, current_epoch, "训练已完成"
            else:
                return False, current_epoch, f"训练进行中(Epoch {current_epoch})"
        
        return False, current_epoch, f"训练进行中(Epoch {current_epoch})"
        
    except Exception as e:
        return False, 0, f"检查失败: {str(e)}"

def plot_training_curves(df, save_dir):
    """绘制训练曲线"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 损失函数曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Box Loss
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Box Loss')
    axes[0, 0].set_title('边界框损失 (Box Loss)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Classification Loss
    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Classification Loss')
    axes[0, 1].set_title('分类损失 (Classification Loss)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # DFL Loss
    axes[1, 0].plot(df['epoch'], df['train/dfl_loss'], label='Train', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('DFL Loss')
    axes[1, 0].set_title('分布焦点损失 (DFL Loss)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 总损失
    if 'train/total_loss' in df.columns:
        axes[1, 1].plot(df['epoch'], df['train/total_loss'], label='Total Loss', linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Total Loss')
        axes[1, 1].set_title('总损失 (Total Loss)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. mAP曲线
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # mAP50
    axes[0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('mAP50')
    axes[0].set_title('平均精度 mAP@0.5')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # mAP50-95
    axes[1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2, color='purple')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP50-95')
    axes[1].set_title('平均精度 mAP@0.5:0.95')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'map_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Precision & Recall
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precision
    axes[0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='green')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('精确率 (Precision)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('召回率 (Recall)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 训练曲线已生成")

def generate_report(df, config):
    """生成Markdown格式的训练报告"""
    
    # 计算统计信息
    best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax()]
    final_epoch = df.iloc[-1]
    
    # 绘制训练曲线
    plot_training_curves(df, "training_plots")
    
    # 生成报告
    report = f"""# YOLO训练报告

生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

---

## 📊 训练配置

| 配置项 | 值 |
|--------|-----|
| 数据集 | {config.get('数据集', 'N/A')} |
| 训练轮次 | {config.get('训练轮次', 'N/A')} |
| 批次大小 | {config.get('批次大小', 'N/A')} |
| 学习率 | {config.get('学习率', 'N/A')} |
| 分类损失权重 | {config.get('分类损失权重', 'N/A')} |
| Dropout | {config.get('Dropout', 'N/A')} |
| 早停轮数 | {config.get('早停轮数', 'N/A')} |
| 优化器 | AdamW |
| 学习率调度 | Cosine Annealing |

---

## 🎯 训练结果总结

### 最佳性能 (Epoch {int(best_epoch['epoch'])})

| 指标 | 值 |
|------|-----|
| **mAP@0.5** | **{best_epoch['metrics/mAP50(B)']:.4f}** |
| **mAP@0.5:0.95** | **{best_epoch['metrics/mAP50-95(B)']:.4f}** |
| **Precision** | **{best_epoch['metrics/precision(B)']:.4f}** |
| **Recall** | **{best_epoch['metrics/recall(B)']:.4f}** |
| Box Loss | {best_epoch['train/box_loss']:.4f} |
| Classification Loss | {best_epoch['train/cls_loss']:.4f} |
| DFL Loss | {best_epoch['train/dfl_loss']:.4f} |

### 最终性能 (Epoch {int(final_epoch['epoch'])})

| 指标 | 值 |
|------|-----|
| mAP@0.5 | {final_epoch['metrics/mAP50(B)']:.4f} |
| mAP@0.5:0.95 | {final_epoch['metrics/mAP50-95(B)']:.4f} |
| Precision | {final_epoch['metrics/precision(B)']:.4f} |
| Recall | {final_epoch['metrics/recall(B)']:.4f} |
| Box Loss | {final_epoch['train/box_loss']:.4f} |
| Classification Loss | {final_epoch['train/cls_loss']:.4f} |
| DFL Loss | {final_epoch['train/dfl_loss']:.4f} |

---

## 📈 训练曲线

### 损失函数变化

![损失曲线](training_plots/loss_curves.png)

**分析:**
- **Box Loss:** {df['train/box_loss'].iloc[0]:.4f} → {df['train/box_loss'].iloc[-1]:.4f} (下降 {((df['train/box_loss'].iloc[0] - df['train/box_loss'].iloc[-1]) / df['train/box_loss'].iloc[0] * 100):.2f}%)
- **Classification Loss:** {df['train/cls_loss'].iloc[0]:.4f} → {df['train/cls_loss'].iloc[-1]:.4f} (下降 {((df['train/cls_loss'].iloc[0] - df['train/cls_loss'].iloc[-1]) / df['train/cls_loss'].iloc[0] * 100):.2f}%)
- **DFL Loss:** {df['train/dfl_loss'].iloc[0]:.4f} → {df['train/dfl_loss'].iloc[-1]:.4f} (下降 {((df['train/dfl_loss'].iloc[0] - df['train/dfl_loss'].iloc[-1]) / df['train/dfl_loss'].iloc[0] * 100):.2f}%)

### 精度指标变化

![mAP曲线](training_plots/map_curves.png)

**分析:**
- **mAP@0.5 最高值:** {df['metrics/mAP50(B)'].max():.4f} (Epoch {df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']:.0f})
- **mAP@0.5:0.95 最高值:** {df['metrics/mAP50-95(B)'].max():.4f} (Epoch {df.loc[df['metrics/mAP50-95(B)'].idxmax(), 'epoch']:.0f})

### 精确率与召回率

![Precision & Recall](training_plots/precision_recall_curves.png)

**分析:**
- **Precision 最高值:** {df['metrics/precision(B)'].max():.4f} (Epoch {df.loc[df['metrics/precision(B)'].idxmax(), 'epoch']:.0f})
- **Recall 最高值:** {df['metrics/recall(B)'].max():.4f} (Epoch {df.loc[df['metrics/recall(B)'].idxmax(), 'epoch']:.0f})

---

## 💡 训练建议

### 模型性能评估
"""

    # 根据mAP给出评估
    final_map50 = final_epoch['metrics/mAP50(B)']
    
    if final_map50 > 0.8:
        report += "\n✅ **优秀**: 模型性能优秀,可以直接部署使用。\n"
    elif final_map50 > 0.6:
        report += "\n✓ **良好**: 模型性能良好,可考虑进一步优化或在特定场景下使用。\n"
    elif final_map50 > 0.4:
        report += "\n⚠ **一般**: 模型性能一般,建议增加训练数据或调整超参数。\n"
    elif final_map50 > 0.2:
        report += "\n⚠️ **较差**: 模型性能较差,需要检查数据质量和标注准确性。\n"
    else:
        report += "\n❌ **很差**: 模型性能很差,建议重新审视数据集、标注和训练策略。\n"
    
    report += f"""
### 可能的改进方向

1. **数据增强**: 当前使用了马赛克、MixUp和随机增强,如果性能不佳可考虑调整增强强度
2. **类别平衡**: 559个类别的数据分布可能不均衡,考虑使用类别权重或重采样
3. **学习率调整**: 当前学习率为{config.get('学习率', 'N/A')},可根据损失曲线调整
4. **早停策略**: 当前patience={config.get('早停轮数', 'N/A')},可根据训练情况调整
5. **数据质量**: 训练中检测到多个损坏的JPEG文件,建议检查数据质量

---

## 📁 模型文件

- **最佳模型**: `runs/detect/train/weights/best.pt`
- **最终模型**: `runs/detect/train/weights/last.pt`
- **训练日志**: `logs/training_log_*.txt`
- **详细结果**: `runs/detect/train/results.csv`

---

## 🔍 数据统计

| 统计项 | 值 |
|--------|-----|
| 总训练轮次 | {len(df)} |
| 最佳Epoch | {int(best_epoch['epoch'])} |
| 训练图像数 | 50 (1张因标签错误被忽略) |
| 验证图像数 | 10 |
| 类别数量 | 559 |

---

*报告由训练监控脚本自动生成*
"""
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"✅ 训练报告已生成: {REPORT_PATH.absolute()}")
    print(f"{'='*60}\n")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("YOLO训练监控程序启动")
    print("="*60 + "\n")
    
    # 解析配置
    config = parse_latest_log()
    if not config:
        print("❌ 未找到训练日志文件")
        return
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    last_epoch = 0
    
    while True:
        is_complete, current_epoch, status = check_training_status()
        
        if current_epoch != last_epoch:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
            last_epoch = current_epoch
        
        if is_complete:
            print(f"\n{'='*60}")
            print("🎉 训练已完成!")
            print(f"{'='*60}\n")
            
            # 读取results.csv
            df = pd.read_csv(RESULTS_CSV)
            df.columns = df.columns.str.strip()
            
            # 生成报告
            print("正在生成训练报告...")
            generate_report(df, config)
            
            print("\n训练监控程序已完成任务")
            break
        
        # 等待下一次检查
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
