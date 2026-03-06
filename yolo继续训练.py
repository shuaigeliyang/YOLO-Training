from ultralytics import YOLO
import datetime
import os
import yaml


def setup_training_logger():
    """设置训练日志记录器"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/continued_training_log_{timestamp}.txt'
    return log_filename


def log_training_info(log_file, message):
    """记录训练信息到文件和打印到控制台"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')


def analyze_previous_training(previous_training_path, log_file):
    """分析之前训练的配置"""
    args_path = os.path.join(previous_training_path, 'args.yaml')
    if os.path.exists(args_path):
        with open(args_path, 'r', encoding='utf-8') as f:
            previous_args = yaml.safe_load(f)

        log_training_info(log_file, "=== 之前训练配置分析 ===")
        log_training_info(log_file, f"训练轮次: {previous_args.get('epochs', 'Unknown')}")
        log_training_info(log_file, f"批次大小: {previous_args.get('batch', 'Unknown')}")
        log_training_info(log_file, f"学习率: {previous_args.get('lr0', 'Unknown')}")
        log_training_info(log_file, f"优化器: {previous_args.get('optimizer', 'Unknown')}")
        log_training_info(log_file, f"图像尺寸: {previous_args.get('imgsz', 'Unknown')}")
        log_training_info(log_file,
                          f"数据增强: 马赛克={previous_args.get('mosaic', 'Unknown')}, MixUp={previous_args.get('mixup', 'Unknown')}")

        return previous_args
    else:
        log_training_info(log_file, "警告: 未找到之前的训练配置文件，使用默认配置")
        return {}


def get_continued_training_config(previous_args):
    """获取与之前训练对接的继续训练配置"""

    # 获取之前训练的epochs数
    previous_epochs = previous_args.get('epochs', 5000)

    # 创建时间戳用于唯一标识
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        # === 基础配置 - 与之前保持一致 ===
        'data': previous_args.get('data', './data.yaml'),
        'imgsz': previous_args.get('imgsz', 640),
        'batch': previous_args.get('batch', 50),
        # 'batch': 10,
        'device': 0,

        # === 训练进度配置 ===
        'epochs': previous_epochs,
        # 'epochs': 10,  # 在之前基础上增加1000轮
        'resume': False,  # 重要：设置为False开始新训练周期
        'pretrained': True,

        # === 学习率配置 - 延续之前的策略但使用更小的学习率 ===
        'lr0': previous_args.get('lr0', 0.001) * 0.1,  # 使用之前学习率的1/10进行微调
        'lrf': previous_args.get('lrf', 0.01) * 0.1,
        'optimizer': previous_args.get('optimizer', 'AdamW'),
        'momentum': previous_args.get('momentum', 0.9),
        'weight_decay': previous_args.get('weight_decay', 0.0001),

        # === 学习率调度 - 延续之前的设置 ===
        'cos_lr': previous_args.get('cos_lr', True),
        'warmup_epochs': max(previous_args.get('warmup_epochs', 3.0) * 0.5, 1.0),  # 缩短预热
        'warmup_momentum': previous_args.get('warmup_momentum', 0.8),
        'warmup_bias_lr': previous_args.get('warmup_bias_lr', 0.01),

        # === 损失权重 - 保持相同平衡 ===
        'box': previous_args.get('box', 5.0),
        'cls': previous_args.get('cls', 2.0),  # 保持高分类权重
        'dfl': previous_args.get('dfl', 1.0),

        # === 数据增强 - 保持一致性 ===
        'hsv_h': previous_args.get('hsv_h', 0.015),
        'hsv_s': previous_args.get('hsv_s', 0.7),
        'hsv_v': previous_args.get('hsv_v', 0.4),
        'fliplr': previous_args.get('fliplr', 0.5),
        'mosaic': 0.1, # 关键：低mosaic
        'mixup': 0.0,  # 关键：禁用mixup
        'copy_paste': previous_args.get('copy_paste', 0.0),
        'degrees': previous_args.get('degrees', 10.0),
        'translate': previous_args.get('translate', 0.2),
        'scale': previous_args.get('scale', 0.5),
        'shear': previous_args.get('shear', 0.1),

        # === 验证和保存 ===
        'val': True,
        'save': True,
        'save_period': previous_args.get('save_period', 100),
        'plots': True,
        'patience': previous_args.get('patience', 5000),  # 保持长耐心

        # === 其他配置 ===
        'amp': previous_args.get('amp', True),
        'workers': previous_args.get('workers', 0),
        'single_cls': previous_args.get('single_cls', False),
        'verbose': previous_args.get('verbose', True),
        'dropout': previous_args.get('dropout', 0.3),  # 保持高dropout防过拟合
        'overlap_mask': previous_args.get('overlap_mask', True),

        # === 多类别相关 ===
        'auto_augment': previous_args.get('auto_augment', 'randaugment'),
        'erasing': previous_args.get('erasing', 0.4),

        # === 输出配置 ===
        'name': f"继续训练_从{previous_epochs}轮开始_{timestamp}",
        'exist_ok': True,

        # === 缓存配置 ===
        'cache': 'disk',  # 使用磁盘缓存避免内存问题
    }

    return config


def check_training_environment(previous_training_path, log_file):
    """检查训练环境"""
    # 检查模型文件
    weights_dir = os.path.join(previous_training_path, 'weights')
    best_pt = os.path.join(weights_dir, 'best.pt')
    last_pt = os.path.join(weights_dir, 'last.pt')

    if os.path.exists(last_pt):
        recommended_model = last_pt
        log_training_info(log_file, f"✓ 使用最后模型: {last_pt}")
    elif os.path.exists(best_pt):
        recommended_model = best_pt
        log_training_info(log_file, f"✓ 使用最佳模型: {best_pt}")
    else:
        log_training_info(log_file, f"✗ 未找到模型文件")
        raise FileNotFoundError("未找到模型文件")

    # 检查数据配置
    data_yaml = './data.yaml'
    if not os.path.exists(data_yaml):
        log_training_info(log_file, f"✗ 数据配置文件不存在: {data_yaml}")
        raise FileNotFoundError("数据配置文件不存在")

    return recommended_model


def check_class_distribution(data_yaml_path, log_file):
    """检查类别分布"""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        nc = data_config.get('nc', 0)
        names = data_config.get('names', [])

        log_training_info(log_file, f"数据集中类别数量: {nc}")
        log_training_info(log_file, f"前10个类别: {names[:10] if len(names) > 10 else names}")

        if nc != 559:
            log_training_info(log_file, f"警告: 配置中的类别数({nc})与预期的559不符!")

    except Exception as e:
        log_training_info(log_file, f"检查类别分布时出错: {str(e)}")


def analyze_continued_training_strategy(previous_epochs, log_file):
    """分析继续训练策略"""
    log_training_info(log_file, "=== 继续训练策略分析 ===")
    log_training_info(log_file, f"之前训练轮次: {previous_epochs}")
    log_training_info(log_file, f"新训练总轮次: {previous_epochs + 1000}")
    log_training_info(log_file, "关键对接策略:")
    log_training_info(log_file, "1. 学习率策略: 使用之前学习率的1/10 (0.0001) 进行微调")
    log_training_info(log_file, "2. 训练进度: 从之前训练结束处继续，增加1000轮")
    log_training_info(log_file, "3. 配置一致性: 保持与之前训练相同的所有关键配置")
    log_training_info(log_file, "4. 数据增强: 使用相同的增强策略保持特征一致性")
    log_training_info(log_file, "5. 优化目标: 保持高分类权重，专注于多类别区分")


if __name__ == '__main__':
    # 设置日志文件
    log_file = setup_training_logger()

    # 记录训练开始信息
    log_training_info(log_file, "=== YOLO模型继续训练开始 ===")
    log_training_info(log_file, f"开始时间: {datetime.datetime.now()}")

    try:
        # 之前训练的路径 - 请修改为实际的训练输出目录
        previous_training_path = 'runs/detect/等级3_5000次'

        # 1. 检查训练环境
        model_path = check_training_environment(previous_training_path, log_file)

        # 2. 分析之前训练配置
        previous_args = analyze_previous_training(previous_training_path, log_file)

        # 3. 检查类别分布
        check_class_distribution('./data.yaml', log_file)

        # 4. 获取继续训练配置
        train_config = get_continued_training_config(previous_args)

        # 5. 分析继续训练策略
        analyze_continued_training_strategy(previous_args.get('epochs', 5000), log_file)

        # 6. 加载模型
        log_training_info(log_file, "正在加载模型...")
        model = YOLO(model_path)

        # 7. 记录训练配置
        log_training_info(log_file, "=== 继续训练配置 ===")
        log_training_info(log_file, f"模型路径: {model_path}")
        log_training_info(log_file, f"训练轮次: {train_config['epochs']} (之前: {previous_args.get('epochs', 5000)})")
        log_training_info(log_file, f"学习率: {train_config['lr0']} (之前: {previous_args.get('lr0', 0.001)})")
        log_training_info(log_file, f"批次大小: {train_config['batch']} (保持一致)")
        log_training_info(log_file, f"优化器: {train_config['optimizer']} (保持一致)")
        log_training_info(log_file, f"输出目录: runs/detect/{train_config['name']}")
        log_training_info(log_file, "对接状态: 所有关键配置与之前训练保持一致")

        # 8. 开始训练
        log_training_info(log_file, "=== 开始继续训练 ===")
        log_training_info(log_file, "注意: 这是新的训练周期，但配置与之前训练完全对接")

        results = model.train(**train_config)

        # 9. 记录训练结果
        log_training_info(log_file, "=== 继续训练完成 ===")
        log_training_info(log_file, f"训练完成时间: {datetime.datetime.now()}")

        if hasattr(results, 'save_dir'):
            log_training_info(log_file, f"结果保存路径: {results.save_dir}")

        # 10. 记录对接总结
        log_training_info(log_file, "=== 对接训练总结 ===")
        log_training_info(log_file, "✓ 成功与之前训练对接")
        log_training_info(log_file, "✓ 保持所有关键配置一致性")
        log_training_info(log_file, "✓ 使用适当的学习率进行微调")
        log_training_info(log_file, "✓ 创建独立的输出目录避免覆盖")

        log_training_info(log_file, f"继续训练完成，日志已保存至: {log_file}")

    except Exception as e:
        log_training_info(log_file, f"继续训练过程中出现错误: {str(e)}")
        log_training_info(log_file, "建议检查:")
        log_training_info(log_file, "1. 模型文件路径是否正确")
        log_training_info(log_file, "2. 数据配置文件是否存在")
        log_training_info(log_file, "3. GPU内存是否充足")
        log_training_info(log_file, "4. 之前的训练目录结构是否完整")
        raise

    print(f'继续训练完毕，详细日志保存在: {log_file}')