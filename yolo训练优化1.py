from ultralytics import YOLO
import datetime
import os


def setup_training_logger():
    """设置训练日志记录器"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/training_log_{timestamp}.txt'
    return log_filename


def log_training_info(log_file, message):
    """记录训练信息到文件和打印到控制台"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')


def get_training_config_for_many_classes():
    """为多类别小数据集优化的训练配置（仅新增 cache='disk'，其余不变）"""
    config = {
        # 基础配置
        'data': './data.yaml',
        'epochs': 1500,
        'imgsz': 640,
        'batch': 50,
        'device': 0,

        # 优化器配置
        'lr0': 0.0005,
        'lrf': 0.001,
        'optimizer': 'AdamW',
        'momentum': 0.9,
        'weight_decay': 0.0001,

        #===训练进度配置===
        'resume': False,  # 重要：设置为False开始新训练周期
        'pretrained': True,

        # 学习率调度
        'cos_lr': True,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.01,

        # 损失权重
        'box': 5.0,
        'cls': 2.0,
        'dfl': 1.0,

        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'fliplr': 0.5,
        'mosaic': 0.1,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'degrees': 10.0,
        'translate': 0.2,
        'scale': 0.5,
        'shear': 0.1,

        # 早停和验证
        'patience': 5000,
        'val': True,
        'save': True,
        'plots': True,

        # 其他优化
        'amp': True,
        'workers': 0,
        'single_cls': False,
        'verbose': True,
        'dropout': 0.3,
        'overlap_mask': True,

        # 多类别相关
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'fraction': 1.0,

        # ★ 新增：因内存不足改为磁盘缓存
        'cache': 'disk',
        # 'workers': '0',
    }
    return config



def analyze_dataset_for_many_classes(log_file):
    """针对多类别数据集的专门分析"""
    log_training_info(log_file, "=== 多类别小数据集分析 ===")
    log_training_info(log_file, f"数据集统计: 40张图片, 每张约20个标记, 总共559个类别")
    log_training_info(log_file, "关键挑战分析:")
    log_training_info(log_file, "1. 极度类别不平衡 - 每个类别平均只有约1.4个实例")
    log_training_info(log_file, "2. 严重过拟合风险 - 数据量远小于类别数")
    log_training_info(log_file, "3. 分类任务困难 - 需要在559个类别中区分")
    log_training_info(log_file, "4. 小样本学习 - 需要强正则化和数据增强")

    log_training_info(log_file, "优化策略:")
    log_training_info(log_file, "1. 大幅增加分类损失权重 (cls=2.0)")
    log_training_info(log_file, "2. 更强的数据增强和正则化")
    log_training_info(log_file, "3. 更低的批次大小和学习率")
    log_training_info(log_file, "4. 使用预训练权重进行迁移学习")


def check_class_distribution(data_yaml_path, log_file):
    """检查类别分布"""
    import yaml
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


if __name__ == '__main__':
    # 设置日志文件
    log_file = setup_training_logger()

    # 记录训练开始信息
    log_training_info(log_file, "=== YOLO模型训练开始 (多类别优化版本) ===")
    log_training_info(log_file, f"开始时间: {datetime.datetime.now()}")

    try:
        # 检查数据配置
        check_class_distribution('./data.yaml', log_file)

        # 加载训练模型 - 使用预训练权重
        log_training_info(log_file, "正在加载预训练模型...")
        model = YOLO(r'yolo11n.pt')

        # 获取专门优化的训练配置
        train_config = get_training_config_for_many_classes()

        # 记录模型信息
        log_training_info(log_file, f"模型名称: YOLOv8 (使用预训练权重)")
        log_training_info(log_file, f"模型类别数: {model.model.nc if hasattr(model.model, 'nc') else 'Unknown'}")

        # 分析数据集特性
        analyze_dataset_for_many_classes(log_file)

        # 记录训练配置
        log_training_info(log_file, "=== 多类别优化训练配置 ===")
        log_training_info(log_file, f"数据集配置文件: {train_config['data']}")
        log_training_info(log_file, f"训练轮次: {train_config['epochs']} (针对多类别增加)")
        log_training_info(log_file, f"批次大小: {train_config['batch']} (进一步减小避免内存问题)")
        log_training_info(log_file, f"学习率: {train_config['lr0']} (更低的学习率)")
        log_training_info(log_file, f"分类损失权重: {train_config['cls']} (大幅提高分类重要性)")
        log_training_info(log_file, f"早停轮数: {train_config['patience']} (更长的耐心)")
        log_training_info(log_file, f"Dropout: {train_config['dropout']} (更高的dropout防过拟合)")
        log_training_info(log_file, f"数据增强: 马赛克+MixUp+复制粘贴+随机增强")

        # 开始训练模型
        log_training_info(log_file, "=== 开始模型训练 ===")
        log_training_info(log_file, "注意: 多类别小数据集训练需要耐心，收敛可能较慢")

        results = model.train(**train_config)

        # 记录训练结果
        log_training_info(log_file, "=== 训练结果汇总 ===")
        log_training_info(log_file, f"训练完成轮次: {results.epochs if hasattr(results, 'epochs') else 'Unknown'}")
        log_training_info(log_file,
                          f"最佳模型保存路径: {results.save_dir if hasattr(results, 'save_dir') else 'Unknown'}")

        # 记录指标信息
        if hasattr(results, 'metrics'):
            log_training_info(log_file, "=== 训练指标 ===")
            for key, value in results.metrics.items():
                log_training_info(log_file, f"{key}: {value}")

        # 记录优化说明
        log_training_info(log_file, "=== 多类别优化措施说明 ===")
        log_training_info(log_file, "1. 分类损失权重调整: cls从0.5提高到2.0，强调分类任务")
        log_training_info(log_file, "2. 批次大小优化: 减小到8，适应小数据集和内存限制")
        log_training_info(log_file, "3. 学习率调整: 降低到0.0005，避免多类别训练震荡")
        log_training_info(log_file, "4. 增强正则化: dropout提高到0.3，防止过拟合")
        log_training_info(log_file, "5. 延长训练: 4000轮次和5000轮早停，给模型更多学习时间")
        log_training_info(log_file, "6. 丰富数据增强: 使用多种增强技术对抗小样本问题")

        log_training_info(log_file, f"模型训练完成，日志已保存至: {log_file}")

    except Exception as e:
        log_training_info(log_file, f"训练过程中出现错误: {str(e)}")
        log_training_info(log_file, "建议检查:")
        log_training_info(log_file, "1. 数据路径和data.yaml配置")
        log_training_info(log_file, "2. GPU内存是否充足 (批次大小已设为40)")
        log_training_info(log_file, "3. 类别数量是否正确配置为559")
        raise

    print(f'模型训练完毕，详细日志保存在: {log_file}')