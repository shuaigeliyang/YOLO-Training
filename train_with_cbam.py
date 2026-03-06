"""
使用CBAM和ASFF增强的YOLO训练脚本
这是一个完整的训练示例,可以直接运行
"""

from custom_yolo_model import EnhancedYOLOTrainer
import datetime
import os


def setup_training_logger():
    """设置训练日志"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/training_cbam_{timestamp}.txt'
    return log_filename


def log_info(log_file, message):
    """记录日志"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')


def get_optimized_config():
    """获取优化的训练配置"""
    config = {
        # 基础配置
        'data': './data.yaml',
        'epochs': 300,
        'imgsz': 640,
        'batch': 16,
        'device': 0,

        # 优化器配置
        'lr0': 0.01,
        'lrf': 0.01,
        'optimizer': 'AdamW',
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 学习率调度
        'cos_lr': True,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,

        # 其他设置
        'patience': 50,
        'save': True,
        'plots': True,
        'workers': 8,
        'amp': True,
        'pretrained': True,
        'verbose': True,
    }
    return config


if __name__ == '__main__':
    # 设置日志
    log_file = setup_training_logger()
    
    log_info(log_file, "="*60)
    log_info(log_file, "YOLO训练 - CBAM注意力机制增强版")
    log_info(log_file, "="*60)
    
    try:
        # 创建增强型训练器
        log_info(log_file, "\n步骤1: 初始化训练器")
        trainer = EnhancedYOLOTrainer(
            model_path='yolo11n.pt',
            use_cbam=True,   # 启用CBAM注意力机制
            use_asff=False   # ASFF需要修改网络结构,暂不启用
        )
        
        # 获取训练配置
        log_info(log_file, "\n步骤2: 加载训练配置")
        config = get_optimized_config()
        
        log_info(log_file, "\n训练配置:")
        log_info(log_file, f"  数据集: {config['data']}")
        log_info(log_file, f"  训练轮次: {config['epochs']}")
        log_info(log_file, f"  批次大小: {config['batch']}")
        log_info(log_file, f"  学习率: {config['lr0']}")
        log_info(log_file, f"  优化器: {config['optimizer']}")
        log_info(log_file, f"  CBAM注意力: ✓ 启用")
        
        # 开始训练
        log_info(log_file, "\n步骤3: 开始训练")
        log_info(log_file, "="*60)
        
        results = trainer.train(**config)
        
        # 训练完成
        log_info(log_file, "="*60)
        log_info(log_file, "✓ 训练完成!")
        log_info(log_file, f"日志文件: {log_file}")
        
        # 记录训练结果
        if hasattr(results, 'save_dir'):
            log_info(log_file, f"模型保存路径: {results.save_dir}")
        
        log_info(log_file, "\n提示:")
        log_info(log_file, "CBAM注意力机制可以:")
        log_info(log_file, "  - 提升特征表达能力")
        log_info(log_file, "  - 增强小目标检测")
        log_info(log_file, "  - 提高模型精度 (预期提升2-3个点mAP)")
        log_info(log_file, "="*60)
        
    except Exception as e:
        log_info(log_file, f"\n✗ 训练出错: {str(e)}")
        log_info(log_file, "\n请检查:")
        log_info(log_file, "  1. data.yaml 配置是否正确")
        log_info(log_file, "  2. GPU内存是否充足")
        log_info(log_file, "  3. 模型文件是否存在")
        raise
    
    print(f"\n详细日志已保存到: {log_file}")
