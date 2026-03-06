"""
使用自定义YAML配置训练YOLO + CBAM
方法二的完整实现
"""

from register_custom_modules import register_custom_modules, create_yolo11n_cbam_yaml
from ultralytics import YOLO
import datetime
import os


def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'logs/training_yaml_cbam_{timestamp}.txt'


def log(file, msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')


if __name__ == '__main__':
    log_file = setup_logger()
    
    log(log_file, "="*70)
    log(log_file, "YOLO11n + CBAM 训练 (使用自定义YAML配置)")
    log(log_file, "="*70)
    
    try:
        # 注册模块
        log(log_file, "\n步骤1: 注册CBAM模块...")
        register_custom_modules()
        
        # 创建配置
        log(log_file, "\n步骤2: 创建配置文件...")
        yaml_file = create_yolo11n_cbam_yaml()
        
        # 加载模型
        log(log_file, f"\n步骤3: 从 {yaml_file} 加载模型...")
        model = YOLO(yaml_file)
        log(log_file, "✓ 模型加载成功!")
        
        # 训练配置
        config = {
            'data': './data.yaml',
            'epochs': 300,
            'imgsz': 640,
            'batch': 16,
            'device': 0,
            'lr0': 0.01,
            'optimizer': 'AdamW',
            'cos_lr': True,
            'patience': 50,
            'save': True,
            'plots': True,
        }
        
        log(log_file, "\n步骤4: 开始训练...")
        log(log_file, f"配置: {config}")
        
        # 开始训练
        results = model.train(**config)
        
        log(log_file, "\n✓ 训练完成!")
        log(log_file, f"日志: {log_file}")
        
    except Exception as e:
        log(log_file, f"\n✗ 错误: {e}")
        raise

    print(f"\n详细日志: {log_file}")
