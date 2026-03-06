"""
方法二实现: 通过继承和扩展ultralytics模块来集成CBAM和ASFF

这个方案不需要修改ultralytics源码,通过Python的模块注册机制实现
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# 导入CBAM和ASFF模块
from cbam_module import CBAM, CBAMBottleneck
from asff_module import ASFF, ASFF_2, ASFFBlock

# 导入ultralytics的核心模块
try:
    from ultralytics.nn.modules import Conv, C3k2, Bottleneck, C2PSA, SPPF
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics import YOLO
except ImportError as e:
    print(f"导入ultralytics模块失败: {e}")
    print("请确保已安装ultralytics: pip install ultralytics")
    sys.exit(1)


def register_custom_modules():
    """
    注册自定义模块到ultralytics
    这样就可以在YAML配置文件中使用这些模块
    """
    import ultralytics.nn.modules as modules
    
    # 注册CBAM模块
    if not hasattr(modules, 'CBAM'):
        setattr(modules, 'CBAM', CBAM)
        print("✓ 已注册 CBAM 模块")
    
    if not hasattr(modules, 'CBAMBottleneck'):
        setattr(modules, 'CBAMBottleneck', CBAMBottleneck)
        print("✓ 已注册 CBAMBottleneck 模块")
    
    # 注册ASFF模块
    if not hasattr(modules, 'ASFF'):
        setattr(modules, 'ASFF', ASFF)
        print("✓ 已注册 ASFF 模块")
    
    if not hasattr(modules, 'ASFF_2'):
        setattr(modules, 'ASFF_2', ASFF_2)
        print("✓ 已注册 ASFF_2 模块")
    
    if not hasattr(modules, 'ASFFBlock'):
        setattr(modules, 'ASFFBlock', ASFFBlock)
        print("✓ 已注册 ASFFBlock 模块")
    
    print("\n所有自定义模块注册完成!")
    return True


class C3k2_CBAM(nn.Module):
    """
    C3k2模块 + CBAM注意力
    在标准C3k2后添加CBAM增强特征
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c3k2 = C3k2(c1, c2, n, shortcut, g, e)
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        x = self.c3k2(x)
        x = self.cbam(x)
        return x


def create_yolo11n_cbam_yaml():
    """创建YOLO11n + CBAM的配置文件"""
    
    yaml_content = """# YOLO11n with CBAM Attention
# 在关键层添加CBAM注意力机制,提升特征表达能力

# Parameters
nc: 559  # 类别数 (根据你的data.yaml修改)
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # YOLO11n

# YOLO11n backbone with CBAM
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, CBAM, [256]]  # 添加CBAM - 层3
  - [-1, 1, Conv, [256, 3, 2]]  # 4-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, CBAM, [512]]  # 添加CBAM - 层6
  - [-1, 1, Conv, [512, 3, 2]]  # 7-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, CBAM, [512]]  # 添加CBAM - 层9
  - [-1, 1, Conv, [1024, 3, 2]]  # 10-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 12
  - [-1, 1, C2PSA, [1024]]  # 13

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 9], 1, Concat, [1]]  # cat backbone P4
  - [-1, 2, C3k2, [512, False]]  # 16
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P3
  - [-1, 2, C3k2, [256, False]]  # 19 (P3/8-small)
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]  # cat head P4
  - [-1, 2, C3k2, [512, False]]  # 22 (P4/16-medium)
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P5
  - [-1, 2, C3k2, [1024, True]]  # 25 (P5/32-large)
  
  - [[19, 22, 25], 1, Detect, [nc]]  # Detect(P3, P4, P5)
"""
    
    yaml_file = 'yolo11n_cbam.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✓ 已创建配置文件: {yaml_file}")
    return yaml_file


def create_yolo11n_cbam_asff_yaml():
    """创建YOLO11n + CBAM + ASFF的完整配置文件"""
    
    yaml_content = """# YOLO11n with CBAM and ASFF
# CBAM: 增强backbone特征
# ASFF: 自适应融合neck特征

# Parameters
nc: 559  # 类别数
scales:
  n: [0.50, 0.25, 1024]

# Backbone with CBAM
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, CBAM, [256]]  # CBAM增强
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, CBAM, [512]]  # CBAM增强
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, CBAM, [512]]  # CBAM增强
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 1, C2PSA, [1024]]

# Head with ASFF (注意: ASFF需要特殊处理,这里先用标准结构)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  
  - [[19, 22, 25], 1, Detect, [nc]]
"""
    
    yaml_file = 'yolo11n_cbam_asff.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✓ 已创建配置文件: {yaml_file}")
    return yaml_file


def train_with_custom_yaml():
    """使用自定义YAML配置进行训练"""
    
    print("="*70)
    print("方法二: 使用自定义YAML配置训练")
    print("="*70)
    
    # 1. 注册自定义模块
    print("\n步骤1: 注册自定义模块到ultralytics")
    register_custom_modules()
    
    # 2. 创建配置文件
    print("\n步骤2: 创建YAML配置文件")
    yaml_file = create_yolo11n_cbam_yaml()
    
    # 3. 从配置文件创建模型
    print(f"\n步骤3: 从 {yaml_file} 加载模型")
    try:
        model = YOLO(yaml_file)
        print("✓ 模型加载成功!")
        
        # 显示模型信息
        print(f"\n模型信息:")
        print(f"  参数量: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"  类别数: {model.model.nc if hasattr(model.model, 'nc') else 'Unknown'}")
        
        return model, yaml_file
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("\n注意: 如果出现模块未找到的错误,可能需要:")
        print("  1. 确保cbam_module.py和asff_module.py在当前目录")
        print("  2. 检查ultralytics版本是否支持自定义模块")
        print("  3. 使用方法一(quick_train_cbam.py)作为替代")
        return None, yaml_file


def create_training_script():
    """创建完整的训练脚本"""
    
    script_content = '''"""
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
        f.write(log_msg + '\\n')


if __name__ == '__main__':
    log_file = setup_logger()
    
    log(log_file, "="*70)
    log(log_file, "YOLO11n + CBAM 训练 (使用自定义YAML配置)")
    log(log_file, "="*70)
    
    try:
        # 注册模块
        log(log_file, "\\n步骤1: 注册CBAM模块...")
        register_custom_modules()
        
        # 创建配置
        log(log_file, "\\n步骤2: 创建配置文件...")
        yaml_file = create_yolo11n_cbam_yaml()
        
        # 加载模型
        log(log_file, f"\\n步骤3: 从 {yaml_file} 加载模型...")
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
        
        log(log_file, "\\n步骤4: 开始训练...")
        log(log_file, f"配置: {config}")
        
        # 开始训练
        results = model.train(**config)
        
        log(log_file, "\\n✓ 训练完成!")
        log(log_file, f"日志: {log_file}")
        
    except Exception as e:
        log(log_file, f"\\n✗ 错误: {e}")
        raise

    print(f"\\n详细日志: {log_file}")
'''
    
    with open('train_yaml_cbam.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✓ 已创建训练脚本: train_yaml_cbam.py")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("方法二: 通过YAML配置集成CBAM和ASFF")
    print("="*70 + "\n")
    
    # 注册模块
    print("测试1: 注册自定义模块")
    register_custom_modules()
    
    # 创建配置文件
    print("\n测试2: 创建配置文件")
    create_yolo11n_cbam_yaml()
    create_yolo11n_cbam_asff_yaml()
    
    # 创建训练脚本
    print("\n测试3: 创建训练脚本")
    create_training_script()
    
    print("\n" + "="*70)
    print("✓ 方法二准备完成!")
    print("\n使用方法:")
    print("1. python register_custom_modules.py  # 测试模块注册")
    print("2. python train_yaml_cbam.py          # 开始训练")
    print("\n或直接:")
    print("python register_custom_modules.py && python train_yaml_cbam.py")
    print("="*70 + "\n")
