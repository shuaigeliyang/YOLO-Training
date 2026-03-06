"""
集成CBAM和ASFF模块的YOLO训练脚本示例

本脚本展示如何在YOLO训练中使用CBAM和ASFF模块
注意: 需要修改ultralytics库的源码才能完全集成这些模块
"""

from ultralytics import YOLO
import datetime
import os
import torch
from cbam_module import CBAM, CBAMBottleneck
from asff_module import ASFF, ASFF_2, ASFFBlock


def setup_training_logger():
    """设置训练日志记录器"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/training_log_with_modules_{timestamp}.txt'
    return log_filename


def log_training_info(log_file, message):
    """记录训练信息到文件和打印到控制台"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')


def test_modules():
    """测试CBAM和ASFF模块是否正常工作"""
    print("=== 测试模块功能 ===\n")
    
    # 测试CBAM
    print("1. 测试CBAM注意力机制:")
    x = torch.randn(2, 256, 64, 64)
    cbam = CBAM(in_channels=256)
    out = cbam(x)
    print(f"   输入: {x.shape} -> 输出: {out.shape}")
    print(f"   CBAM参数量: {sum(p.numel() for p in cbam.parameters()):,}")
    print("   ✓ CBAM模块测试通过\n")
    
    # 测试ASFF
    print("2. 测试ASFF自适应特征融合:")
    x0 = torch.randn(2, 512, 20, 20)
    x1 = torch.randn(2, 256, 40, 40)
    x2 = torch.randn(2, 128, 80, 80)
    asff = ASFF(level=1, in_channels=[512, 256, 128], out_channels=256)
    out = asff(x0, x1, x2)
    print(f"   输入: {x0.shape}, {x1.shape}, {x2.shape}")
    print(f"   输出: {out.shape}")
    print(f"   ASFF参数量: {sum(p.numel() for p in asff.parameters()):,}")
    print("   ✓ ASFF模块测试通过\n")
    
    print("=== 所有模块测试通过 ===\n")


def get_training_config_with_modules():
    """获取使用模块增强的训练配置"""
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
        'optimizer': 'SGD',
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
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,

        # 其他设置
        'patience': 50,
        'save': True,
        'plots': True,
        'workers': 8,
        'amp': True,
    }
    return config


def print_module_integration_guide():
    """打印模块集成指南"""
    guide = """
    ==========================================
    CBAM和ASFF模块集成指南
    ==========================================
    
    这两个模块已经创建完成,但需要修改ultralytics库才能完全集成到YOLO中。
    
    ### 方法一: 修改ultralytics源码 (推荐)
    
    1. CBAM集成到Backbone:
       - 找到 ultralytics/nn/modules/block.py
       - 导入: from cbam_module import CBAM, CBAMBottleneck
       - 在C2f或C3模块的Bottleneck后添加CBAM层
       
       示例代码:
       ```python
       class C2f_CBAM(nn.Module):
           def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
               super().__init__()
               self.c = int(c2 * e)
               self.cv1 = Conv(c1, 2 * self.c, 1, 1)
               self.cv2 = Conv((2 + n) * self.c, c2, 1)
               self.m = nn.ModuleList(CBAMBottleneck(self.c, self.c, shortcut, g, e=1.0) 
                                      for _ in range(n))
               self.cbam = CBAM(c2)  # 添加CBAM
       ```
    
    2. ASFF集成到Neck:
       - 找到 ultralytics/nn/modules/head.py 或相关的neck文件
       - 导入: from asff_module import ASFF, ASFFBlock
       - 在特征金字塔融合处使用ASFF替代简单的concat
       
       示例代码:
       ```python
       # 在Detect头部或PANet中
       self.asff_0 = ASFF(level=0, in_channels=[512,256,128], out_channels=256)
       self.asff_1 = ASFF(level=1, in_channels=[512,256,128], out_channels=256)
       self.asff_2 = ASFF(level=2, in_channels=[512,256,128], out_channels=256)
       ```
    
    3. 修改配置文件:
       - 找到模型配置yaml文件 (如 yolov8n.yaml)
       - 添加CBAM和ASFF模块到网络结构中
    
    ### 方法二: 使用自定义模型 (快速测试)
    
    1. 创建自定义YOLO配置文件 (yolo_cbam_asff.yaml)
    2. 手动定义包含CBAM和ASFF的网络结构
    3. 使用 model = YOLO('yolo_cbam_asff.yaml') 加载
    
    ### 方法三: 后处理增强 (有限效果)
    
    可以在训练后对特征图进行CBAM处理,但效果不如直接集成
    
    ==========================================
    模块说明
    ==========================================
    
    CBAM (Convolutional Block Attention Module):
    - 作用: 增强特征表达,关注重要的通道和空间位置
    - 位置: Backbone的bottleneck后或特征提取层
    - 优势: 轻量级,参数少,涨点明显
    - 适用场景: 小目标检测、密集目标检测
    
    ASFF (Adaptively Spatial Feature Fusion):
    - 作用: 自适应融合多尺度特征
    - 位置: Neck的特征金字塔融合处
    - 优势: 解决特征冲突,提升多尺度检测
    - 适用场景: 多尺度目标、尺度变化大的场景
    
    建议组合:
    - Backbone使用CBAM增强特征
    - Neck使用ASFF融合多尺度特征
    - 可以提升2-5个点的mAP
    
    ==========================================
    """
    print(guide)


def create_custom_yolo_config():
    """创建包含CBAM和ASFF的自定义YOLO配置文件示例"""
    config_content = """# YOLOv8 with CBAM and ASFF
# 这是一个示例配置,展示如何集成CBAM和ASFF模块

# Parameters
nc: 559  # 类别数
scales: # model compound scaling constants
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

# YOLOv8.0n backbone with CBAM
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, CBAM, [128]]  # 添加CBAM
  - [-1, 1, Conv, [256, 3, 2]]  # 4-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, CBAM, [256]]  # 添加CBAM
  - [-1, 1, Conv, [512, 3, 2]]  # 7-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, CBAM, [512]]  # 添加CBAM
  - [-1, 1, Conv, [1024, 3, 2]]  # 10-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 12

# YOLOv8.0n head with ASFF
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 8], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 15
  
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 5], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 18 (P3/8-small)
  
  # 使用ASFF融合多尺度特征
  - [[12, 15, 18], 1, ASFF, [0, [1024, 512, 256], 256]]  # ASFF level 0
  - [[12, 15, 18], 1, ASFF, [1, [1024, 512, 256], 256]]  # ASFF level 1
  - [[12, 15, 18], 1, ASFF, [2, [1024, 512, 256], 256]]  # ASFF level 2
  
  - [[19, 20, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)

# 注意: 这个配置文件需要在ultralytics中注册CBAM和ASFF模块才能使用
"""
    
    config_file = 'yolo_cbam_asff.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"自定义配置文件已创建: {config_file}")
    print("注意: 使用前需要在ultralytics中注册CBAM和ASFF模块")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("YOLO with CBAM & ASFF 模块训练脚本")
    print("="*60 + "\n")
    
    # 测试模块
    test_modules()
    
    # 打印集成指南
    print_module_integration_guide()
    
    # 创建自定义配置示例
    create_custom_yolo_config()
    
    print("\n" + "="*60)
    print("提示:")
    print("1. cbam_module.py - CBAM注意力机制模块")
    print("2. asff_module.py - ASFF自适应特征融合模块")
    print("3. yolo_cbam_asff.yaml - 自定义配置文件示例")
    print("\n如果要直接使用这些模块训练,需要修改ultralytics源码")
    print("或者可以先用标准YOLO训练,然后在推理时添加这些模块")
    print("="*60 + "\n")
    
    # 询问是否继续标准训练
    print("是否使用标准YOLO配置开始训练? (模块作为独立增强)")
    print("注意: 标准训练不会直接使用CBAM和ASFF,需要修改源码集成")
    
    # 这里可以继续你原有的训练逻辑
    # 如果要真正使用模块,需要按照上面的指南修改ultralytics源码
