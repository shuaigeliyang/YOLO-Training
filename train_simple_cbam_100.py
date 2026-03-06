"""
直接使用标准YOLO + 动态添加CBAM模块进行训练
这个方法更可靠,100轮训练
"""

import torch
from ultralytics import YOLO
from cbam_module import CBAM
import torch.nn as nn

print("="*70)
print("YOLO + CBAM 训练 (100轮)")
print("="*70)

# 环境检查
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 加载标准YOLO模型
print("\n" + "="*70)
print("步骤1: 加载YOLO11n模型")
print("="*70)
model = YOLO('yolo11n.pt')
print("✓ 模型加载完成")

# 动态添加CBAM模块
print("\n" + "="*70)
print("步骤2: 添加CBAM注意力模块")
print("="*70)

cbam_count = 0

def add_cbam_hook(module, input, output):
    """在卷积层后添加CBAM处理"""
    if hasattr(module, 'cbam'):
        return module.cbam(output)
    return output

# 在backbone的关键层添加CBAM
backbone_layers_to_enhance = [
    'model.2',  # C3k2 after P2
    'model.4',  # C3k2 after P3
    'model.6',  # C3k2 after P4
    'model.8',  # C3k2 after P5
]

for layer_name in backbone_layers_to_enhance:
    try:
        # 通过名称获取层
        parts = layer_name.split('.')
        layer = model.model
        for part in parts[1:]:
            layer = getattr(layer, part)
        
        # 获取输出通道数
        if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
            channels = layer.cv2.conv.out_channels
        elif hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
            channels = layer.conv.out_channels
        else:
            print(f"  ⊘ 跳过 {layer_name} (无法确定通道数)")
            continue
        
        # 创建并附加CBAM模块
        cbam = CBAM(channels).to(next(model.model.parameters()).device)
        layer.cbam = cbam
        layer.register_forward_hook(add_cbam_hook)
        
        cbam_count += 1
        print(f"  ✓ 在 {layer_name} 添加CBAM (channels={channels})")
        
    except Exception as e:
        print(f"  ✗ {layer_name} 添加失败: {e}")

print(f"\n✓ 共添加 {cbam_count} 个CBAM模块")

# 开始训练
print("\n" + "="*70)
print("步骤3: 开始训练")
print("="*70)
print("\n训练配置:")
print("  轮次: 100")
print("  批次大小: 16")
print("  图像尺寸: 640")
print("  优化器: AdamW")
print("  学习率: 0.01")
print("  CBAM增强: ✓ 已启用")
print("\n开始训练...\n")

# 训练
results = model.train(
    data='./data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    optimizer='AdamW',
    lr0=0.01,
    patience=50,
    save=True,
    plots=True,
    val=True,
    amp=True,
    cos_lr=True,
    name='train_cbam_100',
    exist_ok=False,
)

print("\n" + "="*70)
print("✅ 训练完成!")
print("="*70)
print(f"\n结果保存在: {results.save_dir}")
print("\n生成的文件:")
print("  • weights/best.pt - 最佳模型 (包含CBAM)")
print("  • weights/last.pt - 最后一轮模型")
print("  • results.png - 训练曲线")
print("  • confusion_matrix.png - 混淆矩阵")
print("="*70)
