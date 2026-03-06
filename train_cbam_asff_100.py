"""
使用CBAM+ASFF配置训练YOLO模型
"""

from register_custom_modules import register_custom_modules
from ultralytics import YOLO
import torch

# 注册自定义模块
print("="*70)
print("注册CBAM和ASFF模块...")
print("="*70)
register_custom_modules()
print("\n✓ 模块注册完成\n")

# 检查CUDA
print("="*70)
print("环境检查")
print("="*70)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# 加载配置文件
print("="*70)
print("加载YOLO11n + CBAM + ASFF配置")
print("="*70)

# 使用完整配置（CBAM + ASFF）
model = YOLO('yolo11n_cbam_asff.yaml')

print("\n✓ 模型配置加载完成")
print(f"  配置文件: yolo11n_cbam_asff.yaml")
print(f"  包含模块: CBAM注意力 + ASFF特征融合\n")

# 开始训练
print("="*70)
print("开始训练 - YOLO11n + CBAM + ASFF")
print("="*70)
print("\n训练参数:")
print("  轮次: 100")
print("  批次大小: 16")
print("  图像尺寸: 640")
print("  优化器: AdamW")
print("  学习率: 0.01")
print("  数据增强: ✓")
print("  混合精度: ✓")
print("\n开始训练...\n")

# 训练模型
results = model.train(
    data='./data.yaml',
    epochs=100,  # 100轮
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
    close_mosaic=10,
    name='train_cbam_asff',
    exist_ok=False,
)

print("\n" + "="*70)
print("✅ 训练完成!")
print("="*70)
print(f"\n结果保存在: {results.save_dir}")
print("\n生成的文件:")
print("  • weights/best.pt - 最佳模型")
print("  • weights/last.pt - 最后一轮模型")
print("  • results.png - 训练曲线")
print("  • confusion_matrix.png - 混淆矩阵")
print("="*70)
