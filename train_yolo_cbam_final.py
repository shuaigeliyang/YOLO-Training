"""
YOLO + CBAM 有效训练方案 (简化版)
使用Wrapper包装现有模块,避免重新创建
"""

def add_cbam_to_model(model):
    """
    为模型的关键层添加CBAM
    使用forward hook方式,保证兼容性
    """
    import torch.nn as nn
    from cbam_module import CBAM
    
    print("\n" + "="*70)
    print("为模型添加CBAM模块")
    print("="*70)
    
    cbam_modules = {}
    cbam_count = 0
    
    # 定义hook函数
    def make_cbam_hook(layer_name, cbam_module):
        def hook(module, input, output):
            return cbam_module(output)
        return hook
    
    # 遍历模型层
    for idx, (name, module) in enumerate(model.model.named_modules()):
        # 在C3k2层后添加CBAM
        if 'C3k2' in str(type(module).__name__):
            try:
                # 尝试获取输出通道数
                if hasattr(module, 'cv2'):
                    channels = module.cv2.conv.out_channels
                else:
                    continue
                
                # 创建CBAM模块
                cbam = CBAM(channels)
                device = next(module.parameters()).device
                cbam = cbam.to(device)
                
                # 注册hook
                module.register_forward_hook(make_cbam_hook(name, cbam))
                
                # 保存CBAM模块(防止被垃圾回收)
                # 替换名称中的点号,避免ModuleDict错误
                safe_name = name.replace('.', '_')
                cbam_modules[safe_name] = cbam
                
                cbam_count += 1
                print(f"  ✓ {name} (C3k2, channels={channels}) + CBAM")
                
            except Exception as e:
                print(f"  ✗ {name} 失败: {e}")
    
    # 将CBAM模块附加到模型上
    model.cbam_modules = nn.ModuleDict(cbam_modules)
    
    print(f"\n✓ 成功添加 {cbam_count} 个CBAM模块")
    return model, cbam_count


def main():
    import torch
    from ultralytics import YOLO
    
    print("="*70)
    print("YOLO11n + CBAM 有效训练 (100轮)")
    print("="*70)
    
    # 环境检查
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载模型
    print("\n" + "="*70)
    print("加载YOLO11n模型")
    print("="*70)
    model = YOLO('yolo11n.pt')
    print("✓ 模型加载完成")
    
    original_params = sum(p.numel() for p in model.model.parameters())
    print(f"  原始参数: {original_params:,}")
    
    # 添加CBAM
    model, cbam_count = add_cbam_to_model(model)
    
    # 统计总参数
    total_params = sum(p.numel() for p in model.model.parameters())
    if hasattr(model, 'cbam_modules'):
        total_params += sum(p.numel() for p in model.cbam_modules.parameters())
    
    print(f"\n增强后总参数: {total_params:,}")
    print(f"新增参数: {total_params - original_params:,}")
    
    # 训练配置
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    print("\n配置:")
    print("  轮次: 100")
    print("  批次: 16")
    print("  图像: 640×640")
    print("  优化器: AdamW")
    print("  学习率: 0.01")
    print(f"  CBAM: {cbam_count} 个模块")
    print("  数据集: ./data.yaml (559类)")
    print("\n训练中...\n")
    
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
        workers=4,
        name='yolo_cbam_100epochs',
        exist_ok=False,
    )
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    print(f"\n📁 结果目录: {results.save_dir}")
    print(f"⚡ CBAM模块: {cbam_count} 个")
    print("\n生成的文件:")
    print("  • weights/best.pt - 最佳模型 (含CBAM)")
    print("  • weights/last.pt - 最后模型")
    print("  • results.png - 训练曲线")
    print("  • confusion_matrix.png - 混淆矩阵")
    print("  • F1_curve.png - F1曲线")
    print("  • PR_curve.png - PR曲线")
    print("="*70)


if __name__ == '__main__':
    # Windows多进程保护
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()
