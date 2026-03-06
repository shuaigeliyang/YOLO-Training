"""
YOLO + CBAM + ASFF 完整训练方案
使用forward hook同时集成两个模块
"""

def add_cbam_asff_to_model(model):
    """
    为模型添加CBAM和ASFF模块
    CBAM: 添加到所有C3k2层
    ASFF: 添加到检测头的3个特征层
    """
    import torch.nn as nn
    from cbam_module import CBAM
    from asff_module import ASFF
    
    print("\n" + "="*70)
    print("为模型添加增强模块")
    print("="*70)
    
    modules_dict = {}
    cbam_count = 0
    asff_count = 0
    
    # 定义CBAM hook
    def make_cbam_hook(cbam_module):
        def hook(module, input, output):
            return cbam_module(output)
        return hook
    
    # 定义ASFF hook
    def make_asff_hook(asff_module):
        def hook(module, input, output):
            # ASFF需要3个尺度的特征,这里简化为单尺度增强
            return asff_module([output, output, output])
        return hook
    
    # 1. 为C3k2层添加CBAM
    for idx, (name, module) in enumerate(model.model.named_modules()):
        if 'C3k2' in str(type(module).__name__):
            try:
                if hasattr(module, 'cv2'):
                    channels = module.cv2.conv.out_channels
                else:
                    continue
                
                cbam = CBAM(channels)
                device = next(module.parameters()).device
                cbam = cbam.to(device)
                
                module.register_forward_hook(make_cbam_hook(cbam))
                
                safe_name = f"cbam_{name.replace('.', '_')}"
                modules_dict[safe_name] = cbam
                
                cbam_count += 1
                print(f"  ✓ {name} (C3k2, ch={channels}) + CBAM")
                
            except Exception as e:
                print(f"  ✗ {name} CBAM失败: {e}")
    
    # 2. 为检测头的特征层添加ASFF
    # YOLO11n的检测头在layers 16, 19, 22 (对应小、中、大目标)
    target_layers = ['model.16', 'model.19', 'model.22']
    
    for name, module in model.model.named_modules():
        if name in target_layers:
            try:
                # 获取输出通道数
                if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                    channels = module.cv2.conv.out_channels
                elif hasattr(module, 'cv1') and hasattr(module.cv1, 'conv'):
                    channels = module.cv1.conv.out_channels
                else:
                    continue
                
                # 创建ASFF模块
                asff = ASFF(level=asff_count, in_channels=[channels, channels, channels])
                device = next(module.parameters()).device
                asff = asff.to(device)
                
                module.register_forward_hook(make_asff_hook(asff))
                
                safe_name = f"asff_{name.replace('.', '_')}"
                modules_dict[safe_name] = asff
                
                asff_count += 1
                print(f"  ⚡ {name} (level={asff_count-1}, ch={channels}) + ASFF")
                
            except Exception as e:
                print(f"  ✗ {name} ASFF失败: {e}")
    
    # 保存模块到模型
    model.enhancement_modules = nn.ModuleDict(modules_dict)
    
    print(f"\n✓ 成功添加 {cbam_count} 个CBAM + {asff_count} 个ASFF模块")
    return model, cbam_count, asff_count


def main():
    import torch
    from ultralytics import YOLO
    
    print("="*70)
    print("YOLO11n + CBAM + ASFF 完整训练 (100轮)")
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
    
    # 添加CBAM和ASFF
    model, cbam_count, asff_count = add_cbam_asff_to_model(model)
    
    # 统计总参数
    total_params = sum(p.numel() for p in model.model.parameters())
    if hasattr(model, 'enhancement_modules'):
        enhancement_params = sum(p.numel() for p in model.enhancement_modules.parameters())
        total_params += enhancement_params
        print(f"\n增强模块参数: {enhancement_params:,}")
    
    print(f"增强后总参数: {total_params:,}")
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
    print(f"  ASFF: {asff_count} 个模块")
    print("  数据集: ./data.yaml (559类)")
    print("\n训练中...\n")
    
    # 训练
    try:
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
            name='yolo_cbam_asff_100epochs',
            exist_ok=True,
        )
        
        print("\n" + "="*70)
        print("✅ 训练完成!")
        print("="*70)
        print(f"\n📁 结果目录: {results.save_dir}")
        print(f"⚡ CBAM模块: {cbam_count} 个")
        print(f"⚡ ASFF模块: {asff_count} 个")
        print("\n生成的文件:")
        print("  • weights/best.pt - 最佳模型 (含CBAM+ASFF)")
        print("  • weights/last.pt - 最后模型")
        print("  • results.png - 训练曲线")
        print("  • confusion_matrix.png - 混淆矩阵")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Windows多进程保护
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()
