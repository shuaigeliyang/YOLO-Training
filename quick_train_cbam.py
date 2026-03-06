"""
快速启动脚本 - 使用CBAM增强的YOLO训练
一键启动训练,包含所有优化配置
"""

from ultralytics import YOLO
import torch
import datetime
import os


def add_cbam_to_model(model):
    """
    动态添加CBAM到模型
    通过修改模型的forward hook实现
    """
    from cbam_module import CBAM
    
    cbam_modules = {}
    
    def create_cbam_hook(layer_name, channels):
        """创建CBAM钩子"""
        if layer_name not in cbam_modules:
            cbam_modules[layer_name] = CBAM(channels)
            if torch.cuda.is_available():
                cbam_modules[layer_name] = cbam_modules[layer_name].cuda()
        
        def hook(module, input, output):
            return cbam_modules[layer_name](output)
        
        return hook
    
    # 在backbone的关键层添加CBAM
    hook_count = 0
    for name, module in model.model.named_modules():
        # 在Conv2d层后添加CBAM
        if isinstance(module, torch.nn.Conv2d):
            # 只在特定层添加(避免过多计算)
            if 'model.2' in name or 'model.4' in name or 'model.6' in name:
                channels = module.out_channels
                module.register_forward_hook(create_cbam_hook(name, channels))
                hook_count += 1
                print(f"  ✓ 在 {name} 添加CBAM (channels={channels})")
    
    print(f"\n✓ 共添加 {hook_count} 个CBAM模块")
    return model


def train_with_cbam():
    """使用CBAM训练YOLO"""
    
    print("="*70)
    print("🚀 YOLO + CBAM 注意力机制训练")
    print("="*70)
    
    # 设置日志
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/cbam_training_{timestamp}.txt'
    
    try:
        # 1. 加载模型
        print("\n步骤1: 加载YOLO模型...")
        model = YOLO('yolo11n.pt')
        print("✓ 模型加载完成")
        
        # 2. 添加CBAM模块
        print("\n步骤2: 添加CBAM注意力机制...")
        model = add_cbam_to_model(model)
        
        # 3. 配置训练参数
        print("\n步骤3: 配置训练参数...")
        train_config = {
            # 基础配置
            'data': './data.yaml',
            'epochs': 300,
            'imgsz': 640,
            'batch': 16,
            'device': 0,
            
            # 优化器
            'lr0': 0.01,
            'lrf': 0.01,
            'optimizer': 'AdamW',
            'weight_decay': 0.0005,
            
            # 学习率调度
            'cos_lr': True,
            'warmup_epochs': 3,
            
            # 数据增强
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            
            # 其他
            'patience': 50,
            'save': True,
            'plots': True,
            'amp': True,
            'verbose': True,
        }
        
        print("  训练配置:")
        print(f"    - 数据集: {train_config['data']}")
        print(f"    - 训练轮次: {train_config['epochs']}")
        print(f"    - 批次大小: {train_config['batch']}")
        print(f"    - 学习率: {train_config['lr0']}")
        print(f"    - 优化器: {train_config['optimizer']}")
        print(f"    - CBAM增强: ✓ 已启用")
        
        # 4. 开始训练
        print("\n步骤4: 开始训练...")
        print("="*70)
        
        results = model.train(**train_config)
        
        # 5. 训练完成
        print("\n" + "="*70)
        print("✓ 训练完成!")
        print("="*70)
        
        if hasattr(results, 'save_dir'):
            print(f"\n模型保存位置: {results.save_dir}")
        
        print(f"\n💡 CBAM注意力机制优势:")
        print("  ✓ 增强特征表达能力")
        print("  ✓ 提升小目标检测性能")
        print("  ✓ 改善密集目标识别")
        print("  ✓ 预期提升 2-3 个点 mAP")
        
        # 保存训练信息
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"训练时间: {timestamp}\n")
            f.write(f"模型: YOLO + CBAM\n")
            f.write(f"配置: {train_config}\n")
            f.write(f"结果: 训练完成\n")
        
        print(f"\n日志已保存: {log_file}")
        print("="*70 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\n✗ 训练出错: {str(e)}")
        print("\n请检查:")
        print("  1. data.yaml 文件是否存在且配置正确")
        print("  2. GPU显存是否充足 (建议8GB以上)")
        print("  3. 训练数据是否准备好")
        print("  4. PyTorch和ultralytics是否正确安装")
        raise


if __name__ == '__main__':
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "YOLO + CBAM 训练启动器" + " "*31 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    # 检查环境
    print("环境检查:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 开始训练
    train_with_cbam()
