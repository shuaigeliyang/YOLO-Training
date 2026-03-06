"""
YOLO + CBAM + ASFF 有效训练方案
使用修改后的C3k2模块，在内部集成CBAM
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2
from cbam_module import CBAM
import copy

class C3k2_CBAM(nn.Module):
    """
    C3k2 + CBAM 融合模块
    在C3k2输出后添加CBAM注意力
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c3k2 = C3k2(c1, c2, n, shortcut, g, e)
        self.cbam = CBAM(c2)
        
    def forward(self, x):
        x = self.c3k2(x)
        x = self.cbam(x)
        return x


def inject_cbam_into_model(model):
    """
    将模型中的C3k2替换为C3k2_CBAM
    """
    print("\n" + "="*70)
    print("注入CBAM模块到模型")
    print("="*70)
    
    cbam_count = 0
    
    # 遍历模型的所有子模块
    for name, module in model.model.named_children():
        if isinstance(module, nn.Sequential):
            # 遍历Sequential中的每一层
            for i, layer in enumerate(module):
                if isinstance(layer, C3k2):
                    # 获取原始参数
                    c2 = layer.cv2.conv.out_channels
                    
                    # 创建新的C3k2_CBAM模块
                    new_layer = C3k2_CBAM(
                        c1=layer.cv1.conv.in_channels,
                        c2=c2,
                        n=len(layer.m) if hasattr(layer, 'm') else 1,
                        shortcut=hasattr(layer, 'add'),
                        g=1,
                        e=0.5
                    )
                    
                    # 复制权重
                    new_layer.c3k2.load_state_dict(layer.state_dict())
                    
                    # 移动到正确的设备
                    device = next(layer.parameters()).device
                    new_layer = new_layer.to(device)
                    
                    # 替换模块
                    module[i] = new_layer
                    
                    cbam_count += 1
                    print(f"  ✓ 层 {name}.{i} (C3k2) → C3k2_CBAM (channels={c2})")
    
    print(f"\n✓ 成功注入 {cbam_count} 个CBAM模块")
    return model, cbam_count


def main():
    print("="*70)
    print("YOLO11n + CBAM 训练 (100轮)")
    print("="*70)
    
    # 环境检查
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载模型
    print("\n" + "="*70)
    print("步骤1: 加载YOLO11n模型")
    print("="*70)
    model = YOLO('yolo11n.pt')
    print("✓ 模型加载完成")
    print(f"  原始参数: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # 注入CBAM
    model, cbam_count = inject_cbam_into_model(model)
    
    # 统计新参数
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n增强后参数: {total_params:,}")
    print(f"新增参数: {total_params - 2794677:,}")
    
    # 训练配置
    print("\n" + "="*70)
    print("步骤2: 开始训练")
    print("="*70)
    print("\n训练参数:")
    print("  轮次: 100")
    print("  批次大小: 16")
    print("  图像尺寸: 640")
    print("  优化器: AdamW")
    print("  学习率: 0.01")
    print("  CBAM模块: ✓ 已注入")
    print("  数据集: ./data.yaml (559类)")
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
        workers=4,
        name='train_cbam_effective',
        exist_ok=False,
    )
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    print(f"\n结果目录: {results.save_dir}")
    print(f"CBAM模块数: {cbam_count}")
    print("\n生成的文件:")
    print("  • weights/best.pt - 最佳模型 (含CBAM)")
    print("  • weights/last.pt - 最后模型 (含CBAM)")
    print("  • results.png - 训练曲线")
    print("  • confusion_matrix.png - 混淆矩阵")
    print("="*70)
    
    # 生成结构图
    print("\n正在生成模型结构图...")
    try:
        import sys
        sys.path.insert(0, '.')
        from generate_best_model_graph import generate_trained_model_graph
        
        best_pt = results.save_dir / 'weights' / 'best.pt'
        if best_pt.exists():
            generate_trained_model_graph(str(best_pt))
    except Exception as e:
        print(f"  结构图生成失败: {e}")
        print(f"  可手动运行: python generate_best_model_graph.py")


if __name__ == '__main__':
    main()
