"""
为训练后的模型生成网络结构图
自动保存到模型所在目录
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


def generate_trained_model_graph(model_path, save_dir=None):
    """
    为训练后的模型生成结构图
    
    Args:
        model_path: 模型路径
        save_dir: 保存目录，默认为模型所在目录
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 确定保存目录
    if save_dir is None:
        save_dir = model_path.parent
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"生成模型结构图: {model_path.name}")
    print("="*70)
    
    # 加载模型
    print(f"\n📦 加载模型...")
    model = YOLO(str(model_path))
    
    # 获取模型信息
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    print(f"   模型参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 1. 生成Matplotlib结构图
    print(f"\n🎨 生成结构图...")
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    ax.axis('off')
    
    # 颜色方案
    colors = {
        'input': '#90EE90',
        'backbone': '#87CEEB',
        'cbam': '#FFB6C1',
        'neck': '#FFD700',
        'asff': '#FFA07A',
        'head': '#DDA0DD',
        'output': '#90EE90'
    }
    
    # 绘制标题
    ax.text(5, 29, f'{model_path.stem.upper()} 网络结构', 
            ha='center', va='top', fontsize=24, fontweight='bold',
            fontproperties='SimHei')
    
    ax.text(5, 28.2, f'总参数: {total_params:,}', 
            ha='center', va='top', fontsize=12,
            fontproperties='SimHei', color='gray')
    
    y_pos = 27
    
    # Input
    box = FancyBboxPatch((2, y_pos), 6, 1, 
                        boxstyle="round,pad=0.1", 
                        facecolor=colors['input'], 
                        edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.5, 'Input\n3×640×640', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            fontproperties='SimHei')
    
    y_pos -= 1.5
    
    # Backbone
    ax.text(5, y_pos+0.3, 'Backbone (特征提取)', 
            ha='center', fontsize=14, fontweight='bold',
            fontproperties='SimHei')
    y_pos -= 0.8
    
    backbone_layers = [
        ('Conv', '3→16, P1/2'),
        ('Conv', '16→32, P2/4'),
        ('C3k2', '32→64'),
        ('Conv', '64→128, P3/8'),
        ('C3k2', '128→128'),
        ('Conv', '128→256, P4/16'),
        ('C3k2', '256→256'),
        ('Conv', '256→512, P5/32'),
        ('C3k2', '512'),
        ('SPPF', '512'),
        ('C2PSA', '512'),
    ]
    
    # 检测CBAM
    has_cbam = any('CBAM' in type(m).__name__ or 'cbam' in str(m).lower() 
                   for m in model.model.modules())
    
    for layer_name, layer_info in backbone_layers:
        box = FancyBboxPatch((2.5, y_pos), 5, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['backbone'],
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, y_pos+0.3, f'{layer_name}: {layer_info}',
                ha='center', va='center', fontsize=10,
                fontproperties='SimHei')
        
        # CBAM标记（如果检测到）
        if has_cbam and 'C3k2' in layer_name and any(x in layer_info for x in ['64', '128', '256']):
            cbam_box = FancyBboxPatch((7.8, y_pos+0.1), 1.5, 0.4,
                                     boxstyle="round,pad=0.03",
                                     facecolor=colors['cbam'],
                                     edgecolor='red', linewidth=1)
            ax.add_patch(cbam_box)
            ax.text(8.55, y_pos+0.3, '⚡CBAM',
                    ha='center', va='center', fontsize=8,
                    fontproperties='SimHei')
        
        y_pos -= 0.8
    
    y_pos -= 0.5
    
    # Neck
    ax.text(5, y_pos+0.3, 'Neck (特征融合)', 
            ha='center', fontsize=14, fontweight='bold',
            fontproperties='SimHei')
    y_pos -= 0.8
    
    neck_layers = [
        ('FPN', '自顶向下'),
        ('Upsample', 'P5→P4'),
        ('Concat', 'P5+P4'),
        ('C3k2', '768→512'),
        ('Upsample', 'P4→P3'),
        ('Concat', 'P4+P3'),
        ('C3k2', '640→256'),
        ('PAN', '自底向上'),
        ('Conv', 'P3→P4'),
        ('C3k2', '512'),
        ('Conv', 'P4→P5'),
        ('C3k2', '1024'),
    ]
    
    for layer_name, layer_info in neck_layers:
        if layer_name in ['FPN', 'PAN']:
            ax.text(5, y_pos+0.3, f'• {layer_name} ({layer_info})',
                    ha='center', va='center', fontsize=10, style='italic',
                    fontproperties='SimHei')
            y_pos -= 0.6
        else:
            box = FancyBboxPatch((2.5, y_pos), 5, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['neck'],
                                edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(5, y_pos+0.3, f'{layer_name}: {layer_info}',
                    ha='center', va='center', fontsize=10,
                    fontproperties='SimHei')
            y_pos -= 0.8
    
    # ASFF标记（如果有）
    has_asff = any('ASFF' in type(m).__name__ or 'asff' in str(m).lower() 
                   for m in model.model.modules())
    
    if has_asff:
        asff_box = FancyBboxPatch((1, y_pos), 8, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['asff'],
                                 edgecolor='red', linewidth=2)
        ax.add_patch(asff_box)
        ax.text(5, y_pos+0.3, '🔥 ASFF (自适应特征融合)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                fontproperties='SimHei')
        y_pos -= 1.2
    
    # Head
    ax.text(5, y_pos+0.3, 'Detection Head (检测头)', 
            ha='center', fontsize=14, fontweight='bold',
            fontproperties='SimHei')
    y_pos -= 0.8
    
    # 获取实际类别数
    nc = getattr(model.model, 'nc', 80)
    
    head_layers = [
        ('P3/8', f'小目标 (80×80) - {nc}类'),
        ('P4/16', f'中目标 (40×40) - {nc}类'),
        ('P5/32', f'大目标 (20×20) - {nc}类'),
    ]
    
    for layer_name, layer_info in head_layers:
        box = FancyBboxPatch((2.5, y_pos), 5, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['head'],
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, y_pos+0.3, f'{layer_name}: {layer_info}',
                ha='center', va='center', fontsize=10,
                fontproperties='SimHei')
        y_pos -= 0.8
    
    y_pos -= 0.5
    
    # Output
    box = FancyBboxPatch((2, y_pos), 6, 1,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['output'],
                        edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.5, f'Output\n[Boxes, Scores, {nc} Classes]',
            ha='center', va='center', fontsize=12, fontweight='bold',
            fontproperties='SimHei')
    
    # 添加图例
    legend_y = 1.5
    ax.text(0.5, legend_y, '图例:', fontsize=10, fontweight='bold', fontproperties='SimHei')
    if has_cbam:
        ax.text(0.5, legend_y-0.5, '⚡ CBAM 注意力模块', fontsize=9, fontproperties='SimHei')
    if has_asff:
        ax.text(0.5, legend_y-1, '🔥 ASFF 特征融合', fontsize=9, fontproperties='SimHei')
    
    # 保存图片
    plt.tight_layout()
    
    png_file = save_dir / f'{model_path.stem}_structure.png'
    pdf_file = save_dir / f'{model_path.stem}_structure.pdf'
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ PNG: {png_file}")
    print(f"   ✓ PDF: {pdf_file}")
    
    # 2. 导出ONNX
    print(f"\n📤 导出ONNX模型...")
    try:
        onnx_file = save_dir / f'{model_path.stem}.onnx'
        model.export(format='onnx', imgsz=640, simplify=True)
        
        # 移动到目标目录
        default_onnx = model_path.with_suffix('.onnx')
        if default_onnx.exists():
            import shutil
            shutil.move(str(default_onnx), str(onnx_file))
            print(f"   ✓ ONNX: {onnx_file}")
            print(f"   💡 使用 https://netron.app 查看")
        
    except Exception as e:
        print(f"   ⚠ ONNX导出失败: {e}")
    
    # 3. 生成文本摘要
    print(f"\n📝 生成文本摘要...")
    summary_file = save_dir / f'{model_path.stem}_summary.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"模型结构摘要 - {model_path.name}\n")
        f.write("="*70 + "\n\n")
        
        f.write("【基本信息】\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"总参数量: {total_params:,}\n")
        f.write(f"可训练参数: {trainable_params:,}\n")
        f.write(f"类别数量: {nc}\n")
        f.write(f"CBAM模块: {'✓ 已集成' if has_cbam else '✗ 未检测到'}\n")
        f.write(f"ASFF模块: {'✓ 已集成' if has_asff else '✗ 未检测到'}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("【层统计】\n")
        f.write("="*70 + "\n\n")
        
        # 统计层类型
        layer_types = {}
        for name, module in model.model.named_modules():
            if name:
                module_type = type(module).__name__
                layer_types[module_type] = layer_types.get(module_type, 0) + 1
        
        for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"{layer_type:20s} × {count}\n")
    
    print(f"   ✓ 摘要: {summary_file}")
    
    print("\n" + "="*70)
    print("✅ 完成！")
    print(f"\n📁 保存位置: {save_dir.absolute()}")
    print("\n生成的文件:")
    print(f"  • {png_file.name} - 高清结构图 (PNG)")
    print(f"  • {pdf_file.name} - 矢量图 (PDF)")
    if (save_dir / f'{model_path.stem}.onnx').exists():
        print(f"  • {model_path.stem}.onnx - ONNX模型")
    print(f"  • {summary_file.name} - 文本摘要")
    print("="*70 + "\n")
    
    return png_file


if __name__ == '__main__':
    # 生成训练后最佳模型的结构图
    model_path = Path('runs/detect/train/weights/best.pt')
    
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("\n正在搜索其他训练结果...")
        
        # 查找最新的训练结果
        runs_dir = Path('runs/detect')
        if runs_dir.exists():
            train_dirs = sorted([d for d in runs_dir.glob('train*')], 
                              key=lambda x: x.stat().st_mtime, reverse=True)
            
            for train_dir in train_dirs:
                best_pt = train_dir / 'weights' / 'best.pt'
                if best_pt.exists():
                    print(f"✓ 找到: {best_pt}")
                    model_path = best_pt
                    break
    
    if model_path.exists():
        generate_trained_model_graph(model_path)
    else:
        print("\n未找到训练后的模型文件。")
        print("请确保已完成训练，或指定正确的模型路径。")
