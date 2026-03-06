"""
生成带CBAM+ASFF的YOLO模型结构图
"""
import torch
import os
from pathlib import Path


def generate_enhanced_model_graph(model_path, output_dir=None):
    """
    为增强模型生成结构图
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from datetime import datetime
    
    print("="*70)
    print("生成YOLO+CBAM+ASFF模型结构图")
    print("="*70)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model_dict = checkpoint if isinstance(checkpoint, dict) else {'model': checkpoint}
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(model_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成详细文本摘要
    summary_file = output_dir / "model_structure_summary.txt"
    print(f"\n生成文本摘要: {summary_file}")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YOLO11n + CBAM + ASFF 模型结构摘要\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型文件: {model_path}\n\n")
        
        # 统计信息
        if 'model' in model_dict:
            try:
                model = model_dict['model']
                if hasattr(model, 'model'):
                    f.write("基础模型层:\n")
                    f.write("-" * 80 + "\n")
                    for idx, (name, module) in enumerate(model.model.named_modules()):
                        if len(list(module.children())) == 0:  # 只显示叶子节点
                            module_type = type(module).__name__
                            f.write(f"  [{idx:3d}] {name:30s} {module_type}\n")
                
                # 检查增强模块
                if hasattr(model, 'enhancement_modules'):
                    f.write("\n" + "="*80 + "\n")
                    f.write("增强模块 (CBAM + ASFF):\n")
                    f.write("-" * 80 + "\n")
                    for name, module in model.enhancement_modules.items():
                        module_type = type(module).__name__
                        params = sum(p.numel() for p in module.parameters())
                        f.write(f"  ⚡ {name:30s} {module_type:15s} ({params:,} 参数)\n")
                
                if hasattr(model, 'cbam_modules'):
                    f.write("\n" + "="*80 + "\n")
                    f.write("CBAM模块:\n")
                    f.write("-" * 80 + "\n")
                    for name, module in model.cbam_modules.items():
                        params = sum(p.numel() for p in module.parameters())
                        f.write(f"  🔍 {name:30s} CBAM ({params:,} 参数)\n")
                
                # 总参数统计
                total_params = sum(p.numel() for p in model.parameters())
                f.write("\n" + "="*80 + "\n")
                f.write(f"总参数量: {total_params:,}\n")
                f.write("="*80 + "\n")
                
            except Exception as e:
                f.write(f"\n解析模型结构时出错: {e}\n")
        
        # 检查点信息
        f.write("\n" + "="*80 + "\n")
        f.write("检查点信息:\n")
        f.write("-" * 80 + "\n")
        for key in model_dict.keys():
            if key != 'model':
                value = model_dict[key]
                if isinstance(value, (int, float, str)):
                    f.write(f"  {key}: {value}\n")
                elif isinstance(value, dict) and len(value) < 10:
                    f.write(f"  {key}: {value}\n")
    
    print("✓ 文本摘要生成完成")
    
    # 2. 生成可视化结构图
    print("\n生成可视化结构图...")
    
    try:
        fig, ax = plt.subplots(figsize=(16, 20), dpi=300)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 30)
        ax.axis('off')
        
        # 标题
        ax.text(5, 29, 'YOLO11n + CBAM + ASFF 架构', 
                ha='center', va='top', fontsize=20, fontweight='bold',
                fontproperties='SimHei')
        
        # 图例
        legend_y = 27.5
        cbam_patch = mpatches.Rectangle((0.5, legend_y), 1, 0.3, 
                                        facecolor='#FF6B6B', edgecolor='black', linewidth=2)
        asff_patch = mpatches.Rectangle((3, legend_y), 1, 0.3,
                                        facecolor='#4ECDC4', edgecolor='black', linewidth=2)
        base_patch = mpatches.Rectangle((5.5, legend_y), 1, 0.3,
                                        facecolor='#95E1D3', edgecolor='black', linewidth=2)
        
        ax.add_patch(cbam_patch)
        ax.add_patch(asff_patch)
        ax.add_patch(base_patch)
        
        ax.text(2, legend_y+0.15, 'CBAM层', ha='center', va='center', fontsize=10, fontproperties='SimHei')
        ax.text(4.5, legend_y+0.15, 'ASFF层', ha='center', va='center', fontsize=10, fontproperties='SimHei')
        ax.text(7, legend_y+0.15, '基础层', ha='center', va='center', fontsize=10, fontproperties='SimHei')
        
        # 绘制架构
        y_pos = 25
        x_center = 5
        
        # 输入层
        input_box = mpatches.FancyBboxPatch((x_center-1, y_pos), 2, 0.5,
                                            boxstyle="round,pad=0.1", 
                                            facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(input_box)
        ax.text(x_center, y_pos+0.25, 'Input (640×640×3)', ha='center', va='center', fontsize=9)
        y_pos -= 1
        
        # Backbone (带CBAM)
        layers_info = [
            ('Conv 3×3/2', '#95E1D3', 16),
            ('Conv 3×3/2', '#95E1D3', 32),
            ('C3k2 + CBAM', '#FF6B6B', 64),
            ('Conv 3×3/2', '#95E1D3', 64),
            ('C3k2 + CBAM', '#FF6B6B', 128),
            ('Conv 3×3/2', '#95E1D3', 128),
            ('C3k2 + CBAM', '#FF6B6B', 128),
            ('Conv 3×3/2', '#95E1D3', 256),
            ('C3k2 + CBAM', '#FF6B6B', 256),
            ('SPPF', '#95E1D3', 256),
        ]
        
        ax.text(x_center, y_pos+0.3, '▼ Backbone ▼', ha='center', va='center', 
                fontsize=11, fontweight='bold', fontproperties='SimHei')
        y_pos -= 0.8
        
        for layer_name, color, channels in layers_info:
            box = mpatches.FancyBboxPatch((x_center-1.5, y_pos), 3, 0.4,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x_center, y_pos+0.2, f'{layer_name} ({channels}ch)', 
                   ha='center', va='center', fontsize=8, fontproperties='SimHei')
            
            # 绘制连接线
            ax.plot([x_center, x_center], [y_pos+0.4, y_pos+0.5], 'k-', linewidth=1.5)
            y_pos -= 0.7
        
        # Neck
        y_pos -= 0.5
        ax.text(x_center, y_pos+0.3, '▼ Neck (FPN+PAN) ▼', ha='center', va='center',
                fontsize=11, fontweight='bold', fontproperties='SimHei')
        y_pos -= 0.8
        
        neck_layers = [
            ('C2PSA', '#95E1D3', 256),
            ('Upsample', '#95E1D3', 256),
            ('Concat', '#95E1D3', 384),
            ('C3k2 + CBAM', '#FF6B6B', 128),
            ('Upsample', '#95E1D3', 128),
            ('Concat', '#95E1D3', 256),
            ('C3k2 + CBAM + ASFF', '#4ECDC4', 64),  # 小目标
            ('Conv 3×3/2', '#95E1D3', 64),
            ('Concat', '#95E1D3', 192),
            ('C3k2 + CBAM + ASFF', '#4ECDC4', 128),  # 中目标
            ('Conv 3×3/2', '#95E1D3', 128),
            ('Concat', '#95E1D3', 384),
            ('C3k2 + CBAM + ASFF', '#4ECDC4', 256),  # 大目标
        ]
        
        for layer_name, color, channels in neck_layers:
            box = mpatches.FancyBboxPatch((x_center-1.5, y_pos), 3, 0.4,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x_center, y_pos+0.2, f'{layer_name} ({channels}ch)',
                   ha='center', va='center', fontsize=8, fontproperties='SimHei')
            
            ax.plot([x_center, x_center], [y_pos+0.4, y_pos+0.5], 'k-', linewidth=1.5)
            y_pos -= 0.7
        
        # Head
        y_pos -= 0.5
        ax.text(x_center, y_pos+0.3, '▼ Detection Head ▼', ha='center', va='center',
                fontsize=11, fontweight='bold', fontproperties='SimHei')
        y_pos -= 0.8
        
        detect_box = mpatches.FancyBboxPatch((x_center-1.5, y_pos), 3, 0.5,
                                            boxstyle="round,pad=0.1",
                                            facecolor='#FFE66D', edgecolor='black', linewidth=2)
        ax.add_patch(detect_box)
        ax.text(x_center, y_pos+0.25, 'Detect (559 classes)', ha='center', va='center', 
               fontsize=9, fontweight='bold', fontproperties='SimHei')
        
        # 输出
        y_pos -= 1
        output_box = mpatches.FancyBboxPatch((x_center-1, y_pos), 2, 0.5,
                                            boxstyle="round,pad=0.1",
                                            facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(x_center, y_pos+0.25, 'Output Predictions', ha='center', va='center', fontsize=9)
        
        # 统计信息
        info_y = 1.5
        info_text = f"""
模型统计:
• CBAM模块: 8个 (通道+空间注意力)
• ASFF模块: 3个 (自适应特征融合)
• 总参数: ~2.8M
• 检测类别: 559
        """
        ax.text(0.5, info_y, info_text, ha='left', va='top', fontsize=9, 
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存PNG
        png_file = output_dir / "yolo_cbam_asff_structure.png"
        plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ PNG图片已保存: {png_file}")
        
        # 保存PDF
        pdf_file = output_dir / "yolo_cbam_asff_structure.pdf"
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"✓ PDF文件已保存: {pdf_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 尝试生成ONNX (可选)
    print("\n尝试导出ONNX格式...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        onnx_file = output_dir / "model.onnx"
        model.export(format='onnx', simplify=True)
        
        # 移动生成的ONNX文件
        generated_onnx = Path(str(model_path).replace('.pt', '.onnx'))
        if generated_onnx.exists():
            import shutil
            shutil.move(str(generated_onnx), str(onnx_file))
            print(f"✓ ONNX已保存: {onnx_file}")
            print(f"  可在 https://netron.app 查看交互式结构图")
    except Exception as e:
        print(f"⚠️ ONNX导出跳过: {e}")
    
    print("\n" + "="*70)
    print("✅ 结构图生成完成!")
    print("="*70)
    print(f"\n输出目录: {output_dir}")
    print("生成的文件:")
    print(f"  📄 model_structure_summary.txt - 详细文本摘要")
    print(f"  🖼️ yolo_cbam_asff_structure.png - 高清结构图 (300 DPI)")
    print(f"  📑 yolo_cbam_asff_structure.pdf - 矢量PDF")
    if (output_dir / "model.onnx").exists():
        print(f"  🔗 model.onnx - 交互式ONNX模型")
    print("="*70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 默认使用最新训练的模型
        model_path = "runs/detect/yolo_cbam_asff_100epochs/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("\n用法: python generate_structure_graph.py <model_path>")
        print("示例: python generate_structure_graph.py runs/detect/yolo_cbam_asff_100epochs/weights/best.pt")
    else:
        generate_enhanced_model_graph(model_path)
