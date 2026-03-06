"""
生成YOLO网络结构图 - 图片格式
支持多种可视化方式：ONNX导出 + Netron、PyTorch可视化等
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import subprocess
import sys


def export_to_onnx_and_visualize(model_path='yolo11n.pt', img_size=640):
    """
    导出ONNX格式并使用Netron可视化
    这是最直观的网络结构可视化方法
    """
    print("="*70)
    print("方法1: 导出ONNX模型并使用Netron可视化")
    print("="*70)
    
    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 创建输出目录
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 导出ONNX
    model_name = Path(model_path).stem
    onnx_file = output_dir / f'{model_name}.onnx'
    
    print(f"\n🔄 导出ONNX格式...")
    print(f"   输出文件: {onnx_file}")
    
    try:
        # 导出ONNX
        model.export(
            format='onnx',
            imgsz=img_size,
            simplify=True,
            opset=12,
        )
        
        # 移动ONNX文件到visualizations目录
        default_onnx = Path(model_path).with_suffix('.onnx')
        if default_onnx.exists():
            import shutil
            shutil.move(str(default_onnx), str(onnx_file))
            print(f"   ✓ ONNX模型已保存: {onnx_file}")
        
        # 使用Netron打开
        print(f"\n🌐 使用Netron可视化...")
        print(f"   正在启动Netron服务器...")
        
        try:
            import netron
            # 在浏览器中打开
            netron.start(str(onnx_file), browse=True, port=8080)
            print(f"\n✅ Netron已启动!")
            print(f"   浏览器地址: http://localhost:8080")
            print(f"   按 Ctrl+C 停止服务器")
            
        except Exception as e:
            print(f"\n⚠ Netron启动失败: {e}")
            print(f"\n💡 手动查看方法:")
            print(f"   1. 访问 https://netron.app")
            print(f"   2. 拖拽文件: {onnx_file.absolute()}")
        
        return onnx_file
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return None


def visualize_with_matplotlib(model_path='yolo11n.pt'):
    """
    使用matplotlib绘制网络结构图
    """
    print("\n" + "="*70)
    print("方法2: 使用Matplotlib绘制结构图")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        
        # 创建图形
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
        ax.text(5, 29, 'YOLO11n 网络结构', 
                ha='center', va='top', fontsize=24, fontweight='bold',
                fontproperties='SimHei')
        
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
        
        for layer_name, layer_info in backbone_layers:
            box = FancyBboxPatch((2.5, y_pos), 5, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['backbone'],
                                edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(5, y_pos+0.3, f'{layer_name}: {layer_info}',
                    ha='center', va='center', fontsize=10,
                    fontproperties='SimHei')
            
            # CBAM标记
            if 'C3k2' in layer_name and '64' in layer_info or '128' in layer_info or '256' in layer_info:
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
        
        # ASFF标记
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
        
        head_layers = [
            ('P3/8', '小目标 (80×80)'),
            ('P4/16', '中目标 (40×40)'),
            ('P5/32', '大目标 (20×20)'),
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
        ax.text(5, y_pos+0.5, 'Output\n[Boxes, Scores, Classes]',
                ha='center', va='center', fontsize=12, fontweight='bold',
                fontproperties='SimHei')
        
        # 保存图片
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        
        model_name = Path(model_path).stem
        png_file = output_dir / f'{model_name}_structure.png'
        pdf_file = output_dir / f'{model_name}_structure.pdf'
        
        plt.tight_layout()
        plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
        
        print(f"\n✅ 结构图已保存:")
        print(f"   PNG: {png_file}")
        print(f"   PDF: {pdf_file}")
        
        # 显示图片
        plt.show()
        
        return png_file
        
    except ImportError:
        print("❌ 需要安装 matplotlib")
        print("   运行: pip install matplotlib")
        return None
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_with_pydot(model_path='yolo11n.pt'):
    """
    使用pydot和graphviz生成高质量的结构图
    """
    print("\n" + "="*70)
    print("方法3: 使用Graphviz绘制流程图")
    print("="*70)
    
    try:
        import pydot
        
        # 创建图
        graph = pydot.Dot('YOLO', graph_type='digraph', rankdir='TB')
        graph.set_node_defaults(shape='box', style='rounded,filled', fontname='SimHei')
        
        # Input
        graph.add_node(pydot.Node('input', label='Input\n3×640×640', fillcolor='lightgreen', fontsize=12))
        
        # Backbone
        prev_node = 'input'
        backbone_nodes = [
            ('conv1', 'Conv\n3→16', 'lightblue'),
            ('conv2', 'Conv\n16→32', 'lightblue'),
            ('c3k2_1', 'C3k2\n32→64', 'lightblue'),
            ('cbam1', 'CBAM\n64', 'lightcoral'),
            ('conv3', 'Conv\n64→128', 'lightblue'),
            ('c3k2_2', 'C3k2\n128', 'lightblue'),
            ('cbam2', 'CBAM\n128', 'lightcoral'),
            ('conv4', 'Conv\n128→256', 'lightblue'),
            ('c3k2_3', 'C3k2\n256', 'lightblue'),
            ('cbam3', 'CBAM\n256', 'lightcoral'),
            ('conv5', 'Conv\n256→512', 'lightblue'),
            ('c3k2_4', 'C3k2\n512', 'lightblue'),
            ('sppf', 'SPPF\n512', 'lightyellow'),
            ('c2psa', 'C2PSA\n512', 'lightyellow'),
        ]
        
        for node_id, label, color in backbone_nodes:
            graph.add_node(pydot.Node(node_id, label=label, fillcolor=color))
            graph.add_edge(pydot.Edge(prev_node, node_id))
            prev_node = node_id
        
        # Neck
        neck_nodes = [
            ('fpn', 'FPN\n特征金字塔', 'lightcyan'),
            ('pan', 'PAN\n路径聚合', 'lightcyan'),
            ('asff', 'ASFF\n自适应融合', 'lightsalmon'),
        ]
        
        for node_id, label, color in neck_nodes:
            graph.add_node(pydot.Node(node_id, label=label, fillcolor=color, fontsize=11))
            graph.add_edge(pydot.Edge(prev_node, node_id))
            prev_node = node_id
        
        # Head
        head_nodes = [
            ('head_p3', 'P3/8\n小目标', 'lightpink'),
            ('head_p4', 'P4/16\n中目标', 'lightpink'),
            ('head_p5', 'P5/32\n大目标', 'lightpink'),
        ]
        
        for node_id, label, color in head_nodes:
            graph.add_node(pydot.Node(node_id, label=label, fillcolor=color))
            graph.add_edge(pydot.Edge(prev_node, node_id))
        
        # Output
        graph.add_node(pydot.Node('output', label='Output\nDetections', fillcolor='lightgreen', fontsize=12))
        for head_node, _, _ in head_nodes:
            graph.add_edge(pydot.Edge(head_node, 'output'))
        
        # 保存
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        
        model_name = Path(model_path).stem
        png_file = output_dir / f'{model_name}_graph.png'
        pdf_file = output_dir / f'{model_name}_graph.pdf'
        
        graph.write_png(str(png_file))
        graph.write_pdf(str(pdf_file))
        
        print(f"\n✅ 流程图已保存:")
        print(f"   PNG: {png_file}")
        print(f"   PDF: {pdf_file}")
        
        return png_file
        
    except ImportError:
        print("❌ 需要安装 pydot 和 graphviz")
        print("   运行: pip install pydot")
        print("   系统: 需要安装 Graphviz (https://graphviz.org/download/)")
        return None
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return None


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎨 YOLO网络结构图生成工具 (图片格式)")
    print("="*70)
    
    model_path = 'yolo11n.pt'
    
    print("\n请选择生成方式:")
    print("1. ONNX + Netron (推荐) - 交互式网页查看")
    print("2. Matplotlib - 精美PNG/PDF结构图")
    print("3. Graphviz - 流程图风格")
    print("4. 全部生成")
    
    choice = input("\n输入选择 (1-4, 默认1): ").strip() or '1'
    
    if choice == '1':
        export_to_onnx_and_visualize(model_path)
    elif choice == '2':
        visualize_with_matplotlib(model_path)
    elif choice == '3':
        visualize_with_pydot(model_path)
    elif choice == '4':
        print("\n开始生成所有格式...\n")
        export_to_onnx_and_visualize(model_path)
        visualize_with_matplotlib(model_path)
        visualize_with_pydot(model_path)
    else:
        print("无效选择，使用默认方式...")
        export_to_onnx_and_visualize(model_path)
    
    print("\n" + "="*70)
    print("✅ 完成!")
    print("="*70)
