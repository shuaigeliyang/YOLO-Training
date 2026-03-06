"""
YOLO网络结构可视化工具
自动生成YOLO模型的网络结构图，支持标准YOLO和带CBAM/ASFF的增强版本
"""

import torch
from ultralytics import YOLO
import os
from pathlib import Path

# 可选依赖
try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


def visualize_yolo_architecture(model_path='yolo11n.pt', output_path='yolo_architecture', show_params=True):
    """
    可视化YOLO模型架构
    
    Args:
        model_path: YOLO模型路径
        output_path: 输出文件路径（不含扩展名）
        show_params: 是否显示参数信息
    
    Returns:
        保存的图片路径
    """
    print("="*70)
    print("YOLO网络结构可视化工具")
    print("="*70)
    
    # 加载模型
    print(f"\n1. 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 创建输出目录
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 生成模型摘要
    print("\n2. 生成模型摘要...")
    summary_file = output_dir / f"{output_path}_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"YOLO模型结构摘要 - {model_path}\n")
        f.write("="*70 + "\n\n")
        
        # 模型基本信息
        f.write("【基本信息】\n")
        f.write(f"模型名称: {model_path}\n")
        f.write(f"总参数量: {sum(p.numel() for p in model.model.parameters()):,}\n")
        f.write(f"可训练参数: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}\n")
        
        if hasattr(model.model, 'nc'):
            f.write(f"类别数量: {model.model.nc}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("【网络层结构】\n")
        f.write("="*70 + "\n\n")
        
        # 遍历模型层
        for idx, (name, module) in enumerate(model.model.named_modules()):
            if name:  # 跳过根模块
                # 统计参数
                params = sum(p.numel() for p in module.parameters())
                
                # 获取模块类型
                module_type = type(module).__name__
                
                # 格式化输出
                if params > 0:
                    f.write(f"[{idx:3d}] {name:50s} {module_type:20s} ({params:,} params)\n")
                else:
                    f.write(f"[{idx:3d}] {name:50s} {module_type:20s}\n")
    
    print(f"   ✓ 模型摘要已保存: {summary_file}")
    
    # 使用torchviz生成计算图
    print("\n3. 生成计算图...")
    try:
        # 创建示例输入
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # 前向传播
        output = model.model(dummy_input)
        
        # 生成计算图
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        dot = make_dot(output, params=dict(model.model.named_parameters()))
        dot.format = 'png'
        graph_file = output_dir / f"{output_path}_graph"
        dot.render(graph_file, cleanup=True)
        
        print(f"   ✓ 计算图已保存: {graph_file}.png")
        
    except Exception as e:
        print(f"   ✗ 计算图生成失败: {e}")
        print("   提示: 需要安装 graphviz")
        print("   安装方法: pip install graphviz")
    
    # 生成简化的结构图
    print("\n4. 生成简化结构图...")
    simplified_graph = create_simplified_graph(model, output_path)
    simplified_file = output_dir / f"{output_path}_simplified.png"
    simplified_graph.render(simplified_file.stem, directory=output_dir, cleanup=True, format='png')
    
    print(f"   ✓ 简化结构图已保存: {simplified_file}")
    
    # 生成文本结构树
    print("\n5. 生成文本结构树...")
    tree_file = output_dir / f"{output_path}_tree.txt"
    create_text_tree(model, tree_file)
    print(f"   ✓ 文本结构树已保存: {tree_file}")
    
    print("\n" + "="*70)
    print("✓ 可视化完成!")
    print(f"\n所有文件已保存到: {output_dir.absolute()}")
    print("="*70 + "\n")
    
    return output_dir


def create_simplified_graph(model, output_path):
    """创建简化的网络结构图"""
    
    dot = graphviz.Digraph(comment='YOLO Architecture', format='png')
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # 添加输入节点
    dot.node('input', 'Input\n(3×640×640)', fillcolor='lightgreen')
    
    # 分析模型结构
    backbone_layers = []
    neck_layers = []
    head_layers = []
    cbam_layers = []
    asff_layers = []
    
    for name, module in model.model.named_modules():
        module_type = type(module).__name__
        
        # 识别CBAM层
        if 'CBAM' in module_type or 'cbam' in name.lower():
            cbam_layers.append((name, module_type))
        
        # 识别ASFF层
        elif 'ASFF' in module_type or 'asff' in name.lower():
            asff_layers.append((name, module_type))
        
        # 识别backbone层
        elif any(x in name for x in ['model.0', 'model.1', 'model.2', 'model.3', 'model.4', 
                                       'model.5', 'model.6', 'model.7', 'model.8', 'model.9']):
            if module_type in ['Conv', 'C3k2', 'SPPF', 'C2PSA']:
                backbone_layers.append((name, module_type))
        
        # 识别neck层
        elif any(x in name for x in ['model.1', 'model.11', 'model.12', 'model.13', 
                                      'model.14', 'model.15', 'model.16']):
            if module_type in ['Upsample', 'Concat', 'C3k2', 'Conv']:
                neck_layers.append((name, module_type))
        
        # 识别head层
        elif 'model.23' in name or 'Detect' in module_type:
            head_layers.append((name, module_type))
    
    # 添加Backbone
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Backbone', style='filled', fillcolor='lightyellow')
        prev_node = 'input'
        
        for i, (name, module_type) in enumerate(backbone_layers[:10]):
            node_id = f'backbone_{i}'
            label = f'{module_type}\n{name.split(".")[-1]}'
            c.node(node_id, label, fillcolor='lightblue')
            dot.edge(prev_node, node_id)
            prev_node = node_id
    
    # 添加CBAM (如果有)
    if cbam_layers:
        with dot.subgraph(name='cluster_cbam') as c:
            c.attr(label='CBAM Attention', style='filled', fillcolor='lightcoral')
            for i, (name, module_type) in enumerate(cbam_layers[:3]):
                node_id = f'cbam_{i}'
                c.node(node_id, f'{module_type}', fillcolor='lightcoral')
    
    # 添加Neck
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Neck (FPN+PAN)', style='filled', fillcolor='lightcyan')
        
        node_id = 'neck_fpn'
        c.node(node_id, 'FPN\n(Top-Down)', fillcolor='lightcyan')
        dot.edge(prev_node, node_id)
        
        node_id2 = 'neck_pan'
        c.node(node_id2, 'PAN\n(Bottom-Up)', fillcolor='lightcyan')
        dot.edge(node_id, node_id2)
        prev_node = node_id2
    
    # 添加ASFF (如果有)
    if asff_layers:
        with dot.subgraph(name='cluster_asff') as c:
            c.attr(label='ASFF Fusion', style='filled', fillcolor='lightsalmon')
            for i, (name, module_type) in enumerate(asff_layers[:3]):
                node_id = f'asff_{i}'
                c.node(node_id, f'{module_type}', fillcolor='lightsalmon')
    
    # 添加Head
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Detection Head', style='filled', fillcolor='lightpink')
        
        # P3输出
        c.node('head_p3', 'P3/8\n(Small)', fillcolor='lightpink')
        dot.edge(prev_node, 'head_p3')
        
        # P4输出
        c.node('head_p4', 'P4/16\n(Medium)', fillcolor='lightpink')
        dot.edge(prev_node, 'head_p4')
        
        # P5输出
        c.node('head_p5', 'P5/32\n(Large)', fillcolor='lightpink')
        dot.edge(prev_node, 'head_p5')
    
    # 添加输出节点
    dot.node('output', 'Output\n(Detections)', fillcolor='lightgreen')
    dot.edge('head_p3', 'output')
    dot.edge('head_p4', 'output')
    dot.edge('head_p5', 'output')
    
    return dot


def create_text_tree(model, output_file):
    """创建文本格式的结构树"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("YOLO网络结构树\n")
        f.write("="*70 + "\n\n")
        
        f.write("Input (3×640×640)\n")
        f.write("│\n")
        
        # Backbone
        f.write("├─ Backbone\n")
        f.write("│  ├─ Conv (3→16, stride=2)     # P1/2\n")
        f.write("│  ├─ Conv (16→32, stride=2)    # P2/4\n")
        f.write("│  ├─ C3k2 (32→64)\n")
        f.write("│  │  └─ CBAM (64) ⚡           # 注意力增强\n")
        f.write("│  ├─ Conv (64→128, stride=2)   # P3/8\n")
        f.write("│  ├─ C3k2 (128→128)\n")
        f.write("│  │  └─ CBAM (128) ⚡          # 注意力增强\n")
        f.write("│  ├─ Conv (128→256, stride=2)  # P4/16\n")
        f.write("│  ├─ C3k2 (256→256)\n")
        f.write("│  │  └─ CBAM (256) ⚡          # 注意力增强\n")
        f.write("│  ├─ Conv (256→512, stride=2)  # P5/32\n")
        f.write("│  ├─ C3k2 (512→512)\n")
        f.write("│  ├─ SPPF (512→512)            # 空间金字塔池化\n")
        f.write("│  └─ C2PSA (512→512)           # 位置感知注意力\n")
        f.write("│\n")
        
        # Neck
        f.write("├─ Neck (FPN + PAN)\n")
        f.write("│  ├─ FPN (Top-Down)\n")
        f.write("│  │  ├─ Upsample + Concat (P5→P4)\n")
        f.write("│  │  ├─ C3k2 (768→512)\n")
        f.write("│  │  ├─ Upsample + Concat (P4→P3)\n")
        f.write("│  │  └─ C3k2 (640→256)\n")
        f.write("│  │\n")
        f.write("│  ├─ ASFF 🔥                   # 自适应特征融合\n")
        f.write("│  │  ├─ ASFF Level 0 (P5)\n")
        f.write("│  │  ├─ ASFF Level 1 (P4)\n")
        f.write("│  │  └─ ASFF Level 2 (P3)\n")
        f.write("│  │\n")
        f.write("│  └─ PAN (Bottom-Up)\n")
        f.write("│     ├─ Conv + Concat (P3→P4)\n")
        f.write("│     ├─ C3k2 (512→512)\n")
        f.write("│     ├─ Conv + Concat (P4→P5)\n")
        f.write("│     └─ C3k2 (1024→1024)\n")
        f.write("│\n")
        
        # Head
        f.write("└─ Detection Head\n")
        f.write("   ├─ P3 (256) → Small objects (8×)\n")
        f.write("   ├─ P4 (512) → Medium objects (16×)\n")
        f.write("   └─ P5 (1024) → Large objects (32×)\n")
        f.write("      └─ Output: [boxes, scores, classes]\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("模块说明:\n")
        f.write("  ⚡ CBAM: 通道和空间注意力机制\n")
        f.write("  🔥 ASFF: 自适应空间特征融合\n")
        f.write("  FPN: 特征金字塔网络 (自顶向下)\n")
        f.write("  PAN: 路径聚合网络 (自底向上)\n")
        f.write("="*70 + "\n")


def compare_architectures():
    """对比标准YOLO和增强版YOLO的结构"""
    
    print("\n" + "="*70)
    print("生成对比图: 标准YOLO vs CBAM增强版")
    print("="*70 + "\n")
    
    # 标准YOLO
    print("1. 可视化标准YOLO11n...")
    visualize_yolo_architecture('yolo11n.pt', 'yolo11n_standard')
    
    # CBAM增强版 (如果存在)
    if os.path.exists('yolo11n_cbam.yaml'):
        print("\n2. 可视化CBAM增强版...")
        try:
            from register_custom_modules import register_custom_modules
            register_custom_modules()
            visualize_yolo_architecture('yolo11n_cbam.yaml', 'yolo11n_cbam')
        except:
            print("   ⚠ CBAM版本需要先注册模块")
    
    print("\n✓ 对比图生成完成!")


def create_html_visualization(model_path='yolo11n.pt'):
    """创建交互式HTML可视化"""
    
    print("\n生成交互式HTML可视化...")
    
    model = YOLO(model_path)
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>YOLO网络结构可视化</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .layer {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #4CAF50;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .layer:hover {
            background: #e8f5e9;
            cursor: pointer;
        }
        .backbone { border-left-color: #2196F3; }
        .cbam { border-left-color: #f44336; background: #ffebee; }
        .neck { border-left-color: #FF9800; }
        .asff { border-left-color: #E91E63; background: #fce4ec; }
        .head { border-left-color: #9C27B0; }
        .params {
            float: right;
            color: #666;
            font-size: 0.9em;
        }
        .module-type {
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 YOLO网络结构可视化</h1>
        <p style="text-align: center; color: #666;">
            模型: """ + model_path + """<br>
            总参数: """ + f"{sum(p.numel() for p in model.model.parameters()):,}" + """
        </p>
        <hr>
        <div id="layers">
"""
    
    # 添加层信息
    for idx, (name, module) in enumerate(model.model.named_modules()):
        if name:
            module_type = type(module).__name__
            params = sum(p.numel() for p in module.parameters())
            
            # 确定层类型
            if 'CBAM' in module_type:
                layer_class = 'cbam'
                icon = '⚡'
            elif 'ASFF' in module_type:
                layer_class = 'asff'
                icon = '🔥'
            elif any(x in name for x in ['model.0', 'model.1', 'model.2', 'model.3', 'model.4', 
                                          'model.5', 'model.6', 'model.7', 'model.8', 'model.9']):
                layer_class = 'backbone'
                icon = '🔵'
            elif any(x in name for x in ['model.11', 'model.12', 'model.13', 'model.14', 
                                          'model.15', 'model.16']):
                layer_class = 'neck'
                icon = '🟠'
            elif 'Detect' in module_type:
                layer_class = 'head'
                icon = '🟣'
            else:
                layer_class = ''
                icon = '⚪'
            
            if params > 0:
                html_content += f"""
            <div class="layer {layer_class}">
                {icon} <span class="module-type">{module_type}</span> - {name}
                <span class="params">{params:,} params</span>
            </div>
"""
    
    html_content += """
        </div>
    </div>
    <script>
        document.querySelectorAll('.layer').forEach(layer => {
            layer.addEventListener('click', function() {
                this.style.transform = this.style.transform === 'scale(1.02)' ? 'scale(1)' : 'scale(1.02)';
            });
        });
    </script>
</body>
</html>
"""
    
    # 保存HTML
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    html_file = output_dir / 'yolo_interactive.html'
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ 交互式HTML已保存: {html_file}")
    print(f"  在浏览器中打开查看: file:///{html_file.absolute()}")
    
    return html_file


if __name__ == '__main__':
    print("\n🎨 YOLO网络结构可视化工具\n")
    
    # 检查依赖
    try:
        import graphviz
        import torchviz
    except ImportError as e:
        print("⚠ 缺少依赖包，正在安装...")
        print("\n请运行: pip install graphviz torchviz\n")
        print("注意: graphviz还需要系统级安装:")
        print("  Windows: https://graphviz.org/download/")
        print("  Linux: sudo apt-get install graphviz")
        print("  Mac: brew install graphviz\n")
    
    # 主菜单
    print("请选择:")
    print("1. 可视化标准YOLO (yolo11n.pt)")
    print("2. 对比标准版和CBAM增强版")
    print("3. 生成交互式HTML可视化")
    print("4. 全部生成")
    
    choice = input("\n输入选择 (1-4): ").strip()
    
    if choice == '1':
        visualize_yolo_architecture('yolo11n.pt', 'yolo11n')
    elif choice == '2':
        compare_architectures()
    elif choice == '3':
        create_html_visualization('yolo11n.pt')
    elif choice == '4':
        visualize_yolo_architecture('yolo11n.pt', 'yolo11n')
        create_html_visualization('yolo11n.pt')
        print("\n✓ 所有可视化已完成!")
    else:
        print("无效选择，生成标准YOLO可视化...")
        visualize_yolo_architecture('yolo11n.pt', 'yolo11n')
    
    print("\n" + "="*70)
    print("完成! 查看 visualizations/ 目录")
    print("="*70)
