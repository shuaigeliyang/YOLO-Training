"""
YOLO网络结构可视化工具 - 简化版
无需额外依赖，只生成文本格式的可视化
"""

import torch
from ultralytics import YOLO
from pathlib import Path


def visualize_yolo_simple(model_path='yolo11n.pt', output_path='yolo_architecture'):
    """
    生成YOLO模型的文本可视化
    """
    print("="*70)
    print("YOLO网络结构可视化工具 (简化版)")
    print("="*70)
    
    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 创建输出目录
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 1. 生成模型摘要
    print("\n1️⃣ 生成模型摘要...")
    summary_file = output_dir / f"{output_path}_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"YOLO模型结构摘要 - {model_path}\n")
        f.write("="*70 + "\n\n")
        
        # 模型基本信息
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        f.write("【基本信息】\n")
        f.write(f"模型名称: {model_path}\n")
        f.write(f"总参数量: {total_params:,}\n")
        f.write(f"可训练参数: {trainable_params:,}\n")
        
        if hasattr(model.model, 'nc'):
            f.write(f"类别数量: {model.model.nc}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("【网络层结构】\n")
        f.write("="*70 + "\n\n")
        
        # 遍历模型层
        layer_count = 0
        for name, module in model.model.named_modules():
            if name and '.' in name and name.count('.') <= 2:  # 只显示前两层
                params = sum(p.numel() for p in module.parameters())
                module_type = type(module).__name__
                
                # 添加图标
                icon = get_layer_icon(name, module_type)
                
                if params > 0:
                    f.write(f"{icon} [{layer_count:3d}] {module_type:20s} - {name:40s} ({params:,} params)\n")
                else:
                    f.write(f"{icon} [{layer_count:3d}] {module_type:20s} - {name}\n")
                layer_count += 1
    
    print(f"   ✓ 模型摘要已保存: {summary_file}")
    
    # 2. 生成结构树
    print("\n2️⃣ 生成文本结构树...")
    tree_file = output_dir / f"{output_path}_tree.txt"
    create_text_tree(model, tree_file, model_path)
    print(f"   ✓ 文本结构树已保存: {tree_file}")
    
    # 3. 生成详细层信息
    print("\n3️⃣ 生成详细层信息...")
    detail_file = output_dir / f"{output_path}_details.txt"
    create_layer_details(model, detail_file)
    print(f"   ✓ 详细层信息已保存: {detail_file}")
    
    # 4. 生成ASCII结构图
    print("\n4️⃣ 生成ASCII结构图...")
    ascii_file = output_dir / f"{output_path}_ascii.txt"
    create_ascii_diagram(model, ascii_file)
    print(f"   ✓ ASCII结构图已保存: {ascii_file}")
    
    print("\n" + "="*70)
    print("✅ 可视化完成!")
    print(f"\n📁 所有文件已保存到: {output_dir.absolute()}")
    print("\n生成的文件:")
    print(f"  • {summary_file.name} - 模型摘要")
    print(f"  • {tree_file.name} - 结构树")
    print(f"  • {detail_file.name} - 详细信息")
    print(f"  • {ascii_file.name} - ASCII图")
    print("="*70 + "\n")
    
    return output_dir


def get_layer_icon(name, module_type):
    """根据层类型返回图标"""
    if 'CBAM' in module_type:
        return '⚡'
    elif 'ASFF' in module_type:
        return '🔥'
    elif 'Conv' in module_type:
        return '🔵'
    elif 'C3k2' in module_type or 'C2PSA' in module_type:
        return '🟦'
    elif 'SPPF' in module_type:
        return '💎'
    elif 'Detect' in module_type:
        return '🎯'
    elif 'Upsample' in module_type:
        return '⬆️'
    elif 'Concat' in module_type:
        return '➕'
    else:
        return '⚪'


def create_text_tree(model, output_file, model_path):
    """创建文本格式的结构树"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("YOLO网络结构树\n")
        f.write("="*70 + "\n")
        f.write(f"模型: {model_path}\n")
        f.write("="*70 + "\n\n")
        
        f.write("Input (Batch × 3 × 640 × 640)\n")
        f.write("│\n")
        
        # Backbone
        f.write("├─ 🔵 Backbone (特征提取)\n")
        f.write("│  ├─ Conv (3→16, stride=2)      # P1/2\n")
        f.write("│  ├─ Conv (16→32, stride=2)     # P2/4\n")
        f.write("│  ├─ C3k2 (32→64)\n")
        f.write("│  │  └─ ⚡ CBAM (64)            # 注意力增强\n")
        f.write("│  ├─ Conv (64→128, stride=2)    # P3/8\n")
        f.write("│  ├─ C3k2 (128→128)\n")
        f.write("│  │  └─ ⚡ CBAM (128)           # 注意力增强\n")
        f.write("│  ├─ Conv (128→256, stride=2)   # P4/16\n")
        f.write("│  ├─ C3k2 (256→256)\n")
        f.write("│  │  └─ ⚡ CBAM (256)           # 注意力增强\n")
        f.write("│  ├─ Conv (256→512, stride=2)   # P5/32\n")
        f.write("│  ├─ C3k2 (512→512)\n")
        f.write("│  ├─ 💎 SPPF (512→512)          # 空间金字塔池化\n")
        f.write("│  └─ 🟦 C2PSA (512→512)         # 位置感知注意力\n")
        f.write("│\n")
        
        # Neck
        f.write("├─ 🟠 Neck (特征融合)\n")
        f.write("│  ├─ FPN (自顶向下)\n")
        f.write("│  │  ├─ ⬆️ Upsample (P5→P4)\n")
        f.write("│  │  ├─ ➕ Concat [P5_up, P4]\n")
        f.write("│  │  ├─ C3k2 (768→512)\n")
        f.write("│  │  ├─ ⬆️ Upsample (P4→P3)\n")
        f.write("│  │  ├─ ➕ Concat [P4_up, P3]\n")
        f.write("│  │  └─ C3k2 (640→256)\n")
        f.write("│  │\n")
        f.write("│  ├─ 🔥 ASFF (自适应特征融合)\n")
        f.write("│  │  ├─ ASFF Level 0 (P5)\n")
        f.write("│  │  ├─ ASFF Level 1 (P4)\n")
        f.write("│  │  └─ ASFF Level 2 (P3)\n")
        f.write("│  │\n")
        f.write("│  └─ PAN (自底向上)\n")
        f.write("│     ├─ Conv (P3, stride=2)\n")
        f.write("│     ├─ ➕ Concat [P3_down, P4]\n")
        f.write("│     ├─ C3k2 (512→512)\n")
        f.write("│     ├─ Conv (P4, stride=2)\n")
        f.write("│     ├─ ➕ Concat [P4_down, P5]\n")
        f.write("│     └─ C3k2 (1024→1024)\n")
        f.write("│\n")
        
        # Head
        f.write("└─ 🎯 Detection Head (检测输出)\n")
        f.write("   ├─ P3 (256通道) → 小目标 (80×80)    stride=8\n")
        f.write("   ├─ P4 (512通道) → 中目标 (40×40)    stride=16\n")
        f.write("   └─ P5 (1024通道) → 大目标 (20×20)   stride=32\n")
        f.write("      └─ Output: [boxes, scores, classes]\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("📌 模块说明:\n")
        f.write("  ⚡ CBAM    - 通道和空间双重注意力机制\n")
        f.write("  🔥 ASFF    - 自适应空间特征融合\n")
        f.write("  💎 SPPF    - 空间金字塔池化 (快速版)\n")
        f.write("  🟦 C2PSA   - 通道-位置-空间注意力\n")
        f.write("  🔵 Conv    - 卷积层\n")
        f.write("  ⬆️ Upsample - 上采样\n")
        f.write("  ➕ Concat   - 特征拼接\n")
        f.write("="*70 + "\n")


def create_layer_details(model, output_file):
    """创建详细的层信息"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("YOLO详细层信息\n")
        f.write("="*70 + "\n\n")
        
        # 统计各类型层的数量
        layer_types = {}
        total_params = 0
        
        for name, module in model.model.named_modules():
            if name:
                module_type = type(module).__name__
                params = sum(p.numel() for p in module.parameters())
                
                if module_type not in layer_types:
                    layer_types[module_type] = {'count': 0, 'params': 0}
                
                layer_types[module_type]['count'] += 1
                layer_types[module_type]['params'] = params
                total_params += params
        
        # 写入统计信息
        f.write("【层类型统计】\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'层类型':<20} {'数量':>10} {'参数量':>20}\n")
        f.write("-" * 70 + "\n")
        
        for layer_type, info in sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True):
            f.write(f"{layer_type:<20} {info['count']:>10} {info['params']:>20,}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'总计':<20} {sum(i['count'] for i in layer_types.values()):>10} {total_params:>20,}\n")
        f.write("=" * 70 + "\n\n")
        
        # 分析特殊模块
        f.write("【特殊模块分析】\n")
        f.write("-" * 70 + "\n")
        
        has_cbam = any('CBAM' in type(m).__name__ for _, m in model.model.named_modules())
        has_asff = any('ASFF' in type(m).__name__ for _, m in model.model.named_modules())
        
        if has_cbam:
            f.write("✓ 检测到 CBAM 注意力模块\n")
            f.write("  - 功能: 通道注意力 + 空间注意力\n")
            f.write("  - 优势: 自适应特征增强\n")
        else:
            f.write("✗ 未检测到 CBAM 模块\n")
        
        if has_asff:
            f.write("✓ 检测到 ASFF 特征融合模块\n")
            f.write("  - 功能: 多尺度自适应融合\n")
            f.write("  - 优势: 提升多尺度目标检测\n")
        else:
            f.write("✗ 未检测到 ASFF 模块\n")
        
        f.write("=" * 70 + "\n")


def create_ascii_diagram(model, output_file):
    """创建ASCII艺术结构图"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("YOLO网络结构 ASCII 图\n")
        f.write("="*70 + "\n\n")
        
        diagram = r"""
        ┌─────────────────────────────────────────────────┐
        │           INPUT: 3 × 640 × 640                  │
        └─────────────┬───────────────────────────────────┘
                      │
        ╔═════════════▼════════════════════════════════════╗
        ║               BACKBONE (特征提取)                 ║
        ╠═══════════════════════════════════════════════════╣
        ║  Conv (3→16)   → P1/2                            ║
        ║  Conv (16→32)  → P2/4                            ║
        ║  C3k2 + ⚡CBAM → 64 channels                     ║
        ║  Conv (64→128) → P3/8   ◄── 输出到 Neck         ║
        ║  C3k2 + ⚡CBAM → 128 channels                    ║
        ║  Conv (128→256) → P4/16 ◄── 输出到 Neck         ║
        ║  C3k2 + ⚡CBAM → 256 channels                    ║
        ║  Conv (256→512) → P5/32 ◄── 输出到 Neck         ║
        ║  C3k2 (512)                                      ║
        ║  SPPF (512) - 空间金字塔池化                      ║
        ║  C2PSA (512) - 位置感知注意力                     ║
        ╚═════════════╤═════════════════════════════════════╝
                      │
        ╔═════════════▼════════════════════════════════════╗
        ║                 NECK (特征融合)                   ║
        ╠═══════════════════════════════════════════════════╣
        ║  【FPN - 自顶向下路径】                           ║
        ║    P5 ─┬→ Upsample ─┬→ Concat ─→ C3k2 ─→ P4_out ║
        ║        │             ↑                           ║
        ║        │         P4 ─┘                           ║
        ║        │                                         ║
        ║        └→ Upsample ─┬→ Concat ─→ C3k2 ─→ P3_out ║
        ║                     ↑                            ║
        ║                 P3 ─┘                            ║
        ║                                                  ║
        ║  【🔥 ASFF - 自适应融合】                         ║
        ║    Level 0 (P5) ───┐                            ║
        ║    Level 1 (P4) ───┼─→ 自适应权重融合            ║
        ║    Level 2 (P3) ───┘                            ║
        ║                                                  ║
        ║  【PAN - 自底向上路径】                           ║
        ║    P3_out ─→ Conv ─┬→ Concat ─→ C3k2 ─→ P4_out ║
        ║                    ↑                             ║
        ║                P4 ─┘                             ║
        ║                                                  ║
        ║    P4_out ─→ Conv ─┬→ Concat ─→ C3k2 ─→ P5_out ║
        ║                    ↑                             ║
        ║                P5 ─┘                             ║
        ╚═════════════╤═══════════════════════════════════╝
                      │
        ╔═════════════▼════════════════════════════════════╗
        ║           🎯 DETECTION HEAD (检测头)              ║
        ╠═══════════════════════════════════════════════════╣
        ║  P3 (256) → 小目标检测 (8× downsampling)         ║
        ║             Grid: 80×80                          ║
        ║                                                  ║
        ║  P4 (512) → 中目标检测 (16× downsampling)        ║
        ║             Grid: 40×40                          ║
        ║                                                  ║
        ║  P5 (1024) → 大目标检测 (32× downsampling)       ║
        ║              Grid: 20×20                         ║
        ╚═════════════╤═══════════════════════════════════╝
                      │
        ┌─────────────▼───────────────────────────────────┐
        │      OUTPUT: [Boxes, Scores, Classes]           │
        └─────────────────────────────────────────────────┘
        
        图例:
        ⚡ = CBAM 注意力模块
        🔥 = ASFF 自适应特征融合
        ─→ = 数据流向
        ┬ ┴ ├ ┤ = 连接点
        """
        
        f.write(diagram)
        f.write("\n" + "="*70 + "\n")


if __name__ == '__main__':
    print("\n🎨 YOLO网络结构可视化工具 (简化版)\n")
    print("说明: 此版本无需安装额外依赖，生成文本格式的可视化\n")
    
    # 主菜单
    print("请选择:")
    print("1. 可视化标准YOLO (yolo11n.pt)")
    print("2. 可视化当前目录的所有.pt模型")
    
    choice = input("\n输入选择 (1-2, 默认1): ").strip() or '1'
    
    if choice == '1':
        visualize_yolo_simple('yolo11n.pt', 'yolo11n')
    elif choice == '2':
        from pathlib import Path
        pt_files = list(Path('.').glob('*.pt'))
        
        if not pt_files:
            print("❌ 未找到.pt模型文件!")
        else:
            print(f"\n找到 {len(pt_files)} 个模型文件:")
            for i, pt in enumerate(pt_files, 1):
                print(f"  {i}. {pt.name}")
            
            for pt in pt_files:
                print(f"\n处理: {pt.name}")
                try:
                    visualize_yolo_simple(str(pt), pt.stem)
                except Exception as e:
                    print(f"  ❌ 失败: {e}")
    else:
        print("无效选择，生成标准YOLO可视化...")
        visualize_yolo_simple('yolo11n.pt', 'yolo11n')
    
    print("\n" + "="*70)
    print("✅ 完成! 查看 visualizations/ 目录")
    print("="*70)
