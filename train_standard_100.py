"""
标准YOLO训练 - 100轮
先用标准版本训练，确保流程正常
"""

from ultralytics import YOLO

def main():
    print("="*70)
    print("YOLO11n 标准训练 - 100轮")
    print("="*70)
    
    # 加载模型
    print("\n加载YOLO11n模型...")
    model = YOLO('yolo11n.pt')
    print("✓ 模型加载完成\n")
    
    # 训练配置
    print("="*70)
    print("训练配置")
    print("="*70)
    print("  轮次: 100")
    print("  批次大小: 16")
    print("  图像尺寸: 640")
    print("  设备: CUDA:0")
    print("  优化器: AdamW")
    print("  学习率: 0.01")
    print("  数据集: ./data.yaml (559类)")
    print("\n开始训练...\n")
    
    # 开始训练
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
        workers=4,  # 减少worker数量避免多进程问题
        name='train_standard_100',
        exist_ok=False,
    )
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    print(f"\n结果目录: {results.save_dir}")
    print("\n生成的文件:")
    print("  • weights/best.pt - 最佳模型")
    print("  • weights/last.pt - 最后模型")
    print("  • results.png - 训练曲线")
    print("  • confusion_matrix.png - 混淆矩阵")
    print("="*70)

if __name__ == '__main__':
    main()
