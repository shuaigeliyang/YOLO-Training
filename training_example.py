
# 使用示例1: 标准训练 + CBAM增强
from custom_yolo_model import EnhancedYOLOTrainer

trainer = EnhancedYOLOTrainer(
    model_path='yolo11n.pt',
    use_cbam=True,  # 启用CBAM
    use_asff=False
)

results = trainer.train(
    data='./data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

# 使用示例2: 带CBAM的推理
from custom_yolo_model import CBAMEnhancedPredictor

predictor = CBAMEnhancedPredictor('runs/detect/train/weights/best.pt')
predictor.register_cbam_hooks()  # 启用CBAM增强
results = predictor.predict('test_image.jpg')

# 使用示例3: 完整训练配置
trainer = EnhancedYOLOTrainer('yolo11n.pt', use_cbam=True)
trainer.train(
    data='./data.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
    lr0=0.01,
    optimizer='AdamW',
    cos_lr=True,
    patience=50
)
