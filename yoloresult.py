from ultralytics import YOLO
import cv2
import os
import shutil
from pathlib import Path
import numpy as np

# 加载模型
a1 = YOLO('models/等级3_1500次.pt')

# 读取 新建文件夹_augmented 目录下的所有图片
source_dir = Path('新建文件夹_augmented')
if not source_dir.exists():
    print(f"错误: 目录不存在 {source_dir.absolute()}")
    exit(1)

# 获取所有图片文件
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
images = [f for f in source_dir.iterdir() if f.suffix.lower() in image_extensions]

if not images:
    print(f"未在 {source_dir} 找到任何图片")
    exit(1)

print(f"找到 {len(images)} 张图片")

# 指定保存目录 - 每次运行删除旧结果
save_dir = Path(os.path.dirname(__file__)) / 'picture_result' / 'augmented_results'

# 删除已存在的目录，确保每次都是全新的
if save_dir.exists():
    shutil.rmtree(save_dir)
    print(f"已删除旧结果: {save_dir}")

save_dir.mkdir(parents=True, exist_ok=True)

# 第一步：预检测所有图片，获取平均置信度用于排序
print("\n预检测图片以评估置信度...")
img_confidence = []
for img_path in images:
    result = a1(str(img_path), verbose=False)[0]
    if len(result.boxes) > 0:
        avg_conf = float(result.boxes.conf.mean())
    else:
        avg_conf = 0.0
    img_confidence.append((img_path, avg_conf))

# 按置信度从低到高排序
img_confidence.sort(key=lambda x: x[1])

print("\n图片置信度排序（从低到高）:")
for i, (img_path, conf) in enumerate(img_confidence, 1):
    print(f"  {i}. {img_path.name}: {conf:.2%}")

# 第二步：逐张处理并人为调整置信度阈值，实现"从差到好"的效果
print("\n开始按顺序检测并保存（置信度从低到高）...")
n = len(img_confidence)

for idx, (img_path, original_conf) in enumerate(img_confidence):
    # 动态调整置信度阈值：第一张用高阈值(0.8)只保留少量高置信框，最后一张用低阈值(0.15)保留更多框
    # 线性插值：从 0.8 降到 0.15
    conf_threshold = 0.8 - (idx / max(n-1, 1)) * 0.65
    
    result = a1(str(img_path), conf=conf_threshold, verbose=False)[0]
    
    # 手动保存结果图
    output_img = result.plot()
    output_path = save_dir / img_path.name
    cv2.imwrite(str(output_path), output_img)
    
    detected_count = len(result.boxes)
    avg_conf = float(result.boxes.conf.mean()) if detected_count > 0 else 0.0
    
    print(f"  [{idx+1}/{n}] {img_path.name} | 阈值={conf_threshold:.2f} | 检测框={detected_count} | 平均置信度={avg_conf:.2%}")

print(f"\n检测完成！结果保存在: {save_dir}")
print(f"提示: 结果已按原始置信度从低到高排序，并通过动态阈值实现'从差到好'的视觉效果")