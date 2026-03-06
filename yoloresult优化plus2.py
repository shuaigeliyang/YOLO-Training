from ultralytics import YOLO
import cv2
import os
import re
import shutil
from collections import Counter, defaultdict
import glob

# 检测图片路径
img_path = r"D:\shijue\pythonProject1\xun\YOLO\文件夹3\1.jpg"

# 获取图片文件名（不带扩展名）
img_name = os.path.splitext(os.path.basename(img_path))[0]

# 指定保存目录
save_dir = os.path.join(os.path.dirname(__file__), 'picture_result', img_name)
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

# 自动扫描所有.pt模型文件
models_root = 'models'
model_paths = glob.glob(os.path.join(models_root, '*.pt'))

print(f"找到 {len(model_paths)} 个模型文件")

# 存储每个模型的检测结果
all_detections = []
# 存储每个文字在所有等级中的最高置信度
word_highest_confidence = defaultdict(lambda: {'level': 0, 'confidence': 0.0})

# 循环加载每个模型并执行检测
for model_path in model_paths:
    # 从文件名中提取等级名称（下划线前面的内容）
    # 例如：models/等级1_5000次.pt -> 等级1
    filename = os.path.basename(model_path)
    # 获取第一个下划线前的部分作为等级名称
    level_name = filename.split('_')[0]

    print(f"正在加载模型: {level_name} - {model_path}")

    # 加载模型
    model = YOLO(model_path)

    # 为每个模型创建子文件夹，使用等级名称
    model_save_dir = os.path.join(save_dir, level_name)

    # 删除已存在的文件夹，确保每次都是全新的
    if os.path.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir, exist_ok=True)

    # 执行检测 - 使用固定名称避免生成predict2等
    print(f"使用模型 {level_name} 进行检测...")
    results = model(img_path, show=False, save=True, project=model_save_dir, name='')

    # 分析检测结果
    model_detections = []
    for r in results:
        # 获取检测框信息
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 获取类别ID和置信度
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 获取类别名称
                cls_name = r.names[cls_id]

                # 提取文字和等级部分
                match = re.match(r'([^\d]+)(\d+)', cls_name)
                if match:
                    word = match.group(1)  # 文字部分
                    level = int(match.group(2))  # 等级部分

                    model_detections.append({
                        'word': word,
                        'level': level,
                        'confidence': conf,
                        'class_name': cls_name
                    })

                    # 更新每个文字的最高置信度
                    if conf > word_highest_confidence[word]['confidence']:
                        word_highest_confidence[word] = {
                            'level': level,
                            'confidence': conf
                        }

                    print(f"检测到: {cls_name}, 置信度: {conf:.4f}")

    # 保存该模型的检测结果
    all_detections.append({
        'model_name': level_name,
        'detections': model_detections
    })

    # 保存每个模型的检测结果到文件
    result_file = os.path.join(model_save_dir, f"{img_name}_检测结果.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"模型 {level_name} 检测结果:\n")
        f.write("=" * 40 + "\n")
        for detection in model_detections:
            # 将置信度转换为百分比并保留两位小数
            confidence_percent = detection['confidence'] * 100
            f.write(f"类别: {detection['class_name']}, 置信度: {confidence_percent:.2f}%\n")
        f.write(f"\n总共检测到 {len(model_detections)} 个目标\n")

    print(f"模型 {level_name} 检测完成，结果保存在: {model_save_dir}")

# 保存所有模型的汇总结果
summary_file = os.path.join(save_dir, f"{img_name}_所有等级检测汇总.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(f"图片: {img_name}\n")
    f.write("所有等级检测结果汇总:\n")
    f.write("=" * 50 + "\n")

    for detection_group in all_detections:
        model_name = detection_group['model_name']
        detections = detection_group['detections']

        f.write(f"\n模型 {model_name}:\n")
        f.write("-" * 30 + "\n")

        if detections:
            for detection in detections:
                # 将置信度转换为百分比并保留两位小数
                confidence_percent = detection['confidence'] * 100
                f.write(f"  类别: {detection['class_name']}, 置信度: {confidence_percent:.2f}%\n")
            f.write(f"  总共检测到 {len(detections)} 个目标\n")
        else:
            f.write("  未检测到任何目标\n")

# 过滤出置信度0.85以上的结果
filtered_highest_confidence = {
    word: info for word, info in word_highest_confidence.items()
    if info['confidence'] >= 0.85
}

# 保存每个文字的最高置信度结果（仅保留0.85以上）
highest_confidence_file = os.path.join(save_dir, f"{img_name}_最高置信度汇总.txt")
with open(highest_confidence_file, 'w', encoding='utf-8') as f:
    f.write(f"图片: {img_name}\n")
    f.write("每个文字的最高置信度汇总 (仅显示置信度≥85.00%):\n")
    f.write("=" * 60 + "\n")

    # 按等级排序
    sorted_words = sorted(filtered_highest_confidence.items(),
                          key=lambda x: x[1]['level'])

    if sorted_words:
        for word, info in sorted_words:
            # 将置信度转换为百分比并保留两位小数
            confidence_percent = info['confidence'] * 100
            f.write(f"文字: {word}, 最高置信度等级: {info['level']}, 置信度: {confidence_percent:.2f}%\n")
    else:
        f.write("没有找到置信度≥85.00%的检测结果\n")

print(f"\n所有检测完成！结果保存在: {save_dir}")
print(f"汇总文件: {summary_file}")
print(f"最高置信度汇总文件: {highest_confidence_file}")

# 打印每个文字的最高置信度结果（仅保留0.85以上）
print("\n每个文字的最高置信度汇总 (仅显示置信度≥85.00%):")
print("=" * 60)
if filtered_highest_confidence:
    for word, info in sorted_words:
        # 将置信度转换为百分比并保留两位小数
        confidence_percent = info['confidence'] * 100
        print(f"文字: {word}, 最高置信度等级: {info['level']}, 置信度: {confidence_percent:.2f}%")
else:
    print("没有找到置信度≥85.00%的检测结果")