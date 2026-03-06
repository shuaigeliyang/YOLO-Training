# -*- coding: utf-8 -*-
"""
Simple runner: use a specified YOLO model to detect on images in a folder.
Saves results to `runs/detect/detect_等级3_5000次` by default.
"""
import os
import sys
import shutil

MODEL_PATH = 'models/等级3_5000次.pt'
SOURCE_DIR = '文件夹3'
SAVE_NAME = 'detect_等级3_5000次'
PROJECT_DIR = 'picture_result'


def main():
    print('\n>> 检查路径')
    print('  模型: ', MODEL_PATH)
    print('  图片目录: ', SOURCE_DIR)

    if not os.path.isfile(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        sys.exit(2)

    if not os.path.isdir(SOURCE_DIR):
        print(f"错误: 图片目录不存在: {SOURCE_DIR}")
        sys.exit(3)

    # 列出一些样例文件
    imgs = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    print(f"  找到 {len(imgs)} 张图片 (显示前 10 张):")
    for i, fn in enumerate(imgs[:10], 1):
        print(f"    {i}. {fn}")

    # 延迟导入模型库，避免多进程导入冲突
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('无法导入 ultralytics 库: ', e)
        sys.exit(4)

    print('\n>> 加载模型')
    model = YOLO(MODEL_PATH)
    print('  模型加载成功')

    print('\n>> 运行推理并保存结果（这可能需要一些时间）')
    # 每次运行前删除旧的结果目录，保证 picture_result 下只有最新一次的结果
    target_dir = os.path.join(PROJECT_DIR, SAVE_NAME)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            print(f"已删除旧结果: {target_dir}")
        except Exception as e:
            print(f"删除旧结果失败: {e}")
    # 确保目标 project 目录存在
    os.makedirs(PROJECT_DIR, exist_ok=True)
    res = model.predict(
        source=SOURCE_DIR,
        save=True,
        imgsz=640,
        device=0,
        conf=0.25,
        project=PROJECT_DIR,
        name=SAVE_NAME,
        exist_ok=False,
    )

    # 保存目录: picture_result/<SAVE_NAME>
    save_dir = os.path.join(PROJECT_DIR, SAVE_NAME)
    print(f"\n>> 推理完成，结果已保存到: {save_dir}")
    print('  若需查看请打开该文件夹或使用 generate_model_graph 脚本对模型生成结构图（可选）')


if __name__ == '__main__':
    main()
