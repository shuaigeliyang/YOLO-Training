from ultralytics import YOLO
import cv2
import os

# 加载模型
a1 = YOLO('models/等级3_5000次.pt')

# 检测图片路径
img_path = r"D:\shijue\pythonProject1\xun\YOLO\images\train\2.jpg"

# 获取图片文件名（不带扩展名）
img_name = os.path.splitext(os.path.basename(img_path))[0]

# 指定保存目录（基于图片文件名创建文件夹）
save_dir = os.path.join(os.path.dirname(__file__), 'picture_result', img_name)

# 如果目录不存在则创建
os.makedirs(save_dir, exist_ok=True)

# 执行检测并保存结果
results = a1(img_path, show=False, save=True, project=save_dir)  # 设置保存目录

# 获取保存的图片路径（通常保存在save_dir下的'predict'子文件夹中）
result_img_path = os.path.join(save_dir, 'predict', os.path.basename(img_path))

# 检查结果图片是否存在
if os.path.isfile(result_img_path):
    # 读取并显示保存的图片
    result_img = cv2.imread(result_img_path)
    cv2.imshow('检测结果', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"警告：未能找到保存的结果图片 {result_img_path}")

print(f"处理完成！结果已保存到: {save_dir}")
