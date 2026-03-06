from PIL import Image
import os

def check_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # 验证图片是否损坏
        return True
    except Exception as e:
        print(f"损坏的图片: {file_path}, 错误: {e}")
        return False

def check_and_fix_images_in_directory(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                if not check_image(file_path):
                    # 尝试用PIL重新保存图片来修复
                    try:
                        img = Image.open(file_path)
                        img.save(file_path)
                        print(f"已修复图片: {file_path}")
                    except Exception as e:
                        print(f"修复失败: {file_path}, 错误: {e}")

# 在训练前调用这个函数检查训练和验证集图片
check_and_fix_images_in_directory('./images/train')
check_and_fix_images_in_directory('./images/val')