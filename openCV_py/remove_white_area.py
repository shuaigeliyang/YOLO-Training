import cv2
import numpy as np

def remove_top_white_area(image_path, output_path, threshold=35):
    """
    移除图片顶部的白色区域，将其替换为背景色
    
    参数:
    image_path: 输入图片路径
    output_path: 输出图片路径
    threshold: 白色阈值，像素值大于此值会被视为白色区域
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return False
    
    # 获取图片尺寸，高和宽
    height, width = img.shape[:2]
    
    # 找到顶部的白色区域边界
    # 从顶部开始扫描，找到第一个不是白色的行
    white_top_boundary = 0
    for i in range(height):
        # 检查当前行是否主要是白色
        row_mean = np.mean(img[i])
        if row_mean < threshold:
            # 找到了非白色行,如果这里的行没有找到阈值范围内的白色的行后停止程序
            white_top_boundary = i
            break
    
    # 如果没有找到明显的白色区域边界，使用默认值（比如图片高度的1/4）
    if white_top_boundary == 0:
        white_top_boundary = height // 10
    
    # 创建输出图像的副本
    result = img.copy()
    
    # 获取背景色（取文字区域下方的平均颜色作为背景色）
    # 这里我们取底部10%区域的平均颜色
    bottom_region = img[int(height * 0.9):, :]
    background_color = np.mean(bottom_region, axis=(0, 1)).astype(int)
    
    # 将顶部的白色区域替换为背景色
    result[:white_top_boundary, :] = background_color
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"已保存处理后的图片到: {output_path}")
    return True

if __name__ == "__main__":
    # 处理生成的文本图片
    input_image = "extracted_text_row_1.jpg"  # 输入图片路径
    output_image = "processed_without_white.jpg"  # 输出图片路径
    
    # 调用函数处理图片
    success = remove_top_white_area(input_image, output_image)
    
    if success:
        print("图片处理完成！")
    else:
        print("图片处理失败。")