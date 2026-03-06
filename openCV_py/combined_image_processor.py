import cv2
import numpy as np
from PIL import Image

def extract_text_regions(image_path, output_prefix="extracted_text_row"):
    """
    从图像中提取文本区域并保存为新图像
    
    参数:
    image_path: 输入图像路径
    output_prefix: 输出文件名前缀
    
    返回:
    生成的图像文件列表
    """
    print(f"正在处理图像: {image_path}")
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return []
    
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值进行二值化（增强文字对比）
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # 使用形态学操作去除噪声，连接断开的文字
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选出可能的文字区域（根据面积和宽高比）
    text_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h
        # 过滤掉太小或太细的轮廓
        if area > 100 and 0.3 < aspect_ratio < 5:
            text_contours.append((x, y, w, h))

    # 按 y 坐标排序，分组为行
    text_contours.sort(key=lambda x: x[1])  # 按y排序
    rows = []
    current_row = []

    for i, (x, y, w, h) in enumerate(text_contours):
        if not current_row:
            current_row.append((x, y, w, h))
        else:
            # 如果距离上一行较近，则归为同一行
            if abs(y - current_row[-1][1]) < 40:
                current_row.append((x, y, w, h))
            else:
                rows.append(current_row)
                current_row = [(x, y, w, h)]
    if current_row:
        rows.append(current_row)

    # 合并每行的边界框
    merged_boxes = []
    for row in rows:
        xs = [box[0] for box in row]
        ys = [box[1] for box in row]
        ws = [box[2] for box in row]
        hs = [box[3] for box in row]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs) + max(ws)
        y_max = max(ys) + max(hs)
        merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    # 裁剪出文字区域（可以只取中间的几行）
    # 例如取第1、2行（索引0,1）
    target_boxes = merged_boxes[:2]  # 取前两行

    # 创建新的图像，替换背景为深色（如黑色或深蓝），文字保留白色
    result_images = []
    output_files = []

    for i, box in enumerate(target_boxes):
        x, y, w, h = box
        cropped = img[y:y+h, x:x+w]
        # 将背景变为深色，文字保留白色
        # 先转为灰度，然后用阈值分离文字
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_crop, 180, 255, cv2.THRESH_BINARY)
        
        # 创建新背景（深蓝色）
        background = np.zeros_like(cropped)
        background[:] = (0, 0, 100)  # 深蓝色
        
        # 只保留文字部分（白色），其余为背景
        result = cv2.bitwise_and(cropped, cropped, mask=mask)
        result = cv2.add(result, background)
        
        # 保存
        output_file = f'{output_prefix}_{i+1}.jpg'
        cv2.imwrite(output_file, result)
        result_images.append(result)
        output_files.append(output_file)
        print(f"已保存文本区域图像: {output_file}")
    
    return output_files

def remove_top_white_area(image_path, output_path=None, threshold=35):
    """
    移除图片顶部的白色区域，将其替换为背景色
    
    参数:
    image_path: 输入图片路径
    output_path: 输出图片路径，如果为None则自动生成
    threshold: 白色阈值，像素值大于此值会被视为白色区域
    
    返回:
    处理后的图像路径
    """
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        base_name = image_path.rsplit('.', 1)[0]
        output_path = f"{base_name}_processed.jpg"
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    # 获取图片尺寸，高和宽
    height, width = img.shape[:2]
    
    # 找到顶部的白色区域边界
    # 从顶部开始扫描，找到第一个不是白色的行
    white_top_boundary = 0
    for i in range(height):
        # 检查当前行是否主要是白色
        row_mean = np.mean(img[i])
        if row_mean < threshold:
            # 找到了非白色行
            white_top_boundary = i
            break
    
    # 如果没有找到明显的白色区域边界，使用默认值（比如图片高度的1/10）
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
    return output_path

def process_image_complete(image_path):
    """
    完整处理流程：提取文本区域并移除顶部白色区域
    
    参数:
    image_path: 输入图像路径
    
    返回:
    最终处理后的图像文件列表
    """
    # 第一步：提取文本区域
    extracted_files = extract_text_regions(image_path)
    
    # 第二步：对每个提取的文本区域移除顶部白色区域
    final_files = []
    for file in extracted_files:
        base_name = file.rsplit('.', 1)[0]
        processed_file = f"{base_name}_no_white.jpg"
        result = remove_top_white_area(file, processed_file)
        if result:
            final_files.append(result)
    
    return final_files

if __name__ == "__main__":
    # 设置输入图像路径
    input_image = './image/2-2.jpg'  # 替换为你的图片路径
    
    # 执行完整的处理流程
    print("开始完整的图像处理流程...")
    final_images = process_image_complete(input_image)
    
    # 显示结果
    if final_images:
        print(f"图像处理完成！生成了 {len(final_images)} 个文件:")
        for file in final_images:
            print(f"- {file}")
    else:
        print("图像处理失败。")