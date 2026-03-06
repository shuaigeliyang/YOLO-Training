import cv2
import numpy as np
from PIL import Image

# 读取图像
image_path = './image/5-1.JPG'  # 替换为你的图片路径
img = cv2.imread(image_path)  # 返回值: 图像数组（BGR 格式）
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

for box in target_boxes:
    x, y, w, h = box
    cropped = img[y:y+h, x:x+w]
    # 将背景变为深色，文字保留白色
    # 先转为灰度，然后用阈值分离文字
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_crop, 180, 255, cv2.THRESH_BINARY)
    
    # 创建新背景（比如深蓝色）
    background = np.zeros_like(cropped)
    background[:] = (0, 0, 100)  # 深蓝色
    
    # 只保留文字部分（白色），其余为背景
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    result = cv2.add(result, background)
    
    # 保存或显示
    result_images.append(result)

# 保存到文件
for i, res in enumerate(result_images):
    cv2.imwrite(f'extracted_text_row_{i+1}.jpg', res)

cv2.destroyAllWindows()