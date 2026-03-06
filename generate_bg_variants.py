# -*- coding: utf-8 -*-
"""
从 `文件夹3` 选取第一张图片，使用 GrabCut 抠图并生成 10 个不同纯色背景的变体，
保存回 `文件夹3`，文件名格式: <原名>_bg1.jpg ... _bg10.jpg
"""
import os
import sys
from pathlib import Path
import random

try:
    import cv2
    import numpy as np
except Exception as e:
    print('请先安装依赖: pip install opencv-python numpy')
    sys.exit(1)

SRC = Path('文件夹3')
NUM = 10
GRABCUT_ITERS = 5


def random_bg_color():
    # 返回 BGR 颜色
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))


def grabcut_foreground(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    fg = img * mask2[:, :, np.newaxis]
    return mask2.astype(np.float32), fg


def main():
    if not SRC.exists() or not SRC.is_dir():
        print('找不到目录', SRC)
        return
    imgs = [p for p in SRC.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')]
    if not imgs:
        print('目录没有图片:', SRC)
        return
    img_path = imgs[0]
    print('选择图片:', img_path.name)
    img = cv2.imread(str(img_path))
    if img is None:
        print('读取失败:', img_path)
        return

    mask, fg = grabcut_foreground(img)
    # 平滑掩码边缘
    mask = cv2.GaussianBlur(mask, (11,11), 0)
    mask3 = mask[:, :, None]
    h,w = img.shape[:2]

    created = []
    for i in range(1, NUM+1):
        color = random_bg_color()
        bg = np.zeros((h,w,3), dtype=np.uint8)
        bg[:] = color
        comp = (fg * mask3 + bg * (1 - mask3)).astype(np.uint8)
        out_name = f"{img_path.stem}_bg{i}{img_path.suffix}"
        out_path = SRC / out_name
        ok = cv2.imwrite(str(out_path), comp, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            created.append(out_name)
    print(f'已生成 {len(created)} 张变体，保存在 {SRC}')
    for n in created:
        print(' ', n)

if __name__ == '__main__':
    main()
