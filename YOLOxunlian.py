from ultralytics import YOLO

if __name__ == '__main__':
    #加载训练模型
    a1 = YOLO(r'yolo11n.pt')
    #开始训练模型
    a1.train(
        data='./data.yaml', #数据集配置文件
        epochs=1000,   #训练轮次
        imgsz=640,   #输入图片尺寸 官方推荐640
        batch=40,  #每一次训练的批量
        device=0  #gpu=0,cpu='cpu'
    )
print('模型训练完毕')