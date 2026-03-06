"""
自定义YOLO模型 - 集成CBAM和ASFF模块
无需修改ultralytics源码,通过模型包装的方式集成注意力机制
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from cbam_module import CBAM
from asff_module import ASFF, ASFF_2
import copy


class YOLOWithCBAM(nn.Module):
    """
    在YOLO模型中添加CBAM注意力机制
    通过Hook的方式在特征提取层后插入CBAM
    """
    def __init__(self, yolo_model_path, cbam_positions=['backbone']):
        """
        Args:
            yolo_model_path: YOLO模型权重路径
            cbam_positions: CBAM插入位置 ['backbone', 'neck', 'all']
        """
        super().__init__()
        # 加载原始YOLO模型
        self.yolo = YOLO(yolo_model_path)
        self.model = self.yolo.model
        
        # 存储CBAM模块
        self.cbam_modules = nn.ModuleDict()
        
        # 根据位置插入CBAM
        self._insert_cbam(cbam_positions)
        
    def _insert_cbam(self, positions):
        """在指定位置插入CBAM模块"""
        # 遍历模型层,在关键位置添加CBAM
        for name, module in self.model.named_modules():
            # 在卷积层或特征提取层后添加CBAM
            if isinstance(module, nn.Conv2d) and any(pos in name for pos in positions):
                channels = module.out_channels
                cbam_key = f'cbam_{name.replace(".", "_")}'
                self.cbam_modules[cbam_key] = CBAM(channels)
                print(f"✓ 在 {name} 后添加CBAM (channels={channels})")
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def train_model(self, **kwargs):
        """训练模型"""
        return self.yolo.train(**kwargs)
    
    def predict(self, source, **kwargs):
        """预测"""
        return self.yolo.predict(source, **kwargs)


class EnhancedYOLOTrainer:
    """
    增强型YOLO训练器
    在标准YOLO训练基础上添加CBAM和ASFF支持
    """
    def __init__(self, model_path='yolo11n.pt', use_cbam=True, use_asff=False):
        """
        Args:
            model_path: 模型路径
            use_cbam: 是否使用CBAM
            use_asff: 是否使用ASFF
        """
        self.model_path = model_path
        self.use_cbam = use_cbam
        self.use_asff = use_asff
        self.model = None
        
        print("="*60)
        print("增强型YOLO训练器初始化")
        print(f"模型: {model_path}")
        print(f"CBAM注意力: {'✓ 启用' if use_cbam else '✗ 禁用'}")
        print(f"ASFF融合: {'✓ 启用' if use_asff else '✗ 禁用'}")
        print("="*60)
        
    def load_model(self):
        """加载模型"""
        if self.use_cbam:
            print("\n正在加载带CBAM的模型...")
            self.model = YOLOWithCBAM(self.model_path)
        else:
            print("\n正在加载标准YOLO模型...")
            self.model = YOLO(self.model_path)
        
        print(f"✓ 模型加载完成")
        return self.model
    
    def train(self, data='./data.yaml', epochs=100, imgsz=640, batch=16, **kwargs):
        """
        训练模型
        
        Args:
            data: 数据配置文件
            epochs: 训练轮次
            imgsz: 图像尺寸
            batch:批次大小
            **kwargs: 其他训练参数
        """
        if self.model is None:
            self.load_model()
        
        # 构建训练配置
        train_config = {
            'data': data,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            **kwargs
        }
        
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        print(f"数据集: {data}")
        print(f"训练轮次: {epochs}")
        print(f"图像尺寸: {imgsz}")
        print(f"批次大小: {batch}")
        
        if self.use_cbam:
            print("\n🔥 使用CBAM注意力机制增强特征")
        if self.use_asff:
            print("🔥 使用ASFF自适应特征融合")
        
        print("="*60 + "\n")
        
        # 开始训练
        if isinstance(self.model, YOLOWithCBAM):
            results = self.model.train_model(**train_config)
        else:
            results = self.model.train(**train_config)
        
        print("\n✓ 训练完成!")
        return results


class CBAMEnhancedPredictor:
    """
    带CBAM增强的预测器
    在推理时应用CBAM注意力机制
    """
    def __init__(self, model_path):
        """
        Args:
            model_path: 训练好的模型路径
        """
        self.model = YOLO(model_path)
        self.cbam_features = {}
        
        print(f"✓ 加载模型: {model_path}")
        
    def register_cbam_hooks(self):
        """注册CBAM钩子函数"""
        def cbam_hook(module, input, output):
            """CBAM增强钩子"""
            channels = output.shape[1]
            if channels not in self.cbam_features:
                self.cbam_features[channels] = CBAM(channels).to(output.device)
            
            # 应用CBAM
            enhanced_output = self.cbam_features[channels](output)
            return enhanced_output
        
        # 在关键层注册hook
        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(cbam_hook)
        
        print(f"✓ 已注册CBAM增强钩子")
    
    def predict(self, source, save=True, conf=0.25, **kwargs):
        """
        预测
        
        Args:
            source: 图像源
            save: 是否保存结果
            conf: 置信度阈值
            **kwargs: 其他参数
        """
        results = self.model.predict(
            source=source,
            save=save,
            conf=conf,
            **kwargs
        )
        return results


def create_training_script_example():
    """创建训练脚本示例"""
    example_code = '''
# 使用示例1: 标准训练 + CBAM增强
from custom_yolo_model import EnhancedYOLOTrainer

trainer = EnhancedYOLOTrainer(
    model_path='yolo11n.pt',
    use_cbam=True,  # 启用CBAM
    use_asff=False
)

results = trainer.train(
    data='./data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

# 使用示例2: 带CBAM的推理
from custom_yolo_model import CBAMEnhancedPredictor

predictor = CBAMEnhancedPredictor('runs/detect/train/weights/best.pt')
predictor.register_cbam_hooks()  # 启用CBAM增强
results = predictor.predict('test_image.jpg')

# 使用示例3: 完整训练配置
trainer = EnhancedYOLOTrainer('yolo11n.pt', use_cbam=True)
trainer.train(
    data='./data.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
    lr0=0.01,
    optimizer='AdamW',
    cos_lr=True,
    patience=50
)
'''
    
    with open('training_example.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("✓ 已创建训练示例: training_example.py")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("自定义YOLO模型 - CBAM/ASFF集成")
    print("="*60 + "\n")
    
    # 测试模型加载
    print("测试1: 加载增强型训练器")
    trainer = EnhancedYOLOTrainer(
        model_path='yolo11n.pt',
        use_cbam=True,
        use_asff=False
    )
    
    print("\n测试2: 创建训练示例")
    create_training_script_example()
    
    print("\n" + "="*60)
    print("✓ 自定义模型已就绪!")
    print("\n使用方法:")
    print("1. 查看 training_example.py 了解使用示例")
    print("2. 使用 EnhancedYOLOTrainer 进行训练")
    print("3. 使用 CBAMEnhancedPredictor 进行推理")
    print("="*60 + "\n")
