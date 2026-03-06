"""
CBAM (Convolutional Block Attention Module) 注意力机制模块

CBAM是一个轻量级的注意力模块,结合了通道注意力和空间注意力
可以有效提升特征表达能力,增强YOLO的检测性能
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道压缩比例,默认为16
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        
        # 共享的MLP网络
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x))
        # 融合并归一化
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 卷积核大小,默认为7
        """
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 沿通道维度进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接两个特征图
        concat = torch.cat([avg_out, max_out], dim=1)
        # 卷积并归一化
        out = self.sigmoid(self.conv(concat))
        return out


class CBAM(nn.Module):
    """
    CBAM注意力机制模块
    
    结合通道注意力和空间注意力,依次对特征图进行加权
    可以嵌入到YOLO的Backbone或Neck中
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        """
        Args:
            in_channels: 输入特征图的通道数
            reduction_ratio: 通道注意力的压缩比例
            kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 先进行通道注意力
        x = x * self.channel_attention(x)
        # 再进行空间注意力
        x = x * self.spatial_attention(x)
        return x


class CBAMBottleneck(nn.Module):
    """
    带CBAM的Bottleneck模块
    可以直接替换YOLO中的C2f或C3模块中的Bottleneck
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            shortcut: 是否使用残差连接
            g: 分组卷积的组数
            k: 卷积核大小
            e: 通道扩展因子
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, k[0] // 2, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act1 = nn.SiLU(inplace=True)
        
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU(inplace=True)
        
        # CBAM注意力
        self.cbam = CBAM(c2)
        
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.cv1(x)))
        out = self.bn2(self.cv2(out))
        # 应用CBAM注意力
        out = self.cbam(out)
        out = self.act2(out)
        
        if self.add:
            out = out + residual
        return out


# 测试代码
if __name__ == '__main__':
    # 测试CBAM模块
    print("测试CBAM模块...")
    x = torch.randn(2, 256, 64, 64)  # (batch_size, channels, height, width)
    cbam = CBAM(in_channels=256)
    out = cbam(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in cbam.parameters())}")
    
    print("\n测试CBAMBottleneck模块...")
    bottleneck = CBAMBottleneck(c1=256, c2=256)
    out = bottleneck(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in bottleneck.parameters())}")
    
    print("\nCBAM模块创建成功!")
