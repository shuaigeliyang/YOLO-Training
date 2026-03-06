"""
ASFF (Adaptively Spatial Feature Fusion) 自适应空间特征融合模块

ASFF通过学习不同尺度特征的权重,实现自适应的特征融合
可以有效提升多尺度目标检测性能,特别适合YOLO的特征金字塔网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFF(nn.Module):
    """
    ASFF自适应空间特征融合模块
    
    用于融合来自不同层级的特征图,通过学习权重自适应地选择最优特征
    """
    def __init__(self, level, in_channels=[512, 256, 128], out_channels=256, rfb=False):
        """
        Args:
            level: 当前ASFF模块的层级 (0, 1, 2对应大、中、小特征图)
            in_channels: 三个输入特征图的通道数列表
            out_channels: 输出通道数
            rfb: 是否使用RFB(Receptive Field Block)增强感受野
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [in_channels[0], in_channels[1], in_channels[2]]
        self.inter_dim = out_channels
        
        # 压缩通道数到统一的out_channels
        if level == 0:  # 最大的特征图
            self.stride_level_1 = nn.Conv2d(self.dim[1], self.inter_dim, 3, 2, 1)
            self.stride_level_2 = nn.Conv2d(self.dim[2], self.inter_dim, 3, 2, 1)
            self.expand = nn.Conv2d(self.dim[0], self.inter_dim, 1, 1, 0)
        elif level == 1:  # 中等的特征图
            self.compress_level_0 = nn.Conv2d(self.dim[0], self.inter_dim, 1, 1, 0)
            self.stride_level_2 = nn.Conv2d(self.dim[2], self.inter_dim, 3, 2, 1)
            self.expand = nn.Conv2d(self.dim[1], self.inter_dim, 1, 1, 0)
        elif level == 2:  # 最小的特征图
            self.compress_level_0 = nn.Conv2d(self.dim[0], self.inter_dim, 1, 1, 0)
            self.compress_level_1 = nn.Conv2d(self.dim[1], self.inter_dim, 1, 1, 0)
            self.expand = nn.Conv2d(self.dim[2], self.inter_dim, 1, 1, 0)
        
        # 权重学习层 - 为每个输入特征图学习一个权重
        compress_c = 8 if rfb else 16
        self.weight_level_0 = nn.Conv2d(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_1 = nn.Conv2d(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = nn.Conv2d(self.inter_dim, compress_c, 1, 1, 0)
        
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)
        
    def forward(self, x_level_0, x_level_1, x_level_2):
        """
        Args:
            x_level_0: 来自大特征图的输入 (低分辨率,高语义)
            x_level_1: 来自中特征图的输入
            x_level_2: 来自小特征图的输入 (高分辨率,低语义)
        
        Returns:
            融合后的特征图
        """
        # 调整不同层级特征图到相同尺寸
        if self.level == 0:
            level_0_resized = self.expand(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = F.max_pool2d(self.stride_level_2(x_level_2), 2, 2)
        elif self.level == 1:
            level_0_resized = F.interpolate(self.compress_level_0(x_level_0),
                                           scale_factor=2, mode='nearest')
            level_1_resized = self.expand(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_resized = F.interpolate(self.compress_level_0(x_level_0),
                                           scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(self.compress_level_1(x_level_1),
                                           scale_factor=2, mode='nearest')
            level_2_resized = self.expand(x_level_2)
        
        # 学习每个层级的权重
        level_0_weight = self.weight_level_0(level_0_resized)
        level_1_weight = self.weight_level_1(level_1_resized)
        level_2_weight = self.weight_level_2(level_2_resized)
        
        # 拼接权重并通过softmax归一化
        levels_weight = torch.cat([level_0_weight, level_1_weight, level_2_weight], 1)
        levels_weight = self.weight_levels(levels_weight)
        levels_weight = F.softmax(levels_weight, dim=1)
        
        # 加权融合
        fused_out = (level_0_resized * levels_weight[:, 0:1, :, :] +
                     level_1_resized * levels_weight[:, 1:2, :, :] +
                     level_2_resized * levels_weight[:, 2:, :, :])
        
        return fused_out


class ASFF_2(nn.Module):
    """
    简化版ASFF - 只融合两个层级的特征
    适合只有两个特征层的情况
    """
    def __init__(self, level, in_channels=[256, 128], out_channels=256):
        """
        Args:
            level: 当前层级 (0或1)
            in_channels: 两个输入特征图的通道数
            out_channels: 输出通道数
        """
        super(ASFF_2, self).__init__()
        self.level = level
        
        if level == 0:
            self.stride_level_1 = nn.Conv2d(in_channels[1], out_channels, 3, 2, 1)
            self.expand = nn.Conv2d(in_channels[0], out_channels, 1, 1, 0)
        elif level == 1:
            self.compress_level_0 = nn.Conv2d(in_channels[0], out_channels, 1, 1, 0)
            self.expand = nn.Conv2d(in_channels[1], out_channels, 1, 1, 0)
        
        # 权重学习
        compress_c = 8
        self.weight_level_0 = nn.Conv2d(out_channels, compress_c, 1, 1, 0)
        self.weight_level_1 = nn.Conv2d(out_channels, compress_c, 1, 1, 0)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, 1, 1, 0)
    
    def forward(self, x_level_0, x_level_1):
        """
        Args:
            x_level_0: 第一个输入特征图
            x_level_1: 第二个输入特征图
        """
        if self.level == 0:
            level_0_resized = self.expand(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = F.interpolate(self.compress_level_0(x_level_0),
                                           scale_factor=2, mode='nearest')
            level_1_resized = self.expand(x_level_1)
        
        # 学习权重
        level_0_weight = self.weight_level_0(level_0_resized)
        level_1_weight = self.weight_level_1(level_1_resized)
        
        levels_weight = torch.cat([level_0_weight, level_1_weight], 1)
        levels_weight = self.weight_levels(levels_weight)
        levels_weight = F.softmax(levels_weight, dim=1)
        
        # 加权融合
        fused_out = (level_0_resized * levels_weight[:, 0:1, :, :] +
                     level_1_resized * levels_weight[:, 1:, :, :])
        
        return fused_out


class ASFFBlock(nn.Module):
    """
    ASFF块 - 包含ASFF融合和后续卷积
    可以直接集成到YOLO的neck部分
    """
    def __init__(self, level, in_channels, out_channels, rfb=False):
        super(ASFFBlock, self).__init__()
        self.asff = ASFF(level, in_channels, out_channels, rfb)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x_level_0, x_level_1, x_level_2):
        out = self.asff(x_level_0, x_level_1, x_level_2)
        out = self.conv(out)
        return out


# 测试代码
if __name__ == '__main__':
    print("测试ASFF模块...")
    
    # 模拟三个不同尺度的特征图
    x0 = torch.randn(2, 512, 20, 20)  # 大特征图 (低分辨率)
    x1 = torch.randn(2, 256, 40, 40)  # 中特征图
    x2 = torch.randn(2, 128, 80, 80)  # 小特征图 (高分辨率)
    
    print(f"\n输入特征图尺寸:")
    print(f"Level 0: {x0.shape}")
    print(f"Level 1: {x1.shape}")
    print(f"Level 2: {x2.shape}")
    
    # 测试三个不同层级的ASFF
    in_channels = [512, 256, 128]
    out_channels = 256
    
    for level in range(3):
        print(f"\n测试 ASFF Level {level}:")
        asff = ASFF(level=level, in_channels=in_channels, out_channels=out_channels)
        out = asff(x0, x1, x2)
        print(f"输出形状: {out.shape}")
        print(f"参数量: {sum(p.numel() for p in asff.parameters())}")
    
    # 测试简化版ASFF_2
    print("\n测试简化版 ASFF_2:")
    x0_simple = torch.randn(2, 256, 40, 40)
    x1_simple = torch.randn(2, 128, 80, 80)
    
    asff2 = ASFF_2(level=1, in_channels=[256, 128], out_channels=256)
    out = asff2(x0_simple, x1_simple)
    print(f"输入形状: {x0_simple.shape}, {x1_simple.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in asff2.parameters())}")
    
    # 测试ASFFBlock
    print("\n测试 ASFFBlock:")
    asff_block = ASFFBlock(level=1, in_channels=in_channels, out_channels=out_channels)
    out = asff_block(x0, x1, x2)
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in asff_block.parameters())}")
    
    print("\nASFF模块创建成功!")
