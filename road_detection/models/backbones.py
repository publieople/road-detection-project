#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义Backbone模块
包含ConvNeXt-Tiny、ConvNeXt-Small等现代视觉模型的实现
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class LayerNorm2d(nn.Module):
    """2D层标准化"""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = torch.nn.functional.layer_norm(x, (C,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt基础块 - 现代化的残差连接设计"""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()

        # 深度卷积 (7x7) - ConvNeXt的特征
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)

        # 逐点卷积以扩展维度
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()

        # 逐点卷积以还原维度
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # 层缩放 (Layer Scale)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

        # 随机深度 (Stochastic Depth)
        self.drop_path = nn.Identity()
        if drop_path > 0:
            # 简化实现 - 完整实现可以使用DropPath
            self.drop_path_rate = drop_path
        else:
            self.drop_path_rate = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x

        # 深度卷积分支
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # 应用层缩放
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        # 残差连接
        x = input_tensor + x
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt阶段 - 多个块的序列"""

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: List[float],
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                drop_path=drop_path_rates[i],
                layer_scale_init_value=1e-6,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny 模型 - 替换YOLOv11的CSPDarknet Backbone

    输出通道: [96, 192, 384, 768] (3 levels)
    FLOPs 和参数量都小于CSPDarknet，但准确率相当

    参考: A ConvNet for the 2020s (Liu et al., CVPR 2022)
    """

    def __init__(
        self,
        in_channels: int = 3,
        depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        self.dims = dims

        # 初始层：RGB -> 96通道，步长4
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0]),
            )
        )

        # 下采样层和阶段
        for i in range(3):
            # 下采样层 (2x下采样)
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

            # 生成该阶段的drop_path率
            dp_rates = [
                drop_path_rate * (j + sum(depths[:i])) / (sum(depths) - 1)
                for j in range(depths[i + 1])
            ]

            # ConvNeXt阶段
            stage = ConvNeXtStage(
                dim=dims[i + 1],
                depth=depths[i + 1],
                drop_path_rates=dp_rates,
            )
            self.stages.append(stage)

        # 最后一个阶段（不需要额外下采样）
        dp_rates = [
            drop_path_rate * (j + sum(depths[:3])) / (sum(depths) - 1)
            for j in range(depths[3])
        ]
        self.stages.insert(0, ConvNeXtStage(
            dim=dims[0],
            depth=depths[0],
            drop_path_rates=dp_rates,
        ))

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """初始化网络权重"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回多个尺度的特征图

        Returns:
            包含不同尺度特征的字典:
            - 'stride4': [B, 96, H/4, W/4]
            - 'stride8': [B, 192, H/8, W/8]
            - 'stride16': [B, 384, H/16, W/16]
            - 'stride32': [B, 768, H/32, W/32]
        """
        outputs = {}

        # 初始下采样 (stride 4)
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        outputs['stride4'] = x

        # 阶段1 -> 阶段2 (stride 8)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        outputs['stride8'] = x

        # 阶段2 -> 阶段3 (stride 16)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        outputs['stride16'] = x

        # 阶段3 -> 阶段4 (stride 32)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        outputs['stride32'] = x

        return outputs

    def get_output_channels(self) -> Dict[str, int]:
        """获取各输出的通道数"""
        return {
            'stride4': self.dims[0],
            'stride8': self.dims[1],
            'stride16': self.dims[2],
            'stride32': self.dims[3],
        }


class ConvNeXtSmall(nn.Module):
    """ConvNeXt-Small - 比Tiny更大的模型"""

    def __init__(
        self,
        in_channels: int = 3,
        depths: Tuple[int, int, int, int] = (3, 3, 27, 3),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # 重用ConvNeXtTiny的实现，仅改变depth参数
        self.backbone = ConvNeXtTiny(
            in_channels=in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
        )
        self.dims = dims

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone(x)

    def get_output_channels(self) -> Dict[str, int]:
        return self.backbone.get_output_channels()


def create_convnext_backbone(
    model_name: str = 'tiny',
    in_channels: int = 3,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """
    创建ConvNeXt Backbone

    Args:
        model_name: 'tiny' 或 'small'
        in_channels: 输入通道数
        drop_path_rate: Stochastic depth比率

    Returns:
        ConvNeXt backbone模块
    """
    if model_name.lower() == 'tiny':
        return ConvNeXtTiny(
            in_channels=in_channels,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            drop_path_rate=drop_path_rate,
        )
    elif model_name.lower() == 'small':
        return ConvNeXtSmall(
            in_channels=in_channels,
            depths=(3, 3, 27, 3),
            dims=(96, 192, 384, 768),
            drop_path_rate=drop_path_rate,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == '__main__':
    # 测试代码
    model = ConvNeXtTiny()
    x = torch.randn(1, 3, 640, 640)

    outputs = model(x)
    print("ConvNeXt-Tiny输出:")
    for key, tensor in outputs.items():
        print(f"  {key}: {tensor.shape}")

    print("\n输出通道数:")
    channels = model.get_output_channels()
    for key, ch in channels.items():
        print(f"  {key}: {ch}")
