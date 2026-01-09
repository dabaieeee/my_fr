from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType
from mmengine.model import BaseModule


@MODELS.register_module()
class CrossGatedFusion(BaseModule):
    """交叉门控融合模块，用于几何-语义特征融合。
    
    该模块实现了一种门控融合机制，根据可靠性自适应地组合几何和语义特征：
    - g → 1: 信任几何特征（近距离、密集区域）
    - g → 0: 信任语义特征（远距离、稀疏区域）
    
    公式:
        g = σ(W_g [F_geo; F_sem])
        F_fused = g ⊙ F_geo + (1 - g) ⊙ F_sem
    
    Args:
        geo_channels (int): 几何特征的通道数。
        sem_channels (int): 语义特征的通道数。
        out_channels (int): 输出通道数。
        conv_cfg (dict, optional): 卷积层的配置字典。
        norm_cfg (dict): 归一化层的配置字典。
            默认为 dict(type='BN')。
        act_cfg (dict): 激活层的配置字典。
            默认为 dict(type='LeakyReLU')。
    """

    def __init__(self,
                 geo_channels: int,
                 sem_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: Optional[dict] = None) -> None:
        super(CrossGatedFusion, self).__init__(init_cfg)
        
        self.geo_channels = geo_channels
        self.sem_channels = sem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        # 门控网络：从拼接的特征计算g
        gate_in_channels = geo_channels + sem_channels
        self.gate_conv = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                gate_in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg),
            build_conv_layer(
                conv_cfg,
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.Sigmoid())  # 输出门控值在[0, 1]范围内
        
        # 特征变换网络
        self.geo_transform = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                geo_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg))
        
        self.sem_transform = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                sem_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg))
        
        # 可选：交叉注意力增强
        self.use_cross_attention = False  # 可以在未来启用以增强功能

    def forward(self, geo_feats: torch.Tensor, 
                sem_feats: torch.Tensor) -> torch.Tensor:
        """交叉门控融合的前向传播。
        
        Args:
            geo_feats (Tensor): 几何特征 [B, C_geo, H, W] 或 [N, C_geo]。
            sem_feats (Tensor): 语义特征 [B, C_sem, H, W] 或 [N, C_sem]。
            
        Returns:
            Tensor: 融合后的特征 [B, C_out, H, W] 或 [N, C_out]。
        """
        # 处理2D（距离图像）和1D（点）特征
        is_2d = geo_feats.dim() == 4
        
        if is_2d:
            # 距离图像特征: [B, C, H, W]
            # 沿通道维度拼接
            concat_feats = torch.cat([geo_feats, sem_feats], dim=1)
            
            # 计算门控值
            gate = self.gate_conv(concat_feats)  # [B, C_out, H, W]
            
            # 变换特征
            geo_transformed = self.geo_transform(geo_feats)  # [B, C_out, H, W]
            sem_transformed = self.sem_transform(sem_feats)  # [B, C_out, H, W]
            
            # 门控融合
            fused_feats = gate * geo_transformed + (1 - gate) * sem_transformed
            
        else:
            # 点特征: [N, C]
            # 对于点特征，使用预构建的MLP层
            # 沿特征维度拼接
            concat_feats = torch.cat([geo_feats, sem_feats], dim=1)  # [N, C_geo + C_sem]
            
            # 为点特征构建MLP层（如果不存在）
            if not hasattr(self, 'gate_mlp'):
                gate_in_channels = self.geo_channels + self.sem_channels
                self.gate_mlp = nn.Sequential(
                    nn.Linear(gate_in_channels, self.out_channels, bias=False),
                    build_norm_layer(self.norm_cfg, self.out_channels)[1],
                    build_activation_layer(self.act_cfg),
                    nn.Linear(self.out_channels, self.out_channels, bias=False),
                    build_norm_layer(self.norm_cfg, self.out_channels)[1],
                    nn.Sigmoid())
            
            if not hasattr(self, 'geo_transform_mlp'):
                self.geo_transform_mlp = nn.Sequential(
                    nn.Linear(self.geo_channels, self.out_channels, bias=False),
                    build_norm_layer(self.norm_cfg, self.out_channels)[1],
                    build_activation_layer(self.act_cfg))
            
            if not hasattr(self, 'sem_transform_mlp'):
                self.sem_transform_mlp = nn.Sequential(
                    nn.Linear(self.sem_channels, self.out_channels, bias=False),
                    build_norm_layer(self.norm_cfg, self.out_channels)[1],
                    build_activation_layer(self.act_cfg))
            
            # 计算门控值
            gate = self.gate_mlp(concat_feats)  # [N, C_out]
            
            # 变换特征
            geo_transformed = self.geo_transform_mlp(geo_feats)  # [N, C_out]
            sem_transformed = self.sem_transform_mlp(sem_feats)  # [N, C_out]
            
            # 门控融合
            fused_feats = gate * geo_transformed + (1 - gate) * sem_transformed
        
        return fused_feats

