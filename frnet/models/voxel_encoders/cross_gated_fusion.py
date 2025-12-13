from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from torch import Tensor


@MODELS.register_module()
class CrossGatedFusion(nn.Module):
    """Cross-Gated Fusion Module for Geometry-Semantic Feature Fusion.
    
    This module implements a gated fusion mechanism that adaptively combines
    geometry and semantic features based on their reliability.
    
    Formula:
        g = σ(W_g [F_geo; F_sem])
        F_fused = g ⊙ F_geo + (1 - g) ⊙ F_sem
    
    Where:
        - g → 1: Trust geometry (e.g., close-range, dense regions)
        - g → 0: Trust semantics (e.g., far-range, sparse regions)
    
    Args:
        geo_channels (int): Number of geometry feature channels.
        sem_channels (int): Number of semantic feature channels.
        out_channels (int): Number of output channels. Defaults to None (same as sem_channels).
        norm_cfg (dict or :obj:`ConfigType`): Config dict of normalization layers.
            Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1).
        act_cfg (dict or :obj:`ConfigType`): Config dict of activation layers.
            Defaults to dict(type='Sigmoid').
        fusion_type (str): Type of fusion. 'point' for point-level, 'pixel' for pixel-level.
            Defaults to 'point'.
    """
    
    def __init__(self,
                 geo_channels: int,
                 sem_channels: int,
                 out_channels: Optional[int] = None,
                 norm_cfg: ConfigType = dict(type='BN1d', eps=1e-5, momentum=0.1),
                 act_cfg: ConfigType = dict(type='Sigmoid'),
                 fusion_type: str = 'point') -> None:
        super(CrossGatedFusion, self).__init__()
        
        self.geo_channels = geo_channels
        self.sem_channels = sem_channels
        self.out_channels = out_channels if out_channels is not None else sem_channels
        self.fusion_type = fusion_type
        
        # 如果通道数不同，需要先对齐
        if geo_channels != self.out_channels:
            if fusion_type == 'point':
                self.geo_proj = nn.Sequential(
                    nn.Linear(geo_channels, self.out_channels, bias=False),
                    build_norm_layer(norm_cfg, self.out_channels)[1],
                    nn.ReLU(inplace=True))
            else:  # pixel
                self.geo_proj = nn.Sequential(
                    build_conv_layer(
                        dict(type='Conv2d'),
                        geo_channels,
                        self.out_channels,
                        kernel_size=1,
                        bias=False),
                    build_norm_layer(norm_cfg, self.out_channels)[1],
                    build_activation_layer(dict(type='ReLU', inplace=True)))
        else:
            self.geo_proj = None
        
        if sem_channels != self.out_channels:
            if fusion_type == 'point':
                self.sem_proj = nn.Sequential(
                    nn.Linear(sem_channels, self.out_channels, bias=False),
                    build_norm_layer(norm_cfg, self.out_channels)[1],
                    nn.ReLU(inplace=True))
            else:  # pixel
                self.sem_proj = nn.Sequential(
                    build_conv_layer(
                        dict(type='Conv2d'),
                        sem_channels,
                        self.out_channels,
                        kernel_size=1,
                        bias=False),
                    build_norm_layer(norm_cfg, self.out_channels)[1],
                    build_activation_layer(dict(type='ReLU', inplace=True)))
        else:
            self.sem_proj = None
        
        # 门控网络：g = σ(W_g [F_geo; F_sem])
        if fusion_type == 'point':
            self.gate_net = nn.Sequential(
                nn.Linear(self.out_channels * 2, self.out_channels, bias=False),
                build_norm_layer(norm_cfg, self.out_channels)[1],
                build_activation_layer(act_cfg))  # Sigmoid
        else:  # pixel
            self.gate_net = nn.Sequential(
                build_conv_layer(
                    dict(type='Conv2d'),
                    self.out_channels * 2,
                    self.out_channels,
                    kernel_size=1,
                    bias=False),
                build_norm_layer(norm_cfg, self.out_channels)[1],
                build_activation_layer(act_cfg))  # Sigmoid
    
    def forward(self, geo_feats: Tensor, sem_feats: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            geo_feats (Tensor): Geometry features.
                - Point-level: [N, C_geo]
                - Pixel-level: [B, C_geo, H, W]
            sem_feats (Tensor): Semantic features.
                - Point-level: [N, C_sem]
                - Pixel-level: [B, C_sem, H, W]
                
        Returns:
            Tensor: Fused features.
                - Point-level: [N, C_out]
                - Pixel-level: [B, C_out, H, W]
        """
        # 对齐通道数
        if self.geo_proj is not None:
            geo_feats = self.geo_proj(geo_feats)
        if self.sem_proj is not None:
            sem_feats = self.sem_proj(sem_feats)
        
        # 拼接特征
        if self.fusion_type == 'point':
            # [N, C_geo] + [N, C_sem] -> [N, C_geo + C_sem]
            concat_feats = torch.cat([geo_feats, sem_feats], dim=1)  # [N, 2*C_out]
        else:  # pixel
            # [B, C_geo, H, W] + [B, C_sem, H, W] -> [B, C_geo + C_sem, H, W]
            concat_feats = torch.cat([geo_feats, sem_feats], dim=1)  # [B, 2*C_out, H, W]
        
        # 计算门控值 g
        gate = self.gate_net(concat_feats)  # [N, C_out] or [B, C_out, H, W]
        
        # 门控融合：F_fused = g ⊙ F_geo + (1 - g) ⊙ F_sem
        fused_feats = gate * geo_feats + (1 - gate) * sem_feats
        
        return fused_feats

