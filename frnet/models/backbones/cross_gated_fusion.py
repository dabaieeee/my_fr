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
    """Cross-Gated Fusion Module for Geometry-Semantic Feature Fusion.
    
    This module implements a gated fusion mechanism that adaptively combines
    geometric and semantic features based on their reliability:
    - g → 1: Trust geometry (nearby, dense regions)
    - g → 0: Trust semantics (distant, sparse regions)
    
    Formula:
        g = σ(W_g [F_geo; F_sem])
        F_fused = g ⊙ F_geo + (1 - g) ⊙ F_sem
    
    Args:
        geo_channels (int): Number of channels in geometric features.
        sem_channels (int): Number of channels in semantic features.
        out_channels (int): Number of output channels.
        conv_cfg (dict, optional): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU').
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
        
        # Gate network: computes g from concatenated features
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
            nn.Sigmoid())  # Output gate values in [0, 1]
        
        # Feature transformation networks
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
        
        # Optional: Cross-attention enhancement
        self.use_cross_attention = False  # Can be enabled for future enhancement
        
        # Pre-build MLP layers for point features (fix dynamic creation issue)
        gate_in_channels = geo_channels + sem_channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg),
            nn.Linear(out_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.Sigmoid())
        
        self.geo_transform_mlp = nn.Sequential(
            nn.Linear(geo_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg))
        
        self.sem_transform_mlp = nn.Sequential(
            nn.Linear(sem_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            build_activation_layer(act_cfg))

    def forward(self, geo_feats: torch.Tensor, 
                sem_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass of Cross-Gated Fusion.
        
        Args:
            geo_feats (Tensor): Geometric features [B, C_geo, H, W] or [N, C_geo].
            sem_feats (Tensor): Semantic features [B, C_sem, H, W] or [N, C_sem].
            
        Returns:
            Tensor: Fused features [B, C_out, H, W] or [N, C_out].
        """
        # Handle both 2D (range image) and 1D (point) features
        is_2d = geo_feats.dim() == 4
        
        if is_2d:
            # Range image features: [B, C, H, W]
            # Concatenate along channel dimension
            concat_feats = torch.cat([geo_feats, sem_feats], dim=1)
            
            # Compute gate
            gate = self.gate_conv(concat_feats)  # [B, C_out, H, W]
            
            # Transform features
            geo_transformed = self.geo_transform(geo_feats)  # [B, C_out, H, W]
            sem_transformed = self.sem_transform(sem_feats)  # [B, C_out, H, W]
            
            # Gated fusion
            fused_feats = gate * geo_transformed + (1 - gate) * sem_transformed
            
        else:
            # Point features: [N, C]
            # Concatenate along feature dimension
            concat_feats = torch.cat([geo_feats, sem_feats], dim=1)  # [N, C_geo + C_sem]
            
            # Compute gate
            gate = self.gate_mlp(concat_feats)  # [N, C_out]
            
            # Transform features
            geo_transformed = self.geo_transform_mlp(geo_feats)  # [N, C_out]
            sem_transformed = self.sem_transform_mlp(sem_feats)  # [N, C_out]
            
            # Gated fusion
            fused_feats = gate * geo_transformed + (1 - gate) * sem_transformed
        
        return fused_feats

