from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor

from .frnet_backbone import BasicBlock
from .cross_gated_fusion import CrossGatedFusion


@MODELS.register_module()
class DualPathFRNetBackbone(BaseModule):
    """Dual-Path FRNet Backbone with Geometry-Semantic Decoupling.
    
    This backbone implements a dual-path architecture:
    1. Geometry Path: Extracts structure-preserving geometric features
    2. Semantic Path: Extracts context-aware semantic features using FFE and FPFM
    3. Cross-Gated Fusion: Adaptively fuses geometry and semantic features
    
    The FPFM (Frustum-Point Fusion Module) is preserved and operates on
    the semantic path to maintain scene consistency.
    
    Args:
        geo_channels (int): Number of channels from geometry encoder.
        sem_channels (int): Number of channels from semantic encoder.
        output_shape (Sequence[int]): Output shape of range image [H, W].
        depth (int): Depth of ResNet backbone (18 or 34).
        stem_channels (int): Number of channels in stem layer.
        num_stages (int): Number of ResNet stages.
        out_channels (Sequence[int]): Output channels for each stage.
        strides (Sequence[int]): Strides for each stage.
        dilations (Sequence[int]): Dilations for each stage.
        fuse_channels (Sequence[int]): Channels for fusion layers.
        conv_cfg (dict, optional): Config for convolution layers.
        norm_cfg (dict): Config for normalization layers.
        point_norm_cfg (dict): Config for point normalization layers.
        act_cfg (dict): Config for activation layers.
        use_cross_gated_fusion (bool): Whether to use cross-gated fusion.
            Defaults to True.
        fusion_channels (int): Number of channels after fusion.
            Defaults to None (uses stem_channels).
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
    }

    def __init__(self,
                 geo_channels: int,
                 sem_channels: int,
                 output_shape: Sequence[int],
                 depth: int,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 point_norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 use_cross_gated_fusion: bool = True,
                 fusion_channels: Optional[int] = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(DualPathFRNetBackbone, self).__init__(init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for DualPathFRNetBackbone.')

        self.block, stage_blocks = self.arch_settings[depth]
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(
            dilations) == num_stages, \
            'The length of stage_blocks, out_channels, strides and ' \
            'dilations should be equal to num_stages.'
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.point_norm_cfg = point_norm_cfg
        self.act_cfg = act_cfg
        self.use_cross_gated_fusion = use_cross_gated_fusion
        
        # Geometry path: small receptive field, structure-preserving
        self.geo_stem = self._make_stem_layer(geo_channels, stem_channels)
        
        # Semantic path: uses FFE output, context-aware
        self.sem_stem = self._make_stem_layer(sem_channels, stem_channels)
        
        # Cross-Gated Fusion at stem level
        if use_cross_gated_fusion:
            fusion_ch = fusion_channels if fusion_channels is not None else stem_channels
            self.cross_gated_fusion = CrossGatedFusion(
                geo_channels=stem_channels,
                sem_channels=stem_channels,
                out_channels=fusion_ch,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            fused_channels = fusion_ch
        else:
            # Simple concatenation if fusion is disabled
            self.fusion_stem = self._make_fusion_layer(
                stem_channels * 2, stem_channels)
            fused_channels = stem_channels
        
        # Point feature stems
        self.geo_point_stem = self._make_point_layer(geo_channels, stem_channels)
        self.sem_point_stem = self._make_point_layer(sem_channels, stem_channels)
        
        # Fused point stem (after cross-gated fusion at point level)
        if use_cross_gated_fusion:
            self.point_cross_gated_fusion = CrossGatedFusion(
                geo_channels=stem_channels,
                sem_channels=stem_channels,
                out_channels=fusion_ch,
                conv_cfg=None,  # Use MLP for point features
                norm_cfg=point_norm_cfg,
                act_cfg=act_cfg)
            fused_point_channels = fusion_ch
        else:
            self.point_fusion_stem = self._make_point_layer(
                stem_channels * 2, stem_channels)
            fused_point_channels = stem_channels
        
        # Stem-level FPFM fusion layers
        self.stem_point_fusion = self._make_gated_point_fusion_layer(
            fused_channels + fused_point_channels, fused_point_channels)
        self.stem_pixel_fusion = self._make_gated_fusion_layer(
            fused_channels * 2, fused_channels)

        inplanes = fused_channels
        self.res_layers = []
        # FPFM layers (operate on semantic path, then fuse with geometry)
        self.point_fusion_layers = nn.ModuleList()  # Frustum-to-Point fusion
        self.pixel_fusion_layers = nn.ModuleList()  # Point-to-Frustum fusion
        self.attention_layers = nn.ModuleList()
        self.strides = []
        overall_stride = 1
        
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            
            # FPFM: Frustum-to-Point fusion (on semantic path)
            self.point_fusion_layers.append(
                self._make_gated_point_fusion_layer(inplanes + planes, planes))
            
            # FPFM: Point-to-Frustum fusion
            self.pixel_fusion_layers.append(
                self._make_gated_fusion_layer(planes * 2, planes))
            
            # Attention module
            self.attention_layers.append(self._make_attention_layer(planes))
            
            inplanes = planes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # Multi-scale feature fusion
        in_channels = fused_channels + sum(out_channels)
        self.fuse_layers = []
        self.point_fuse_layers = []
        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = ConvModule(
                in_channels,
                fuse_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            point_fuse_layer = self._make_point_layer(in_channels, fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

    def _make_stem_layer(self, in_channels: int,
                         out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels // 2)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels // 2,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_point_layer(self, in_channels: int,
                          out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            build_norm_layer(self.point_norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True))

    def _make_fusion_layer(self, in_channels: int,
                           out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_gated_fusion_layer(self, in_channels: int,
                                 out_channels: int) -> nn.Module:
        """Create gated fusion layer for pixel features."""
        return nn.ModuleDict({
            'fusion': nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1],
                build_activation_layer(self.act_cfg)),
            'gate': nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1],
                nn.Sigmoid())
        })

    def _make_gated_point_fusion_layer(self, in_channels: int,
                                       out_channels: int) -> nn.Module:
        """Create gated fusion layer for point features."""
        return nn.ModuleDict({
            'fusion': nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                build_norm_layer(self.point_norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True)),
            'gate': nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                build_norm_layer(self.point_norm_cfg, out_channels)[1],
                nn.Sigmoid())
        })

    def _make_attention_layer(self, channels: int) -> nn.Module:
        """Enhanced attention layer with channel and spatial attention.
        
        Uses both channel attention and spatial attention for better feature
        refinement.
        """
        return nn.ModuleDict({
            # Channel attention: focuses on 'what' features are important
            'channel_attention': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                build_conv_layer(
                    self.conv_cfg,
                    channels,
                    channels // 4,
                    kernel_size=1,
                    padding=0,
                    bias=False),
                build_activation_layer(self.act_cfg),
                build_conv_layer(
                    self.conv_cfg,
                    channels // 4,
                    channels,
                    kernel_size=1,
                    padding=0,
                    bias=False),
                build_norm_layer(self.norm_cfg, channels)[1],
                nn.Sigmoid()),
            # Spatial attention: focuses on 'where' features are important
            'spatial_attention': nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    channels,
                    channels // 4,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, channels // 4)[1],
                build_activation_layer(self.act_cfg),
                build_conv_layer(
                    self.conv_cfg,
                    channels // 4,
                    1,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                nn.Sigmoid())
        })

    def make_res_layer(self,
                      block: nn.Module,
                      inplanes: int,
                      planes: int,
                      num_blocks: int,
                      stride: int,
                      dilation: int,
                      conv_cfg: OptConfigType = None,
                      norm_cfg: ConfigType = dict(type='BN'),
                      act_cfg: ConfigType = dict(type='LeakyReLU')
                      ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes)[1])

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        return nn.Sequential(*layers)

    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass of Dual-Path FRNet Backbone.
        
        Args:
            voxel_dict (dict): Dictionary containing:
                - 'geo_voxel_feats': Geometric frustum features [M, C_geo]
                - 'geo_voxel_coors': Geometric frustum coordinates [M, 4]
                - 'geo_point_feats': Geometric point features [N, C_geo]
                - 'sem_voxel_feats': Semantic frustum features [M, C_sem]
                - 'sem_voxel_coors': Semantic frustum coordinates [M, 4]
                - 'sem_point_feats': Semantic point features [N, C_sem]
                - 'coors': Point coordinates [N, 3]
                
        Returns:
            dict: Updated voxel_dict with backbone features.
        """
        # Extract inputs
        geo_voxel_feats = voxel_dict['geo_voxel_feats']
        geo_voxel_coors = voxel_dict['geo_voxel_coors']
        geo_point_feats = voxel_dict['geo_point_feats']
        
        sem_voxel_feats = voxel_dict['sem_voxel_feats']
        sem_voxel_coors = voxel_dict['sem_voxel_coors']
        sem_point_feats = voxel_dict['sem_point_feats']
        
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1

        # Geometry Path: Convert to range image and extract features
        geo_pixel = self.frustum2pixel(
            geo_voxel_feats, geo_voxel_coors, batch_size, stride=1)
        geo_pixel = self.geo_stem(geo_pixel)  # [B, C, H, W]
        
        # Semantic Path: Convert to range image and extract features
        sem_pixel = self.frustum2pixel(
            sem_voxel_feats, sem_voxel_coors, batch_size, stride=1)
        sem_pixel = self.sem_stem(sem_pixel)  # [B, C, H, W]
        
        # Cross-Gated Fusion at pixel level
        if self.use_cross_gated_fusion:
            fused_pixel = self.cross_gated_fusion(geo_pixel, sem_pixel)
        else:
            fused_pixel = torch.cat([geo_pixel, sem_pixel], dim=1)
            fused_pixel = self.fusion_stem(fused_pixel)
        
        # Point-level fusion
        # Map fused pixel features to points
        map_point_feats = self.pixel2point(fused_pixel, pts_coors, stride=1)
        
        # Process point features from both paths
        geo_point_processed = self.geo_point_stem(geo_point_feats)
        sem_point_processed = self.sem_point_stem(sem_point_feats)
        
        # Cross-Gated Fusion at point level
        if self.use_cross_gated_fusion:
            fused_point_feats = self.point_cross_gated_fusion(
                geo_point_processed, sem_point_processed)
        else:
            fused_point_feats = torch.cat(
                [geo_point_processed, sem_point_processed], dim=1)
            fused_point_feats = self.point_fusion_stem(fused_point_feats)
        
        # FPFM: Frustum-to-Point fusion (eq.4) - Initial fusion at stem level
        fusion_point_feats = torch.cat(
            [map_point_feats, fused_point_feats], dim=1)
        point_fused = self.stem_point_fusion['fusion'](fusion_point_feats)
        point_gate = self.stem_point_fusion['gate'](fusion_point_feats)
        point_feats = point_fused * point_gate + fused_point_feats
        
        # FPFM: Point-to-Frustum fusion (eq.5) - Initial fusion at stem level
        stride_voxel_coors, frustum_feats, _ = self.point2frustum(
            point_feats, pts_coors, stride=1)
        pixel_feats = self.frustum2pixel(
            frustum_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat([pixel_feats, fused_pixel], dim=1)
        fuse_out = self.stem_pixel_fusion['fusion'](fusion_pixel_feats)
        fuse_gate = self.stem_pixel_fusion['gate'](fusion_pixel_feats)
        x = fuse_out * fuse_gate + fused_pixel

        outs = [x]
        out_points = [point_feats]
        
        # Process through ResNet stages with FPFM
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # FPFM: Frustum-to-Point fusion
            map_point_feats = self.pixel2point(
                x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat([map_point_feats, point_feats], dim=1)
            
            gated_point_fusion = self.point_fusion_layers[i]
            point_fused = gated_point_fusion['fusion'](fusion_point_feats)
            point_gate = gated_point_fusion['gate'](fusion_point_feats)
            point_feats = point_fused * point_gate + point_feats

            # FPFM: Point-to-Frustum fusion
            stride_voxel_coors, frustum_feats, _ = self.point2frustum(
                point_feats, pts_coors, stride=self.strides[i])
            pixel_feats = self.frustum2pixel(
                frustum_feats, stride_voxel_coors, batch_size,
                stride=self.strides[i])
            fusion_pixel_feats = torch.cat([pixel_feats, x], dim=1)
            
            gated_pixel_fusion = self.pixel_fusion_layers[i]
            fuse_out = gated_pixel_fusion['fusion'](fusion_pixel_feats)
            fuse_gate = gated_pixel_fusion['gate'](fusion_pixel_feats)
            fuse_out = fuse_out * fuse_gate
            
            # Enhanced residual-attentive fusion with channel and spatial attention
            attention_module = self.attention_layers[i]
            # Channel attention
            channel_att = attention_module['channel_attention'](fuse_out)
            fuse_out = fuse_out * channel_att
            # Spatial attention
            spatial_att = attention_module['spatial_attention'](fuse_out)
            x = fuse_out * spatial_att + x

            outs.append(x)
            out_points.append(point_feats)

        # Multi-scale feature fusion
        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i],
                    size=outs[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers,
                                                self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict

    def frustum2pixel(self,
                     frustum_features: Tensor,
                     coors: Tensor,
                     batch_size: int,
                     stride: int = 1) -> Tensor:
        """Convert frustum features to range image (pixel)."""
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype,
            device=frustum_features.device)
        pixel_features[coors[:, 0], coors[:, 1], coors[:, 2]] = frustum_features
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
        return pixel_features

    def pixel2point(self,
                    pixel_features: Tensor,
                    coors: Tensor,
                    stride: int = 1) -> Tensor:
        """Convert range image (pixel) to point features."""
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride,
                                     coors[:, 2] // stride]
        return point_feats

    def point2frustum(self,
                     point_features: Tensor,
                     pts_coors: Tensor,
                     stride: int = 1) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert point features to frustum features."""
        coors = pts_coors.clone()
        coors[:, 1] = pts_coors[:, 1] // stride
        coors[:, 2] = pts_coors[:, 2] // stride
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)
        frustum_features = torch_scatter.scatter_mean(
            point_features.float(), inverse_map, dim=0).to(point_features.dtype)
        point_counts = torch_scatter.scatter_sum(
            torch.ones(point_features.shape[0], 1, 
                      device=point_features.device, 
                      dtype=point_features.dtype),
            inverse_map, dim=0).squeeze(-1)
        return voxel_coors, frustum_features, point_counts

