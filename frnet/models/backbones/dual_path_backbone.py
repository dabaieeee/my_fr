from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType

from ..voxel_encoders.cross_gated_fusion import CrossGatedFusion
from .frnet_backbone import FRNetBackbone


@MODELS.register_module()
class DualPathBackbone(nn.Module):
    """Dual-Path Backbone with Geometry-Semantic Decoupling.
    
    This backbone implements a structure-level decoupling approach:
    - Geometry Path: Processes geometric features (normals, curvature) with
      local, structure-preserving operations
    - Semantic Path: Processes semantic context using FFE and FPFM with
      larger receptive fields
    - Cross-Gated Fusion: Adaptively combines both paths
    
    Args:
        geometry_encoder (dict): Config for GeometryEncoder.
        semantic_backbone (dict): Config for FRNetBackbone (semantic path).
        fusion_cfg (dict): Config for CrossGatedFusion.
        output_shape (Sequence[int]): Output shape [H, W] for range image.
    """
    
    def __init__(self,
                 geometry_encoder: ConfigType,
                 semantic_backbone: ConfigType,
                 fusion_cfg: ConfigType,
                 output_shape: Sequence[int],
                 init_cfg: OptConfigType = None) -> None:
        super(DualPathBackbone, self).__init__()
        
        from mmdet3d.registry import MODELS as MODELS_REGISTRY
        
        # Geometry Path: Structure-preserving encoder
        self.geometry_encoder = MODELS_REGISTRY.build(geometry_encoder)
        geo_channels = self.geometry_encoder.output_channels
        
        # Semantic Path: Context-aware backbone (preserves FFE and FPFM)
        # 注意：semantic_backbone需要配置为使用FFE和FPFM
        self.semantic_backbone = MODELS_REGISTRY.build(semantic_backbone)
        
        # 获取语义路径的输出通道数
        # 从semantic_backbone的配置中获取fuse_channels
        sem_channels = semantic_backbone.get('fuse_channels', [128])
        if isinstance(sem_channels, (list, tuple)) and len(sem_channels) > 0:
            sem_channels = sem_channels[-1]
        else:
            # 如果fuse_channels不存在，尝试从out_channels获取
            out_channels = semantic_backbone.get('out_channels', [128])
            if isinstance(out_channels, (list, tuple)) and len(out_channels) > 0:
                sem_channels = out_channels[-1]
            else:
                sem_channels = 128  # 默认值
        
        # Cross-Gated Fusion
        fusion_cfg['geo_channels'] = geo_channels
        fusion_cfg['sem_channels'] = sem_channels
        self.fusion = MODELS_REGISTRY.build(fusion_cfg)
        
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
    
    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass.
        
        Args:
            voxel_dict (dict): Input dictionary containing:
                - 'voxels': Point features [N, C]
                - 'coors': Point coordinates [N, 3]
                - 'voxel_feats': Frustum features from FFE
                - 'voxel_coors': Frustum coordinates
                - 'point_feats': Point features from FFE
                
        Returns:
            dict: Output dictionary with fused features.
        """
        # ========== Geometry Path ==========
        # 提取几何特征（只使用xyz，不混合语义信息）
        # 注意：geometry_encoder已经在segmentor中调用，这里直接使用结果
        if 'geometry_feats' not in voxel_dict:
            # 如果没有几何特征，在这里计算
            geo_voxel_dict = {
                'voxels': voxel_dict['voxels'],
                'coors': voxel_dict['coors']
            }
            geo_voxel_dict = self.geometry_encoder(geo_voxel_dict)
            geometry_feats = geo_voxel_dict['geometry_feats']  # [N, C_geo]
            geometry_point_feats = geo_voxel_dict.get('geometry_point_feats', [geometry_feats])
        else:
            # 使用已有的几何特征
            geometry_feats = voxel_dict['geometry_feats']
            geometry_point_feats = voxel_dict.get('geometry_point_feats', [geometry_feats])
        
        # ========== Semantic Path ==========
        # 使用原有的FFE和FPFM结构（已经在voxel_dict中）
        # semantic_backbone会处理frustum和point的双向融合
        semantic_voxel_dict = self.semantic_backbone(voxel_dict)
        
        # 获取语义路径的特征
        semantic_voxel_feats = semantic_voxel_dict['voxel_feats']  # List of [B, C, H, W]
        semantic_point_feats = semantic_voxel_dict.get('point_feats_backbone', 
                                                       semantic_voxel_dict.get('point_feats', []))
        
        # ========== Cross-Gated Fusion ==========
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1
        
        # 在像素级别融合（range image）- 主要融合位置
        if len(semantic_voxel_feats) > 0:
            # 将几何特征映射到range image
            # 将几何点特征聚合到frustum
            geo_frustum_feats = self._point2frustum(geometry_feats, pts_coors)
            geo_pixel_feats = self._frustum2pixel(geo_frustum_feats, pts_coors, batch_size)
            
            # 使用像素级别的融合
            # 创建像素级融合模块（如果还没有）
            if not hasattr(self, 'pixel_fusion'):
                from mmdet3d.registry import MODELS as MODELS_REGISTRY
                fusion_cfg_pixel = {
                    'type': 'CrossGatedFusion',
                    'geo_channels': geo_pixel_feats.shape[1],
                    'sem_channels': semantic_voxel_feats[0].shape[1],
                    'out_channels': semantic_voxel_feats[0].shape[1],
                    'fusion_type': 'pixel'
                }
                self.pixel_fusion = MODELS_REGISTRY.build(fusion_cfg_pixel)
            
            fused_pixel_feats = self.pixel_fusion(geo_pixel_feats, semantic_voxel_feats[0])
            
            # 更新voxel_feats
            semantic_voxel_feats[0] = fused_pixel_feats
            
            # 将融合后的像素特征映射回点特征
            fused_point_feats = self._pixel2point(fused_pixel_feats, pts_coors)
        else:
            # 如果没有语义voxel特征，只使用几何特征
            fused_point_feats = geometry_feats
        
        # 在点级别也进行融合（作为补充）
        if len(semantic_point_feats) > 0:
            # 使用backbone输出的点特征
            sem_point_feat = semantic_point_feats[0] if isinstance(semantic_point_feats, list) else semantic_point_feats
            # 确保维度匹配
            if sem_point_feat.shape[0] == geometry_feats.shape[0]:
                # 点级别融合（使用点级融合模块）
                if not hasattr(self, 'point_fusion'):
                    # 如果还没有点级融合模块，使用像素级融合的结果
                    fused_point_feats = self.fusion(geometry_feats, sem_point_feat)
                else:
                    fused_point_feats = self.fusion(geometry_feats, sem_point_feat)
        
        # 更新输出
        semantic_voxel_dict['voxel_feats'] = semantic_voxel_feats
        semantic_voxel_dict['point_feats_backbone'] = [fused_point_feats]
        semantic_voxel_dict['geometry_feats'] = geometry_feats
        semantic_voxel_dict['fused_point_feats'] = fused_point_feats
        
        return semantic_voxel_dict
    
    def _point2frustum(self, point_features: torch.Tensor, pts_coors: torch.Tensor) -> torch.Tensor:
        """将点特征聚合到frustum。
        
        Args:
            point_features (Tensor): 点特征 [N, C]
            pts_coors (Tensor): 点坐标 [N, 3]
            
        Returns:
            Tensor: Frustum特征 [N_frustum, C]
        """
        import torch_scatter
        
        # 使用frustum坐标聚合
        coors = pts_coors.clone()
        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)
        
        # 使用scatter_mean聚合
        frustum_features = torch_scatter.scatter_mean(
            point_features.float(), inverse_map, dim=0).to(point_features.dtype)
        
        return frustum_features
    
    def _frustum2pixel(self, frustum_features: torch.Tensor, coors: torch.Tensor, 
                      batch_size: int) -> torch.Tensor:
        """将frustum特征转换为range image。
        
        Args:
            frustum_features (Tensor): Frustum特征 [N_frustum, C]
            coors (Tensor): Frustum坐标 [N_frustum, 3]
            batch_size (int): Batch大小
            
        Returns:
            Tensor: Range image特征 [B, C, H, W]
        """
        nx = self.nx
        ny = self.ny
        
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype,
            device=frustum_features.device)
        
        # 确保coors索引在有效范围内
        valid_mask = (coors[:, 1] >= 0) & (coors[:, 1] < ny) & \
                     (coors[:, 2] >= 0) & (coors[:, 2] < nx)
        if valid_mask.sum() > 0:
            valid_coors = coors[valid_mask]
            valid_feats = frustum_features[valid_mask]
            pixel_features[valid_coors[:, 0], valid_coors[:, 1], valid_coors[:, 2]] = valid_feats
        
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
        return pixel_features
    
    def _pixel2point(self, pixel_features: torch.Tensor, coors: torch.Tensor) -> torch.Tensor:
        """将range image特征转换为点特征。
        
        Args:
            pixel_features (Tensor): Range image特征 [B, C, H, W]
            coors (Tensor): 点坐标 [N, 3]
            
        Returns:
            Tensor: 点特征 [N, C]
        """
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        point_feats = pixel_features[coors[:, 0], coors[:, 1], coors[:, 2]]
        return point_feats

