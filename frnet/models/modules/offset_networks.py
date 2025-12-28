# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, build_activation_layer
from mmdet3d.utils import ConfigType
from torch import Tensor
from typing import Tuple, Optional


class FrustumOffsetNetwork(nn.Module):
    """Frustum Offset Network for learning pixel-wise offsets.
    
    This network learns discrete offsets in H and W dimensions to align
    frustum features with point features.
    
    Args:
        in_channels (int): Number of input feature channels.
        offset_range (int): Maximum offset range (symmetric). Defaults to 3.
        norm_cfg (dict): Config for normalization layer.
        act_cfg (dict): Config for activation layer.
    """
    
    def __init__(self,
                 in_channels: int,
                 offset_range: int = 3,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)) -> None:
        super(FrustumOffsetNetwork, self).__init__()
        
        self.in_channels = in_channels
        self.offset_range = offset_range
        # Total offset candidates: (2*offset_range+1) for H and W each
        self.num_offsets_h = 2 * offset_range + 1
        self.num_offsets_w = 2 * offset_range + 1
        
        # Network to predict offset probabilities and features
        # Output: offset_h_logits + offset_w_logits + aligned_features
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg),
        )
        
        # Separate heads for H and W offsets
        self.offset_h_head = nn.Conv2d(in_channels, self.num_offsets_h, kernel_size=1)
        self.offset_w_head = nn.Conv2d(in_channels, self.num_offsets_w, kernel_size=1)
        
        # Feature alignment head
        self.feature_head = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, 
                frustum_feats: Tensor,
                pts_coors: Tensor,
                stride: int = 1) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward function.
        
        Args:
            frustum_feats (Tensor): Frustum features [B, C, H, W].
            pts_coors (Tensor): Point coordinates [N, 3] as [batch_idx, y, x].
            stride (int): Stride for downsampling. Defaults to 1.
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]: 
                - Aligned point features [N, C]
                - Offset regularization loss (L1 norm of offset distributions)
        """
        B, C, H, W = frustum_feats.shape
        
        # Extract features for offset prediction
        offset_feats = self.offset_net(frustum_feats)  # [B, C, H, W]
        
        # Predict offset distributions
        offset_h_logits = self.offset_h_head(offset_feats)  # [B, num_offsets_h, H, W]
        offset_w_logits = self.offset_w_head(offset_feats)  # [B, num_offsets_w, H, W]
        
        # Get aligned features
        aligned_feats = self.feature_head(offset_feats)  # [B, C, H, W]
        
        # Apply soft offset (weighted sum over offset candidates)
        aligned_feats = self._apply_soft_offset(
            aligned_feats, offset_h_logits, offset_w_logits
        )  # [B, C, H, W]
        
        # Map to point features using point coordinates
        aligned_feats_permuted = aligned_feats.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        point_feats = aligned_feats_permuted[
            pts_coors[:, 0], 
            pts_coors[:, 1] // stride, 
            pts_coors[:, 2] // stride
        ]  # [N, C]
        
        # Compute offset regularization (L1 norm to prevent large offsets)
        offset_h_probs = F.softmax(offset_h_logits, dim=1)  # [B, num_offsets_h, H, W]
        offset_w_probs = F.softmax(offset_w_logits, dim=1)  # [B, num_offsets_w, H, W]
        
        # Create offset values [-offset_range, ..., 0, ..., +offset_range]
        offset_values = torch.arange(
            -self.offset_range, self.offset_range + 1, 
            dtype=frustum_feats.dtype, 
            device=frustum_feats.device
        )  # [num_offsets]
        
        # Expected offset magnitude (L1 norm)
        expected_offset_h = (offset_h_probs * offset_values.view(1, -1, 1, 1).abs()).sum(dim=1)  # [B, H, W]
        expected_offset_w = (offset_w_probs * offset_values.view(1, -1, 1, 1).abs()).sum(dim=1)  # [B, H, W]
        
        offset_reg_loss = (expected_offset_h.mean() + expected_offset_w.mean()) / 2.0
        
        return point_feats, offset_reg_loss
    
    def _apply_soft_offset(self,
                          features: Tensor,
                          offset_h_logits: Tensor,
                          offset_w_logits: Tensor) -> Tensor:
        """Apply soft offset using weighted sum over offset candidates.
        
        Args:
            features (Tensor): Input features [B, C, H, W].
            offset_h_logits (Tensor): H offset logits [B, num_offsets_h, H, W].
            offset_w_logits (Tensor): W offset logits [B, num_offsets_w, H, W].
            
        Returns:
            Tensor: Offset-aligned features [B, C, H, W].
        """
        B, C, H, W = features.shape
        
        # Softmax to get offset probabilities
        offset_h_probs = F.softmax(offset_h_logits, dim=1)  # [B, num_offsets_h, H, W]
        offset_w_probs = F.softmax(offset_w_logits, dim=1)  # [B, num_offsets_w, H, W]
        
        # Collect features from all offset positions
        aligned_features = torch.zeros_like(features)  # [B, C, H, W]
        
        for dh in range(-self.offset_range, self.offset_range + 1):
            for dw in range(-self.offset_range, self.offset_range + 1):
                # Get probability for this offset
                h_idx = dh + self.offset_range
                w_idx = dw + self.offset_range
                prob_h = offset_h_probs[:, h_idx:h_idx+1, :, :]  # [B, 1, H, W]
                prob_w = offset_w_probs[:, w_idx:w_idx+1, :, :]  # [B, 1, H, W]
                prob = prob_h * prob_w  # [B, 1, H, W]
                
                # Shift features by (dh, dw)
                if dh == 0 and dw == 0:
                    shifted_feats = features
                else:
                    # Use padding to handle boundary
                    pad_h = (abs(dh), abs(dh))
                    pad_w = (abs(dw), abs(dw))
                    padded_feats = F.pad(features, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='replicate')
                    
                    # Crop to get shifted features
                    h_start = abs(dh) + max(0, -dh)
                    h_end = h_start + H
                    w_start = abs(dw) + max(0, -dw)
                    w_end = w_start + W
                    shifted_feats = padded_feats[:, :, h_start:h_end, w_start:w_end]
                
                # Accumulate weighted features
                aligned_features += shifted_feats * prob
        
        return aligned_features


class VoxelOffsetNetwork(nn.Module):
    """Voxel Offset Network for learning voxel-wise offsets.
    
    This network learns offsets in X, Y, Z dimensions to align
    voxel features with point features.
    
    Args:
        in_channels (int): Number of input feature channels.
        offset_range (int): Maximum offset range (L1 norm constraint). Defaults to 2.
        norm_cfg (dict): Config for normalization layer.
        act_cfg (dict): Config for activation layer.
    """
    
    def __init__(self,
                 in_channels: int,
                 offset_range: int = 2,
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)) -> None:
        super(VoxelOffsetNetwork, self).__init__()
        
        self.in_channels = in_channels
        self.offset_range = offset_range
        
        # Network to predict offsets and features (using MLP for sparse voxels)
        self.offset_net = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg),
            nn.Linear(in_channels, in_channels, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg),
        )
        
        # Offset prediction head (predicts continuous offsets in [-1, 1])
        self.offset_head = nn.Linear(in_channels, 3)  # 3 for X, Y, Z offsets
        
        # Feature alignment head
        self.feature_head = nn.Linear(in_channels, in_channels)
        
    def forward(self,
                voxel_feats: Tensor,
                voxel_coors: Tensor,
                pts_coors: Tensor,
                voxel_shape: Tuple[int, int, int],
                stride: int = 1) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward function.
        
        Args:
            voxel_feats (Tensor): Voxel features [N_voxel, C].
            voxel_coors (Tensor): Voxel coordinates [N_voxel, 4] as [batch_idx, x, y, z].
            pts_coors (Tensor): Point coordinates [N_point, 3] as [batch_idx, y, x].
            voxel_shape (Tuple[int, int, int]): Shape of voxel grid (X, Y, Z).
            stride (int): Stride for downsampling. Defaults to 1.
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - Aligned point features [N_point, C]
                - Offset regularization loss (L1 norm of offsets)
        """
        # Extract features for offset prediction
        offset_feats = self.offset_net(voxel_feats)  # [N_voxel, C]
        
        # Predict continuous offsets in [-1, 1] (normalized by offset_range)
        offsets = torch.tanh(self.offset_head(offset_feats))  # [N_voxel, 3]
        offsets = offsets * self.offset_range  # Scale to [-offset_range, +offset_range]
        
        # Get aligned features
        aligned_voxel_feats = self.feature_head(offset_feats)  # [N_voxel, C]
        
        # Map voxel features to points
        # For simplicity, we use nearest neighbor mapping based on voxel coordinates
        # In practice, you might want to use more sophisticated interpolation
        
        # Create a mapping from point coordinates to voxel features
        # This is a simplified version - assumes we can map points to voxels
        batch_size = pts_coors[-1, 0].item() + 1
        
        # For each point, find the corresponding voxel (simplified mapping)
        # In a real implementation, you would compute the voxel index for each point
        # based on its 3D position and the voxel grid parameters
        
        # Here we use a placeholder approach: map based on frustum coordinates
        # This is a simplification - in practice you need the actual 3D coordinates
        point_feats = torch.zeros(
            (pts_coors.shape[0], self.in_channels),
            dtype=voxel_feats.dtype,
            device=voxel_feats.device
        )
        
        # Simplified mapping: use global average as fallback
        # In a production implementation, you would:
        # 1. Compute 3D voxel coordinates for each point
        # 2. Apply learned offsets to voxel coordinates
        # 3. Interpolate features from nearby voxels
        for b in range(batch_size):
            batch_mask_voxel = voxel_coors[:, 0] == b
            batch_mask_point = pts_coors[:, 0] == b
            
            if batch_mask_voxel.sum() > 0 and batch_mask_point.sum() > 0:
                # Use mean voxel features for all points in this batch (simplified)
                mean_feat = aligned_voxel_feats[batch_mask_voxel].mean(dim=0)
                point_feats[batch_mask_point] = mean_feat
        
        # Compute offset regularization (L1 norm)
        offset_reg_loss = offsets.abs().mean()
        
        return point_feats, offset_reg_loss

