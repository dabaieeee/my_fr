from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from torch import Tensor


def compute_normals(points: Tensor, k: int = 10) -> Tensor:
    """计算点云的法向量（使用局部PCA方法）。
    
    Args:
        points (Tensor): 点云坐标 [N, 3]
        k (int): 用于计算法向量的最近邻点数。Defaults to 10.
        
    Returns:
        Tensor: 法向量 [N, 3]
    """
    device = points.device
    N = points.shape[0]
    normals = torch.zeros_like(points)
    
    # 批量计算：对每个点找到k近邻
    # 使用广播计算所有点对之间的距离
    # points: [N, 3], points.unsqueeze(1): [N, 1, 3]
    # points.unsqueeze(0): [1, N, 3]
    dist_matrix = torch.cdist(points, points, p=2)  # [N, N]
    
    # 对每个点，找到k+1个最近邻（包括自身）
    _, indices = torch.topk(dist_matrix, k=min(k+1, N), dim=1, largest=False)  # [N, k+1]
    
    # 排除自身（第一个最近邻是自己）
    neighbor_indices = indices[:, 1:]  # [N, k]
    
    # 批量计算法向量
    for i in range(N):
        neighbor_idx = neighbor_indices[i]  # [k]
        if len(neighbor_idx) > 2:
            # 获取邻居点
            neighbors = points[neighbor_idx]  # [k, 3]
            center = points[i:i+1]  # [1, 3]
            
            # 计算局部协方差矩阵
            local_points = neighbors - center  # [k, 3]
            if local_points.shape[0] > 0:
                # 使用更稳定的协方差计算
                cov = torch.mm(local_points.t(), local_points) / (local_points.shape[0] - 1)  # [3, 3]
                
                # 计算特征值和特征向量
                try:
                    eigenvals, eigenvecs = torch.linalg.eigh(cov)
                    # 最小特征值对应的特征向量就是法向量
                    normals[i] = eigenvecs[:, 0]
                except:
                    # 如果计算失败，使用默认法向量
                    normals[i] = torch.tensor([0, 0, 1], device=device, dtype=points.dtype)
            else:
                normals[i] = torch.tensor([0, 0, 1], device=device, dtype=points.dtype)
        else:
            normals[i] = torch.tensor([0, 0, 1], device=device, dtype=points.dtype)
    
    # 归一化
    normals = F.normalize(normals, p=2, dim=1)
    return normals


def compute_curvature(points: Tensor, normals: Tensor, k: int = 10) -> Tensor:
    """计算点云的曲率（使用法向量变化）。
    
    Args:
        points (Tensor): 点云坐标 [N, 3]
        normals (Tensor): 法向量 [N, 3]
        k (int): 用于计算曲率的最近邻点数。Defaults to 10.
        
    Returns:
        Tensor: 曲率 [N, 1]
    """
    N = points.shape[0]
    curvatures = torch.zeros(N, 1, device=points.device, dtype=points.dtype)
    
    # 批量计算距离矩阵
    dist_matrix = torch.cdist(points, points, p=2)  # [N, N]
    _, indices = torch.topk(dist_matrix, k=min(k+1, N), dim=1, largest=False)  # [N, k+1]
    neighbor_indices = indices[:, 1:]  # [N, k]
    
    # 批量计算曲率
    for i in range(N):
        neighbor_idx = neighbor_indices[i]  # [k]
        if len(neighbor_idx) > 0:
            # 计算局部曲率（使用法向量变化）
            local_normals = normals[neighbor_idx]  # [k, 3]
            normal_variation = torch.norm(local_normals - normals[i:i+1], dim=1).mean()
            curvatures[i] = normal_variation
        else:
            curvatures[i] = 0.0
    
    return curvatures


@MODELS.register_module()
class GeometryEncoder(nn.Module):
    """Geometry Encoder for structure-preserving geometric features.
    
    This encoder focuses on local geometric structures (normals, curvature, 
    plane-aware features) without semantic context mixing.
    
    Args:
        in_channels (int): Number of input features (xyz). Defaults to 3.
        feat_channels (Sequence[int]): Number of features in each layer.
            Defaults to (32, 64, 128).
        with_normals (bool): Whether to compute and use normals. Defaults to True.
        with_curvature (bool): Whether to compute and use curvature. Defaults to True.
        with_distance (bool): Whether to include Euclidean distance. Defaults to True.
        norm_cfg (dict or :obj:`ConfigDict`): Config dict of normalization
            layers. Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1).
        k_neighbors (int): Number of neighbors for computing normals/curvature.
            Defaults to 10.
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 feat_channels: Sequence[int] = (32, 64, 128),
                 with_normals: bool = True,
                 with_curvature: bool = True,
                 with_distance: bool = True,
                 norm_cfg: ConfigType = dict(type='BN1d', eps=1e-5, momentum=0.1),
                 k_neighbors: int = 10) -> None:
        super(GeometryEncoder, self).__init__()
        
        assert len(feat_channels) > 0
        self.with_normals = with_normals
        self.with_curvature = with_curvature
        self.with_distance = with_distance
        self.k_neighbors = k_neighbors
        
        # 计算输入通道数
        actual_in_channels = in_channels  # xyz
        if with_normals:
            actual_in_channels += 3  # nx, ny, nz
        if with_curvature:
            actual_in_channels += 1  # curvature
        if with_distance:
            actual_in_channels += 1  # distance
        
        self.actual_in_channels = actual_in_channels
        
        # 构建MLP层（使用小感受野，保持局部性）
        feat_channels = [actual_in_channels] + list(feat_channels)
        self.mlp_layers = nn.ModuleList()
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                # 最后一层不加激活函数
                self.mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters),
                        norm_layer))
            else:
                self.mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters, bias=False),
                        norm_layer,
                        nn.ReLU(inplace=True)))
        
        self.output_channels = feat_channels[-1]
    
    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass.
        
        Args:
            voxel_dict (dict): Input dictionary containing:
                - 'voxels': Point features [N, C] (should contain xyz)
                - 'coors': Point coordinates [N, 3]
                
        Returns:
            dict: Output dictionary containing:
                - 'geometry_feats': Geometry features [N, C_geo]
                - 'geometry_point_feats': Per-point geometry features (for multi-scale)
        """
        points = voxel_dict['voxels']
        coors = voxel_dict['coors']
        
        # 提取xyz坐标
        xyz = points[:, :3]  # [N, 3]
        
        # 计算几何特征
        features_list = [xyz]
        
        # 计算法向量
        if self.with_normals:
            normals = compute_normals(xyz, k=self.k_neighbors)  # [N, 3]
            features_list.append(normals)
        
        # 计算曲率
        if self.with_curvature:
            if self.with_normals:
                curvatures = compute_curvature(xyz, normals, k=self.k_neighbors)  # [N, 1]
            else:
                # 如果没有法向量，先计算
                normals = compute_normals(xyz, k=self.k_neighbors)
                curvatures = compute_curvature(xyz, normals, k=self.k_neighbors)
            features_list.append(curvatures)
        
        # 计算距离
        if self.with_distance:
            distances = torch.norm(xyz, dim=1, keepdim=True)  # [N, 1]
            features_list.append(distances)
        
        # 拼接所有几何特征
        geometry_input = torch.cat(features_list, dim=-1)  # [N, C_in]
        
        # 通过MLP提取几何特征
        geometry_feats = geometry_input
        geometry_point_feats = []
        for mlp_layer in self.mlp_layers:
            geometry_feats = mlp_layer(geometry_feats)
            geometry_point_feats.append(geometry_feats)
        
        # 保存结果
        voxel_dict['geometry_feats'] = geometry_feats  # [N, C_geo]
        voxel_dict['geometry_point_feats'] = geometry_point_feats  # 多尺度特征
        
        return voxel_dict

