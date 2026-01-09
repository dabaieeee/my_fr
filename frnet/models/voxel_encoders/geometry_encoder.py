from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch_scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class GeometryEncoder(nn.Module):
    """几何编码器，用于结构保持的几何特征提取。
    
    该编码器专注于提取局部几何特征，例如：
    - 点坐标 (xyz)
    - 表面法向量
    - 曲率信息
    - 局部平面置信度
    
    使用小感受野以保持几何结构并避免语义污染。
    
    Args:
        in_channels (int): 输入特征数量 (xyz + 可选特征)。
            默认为 3。
        feat_channels (Sequence[int]): 每个MLP层的特征数量。
            默认为 [64, 128, 128]。
        with_normals (bool): 是否计算和使用表面法向量。
            默认为 True。
        with_curvature (bool): 是否计算曲率特征。
            默认为 True。
        norm_cfg (dict): 归一化层的配置字典。
            默认为 dict(type='BN1d', eps=1e-5, momentum=0.1)。
        k_neighbors (int): 用于法向量/曲率计算的邻居点数量。
            默认为 10。
    """

    def __init__(self,
                 in_channels: int = 3,
                 feat_channels: Sequence[int] = [64, 128, 128],
                 with_normals: bool = True,
                 with_curvature: bool = True,
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-5, momentum=0.1),
                 k_neighbors: int = 10) -> None:
        super(GeometryEncoder, self).__init__()
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        self._with_normals = with_normals
        self._with_curvature = with_curvature
        self.k_neighbors = k_neighbors

        # 计算添加几何特征后的实际输入通道数
        actual_in_channels = in_channels
        if with_normals:
            actual_in_channels += 3  # 法向量 (nx, ny, nz)
        if with_curvature:
            actual_in_channels += 3  # 曲率特征 (线性度, 平面度, 球面度)

        # 构建用于几何特征提取的MLP层
        feat_channels = [actual_in_channels] + list(feat_channels)
        geo_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                geo_layers.append(nn.Linear(in_filters, out_filters))
            else:
                geo_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters, bias=False),
                        norm_layer, nn.ReLU(inplace=True)))

        self.geo_layers = nn.ModuleList(geo_layers)

    def compute_normals(self, points: torch.Tensor, 
                       coors: torch.Tensor, 
                       k: int = 10) -> torch.Tensor:
        """使用局部邻域计算表面法向量。
        
        使用基于PyTorch的k-NN搜索以提高效率。
        
        Args:
            points (Tensor): 点坐标 [N, 3]。
            coors (Tensor): 视锥坐标 [N, 4] (batch, y, x, z)。
            k (int): 用于法向量计算的邻居点数量。
            
        Returns:
            Tensor: 法向量 [N, 3]。
        """
        device = points.device
        N = points.shape[0]
        k = min(k + 1, N)  # +1 以排除自身
        
        # 计算成对距离
        dists = torch.cdist(points, points)  # [N, N]
        
        # 获取k个最近邻（包括自身）
        _, indices = torch.topk(dists, k, dim=1, largest=False)  # [N, k]
        
        normals = torch.zeros_like(points)
        for i in range(N):
            if k > 1:
                # 获取邻居点（排除自身）
                neighbor_indices = indices[i, 1:]  # 跳过第一个（自身）
                neighbor_points = points[neighbor_indices]  # [k-1, 3]
                center = points[i:i+1]  # [1, 3]
                
                if len(neighbor_points) > 0:
                    # 计算协方差矩阵
                    centered = neighbor_points - center  # [k-1, 3]
                    cov = torch.mm(centered.t(), centered) / len(centered)  # [3, 3]
                    
                    # 特征值分解
                    try:
                        eigenvals, eigenvecs = torch.linalg.eigh(cov)
                        # 法向量是特征值最小的特征向量
                        normal = eigenvecs[:, 0]
                        # 确保方向一致（指向原点）
                        center_vec = center.squeeze()
                        if torch.dot(normal, center_vec) > 0:
                            normal = -normal
                        normals[i] = normal
                    except:
                        # 备用方案：使用简单的法向量估计
                        if len(neighbor_points) >= 2:
                            v1 = neighbor_points[0] - center_vec
                            if len(neighbor_points) >= 3:
                                v2 = neighbor_points[1] - center_vec
                            else:
                                v2 = neighbor_points[-1] - center_vec
                            normal = torch.cross(v1, v2)
                            norm = torch.norm(normal)
                            if norm > 1e-6:
                                normals[i] = normal / norm
        
        return normals

    def compute_curvature(self, points: torch.Tensor,
                         coors: torch.Tensor,
                         k: int = 10) -> torch.Tensor:
        """计算曲率特征（线性度、平面度、球面度）。
        
        使用基于PyTorch的k-NN搜索以提高效率。
        
        Args:
            points (Tensor): 点坐标 [N, 3]。
            coors (Tensor): 视锥坐标 [N, 4]。
            k (int): 用于曲率计算的邻居点数量。
            
        Returns:
            Tensor: 曲率特征 [N, 3] (线性度, 平面度, 球面度)。
        """
        device = points.device
        N = points.shape[0]
        k = min(k + 1, N)  # +1 以排除自身
        
        # 计算成对距离
        dists = torch.cdist(points, points)  # [N, N]
        
        # 获取k个最近邻（包括自身）
        _, indices = torch.topk(dists, k, dim=1, largest=False)  # [N, k]
        
        curvature_features = torch.zeros((N, 3), device=device)
        for i in range(N):
            if k > 1:
                neighbor_indices = indices[i, 1:]  # 跳过第一个（自身）
                neighbor_points = points[neighbor_indices]
                center = points[i:i+1]
                
                if len(neighbor_points) > 0:
                    centered = neighbor_points - center
                    cov = torch.mm(centered.t(), centered) / len(centered)
                    try:
                        eigenvals, _ = torch.linalg.eigh(cov)
                        eigenvals = torch.abs(eigenvals)
                        eigenvals = torch.sort(eigenvals, descending=True)[0]
                        
                        # 归一化特征值
                        lambda_sum = eigenvals.sum()
                        if lambda_sum > 1e-6:
                            eigenvals = eigenvals / lambda_sum
                            
                            # 线性度: (lambda1 - lambda2) / lambda1
                            linearity = (eigenvals[0] - eigenvals[1]) / (eigenvals[0] + 1e-6)
                            # 平面度: (lambda2 - lambda3) / lambda1
                            planarity = (eigenvals[1] - eigenvals[2]) / (eigenvals[0] + 1e-6)
                            # 球面度: lambda3 / lambda1
                            sphericity = eigenvals[2] / (eigenvals[0] + 1e-6)
                            
                            curvature_features[i] = torch.stack([
                                linearity, planarity, sphericity
                            ])
                    except:
                        pass
        
        return curvature_features

    def forward(self, voxel_dict: dict) -> dict:
        """几何编码器的前向传播。
        
        Args:
            voxel_dict (dict): 包含以下键的字典：
                - 'voxels': 点特征 [N, C]
                - 'coors': 视锥坐标 [N, 4]
                
        Returns:
            dict: 更新后的voxel_dict，包含：
                - 'geo_point_feats': 几何点特征 [N, C_geo]
                - 'geo_voxel_feats': 几何视锥特征 [M, C_geo]
                - 'geo_voxel_coors': 视锥坐标 [M, 4]
        """
        features = voxel_dict['voxels']
        coors = voxel_dict['coors']
        
        # 提取xyz坐标（前3个通道）
        xyz = features[:, :3]
        
        # 构建几何特征
        geo_features = [xyz]
        
        # 如果启用，添加法向量
        if self._with_normals:
            normals = self.compute_normals(xyz, coors, k=self.k_neighbors)
            geo_features.append(normals)
        
        # 如果启用，添加曲率
        if self._with_curvature:
            curvature = self.compute_curvature(xyz, coors, k=self.k_neighbors)
            geo_features.append(curvature)
        
        # 拼接所有几何特征
        geo_input = torch.cat(geo_features, dim=-1)
        
        # 通过MLP层提取几何特征
        geo_feats = geo_input
        for geo_layer in self.geo_layers:
            geo_feats = geo_layer(geo_feats)
        
        # 聚合到视锥级别（使用最大池化以保持结构）
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)
        
        # 使用最大池化以保持几何结构
        geo_voxel_feats = torch_scatter.scatter_max(
            geo_feats.float(), inverse_map, dim=0)[0].to(geo_feats.dtype)
        
        voxel_dict['geo_point_feats'] = geo_feats
        voxel_dict['geo_voxel_feats'] = geo_voxel_feats
        voxel_dict['geo_voxel_coors'] = voxel_coors
        
        return voxel_dict

