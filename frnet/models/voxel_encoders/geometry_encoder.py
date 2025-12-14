from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch_scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class GeometryEncoder(nn.Module):
    """Geometry Encoder for structure-preserving geometric feature extraction.
    
    This encoder focuses on extracting local geometric features such as:
    - Point coordinates (xyz)
    - Surface normals
    - Curvature information
    - Local plane confidence
    
    It uses small receptive fields to preserve geometric structure and
    avoid semantic contamination.
    
    Args:
        in_channels (int): Number of input features (xyz + optional features).
            Defaults to 3.
        feat_channels (Sequence[int]): Number of features in each MLP layer.
            Defaults to [64, 128, 128].
        with_normals (bool): Whether to compute and use surface normals.
            Defaults to True.
        with_curvature (bool): Whether to compute curvature features.
            Defaults to True.
        norm_cfg (dict): Config dict of normalization layers.
            Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1).
        k_neighbors (int): Number of neighbors for normal/curvature computation.
            Defaults to 10.
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

        # Calculate actual input channels after adding geometric features
        actual_in_channels = in_channels
        if with_normals:
            actual_in_channels += 3  # normal vector (nx, ny, nz)
        if with_curvature:
            actual_in_channels += 3  # curvature features (linearity, planarity, sphericity)

        # Build MLP layers for geometric feature extraction
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
        """Compute surface normals using local neighborhood.
        
        Optimized version using frustum-based spatial search instead of 
        expensive pairwise distance computation.
        
        Args:
            points (Tensor): Point coordinates [N, 3].
            coors (Tensor): Frustum coordinates [N, 4] (batch, y, x, z).
            k (int): Number of neighbors for normal computation.
            
        Returns:
            Tensor: Normal vectors [N, 3].
        """
        device = points.device
        N = points.shape[0]
        normals = torch.zeros_like(points)
        
        if N == 0:
            return normals
        
        # Use frustum-based spatial search for efficiency
        # Group points by batch and use frustum coordinates (y, x) for fast neighbor search
        batch_indices = coors[:, 0].long()
        y_coords = coors[:, 1].long()
        x_coords = coors[:, 2].long()
        
        # Vectorized normal computation using frustum neighbors
        # For each point, search neighbors in a local window in frustum space
        search_radius = max(1, int(k ** 0.5))  # Adaptive search radius
        
        for batch_idx in torch.unique(batch_indices):
            batch_mask = batch_indices == batch_idx
            batch_points = points[batch_mask]
            batch_y = y_coords[batch_mask]
            batch_x = x_coords[batch_mask]
            batch_indices_local = torch.where(batch_mask)[0]
            
            if len(batch_points) == 0:
                continue
            
            # Create a grid-based neighbor search
            # For each point, find neighbors within search_radius in frustum space
            for i, (y, x) in enumerate(zip(batch_y, batch_x)):
                # Find neighbors within search radius
                y_diff = torch.abs(batch_y - y)
                x_diff = torch.abs(batch_x - x)
                neighbor_mask = (y_diff <= search_radius) & (x_diff <= search_radius)
                neighbor_mask[i] = False  # Exclude self
                
                neighbor_indices = torch.where(neighbor_mask)[0]
                
                if len(neighbor_indices) >= 3:
                    # Get neighbor points
                    neighbor_points = batch_points[neighbor_indices]
                    center = batch_points[i:i+1]
                    
                    # Compute covariance matrix (vectorized)
                    centered = neighbor_points - center  # [k_neighbors, 3]
                    cov = torch.mm(centered.t(), centered) / len(centered)  # [3, 3]
                    
                    # Eigenvalue decomposition
                    try:
                        eigenvals, eigenvecs = torch.linalg.eigh(cov)
                        # Normal is the eigenvector with smallest eigenvalue
                        normal = eigenvecs[:, 0]
                        # Ensure consistent orientation (pointing towards origin)
                        center_vec = center.squeeze()
                        if torch.dot(normal, center_vec) > 0:
                            normal = -normal
                        normals[batch_indices_local[i]] = normal
                    except:
                        # Fallback: use simple normal estimation
                        if len(neighbor_points) >= 2:
                            v1 = neighbor_points[0] - center_vec
                            if len(neighbor_points) >= 3:
                                v2 = neighbor_points[1] - center_vec
                            else:
                                v2 = neighbor_points[-1] - center_vec
                            normal = torch.cross(v1, v2)
                            norm = torch.norm(normal)
                            if norm > 1e-6:
                                normals[batch_indices_local[i]] = normal / norm
        
        return normals

    def compute_curvature(self, points: torch.Tensor,
                         coors: torch.Tensor,
                         k: int = 10) -> torch.Tensor:
        """Compute curvature features (linearity, planarity, sphericity).
        
        Optimized version using frustum-based spatial search.
        
        Args:
            points (Tensor): Point coordinates [N, 3].
            coors (Tensor): Frustum coordinates [N, 4].
            k (int): Number of neighbors for curvature computation.
            
        Returns:
            Tensor: Curvature features [N, 3] (linearity, planarity, sphericity).
        """
        device = points.device
        N = points.shape[0]
        curvature_features = torch.zeros((N, 3), device=device)
        
        if N == 0:
            return curvature_features
        
        # Use frustum-based spatial search for efficiency
        batch_indices = coors[:, 0].long()
        y_coords = coors[:, 1].long()
        x_coords = coors[:, 2].long()
        
        search_radius = max(1, int(k ** 0.5))  # Adaptive search radius
        
        for batch_idx in torch.unique(batch_indices):
            batch_mask = batch_indices == batch_idx
            batch_points = points[batch_mask]
            batch_y = y_coords[batch_mask]
            batch_x = x_coords[batch_mask]
            batch_indices_local = torch.where(batch_mask)[0]
            
            if len(batch_points) == 0:
                continue
            
            # Create a grid-based neighbor search
            for i, (y, x) in enumerate(zip(batch_y, batch_x)):
                # Find neighbors within search radius
                y_diff = torch.abs(batch_y - y)
                x_diff = torch.abs(batch_x - x)
                neighbor_mask = (y_diff <= search_radius) & (x_diff <= search_radius)
                neighbor_mask[i] = False  # Exclude self
                
                neighbor_indices = torch.where(neighbor_mask)[0]
                
                if len(neighbor_indices) >= 3:
                    neighbor_points = batch_points[neighbor_indices]
                    center = batch_points[i:i+1]
                    
                    centered = neighbor_points - center
                    cov = torch.mm(centered.t(), centered) / len(centered)
                    try:
                        eigenvals, _ = torch.linalg.eigh(cov)
                        eigenvals = torch.abs(eigenvals)
                        eigenvals = torch.sort(eigenvals, descending=True)[0]
                        
                        # Normalize eigenvalues
                        lambda_sum = eigenvals.sum()
                        if lambda_sum > 1e-6:
                            eigenvals = eigenvals / lambda_sum
                            
                            # Linearity: (lambda1 - lambda2) / lambda1
                            linearity = (eigenvals[0] - eigenvals[1]) / (eigenvals[0] + 1e-6)
                            # Planarity: (lambda2 - lambda3) / lambda1
                            planarity = (eigenvals[1] - eigenvals[2]) / (eigenvals[0] + 1e-6)
                            # Sphericity: lambda3 / lambda1
                            sphericity = eigenvals[2] / (eigenvals[0] + 1e-6)
                            
                            curvature_features[batch_indices_local[i]] = torch.stack([
                                linearity, planarity, sphericity
                            ])
                    except:
                        pass
        
        return curvature_features

    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass of Geometry Encoder.
        
        Args:
            voxel_dict (dict): Dictionary containing:
                - 'voxels': Point features [N, C]
                - 'coors': Frustum coordinates [N, 4]
                
        Returns:
            dict: Updated voxel_dict with:
                - 'geo_point_feats': Geometric point features [N, C_geo]
                - 'geo_voxel_feats': Geometric frustum features [M, C_geo]
                - 'geo_voxel_coors': Frustum coordinates [M, 4]
        """
        features = voxel_dict['voxels']
        coors = voxel_dict['coors']
        
        # Extract xyz coordinates (first 3 channels)
        xyz = features[:, :3]
        
        # Build geometric features
        geo_features = [xyz]
        
        # Add normals if enabled
        if self._with_normals:
            normals = self.compute_normals(xyz, coors, k=self.k_neighbors)
            geo_features.append(normals)
        
        # Add curvature if enabled
        if self._with_curvature:
            curvature = self.compute_curvature(xyz, coors, k=self.k_neighbors)
            geo_features.append(curvature)
        
        # Concatenate all geometric features
        geo_input = torch.cat(geo_features, dim=-1)
        
        # Extract geometric features through MLP layers
        geo_feats = geo_input
        for geo_layer in self.geo_layers:
            geo_feats = geo_layer(geo_feats)
        
        # Aggregate to frustum level (max pooling to preserve structure)
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)
        
        # Use max pooling to preserve geometric structure
        geo_voxel_feats = torch_scatter.scatter_max(
            geo_feats.float(), inverse_map, dim=0)[0].to(geo_feats.dtype)
        
        voxel_dict['geo_point_feats'] = geo_feats
        voxel_dict['geo_voxel_feats'] = geo_voxel_feats
        voxel_dict['geo_voxel_coors'] = voxel_coors
        
        return voxel_dict

