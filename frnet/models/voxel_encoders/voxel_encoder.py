from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from torch import Tensor


@MODELS.register_module()
class VoxelFeatureEncoder(nn.Module):
    """Voxel Feature Encoder using 3D Convolutions.
    
    Args:
        in_channels (int): Number of input features. Defaults to 4.
        feat_channels (Sequence[int]): Number of features in each 3D conv layer.
            Defaults to (64, 128, 256).
        voxel_size (Sequence[float]): Size of each voxel in [x, y, z]. 
            Defaults to (0.1, 0.1, 0.1).
        point_cloud_range (Sequence[float]): Range of point cloud in 
            [x_min, y_min, z_min, x_max, y_max, z_max]. 
            Defaults to (-50.0, -50.0, -3.0, 50.0, 50.0, 3.0).
        norm_cfg (dict or :obj:`ConfigDict`): Config dict of normalization
            layers. Defaults to dict(type='BN', eps=1e-5, momentum=0.1).
        act_cfg (dict or :obj:`ConfigDict`): Config dict of activation layers.
            Defaults to dict(type='ReLU', inplace=True).
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = (64, 128, 256),
                 voxel_size: Sequence[float] = (0.2, 0.2, 0.2),
                 point_cloud_range: Sequence[float] = (-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
                 norm_cfg: ConfigType = dict(type='BN', eps=1e-5, momentum=0.1),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 use_sparse: bool = True) -> None:
        super(VoxelFeatureEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.use_sparse = use_sparse
        
        # 计算体素网格的尺寸（用于参考，实际不使用密集网格）
        voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32)
        point_cloud_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.voxel_shape = (
            int((point_cloud_range_tensor[3] - point_cloud_range_tensor[0]) / voxel_size_tensor[0]),
            int((point_cloud_range_tensor[4] - point_cloud_range_tensor[1]) / voxel_size_tensor[1]),
            int((point_cloud_range_tensor[5] - point_cloud_range_tensor[2]) / voxel_size_tensor[2])
        )
        
        self.use_sparse = use_sparse
        if use_sparse:
            # 使用MLP处理稀疏体素特征，避免创建密集网格
            self.mlp_layers = nn.ModuleList()
            in_ch = in_channels
            for out_ch in feat_channels:
                self.mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(in_ch, out_ch, bias=False),
                        build_norm_layer(norm_cfg, out_ch)[1],
                        build_activation_layer(act_cfg)))
                in_ch = out_ch
            # 不创建conv_layers，避免checkpoint加载时的参数不匹配
            self.conv_layers = None
        else:
            # 构建3D卷积层（仅在需要时使用，显存消耗大）
            self.conv_layers = nn.ModuleList()
            in_ch = in_channels
            for out_ch in feat_channels:
                self.conv_layers.append(
                    ConvModule(
                        in_ch,
                        out_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
                in_ch = out_ch
            # 不创建mlp_layers，避免checkpoint加载时的参数不匹配
            self.mlp_layers = None
        
        # 输出特征压缩层（可选）
        self.output_channels = feat_channels[-1]
    
    def voxelize(self, points: Tensor, coors: Tensor) -> Tuple[Tensor, Tensor]:
        """将点云体素化。
        
        Args:
            points (Tensor): 点云特征 [N, C]，其中C包含xyz和反射率等
            coors (Tensor): 点云坐标 [N, 3]，格式为 [batch_idx, y, x] (frustum坐标)
            
        Returns:
            Tuple[Tensor, Tensor]: 体素特征和体素坐标 [batch_idx, x, y, z]
        """
        device = points.device
        if self.voxel_size.device != device:
            self.voxel_size = self.voxel_size.to(device)
        if self.point_cloud_range.device != device:
            self.point_cloud_range = self.point_cloud_range.to(device)
        
        # 获取点云坐标 (x, y, z)
        xyz = points[:, :3]  # [N, 3]
        
        # 计算体素坐标 (x, y, z)
        voxel_coors_xyz = torch.floor(
            (xyz - self.point_cloud_range[:3]) / self.voxel_size
        ).long()  # [N, 3]
        
        # 限制在有效范围内
        voxel_coors_xyz[:, 0] = torch.clamp(
            voxel_coors_xyz[:, 0], min=0, max=self.voxel_shape[0] - 1)
        voxel_coors_xyz[:, 1] = torch.clamp(
            voxel_coors_xyz[:, 1], min=0, max=self.voxel_shape[1] - 1)
        voxel_coors_xyz[:, 2] = torch.clamp(
            voxel_coors_xyz[:, 2], min=0, max=self.voxel_shape[2] - 1)
        
        # 添加batch索引
        batch_indices = coors[:, 0:1]  # [N, 1]
        voxel_coors_with_batch = torch.cat([batch_indices, voxel_coors_xyz], dim=1)  # [N, 4]
        
        # 使用scatter_mean聚合同一体素内的点
        unique_voxel_coors, inverse_map = torch.unique(
            voxel_coors_with_batch, return_inverse=True, dim=0
        )
        
        # 聚合特征（对每个通道分别聚合）
        voxel_feats = torch.zeros(
            (unique_voxel_coors.shape[0], self.in_channels),
            dtype=points.dtype,
            device=device
        )
        
        for i in range(self.in_channels):
            voxel_feats[:, i] = torch_scatter.scatter_mean(
                points[:, i].float(), inverse_map, dim=0
            ).to(points.dtype)
        
        return voxel_feats, unique_voxel_coors
    
    def forward(self, voxel_dict: dict) -> dict:
        """前向传播。
        
        Args:
            voxel_dict (dict): 包含点云数据的字典
                - 'voxels': 点云特征 [N, C]
                - 'coors': 点云坐标 [N, 3]
                
        Returns:
            dict: 包含体素特征的字典
                - 'voxel_3d_feats': 3D体素特征（稀疏表示或密集网格）
                - 'voxel_3d_coors': 3D体素坐标
        """
        points = voxel_dict['voxels']
        coors = voxel_dict['coors']
        
        # 体素化
        voxel_feats, voxel_coors = self.voxelize(points, coors)
        
        if self.use_sparse:
            # 稀疏模式：直接对体素特征使用MLP，避免创建密集网格
            x = voxel_feats  # [N_voxel, C]
            # for mlp_layer in self.mlp_layers:
            #     x = mlp_layer(x)
            if self.mlp_layers is not None:
                for mlp_layer in self.mlp_layers:
                    x = mlp_layer(x)
            
            # 将稀疏体素特征保存，后续在backbone中映射到range image
            # 这里保存为点级特征，格式与backbone期望的格式兼容
            voxel_dict['voxel_3d_feats'] = x  # [N_voxel, C] 稀疏体素特征
            voxel_dict['voxel_3d_coors'] = voxel_coors  # [N_voxel, 4]
            voxel_dict['voxel_3d_sparse'] = True  # 标记为稀疏模式
        else:
            # 密集模式：创建3D网格并使用3D卷积（显存消耗大）
            batch_size = coors[-1, 0].item() + 1
            voxel_grid = torch.zeros(
                (batch_size, *self.voxel_shape, self.in_channels),
                dtype=points.dtype,
                device=points.device
            )
            
            # 填充体素网格 [batch, x, y, z, channels]
            # voxel_coors格式: [batch_idx, x_idx, y_idx, z_idx]
            voxel_grid[
                voxel_coors[:, 0],  # batch
                voxel_coors[:, 1],  # x
                voxel_coors[:, 2],  # y
                voxel_coors[:, 3],  # z
                :
            ] = voxel_feats
            
            # 转换为 [B, C, X, Y, Z] 格式用于3D卷积
            voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, X, Y, Z]
            
            # 通过3D卷积层提取特征
            x = voxel_grid
            # for conv_layer in self.conv_layers:
            #     x = conv_layer(x)
            if self.conv_layers is not None:
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)            
            voxel_dict['voxel_3d_feats'] = x  # [B, C, X, Y, Z]
            voxel_dict['voxel_3d_sparse'] = False
        
        voxel_dict['voxel_shape'] = self.voxel_shape
        # 记录体素尺寸与点云范围，便于后续 BEV 投影/蒸馏
        voxel_dict['voxel_size'] = self.voxel_size
        voxel_dict['point_cloud_range'] = self.point_cloud_range
        voxel_dict['bev_shape'] = self.voxel_shape[:2]
        
        return voxel_dict

