from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch_scatter
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from torch import Tensor


@MODELS.register_module()
class MultiScaleVoxelFeatureEncoder(nn.Module):
    """多尺度体素特征编码器。
    
    通过并行处理多个不同分辨率的体素网格，实现对点云的多尺度特征提取。
    支持三种不同分辨率的体素化，然后融合多尺度特征。
    
    Args:
        in_channels (int): 输入特征通道数。默认为4。
        feat_channels (Sequence[int]): 每个3D卷积层的特征通道数。
            默认为(64, 128, 256)。
        voxel_sizes (Sequence[Sequence[float]]): 多个体素尺寸，每个为[x, y, z]。
            默认为[(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4)]。
        point_cloud_range (Sequence[float]): 点云范围
            [x_min, y_min, z_min, x_max, y_max, z_max]。
            默认为(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0)。
        norm_cfg (dict or :obj:`ConfigDict`): 归一化层配置。
            默认为dict(type='BN', eps=1e-5, momentum=0.1)。
        act_cfg (dict or :obj:`ConfigDict`): 激活层配置。
            默认为dict(type='ReLU', inplace=True)。
        fusion_method (str): 多尺度特征融合方法。可选'concat'或'attention'。
            默认为'concat'。
        use_sparse (bool): 是否使用稀疏模式。默认为True。
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = (64, 128, 256),
                 voxel_sizes: Optional[Sequence[Sequence[float]]] = None,
                 voxel_size: Optional[Sequence[float]] = None,  # 兼容单尺度参数
                 point_cloud_range: Sequence[float] = (-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
                 norm_cfg: ConfigType = dict(type='BN', eps=1e-5, momentum=0.1),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 fusion_method: str = 'concat',
                 use_sparse: bool = True) -> None:
        super(MultiScaleVoxelFeatureEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        
        # 处理voxel_sizes参数：如果提供了voxel_size（单尺度），自动转换为多尺度
        if voxel_sizes is None:
            if voxel_size is not None:
                # 从单尺度参数自动生成多尺度：使用0.5x, 1x, 2x的倍数
                base_size = tuple(voxel_size)
                voxel_sizes = (
                    tuple(s * 0.5 for s in base_size),  # 0.5倍（更细粒度）
                    base_size,                           # 1倍（原始尺寸）
                    tuple(s * 2.0 for s in base_size)   # 2倍（更粗粒度）
                )
            else:
                # 使用默认值
                voxel_sizes = ((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4))
        
        self.voxel_sizes = [torch.tensor(vs, dtype=torch.float32) for vs in voxel_sizes]
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.num_scales = len(voxel_sizes)
        self.fusion_method = fusion_method
        self.use_sparse = use_sparse
        
        # 计算每个尺度的体素网格尺寸
        self.voxel_shapes = []
        for voxel_size in self.voxel_sizes:
            voxel_shape = (
                int((self.point_cloud_range[3] - self.point_cloud_range[0]) / voxel_size[0]),
                int((self.point_cloud_range[4] - self.point_cloud_range[1]) / voxel_size[1]),
                int((self.point_cloud_range[5] - self.point_cloud_range[2]) / voxel_size[2])
            )
            self.voxel_shapes.append(voxel_shape)
        
        # 为每个尺度创建独立的特征提取MLP层
        self.scale_mlps = nn.ModuleList()
        for scale_idx in range(self.num_scales):
            scale_mlp = nn.ModuleList()
            in_ch = in_channels
            for out_ch in feat_channels:
                scale_mlp.append(
                    nn.Sequential(
                        nn.Linear(in_ch, out_ch, bias=False),
                        build_norm_layer(norm_cfg, out_ch)[1],
                        build_activation_layer(act_cfg)))
                in_ch = out_ch
            self.scale_mlps.append(scale_mlp)
        
        # 多尺度特征融合层
        if fusion_method == 'concat':
            # 简单拼接后通过MLP融合
            fusion_in_channels = feat_channels[-1] * self.num_scales
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_in_channels, feat_channels[-1], bias=False),
                build_norm_layer(norm_cfg, feat_channels[-1])[1],
                build_activation_layer(act_cfg))
        elif fusion_method == 'attention':
            # 使用注意力机制融合多尺度特征
            self.fusion_mlp = None
            self.scale_attention = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_channels[-1] // 4),
                build_activation_layer(act_cfg),
                nn.Linear(feat_channels[-1] // 4, self.num_scales),
                nn.Softmax(dim=-1))
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")
        
        # 输出特征通道数
        self.output_channels = feat_channels[-1]
    
    def voxelize(self, points: Tensor, coors: Tensor, voxel_size: Tensor, voxel_shape: Tuple[int, int, int]) -> Tuple[Tensor, Tensor]:
        """将点云体素化。
        
        Args:
            points (Tensor): 点云特征 [N, C]，其中C包含xyz和反射率等
            coors (Tensor): 点云坐标 [N, 3]，格式为 [batch_idx, y, x] (frustum坐标)
            voxel_size (Tensor): 体素尺寸 [3]
            voxel_shape (Tuple[int, int, int]): 体素网格形状 (X, Y, Z)
            
        Returns:
            Tuple[Tensor, Tensor]: 体素特征和体素坐标 [batch_idx, x, y, z]
        """
        device = points.device
        if voxel_size.device != device:
            voxel_size = voxel_size.to(device)
        if self.point_cloud_range.device != device:
            self.point_cloud_range = self.point_cloud_range.to(device)
        
        # 获取点云坐标 (x, y, z)
        xyz = points[:, :3]  # [N, 3]
        
        # 计算体素坐标 (x, y, z)
        voxel_coors_xyz = torch.floor(
            (xyz - self.point_cloud_range[:3]) / voxel_size
        ).long()  # [N, 3]
        
        # 限制在有效范围内
        voxel_coors_xyz[:, 0] = torch.clamp(
            voxel_coors_xyz[:, 0], min=0, max=voxel_shape[0] - 1)
        voxel_coors_xyz[:, 1] = torch.clamp(
            voxel_coors_xyz[:, 1], min=0, max=voxel_shape[1] - 1)
        voxel_coors_xyz[:, 2] = torch.clamp(
            voxel_coors_xyz[:, 2], min=0, max=voxel_shape[2] - 1)
        
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
            dict: 包含多尺度体素特征的字典
                - 'voxel_3d_feats': 融合后的3D体素特征 [N_voxel, C] (稀疏模式)
                - 'voxel_3d_coors': 3D体素坐标 [N_voxel, 4] (使用中等分辨率的坐标)
                - 'voxel_3d_sparse': True (标记为稀疏模式)
        """
        points = voxel_dict['voxels']
        coors = voxel_dict['coors']
        
        # 对每个尺度并行进行体素化和特征提取
        scale_features = []
        scale_coors = []
        
        for scale_idx in range(self.num_scales):
            # 体素化
            voxel_feats, voxel_coors = self.voxelize(
                points, coors, 
                self.voxel_sizes[scale_idx], 
                self.voxel_shapes[scale_idx]
            )
            
            # 通过MLP提取特征
            x = voxel_feats  # [N_voxel_k, C]
            for mlp_layer in self.scale_mlps[scale_idx]:
                x = mlp_layer(x)
            
            scale_features.append(x)  # [N_voxel_k, feat_channels[-1]]
            scale_coors.append(voxel_coors)  # [N_voxel_k, 4]
        
        # 多尺度特征融合
        # 使用中等分辨率（中间尺度）的体素坐标作为参考
        ref_scale_idx = self.num_scales // 2  # 使用中间尺度作为参考
        ref_coors = scale_coors[ref_scale_idx]  # [N_ref, 4]
        
        if self.fusion_method == 'concat':
            # 方法1: 拼接融合
            # 将其他尺度的特征对齐到参考尺度的体素坐标
            aligned_features = self._align_scale_features(scale_features, scale_coors, ref_coors)
            
            # 拼接所有尺度的特征
            fused_features = torch.cat(aligned_features, dim=-1)  # [N_ref, C * num_scales]
            
            # 通过融合MLP进一步处理
            fused_features = self.fusion_mlp(fused_features)  # [N_ref, C]
                    
        elif self.fusion_method == 'attention':
            # 方法2: 注意力融合
            # 对齐所有尺度的特征到参考尺度
            aligned_features = self._align_scale_features(scale_features, scale_coors, ref_coors)
            
            # 计算注意力权重
            stacked_features = torch.stack(aligned_features, dim=0)  # [num_scales, N_ref, C]
            # 对每个体素计算多尺度特征的注意力权重
            # 使用平均池化后的特征计算注意力
            mean_features = stacked_features.mean(dim=0)  # [N_ref, C]
            attention_weights = self.scale_attention(mean_features)  # [N_ref, num_scales]
            attention_weights = attention_weights.unsqueeze(-1)  # [N_ref, num_scales, 1]
            
            # 加权融合
            stacked_features = stacked_features.permute(1, 0, 2)  # [N_ref, num_scales, C]
            fused_features = (stacked_features * attention_weights).sum(dim=1)  # [N_ref, C]
        else:
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")
        
        # 保存结果
        voxel_dict['voxel_3d_feats'] = fused_features  # [N_ref, C]
        voxel_dict['voxel_3d_coors'] = ref_coors  # [N_ref, 4]
        voxel_dict['voxel_3d_sparse'] = True  # 标记为稀疏模式
        voxel_dict['voxel_shape'] = self.voxel_shapes[ref_scale_idx]
        
        return voxel_dict
    
    def _align_scale_features(self, scale_features: list, scale_coors: list, ref_coors: Tensor) -> list:
        """将不同尺度的特征对齐到参考坐标（向量化实现，高效版本）。
        
        通过体素坐标的对应关系，将各个尺度的特征映射到参考尺度的体素坐标上。
        使用向量化操作替代Python循环，大幅提升性能。
        
        Args:
            scale_features: 每个尺度的特征列表，每个为 [N_k, C]
            scale_coors: 每个尺度的坐标列表，每个为 [N_k, 4]
            ref_coors: 参考坐标 [N_ref, 4]
            
        Returns:
            list: 对齐后的特征列表，每个为 [N_ref, C]
        """
        device = ref_coors.device
        aligned_features = []
        
        # 获取参考尺度的体素尺寸（用于坐标转换）
        ref_scale_idx = self.num_scales // 2
        ref_voxel_size = self.voxel_sizes[ref_scale_idx].to(device)
        point_cloud_range_tensor = self.point_cloud_range.to(device)
        
        for scale_idx, (feat, coors) in enumerate(zip(scale_features, scale_coors)):
            if scale_idx == ref_scale_idx:
                # 参考尺度直接使用
                aligned_features.append(feat)
            else:
                # 向量化实现：将参考体素坐标转换为当前尺度的体素坐标
                current_voxel_size = self.voxel_sizes[scale_idx].to(device)
                
                # 将参考体素的中心坐标转换为3D空间坐标（向量化）
                ref_coors_float = ref_coors.float()  # [N_ref, 4]
                ref_centers = (ref_coors_float[:, 1:4] + 0.5) * ref_voxel_size.unsqueeze(0) + point_cloud_range_tensor[:3].unsqueeze(0)  # [N_ref, 3]
                
                # 计算在当前尺度下的体素坐标（向量化）
                curr_coors_float = (ref_centers - point_cloud_range_tensor[:3].unsqueeze(0)) / current_voxel_size.unsqueeze(0)  # [N_ref, 3]
                curr_coors = curr_coors_float.long()  # [N_ref, 3]
                
                # 限制在有效范围内（向量化）
                curr_coors[:, 0] = torch.clamp(curr_coors[:, 0], min=0, max=self.voxel_shapes[scale_idx][0] - 1)
                curr_coors[:, 1] = torch.clamp(curr_coors[:, 1], min=0, max=self.voxel_shapes[scale_idx][1] - 1)
                curr_coors[:, 2] = torch.clamp(curr_coors[:, 2], min=0, max=self.voxel_shapes[scale_idx][2] - 1)
                
                # 构建当前尺度的坐标到特征的映射（使用tensor操作）
                batch_indices = ref_coors[:, 0:1].long()  # [N_ref, 1]
                curr_coors_with_batch = torch.cat([batch_indices, curr_coors], dim=1)  # [N_ref, 4]
                
                # 使用向量化的坐标匹配方法
                aligned_feat = torch.zeros(
                    (ref_coors.shape[0], feat.shape[1]),
                    dtype=feat.dtype,
                    device=device
                )
                
                # 构建coors的哈希键（使用tensor操作，避免Python循环）
                # 使用更大的乘数避免冲突
                coors_keys = (coors[:, 0].long() * 1000000000 + 
                             coors[:, 1].long() * 1000000 + 
                             coors[:, 2].long() * 1000 + 
                             coors[:, 3].long())  # [N_k]
                curr_keys = (curr_coors_with_batch[:, 0].long() * 1000000000 + 
                            curr_coors_with_batch[:, 1].long() * 1000000 + 
                            curr_coors_with_batch[:, 2].long() * 1000 + 
                            curr_coors_with_batch[:, 3].long())  # [N_ref]
                
                # 使用searchsorted进行快速查找（需要排序）
                sorted_indices = torch.argsort(coors_keys)
                sorted_keys = coors_keys[sorted_indices]
                
                # 查找每个curr_key在sorted_keys中的位置
                search_indices = torch.searchsorted(sorted_keys, curr_keys, right=False)  # [N_ref]
                
                # 检查是否找到精确匹配
                clamped_indices = torch.clamp(search_indices, max=len(sorted_keys) - 1)
                valid_mask = (search_indices < len(sorted_keys)) & (sorted_keys[clamped_indices] == curr_keys)
                
                # 获取匹配的索引并赋值
                matched_indices = sorted_indices[clamped_indices]  # [N_ref]
                aligned_feat[valid_mask] = feat[matched_indices[valid_mask]]
                
                aligned_features.append(aligned_feat)
        
        return aligned_features

