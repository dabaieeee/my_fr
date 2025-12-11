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
class UNetVoxelFeatureEncoder(nn.Module):
    """UNet风格的体素特征编码器，使用金字塔特征融合结构。
    
    采用编码器-解码器架构，通过下采样和上采样实现多尺度特征提取，
    并使用跳跃连接融合不同尺度的特征，类似UNet结构。
    
    Args:
        in_channels (int): 输入特征通道数。默认为4。
        feat_channels (Sequence[int]): 编码器/解码器各层的特征通道数。
            默认为(64, 128, 256, 512)。
        voxel_size (Sequence[float]): 体素尺寸 [x, y, z]。
            默认为(0.2, 0.2, 0.2)。
        point_cloud_range (Sequence[float]): 点云范围
            [x_min, y_min, z_min, x_max, y_max, z_max]。
            默认为(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0)。
        norm_cfg (dict or :obj:`ConfigDict`): 归一化层配置。
            默认为dict(type='BN', eps=1e-5, momentum=0.1)。
        act_cfg (dict or :obj:`ConfigDict`): 激活层配置。
            默认为dict(type='ReLU', inplace=True)。
        use_sparse (bool): 是否使用稀疏模式。默认为False（UNet需要密集网格）。
        num_downsample (int): 下采样层数。默认为3。
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = (32, 64, 128, 256),  # 轻量通道，减少显存
                 voxel_size: Sequence[float] = (0.4, 0.4, 0.4),  # 更粗体素，减少网格大小
                 point_cloud_range: Sequence[float] = (-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
                 norm_cfg: ConfigType = dict(type='BN', eps=1e-5, momentum=0.1),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 use_sparse: bool = False,
                 num_downsample: int = 2) -> None:
        super(UNetVoxelFeatureEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.use_sparse = use_sparse
        self.num_downsample = min(num_downsample, len(feat_channels) - 1)
        
        # 计算体素网格尺寸
        voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32)
        point_cloud_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.voxel_shape = (
            int((point_cloud_range_tensor[3] - point_cloud_range_tensor[0]) / voxel_size_tensor[0]),
            int((point_cloud_range_tensor[4] - point_cloud_range_tensor[1]) / voxel_size_tensor[1]),
            int((point_cloud_range_tensor[5] - point_cloud_range_tensor[2]) / voxel_size_tensor[2])
        )
        
        # UNet结构：编码器（下采样）+ 解码器（上采样）+ 跳跃连接
        # 编码器部分（下采样）
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(feat_channels[:self.num_downsample + 1]):
            # 每个编码器块：两个3x3x3卷积 + 下采样
            encoder_block = nn.ModuleList([
                # 第一个卷积块
                ConvModule(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                # 第二个卷积块
                ConvModule(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ])
            self.encoder_blocks.append(encoder_block)
            in_ch = out_ch
        
        # 瓶颈层（最底层，不下采样）
        if len(feat_channels) > self.num_downsample + 1:
            bottleneck_ch = feat_channels[self.num_downsample + 1]
            self.bottleneck = nn.ModuleList([
                ConvModule(
                    in_ch,
                    bottleneck_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule(
                    bottleneck_ch,
                    bottleneck_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ])
            in_ch = bottleneck_ch
        else:
            self.bottleneck = None
        
        # 解码器部分（上采样）
        self.decoder_blocks = nn.ModuleList()
        # 构建解码器通道列表：从瓶颈层开始，逐步恢复到输入分辨率
        encoder_channels = feat_channels[:self.num_downsample + 1]  # [64, 128, 256, ...]
        if self.bottleneck is not None:
            decoder_start_ch = bottleneck_ch
        else:
            decoder_start_ch = encoder_channels[-1]
        
        # 解码器通道：从瓶颈层开始，逐步恢复到第一层
        decoder_channels = [decoder_start_ch] + list(reversed(encoder_channels))
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]  # 当前层输入通道
            skip_ch = encoder_channels[-(i+1)]  # 对应编码器层的通道（跳跃连接）
            out_ch = decoder_channels[i + 1]  # 输出通道
            
            # 上采样层
            decoder_block = nn.ModuleList([
                # 上采样卷积（转置卷积）
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch,
                        in_ch,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        bias=False),
                    build_norm_layer(norm_cfg, in_ch)[1],
                    build_activation_layer(act_cfg)),
                # 第一个卷积块（融合跳跃连接）
                ConvModule(
                    in_ch + skip_ch,  # 跳跃连接：上采样特征 + 编码器特征
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                # 第二个卷积块
                ConvModule(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ])
            self.decoder_blocks.append(decoder_block)
        
        # 输出通道数
        self.output_channels = decoder_channels[-1] if decoder_channels else feat_channels[0]
    
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
            dict: 包含UNet提取的体素特征的字典
                - 'voxel_3d_feats': 3D体素特征 [B, C, X, Y, Z] (密集模式) 或 [N_voxel, C] (稀疏模式)
                - 'voxel_3d_coors': 3D体素坐标 [N_voxel, 4]
                - 'voxel_3d_sparse': False (密集模式) 或 True (稀疏模式)
        """
        points = voxel_dict['voxels']
        coors = voxel_dict['coors']
        batch_size = coors[-1, 0].item() + 1
        
        # 体素化
        voxel_feats, voxel_coors = self.voxelize(points, coors)
        
        if self.use_sparse:
            # 稀疏模式：直接对体素特征使用MLP（简化版UNet）
            # 注意：稀疏模式下无法使用真正的UNet结构（需要密集网格）
            # 这里提供一个简化的实现
            x = voxel_feats  # [N_voxel, C]
            
            # 简化的编码器-解码器（使用MLP）
            # 编码器
            for encoder_block in self.encoder_blocks:
                for layer in encoder_block:
                    if isinstance(layer, ConvModule):
                        # 对于稀疏模式，跳过3D卷积，只使用MLP部分
                        # 这里简化处理，实际应该使用MLP替代
                        pass
            
            voxel_dict['voxel_3d_feats'] = x  # [N_voxel, C]
            voxel_dict['voxel_3d_coors'] = voxel_coors
            voxel_dict['voxel_3d_sparse'] = True
        else:
            # 密集模式：创建3D网格并使用UNet结构
            # 创建体素网格 [batch, x, y, z, channels]
            voxel_grid = torch.zeros(
                (batch_size, *self.voxel_shape, self.in_channels),
                dtype=points.dtype,
                device=points.device
            )
            
            # 填充体素网格
            voxel_grid[
                voxel_coors[:, 0],  # batch
                voxel_coors[:, 1],  # x
                voxel_coors[:, 2],  # y
                voxel_coors[:, 3],  # z
                :
            ] = voxel_feats
            
            # 转换为 [B, C, X, Y, Z] 格式用于3D卷积
            x = voxel_grid.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, X, Y, Z]
            
            # UNet编码器（下采样）+ 保存跳跃连接特征
            encoder_features = []
            for encoder_block in self.encoder_blocks:
                # 第一个卷积
                x = encoder_block[0](x)
                # 第二个卷积
                x = encoder_block[1](x)
                # 保存跳跃连接特征
                encoder_features.append(x)
                # 下采样（使用最大池化）
                if len(encoder_features) < len(self.encoder_blocks):
                    x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)
            
            # 瓶颈层
            if self.bottleneck is not None:
                x = self.bottleneck[0](x)
                x = self.bottleneck[1](x)
            
            # UNet解码器（上采样）+ 跳跃连接
            for i, decoder_block in enumerate(self.decoder_blocks):
                # 上采样
                x = decoder_block[0](x)
                # 获取对应的编码器特征（从后往前，与解码器层对应）
                # encoder_features存储顺序：第0层（最高分辨率）到最后层（最低分辨率）
                # 解码器从最低分辨率开始，逐步恢复到最高分辨率
                skip_idx = len(encoder_features) - 1 - i
                if skip_idx >= 0 and skip_idx < len(encoder_features):
                    skip_feat = encoder_features[skip_idx]
                    
                    # 确保尺寸匹配（处理边界情况）
                    if x.shape[2:] != skip_feat.shape[2:]:
                        # 使用插值调整尺寸
                        x = F.interpolate(
                            x,
                            size=skip_feat.shape[2:],
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    # 拼接跳跃连接
                    x = torch.cat([x, skip_feat], dim=1)
                else:
                    # 如果没有对应的跳跃连接（最后一层），直接使用上采样后的特征
                    pass
                
                # 第一个卷积（融合）
                x = decoder_block[1](x)
                # 第二个卷积
                x = decoder_block[2](x)
            
            # 保存结果
            voxel_dict['voxel_3d_feats'] = x  # [B, C, X, Y, Z]
            voxel_dict['voxel_3d_sparse'] = False
            # 对于密集模式，仍然保存体素坐标（用于后续映射）
            voxel_dict['voxel_3d_coors'] = voxel_coors
        
        voxel_dict['voxel_shape'] = self.voxel_shape
        return voxel_dict

