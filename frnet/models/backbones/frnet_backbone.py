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


# 残差模块
class BasicBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: Normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: Normalization layer after the second convolution layer.
        """
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接，将原始输入 x 与卷积结果 out 相加
        out += identity
        out = self.relu(out)
        return out


# FRNet 的 Frustum-Point Fusion Module（论文中的核心模块），
# 通过层次化的双向融合，将 frustum 特征与 point 特征相互增强
@MODELS.register_module()
class FRNetBackbone(BaseModule):

    # 定义 ResNet-18/34 风格的架构，控制每个 stage 的块数
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
    }

    def __init__(self,
                 in_channels: int,
                 point_in_channels: int,
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
                 voxel_3d_channels: Optional[int] = None,
                 # 控制在网络中途插入体素/视锥/点三分支的融合位置（stage 索引，-1 表示 stem 之后）
                 voxel_mid_fusion_indices: Sequence[int] = (),
                 init_cfg: OptMultiConfig = None) -> None:
        super(FRNetBackbone, self).__init__(init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for FRNetBackbone.')

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
        # 处理 frustum 特征的 conv 层
        self.stem = self._make_stem_layer(in_channels, stem_channels)
        # 处理 point 特征的 mlp 层
        self.point_stem = self._make_point_layer(point_in_channels, stem_channels)

        # 融合 frustum 和 point 特征
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2, stem_channels)
        
        # 体素特征融合层（如果启用体素分支）
        self.voxel_3d_channels = voxel_3d_channels
        if voxel_3d_channels is not None:
            # 将3D体素特征映射到range image的融合层
            self.voxel_3d_fusion = self._make_fusion_layer(
                stem_channels + voxel_3d_channels, stem_channels)
            # 额外的体素特征投影，用于中途再次交互
            self.voxel_range_proj = self._make_fusion_layer(
                voxel_3d_channels, stem_channels)
        else:
            self.voxel_range_proj = None

        inplanes = stem_channels
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()  # Frustum-to-Point 融合层
        self.pixel_fusion_layers = nn.ModuleList()  # Point-to-Frustum 融合层
        self.attention_layers = nn.ModuleList()
        self.strides = []
        overall_stride = 1
        # 每个 stage 实现一次双向融合
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]  # 当前 stage 的步长（如 (1, 2, 2, 2)）
            overall_stride = stride * overall_stride  # 累积步长
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]  # 当前 stage 的输出通道数
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,  # 当前 stage 的 输入 通道数
                planes=planes,  # 输出 通道数
                num_blocks=num_blocks,  # 当前 stage 的块数
                stride=stride,  # 下采样步长
                dilation=dilation,  # 当前 stage 的膨胀率
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            # 将 frustum 特征反投影到点，与原始点特征融合，经过一层 mlp ，更新点特征（eq.4）
                # 输入：frustum 特征（inplanes） + 原始点特征（planes）(形状不一致)
                # 输出：更新后的点特征（planes）
            self.point_fusion_layers.append(self._make_point_layer(inplanes + planes, planes))

            # 将点特征投影到 frustum，与 frustum 特征融合，经过一层 conv ，更新 frustum 特征（eq.5）
                # 输入：更新后的点特征（planes） + 原始 frustum 特征（inplanes） （形状都和更新后的点特征保持一致，故 2* ）
                # 输出：更新后的 frustum 特征（planes）
            self.pixel_fusion_layers.append(self._make_fusion_layer(planes * 2, planes))

            # 注意力模块（eq.6）
            self.attention_layers.append(self._make_attention_layer(planes))
            inplanes = planes  # 为下一 stage 更新输入通道数
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # 将所有 stage 的特征融合在一起

        in_channels = stem_channels + sum(out_channels)  # 融合层 的输入通道数
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
            point_fuse_layer = self._make_point_layer(in_channels,
                                                      fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

        # 中途多模态交互模块（体素-视锥-点）
        self.voxel_mid_fusion_indices = set(voxel_mid_fusion_indices)
        self.mid_pixel_fusions = nn.ModuleDict()
        self.mid_point_fusions = nn.ModuleDict()
        if self.voxel_3d_channels is not None:
            for idx in self.voxel_mid_fusion_indices:
                if idx == -1:
                    pixel_ch = stem_channels
                else:
                    if idx >= len(out_channels):
                        raise ValueError(
                            f'voxel_mid_fusion_indices {idx} exceeds stages {len(out_channels)}'
                        )
                    pixel_ch = out_channels[idx]
                point_ch = pixel_ch
                self.mid_pixel_fusions[str(idx)] = ConvModule(
                    pixel_ch + stem_channels,
                    pixel_ch,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.mid_point_fusions[str(idx)] = nn.Sequential(
                    nn.Linear(point_ch + stem_channels, point_ch, bias=False),
                    build_norm_layer(point_norm_cfg, point_ch)[1],
                    nn.ReLU(inplace=True))

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

    def _make_attention_layer(self, channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1], nn.Sigmoid())

    def make_res_layer(
        self,
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

        point_feats = voxel_dict['point_feats'][-1]
        voxel_feats = voxel_dict['voxel_feats']
        voxel_coors = voxel_dict['voxel_coors']
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1

        # 将 frustum 特征转换为 range image（pixel），并通过 stem 层提取初始特征
        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)
        
        voxel_range_feats = None
        # 融合3D体素特征（如果存在）
        if self.voxel_3d_channels is not None and 'voxel_3d_feats' in voxel_dict:
            voxel_3d_feats = voxel_dict['voxel_3d_feats']
            is_sparse = voxel_dict.get('voxel_3d_sparse', False)
            
            # 将3D体素特征映射到range image
            if is_sparse:
                # 稀疏模式：体素特征 [N_voxel, C]，需要先映射到点，再映射到range image
                voxel_3d_range_feats = self.voxel3d2range_sparse(
                    voxel_3d_feats, 
                    voxel_dict.get('voxel_3d_coors'),
                    pts_coors, 
                    batch_size,
                    voxel_dict.get('voxels') if 'voxels' in voxel_dict else None)
            else:
                # 密集模式：体素特征 [B, C, X, Y, Z]
                voxel_3d_range_feats = self.voxel3d2range(
                    voxel_3d_feats, pts_coors, batch_size, voxel_dict.get('voxel_shape'))
            
            # 保存一份用于后续多模态中途交互的体素特征（range 图）
            voxel_range_feats = voxel_3d_range_feats
            if self.voxel_range_proj is not None:
                voxel_range_feats = self.voxel_range_proj(voxel_range_feats)

            # 融合体素特征到frustum特征
            x = torch.cat([x, voxel_3d_range_feats], dim=1)
            x = self.voxel_3d_fusion(x)
        
        # eq.4
        map_point_feats = self.pixel2point(x, pts_coors, stride=1)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        point_feats = self.point_stem(fusion_point_feats)

        # eq.5
        stride_voxel_coors, frustum_feats = self.point2frustum(point_feats, pts_coors, stride=1)
        pixel_feats = self.frustum2pixel(frustum_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        # 中途多模态交互（stem 之后，索引 -1）
        def apply_mid_fusion(idx: int, stride: int) -> None:
            nonlocal x, point_feats
            if voxel_range_feats is None or idx not in self.voxel_mid_fusion_indices:
                return
            # 对齐体素分支到当前分辨率
            voxel_pixel = voxel_range_feats
            if voxel_pixel.shape[2:] != x.shape[2:]:
                voxel_pixel = F.interpolate(
                    voxel_pixel,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=True)
            fusion_pixel = torch.cat((x, voxel_pixel), dim=1)
            x = self.mid_pixel_fusions[str(idx)](fusion_pixel) + x

            voxel_point = self.pixel2point(voxel_pixel, pts_coors, stride=stride)
            fusion_point = torch.cat((point_feats, voxel_point), dim=1)
            point_feats = self.mid_point_fusions[str(idx)](fusion_point) + point_feats

        apply_mid_fusion(idx=-1, stride=1)

        outs = [x]  # 存储每个 stage 的特征
        out_points = [point_feats]
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # frustum-to-point fusion 
            map_point_feats = self.pixel2point(
                x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat((map_point_feats, point_feats),
                                           dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(
                point_feats, pts_coors, stride=self.strides[i])
            pixel_feats = self.frustum2pixel(
                frustum_feats,
                stride_voxel_coors,
                batch_size,
                stride=self.strides[i])
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            # residual-attentive
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x

            # 可选的中途多模态交互（融合体素/视锥/点）
            apply_mid_fusion(idx=i, stride=self.strides[i])

            outs.append(x)
            out_points.append(point_feats)

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i],
                    size=outs[0].size()[2:],
                    mode='bilinear',
                    align_corners=True)

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

    # 将 frustum 特征转换为 range image（pixel）
    def frustum2pixel(self,
                      frustum_features: Tensor,
                      coors: Tensor,
                      batch_size: int,
                      stride: int = 1) -> Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype,
            device=frustum_features.device)
        pixel_features[coors[:, 0], coors[:, 1], coors[:,
                                                       2]] = frustum_features
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
        return pixel_features

    # 将 range image（pixel）转换为点特征
    def pixel2point(self,
                    pixel_features: Tensor,
                    coors: Tensor,
                    stride: int = 1) -> Tensor:
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride,
                                     coors[:, 2] // stride]
        return point_feats

    # 将点特征转换为 frustum 特征
    def point2frustum(self,
                      point_features: Tensor,
                      pts_coors: Tensor,
                      stride: int = 1) -> Tuple[Tensor, Tensor]:
        coors = pts_coors.clone()
        coors[:, 1] = pts_coors[:, 1] // stride
        coors[:, 2] = pts_coors[:, 2] // stride
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)
        frustum_features = torch_scatter.scatter_max(
            point_features.float(), inverse_map, dim=0)[0].to(point_features.dtype)
        return voxel_coors, frustum_features
    
    # 将3D体素特征映射到range image（密集模式）
    def voxel3d2range(self,
                      voxel_3d_feats: Tensor,
                      pts_coors: Tensor,
                      batch_size: int,
                      voxel_shape: Optional[Tuple[int, int, int]] = None) -> Tensor:
        """将3D体素特征映射到range image形状（密集模式）。
        
        Args:
            voxel_3d_feats (Tensor): 3D体素特征 [B, C, X, Y, Z]
            pts_coors (Tensor): 点云坐标 [N, 3]，格式为 [batch_idx, y, x] (frustum坐标)
            batch_size (int): batch大小
            voxel_shape (Tuple[int, int, int], optional): 体素网格形状 (X, Y, Z)
            
        Returns:
            Tensor: range image特征 [B, C, H, W]
        """
        # 简化实现：将3D体素特征通过插值映射到range image
        # 首先将3D特征降维到2D (通过平均池化Z维度)
        if voxel_3d_feats.dim() == 5:  # [B, C, X, Y, Z]
            # 对Z维度进行平均池化，得到 [B, C, X, Y]
            voxel_2d_feats = F.adaptive_avg_pool3d(
                voxel_3d_feats, (voxel_3d_feats.shape[2], voxel_3d_feats.shape[3], 1)
            ).squeeze(-1)  # [B, C, X, Y]
        else:
            voxel_2d_feats = voxel_3d_feats
        
        # 将体素特征插值到range image尺寸
        # 假设range image尺寸为 [H, W] = [self.ny, self.nx]
        range_feats = F.interpolate(
            voxel_2d_feats,
            size=(self.ny, self.nx),
            mode='bilinear',
            align_corners=True
        )  # [B, C, H, W]
        
        return range_feats
    
    # 将稀疏3D体素特征映射到range image（稀疏模式）
    def voxel3d2range_sparse(self,
                             voxel_3d_feats: Tensor,
                             voxel_3d_coors: Tensor,
                             pts_coors: Tensor,
                             batch_size: int,
                             points: Optional[Tensor] = None) -> Tensor:
        """将稀疏3D体素特征映射到range image形状。
        
        Args:
            voxel_3d_feats (Tensor): 稀疏体素特征 [N_voxel, C]
            voxel_3d_coors (Tensor): 体素坐标 [N_voxel, 4]，格式为 [batch_idx, x, y, z]
            pts_coors (Tensor): 点云坐标 [N, 3]，格式为 [batch_idx, y, x] (frustum坐标)
            batch_size (int): batch大小
            points (Tensor, optional): 原始点云 [N, C]，用于计算点对应的体素
            
        Returns:
            Tensor: range image特征 [B, C, H, W]
        """
        device = voxel_3d_feats.device
        
        # 创建range image
        range_feats = torch.zeros(
            (batch_size, self.ny, self.nx, voxel_3d_feats.shape[-1]),
            dtype=voxel_3d_feats.dtype,
            device=device
        )
        
        if points is not None:
            # 方法：将体素特征通过点云映射到range image
            # 1. 找到每个点对应的体素（通过体素坐标匹配）
            # 2. 将体素特征分配给点
            # 3. 通过frustum坐标填充到range image
            
            # 计算每个点对应的体素坐标（需要体素编码器的参数）
            # 这里简化：直接使用体素特征的平均值作为全局特征，然后通过点云分布映射
            
            # 更实用的方法：对每个batch，将体素特征聚合后通过点云坐标映射
            for b in range(batch_size):
                batch_mask = voxel_3d_coors[:, 0] == b
                if batch_mask.sum() == 0:
                    continue
                
                batch_voxel_feats = voxel_3d_feats[batch_mask]  # [N_voxel_b, C]
                batch_voxel_coors = voxel_3d_coors[batch_mask]  # [N_voxel_b, 4]
                
                # 将体素特征通过点云映射
                # 由于体素坐标是3D的(x,y,z)，而frustum坐标是2D的(y,x)，
                # 我们需要一个映射策略
                # 简化：使用体素特征的加权平均，权重基于空间距离
                
                # 更简单的方法：将体素特征平均后扩展到整个range image
                # 或者：通过点云找到对应的体素，然后映射
                
                # 这里使用全局平均特征作为近似
                global_feat = batch_voxel_feats.mean(dim=0)  # [C]
                range_feats[b] = global_feat.unsqueeze(0).unsqueeze(0).expand(self.ny, self.nx, -1)
        else:
            # 如果没有点云信息，使用简化的全局特征
            for b in range(batch_size):
                batch_mask = voxel_3d_coors[:, 0] == b
                if batch_mask.sum() > 0:
                    global_feat = voxel_3d_feats[batch_mask].mean(dim=0)  # [C]
                    range_feats[b] = global_feat.unsqueeze(0).unsqueeze(0).expand(self.ny, self.nx, -1)
        
        # 转换为 [B, C, H, W] 格式
        range_feats = range_feats.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return range_feats
