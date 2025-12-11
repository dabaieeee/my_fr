from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType
from torch import Tensor


def _hash_keys(keys: Tensor) -> Tensor:
    """将 (b, x, y) 整数坐标映射为唯一 hash，便于对齐."""
    return (keys[:, 0].long() * 10_000_000_000
            + keys[:, 1].long() * 10_000
            + keys[:, 2].long())


def _align_by_keys(src_feat: Tensor, src_keys: Tensor,
                   tgt_feat: Tensor, tgt_keys: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
    """根据 hash key 对齐两个视角的 BEV 单元，返回匹配的特征对."""
    if src_keys.numel() == 0 or tgt_keys.numel() == 0:
        return None

    src_hash = _hash_keys(src_keys)
    tgt_hash = _hash_keys(tgt_keys)

    # 在目标中查找源哈希
    sorted_tgt_idx = torch.argsort(tgt_hash)
    sorted_tgt_hash = tgt_hash[sorted_tgt_idx]
    pos = torch.searchsorted(sorted_tgt_hash, src_hash)
    pos_clamped = torch.clamp(pos, max=sorted_tgt_hash.shape[0] - 1)
    match_mask = (pos < sorted_tgt_hash.shape[0]) & (sorted_tgt_hash[pos_clamped] == src_hash)
    if match_mask.sum() == 0:
        return None

    matched_src = src_feat[match_mask]
    matched_tgt = tgt_feat[sorted_tgt_idx[pos_clamped[match_mask]]]
    return matched_src, matched_tgt


@MODELS.register_module()
class BEVDistillHead(Base3DDecodeHead):
    """Frustum ↔ BEV ↔ Voxel 三视角互蒸馏的轻量辅助头。

    仅在训练阶段工作，推理可裁剪（不影响主干、分割头）。
    """

    def __init__(self,
                 point_channels: int,
                 frustum_channels: int,
                 voxel_channels: int,
                 bev_channels: int = 96,
                 loss_l1_weight: float = 0.5,
                 loss_frustum_weight: float = 0.5,
                 loss_nce_weight: float = 0.1,
                 temperature: float = 0.2,
                 with_frustum_view: bool = True,
                 detach_teacher: bool = True,
                 **kwargs) -> None:
        # Base3DDecodeHead 需要 num_classes/ignore_index 等参数，沿用配置默认值
        super().__init__(**kwargs)

        self.loss_l1_weight = loss_l1_weight
        self.loss_frustum_weight = loss_frustum_weight
        self.loss_nce_weight = loss_nce_weight
        self.temperature = temperature
        self.with_frustum_view = with_frustum_view
        self.detach_teacher = detach_teacher

        self.point_proj = nn.Sequential(
            nn.Linear(point_channels, bev_channels, bias=False),
            nn.LayerNorm(bev_channels),
            nn.ReLU(inplace=True))
        self.frustum_proj = nn.Sequential(
            nn.Linear(frustum_channels, bev_channels, bias=False),
            nn.LayerNorm(bev_channels),
            nn.ReLU(inplace=True))
        self.voxel_proj = nn.Sequential(
            nn.Linear(voxel_channels, bev_channels, bias=False),
            nn.LayerNorm(bev_channels),
            nn.ReLU(inplace=True))

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        # 蒸馏头不输出分割 logits
        return nn.Identity()

    def forward(self, voxel_dict: dict) -> dict:
        bev_views = self._build_bev_views(voxel_dict)
        voxel_dict['bev_distill_views'] = bev_views
        return voxel_dict

    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> Dict[str, Tensor]:
        if 'bev_distill_views' not in voxel_dict:
            voxel_dict = self.forward(voxel_dict)
        views = voxel_dict['bev_distill_views']

        losses: Dict[str, Tensor] = {}
        point_view = views.get('point')
        voxel_view = views.get('voxel')
        frustum_view = views.get('frustum')

        # voxel (teacher) ↔ point (student)
        if point_view and voxel_view:
            match = _align_by_keys(point_view['feats'], point_view['keys'],
                                   voxel_view['feats'], voxel_view['keys'])
            if match is not None:
                s_feat, t_feat = match
                if self.detach_teacher:
                    t_feat = t_feat.detach()
                losses['loss_bev_l1'] = F.smooth_l1_loss(
                    s_feat, t_feat, reduction='mean') * self.loss_l1_weight
                if self.loss_nce_weight > 0 and s_feat.shape[0] > 1:
                    s_norm = F.normalize(s_feat, dim=-1)
                    t_norm = F.normalize(t_feat, dim=-1)
                    logits = torch.mm(s_norm, t_norm.t()) / self.temperature
                    labels = torch.arange(
                        s_norm.shape[0], device=s_norm.device, dtype=torch.long)
                    losses['loss_bev_nce'] = F.cross_entropy(
                        logits, labels) * self.loss_nce_weight

        # frustum ↔ point（一致性约束，轻量）
        if self.with_frustum_view and frustum_view and point_view:
            match = _align_by_keys(frustum_view['feats'], frustum_view['keys'],
                                   point_view['feats'], point_view['keys'])
            if match is not None:
                f_feat, p_feat = match
                losses['loss_frustum_bev'] = F.smooth_l1_loss(
                    p_feat, f_feat.detach() if self.detach_teacher else f_feat,
                    reduction='mean') * self.loss_frustum_weight

        # 防止 DDP 认为本模块未参与反向：若当前 batch 无匹配对，返回一个零损失占位
        if len(losses) == 0:
            # 使用一个与现有特征同设备的标量零，保持图连通性
            if point_view:
                losses['loss_bev_dummy'] = point_view['feats'].sum() * 0
            elif frustum_view:
                losses['loss_bev_dummy'] = frustum_view['feats'].sum() * 0
            elif voxel_view:
                losses['loss_bev_dummy'] = voxel_view['feats'].sum() * 0
            else:
                losses['loss_bev_dummy'] = torch.tensor(0.0, device=device)
        return losses

    # ========== BEV 视角构建 ==========
    def _build_bev_views(self, voxel_dict: dict) -> Dict[str, dict]:
        device = voxel_dict['point_feats_backbone'][0].device
        voxel_size = voxel_dict.get('voxel_size')
        point_cloud_range = voxel_dict.get('point_cloud_range')
        bev_shape = self._get_bev_shape(voxel_dict)

        if voxel_size is None or point_cloud_range is None:
            # 回退：使用常见 KITTI 范围，避免训练中断
            voxel_size = torch.tensor([0.2, 0.2, 0.2],
                                      device=device, dtype=torch.float32)
            point_cloud_range = torch.tensor(
                [-50.0, -50.0, -3.0, 50.0, 50.0, 3.0],
                device=device, dtype=torch.float32)
        else:
            voxel_size = voxel_size.to(device).float()
            point_cloud_range = point_cloud_range.to(device).float()

        views: Dict[str, dict] = {}

        # 1) point 视角（学生）：backbone 点特征 -> BEV
        point_view = self._point_bev(voxel_dict, bev_shape, voxel_size,
                                     point_cloud_range)
        if point_view is not None:
            views['point'] = point_view

        # 2) frustum 视角：range 特征采样到点，再落到 BEV
        if self.with_frustum_view:
            frustum_view = self._frustum_bev(voxel_dict, bev_shape, voxel_size,
                                             point_cloud_range)
            if frustum_view is not None:
                views['frustum'] = frustum_view

        # 3) voxel 视角（教师）：3D 体素特征池化到 BEV
        if 'voxel_3d_feats' in voxel_dict and 'voxel_3d_coors' in voxel_dict:
            voxel_view = self._voxel_bev(voxel_dict, bev_shape)
            if voxel_view is not None:
                views['voxel'] = voxel_view

        return views

    def _get_bev_shape(self, voxel_dict: dict) -> Tuple[int, int]:
        if 'bev_shape' in voxel_dict:
            return tuple(voxel_dict['bev_shape'])
        if 'voxel_shape' in voxel_dict:
            shape = voxel_dict['voxel_shape']
            return int(shape[0]), int(shape[1])
        # 回退到常用 KITTI 范围
        return 500, 500

    def _point_bev(self, voxel_dict: dict, bev_shape: Tuple[int, int],
                   voxel_size: Tensor, point_cloud_range: Tensor) -> Optional[dict]:
        points = voxel_dict.get('voxels')
        coors = voxel_dict.get('coors')
        point_feats = voxel_dict['point_feats_backbone'][0]
        if points is None or coors is None:
            return None

        xy = points[:, :2]
        batch_idx = coors[:, 0].long()
        ix = torch.floor((xy[:, 0] - point_cloud_range[0]) / voxel_size[0]).long()
        iy = torch.floor((xy[:, 1] - point_cloud_range[1]) / voxel_size[1]).long()

        valid = (ix >= 0) & (ix < bev_shape[0]) & (iy >= 0) & (iy < bev_shape[1])
        if valid.sum() == 0:
            return None

        keys = torch.stack([batch_idx[valid], ix[valid], iy[valid]], dim=1)
        feats = self.point_proj(point_feats[valid])
        uniq_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        bev_feats = torch_scatter.scatter_mean(feats.float(), inverse, dim=0).to(feats.dtype)
        return {'feats': bev_feats, 'keys': uniq_keys}

    def _frustum_bev(self, voxel_dict: dict, bev_shape: Tuple[int, int],
                     voxel_size: Tensor, point_cloud_range: Tensor) -> Optional[dict]:
        voxel_feats = voxel_dict.get('voxel_feats')
        coors = voxel_dict.get('coors')
        points = voxel_dict.get('voxels')
        if voxel_feats is None or coors is None or points is None:
            return None

        # 提取每个点对应的 frustum/range 特征
        pixel_feats = voxel_feats[0].permute(0, 2, 3, 1).contiguous()
        frustum_point_feats = pixel_feats[coors[:, 0], coors[:, 1], coors[:, 2]]

        xy = points[:, :2]
        batch_idx = coors[:, 0].long()
        ix = torch.floor((xy[:, 0] - point_cloud_range[0]) / voxel_size[0]).long()
        iy = torch.floor((xy[:, 1] - point_cloud_range[1]) / voxel_size[1]).long()
        valid = (ix >= 0) & (ix < bev_shape[0]) & (iy >= 0) & (iy < bev_shape[1])
        if valid.sum() == 0:
            return None

        keys = torch.stack([batch_idx[valid], ix[valid], iy[valid]], dim=1)
        feats = self.frustum_proj(frustum_point_feats[valid])
        uniq_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        bev_feats = torch_scatter.scatter_mean(feats.float(), inverse, dim=0).to(feats.dtype)
        return {'feats': bev_feats, 'keys': uniq_keys}

    def _voxel_bev(self, voxel_dict: dict,
                   bev_shape: Tuple[int, int]) -> Optional[dict]:
        voxel_feats = voxel_dict.get('voxel_3d_feats')
        voxel_coors = voxel_dict.get('voxel_3d_coors')
        if voxel_feats is None or voxel_coors is None:
            return None

        batch = voxel_coors[:, 0]
        vx = voxel_coors[:, 1]
        vy = voxel_coors[:, 2]
        keys = torch.stack([batch, vx, vy], dim=1)
        valid = (vx >= 0) & (vx < bev_shape[0]) & (vy >= 0) & (vy < bev_shape[1])
        if valid.sum() == 0:
            return None

        keys = keys[valid]
        feats = voxel_feats[valid]
        uniq_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        bev_feats = torch_scatter.scatter_mean(feats.float(), inverse, dim=0).to(feats.dtype)
        bev_feats = self.voxel_proj(bev_feats)
        return {'feats': bev_feats, 'keys': uniq_keys}

