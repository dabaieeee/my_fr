from typing import List, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType
from torch import Tensor


@MODELS.register_module()
class FRHead(Base3DDecodeHead):

    def __init__(self,
                 in_channels: int,
                 middle_channels: Sequence[int],
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 use_curriculum_learning: bool = False,
                 **kwargs) -> None:
        super(FRHead, self).__init__(**kwargs)

        self.loss_ce = MODELS.build(loss_ce)
        self.use_curriculum_learning = use_curriculum_learning
        
        # 初始化课程学习参数
        self.curriculum_learning = {
            'enabled': use_curriculum_learning,
            'sample_weights': None,
            'curriculum_progress': 0.0,
            'difficulty_threshold': 0.5,
            'min_weight': 0.1,
            'max_weight': 1.0
        }

        self.mlps = nn.ModuleList()
        for i in range(len(middle_channels)):
            out_channels = middle_channels[i]
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU(inplace=True)))
            in_channels = out_channels

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return nn.Linear(channels, num_classes)

    def forward(self, voxel_dict: dict) -> dict:
        point_feats_backbone = voxel_dict['point_feats_backbone'][0]
        point_feats = voxel_dict['point_feats'][:-1]
        voxel_feats = voxel_dict['voxel_feats'][0]
        voxel_feats = voxel_feats.permute(0, 2, 3, 1)
        pts_coors = voxel_dict['coors']
        map_point_feats = voxel_feats[pts_coors[:, 0], pts_coors[:, 1],
                                      pts_coors[:, 2]]

        for i, mlp in enumerate(self.mlps):
            map_point_feats = mlp(map_point_feats)
            # 第 1 层：与 backbone 输出的点特征相加（残差连接）
            if i == 0:
                map_point_feats = map_point_feats + point_feats_backbone
            # 后续层：与 FFE 提取的对应层级点特征相加（残差连接）
            else:
                map_point_feats = map_point_feats + point_feats[-i]
        seg_logit = self.cls_seg(map_point_feats)
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict

    # 将 batch 中所有样本的点级真实标签拼接成一个张量。
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ]
        return torch.cat(gt_semantic_segs, dim=0)

    # 交叉熵
    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> dict:
        seg_logit = voxel_dict['seg_logit']
        seg_label = self._stack_batch_gt(batch_data_samples)

        loss = dict()
        
        # 如果启用课程学习，计算样本级别的权重损失
        if (self.use_curriculum_learning and 
            hasattr(self, 'curriculum_learning') and 
            self.curriculum_learning.get('enabled', False)):
            
            # 计算每个点的损失（不进行平均）
            loss_per_point = self._compute_pointwise_loss(
                seg_logit, seg_label, ignore_index=self.ignore_index)
            
            # 计算样本权重
            sample_weights = self._compute_sample_weights(
                loss_per_point, seg_label, ignore_index=self.ignore_index)
            
            # 应用权重并计算最终损失
            weighted_loss = (loss_per_point * sample_weights).mean()
            loss['loss_ce'] = weighted_loss
        else:
            # 标准损失计算
            loss['loss_ce'] = self.loss_ce(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        
        return loss
    
    def _compute_pointwise_loss(self, seg_logit: Tensor, seg_label: Tensor, 
                                ignore_index: int) -> Tensor:
        """计算每个点的损失值（用于课程学习）。
        
        Args:
            seg_logit: 预测logits，形状为 [N, num_classes]
            seg_label: 真实标签，形状为 [N]
            ignore_index: 忽略的标签索引
            
        Returns:
            Tensor: 每个点的损失值，形状为 [N]
        """
        # 使用交叉熵损失，但返回每个点的损失
        log_softmax = torch.nn.functional.log_softmax(seg_logit, dim=1)
        
        # 创建one-hot编码
        num_classes = seg_logit.size(1)
        one_hot = torch.zeros_like(seg_logit)
        valid_mask = (seg_label != ignore_index)
        if valid_mask.any():
            one_hot[valid_mask, seg_label[valid_mask]] = 1.0
        
        # 计算每个点的负对数似然
        pointwise_loss = -torch.sum(one_hot * log_softmax, dim=1)
        
        # 对忽略的标签，损失设为0
        pointwise_loss = pointwise_loss * valid_mask.float()
        
        return pointwise_loss
    
    def _compute_sample_weights(self, loss_per_point: Tensor, seg_label: Tensor,
                               ignore_index: int) -> Tensor:
        """根据课程学习策略计算样本权重。
        
        Args:
            loss_per_point: 每个点的损失值，形状为 [N]
            seg_label: 真实标签，形状为 [N]
            ignore_index: 忽略的标签索引
            
        Returns:
            Tensor: 每个点的权重，形状为 [N]
        """
        curriculum_params = self.curriculum_learning
        progress = curriculum_params.get('curriculum_progress', 0.0)
        threshold = curriculum_params.get('difficulty_threshold', 0.5)
        min_weight = curriculum_params.get('min_weight', 0.1)
        max_weight = curriculum_params.get('max_weight', 1.0)
        
        # 归一化损失值（相对于当前batch）
        valid_mask = (seg_label != ignore_index)
        if not valid_mask.any():
            return torch.ones_like(loss_per_point)
        
        valid_loss = loss_per_point[valid_mask]
        if valid_loss.numel() == 0:
            return torch.ones_like(loss_per_point)
        
        # 计算损失的分位数作为阈值
        loss_mean = valid_loss.mean()
        loss_std = valid_loss.std()
        if loss_std > 0:
            normalized_loss = (loss_per_point - loss_mean) / (loss_std + 1e-8)
        else:
            normalized_loss = torch.zeros_like(loss_per_point)
        
        # 根据损失值判断样本难度
        # 损失值高于阈值的是困难样本
        difficulty_mask = normalized_loss > threshold
        
        # 计算权重：困难样本的权重从min_weight逐渐增加到max_weight
        weights = torch.ones_like(loss_per_point)
        
        # 简单样本：权重始终为1.0
        # 困难样本：权重从min_weight线性增加到max_weight
        if difficulty_mask.any():
            hard_weights = min_weight + (max_weight - min_weight) * progress
            weights[difficulty_mask] = hard_weights
        
        # 对忽略的标签，权重设为0
        weights = weights * valid_mask.float()
        
        return weights

    def predict(self, voxel_dict: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        voxel_dict = self.forward(voxel_dict)

        seg_pred_list = self.predict_by_feat(voxel_dict, batch_input_metas)

        final_seg_pred_list = []
        for seg_pred, input_metas in zip(seg_pred_list, batch_input_metas):
            if 'num_points' in input_metas:
                num_points = input_metas['num_points']
                seg_pred = seg_pred[:num_points]
            final_seg_pred_list.append(seg_pred)
        return final_seg_pred_list

    def predict_by_feat(self, voxel_dict: dict,
                        batch_input_metas: List[dict]) -> List[Tensor]:
        seg_logits = voxel_dict['seg_logit']

        coors = voxel_dict['coors']
        seg_pred_list = []
        for batch_idx in range(len(batch_input_metas)):
            batch_mask = coors[:, 0] == batch_idx
            seg_pred_list.append(seg_logits[batch_mask])
        return seg_pred_list
