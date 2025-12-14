"""FRNet中的Focal Loss，用于困难样本挖掘。

Focal Loss通过降低简单样本的权重，让模型更关注困难样本，
从而解决类别不平衡问题。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class FocalLoss(nn.Module):
    """用于语义分割的Focal Loss。
    
    Focal Loss设计用于解决一阶段目标检测中的类别不平衡问题。
    它通过降低简单样本的权重，让模型更关注困难样本。
    
    Args:
        alpha (float): 稀有类别的权重因子。默认为0.25。
        gamma (float): 聚焦参数。默认为2.0。
        reduction (str): 指定对输出的归约方式。
            选项: 'none', 'mean', 'sum'。默认为'mean'。
        ignore_index (int): 指定被忽略且不参与梯度计算的目标值。
            默认为255。
        loss_weight (float): Focal Loss的权重。默认为1.0。
    """
    
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = 255,
                 loss_weight: float = 1.0) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        
        assert reduction in ['none', 'mean', 'sum'], \
            f'reduction must be one of ["none", "mean", "sum"], but got {reduction}'
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """前向传播函数。
        
        Args:
            pred (Tensor): 预测的logits。形状: [N, num_classes]
            target (Tensor): 真实标签。形状: [N]
            
        Returns:
            Tensor: Focal Loss值。
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            pred, target, 
            reduction='none', 
            ignore_index=self.ignore_index)
        
        # 计算p_t（预测概率）
        p = F.softmax(pred, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 创建有效样本掩码（非忽略样本）
        valid_mask = (target != self.ignore_index).float()
        
        # 计算alpha_t
        alpha_t = torch.ones_like(p_t) * self.alpha
        # 如果需要，可以在这里为每个类别自定义alpha_t
        
        # 计算focal loss
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # 应用有效掩码
        focal_loss = focal_loss * valid_mask
        
        # 应用归约操作
        if self.reduction == 'mean':
            if valid_mask.sum() > 0:
                focal_loss = focal_loss.sum() / valid_mask.sum()
            else:
                focal_loss = focal_loss.sum() * 0.0
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        # 'none'归约：直接返回
        
        return self.loss_weight * focal_loss

