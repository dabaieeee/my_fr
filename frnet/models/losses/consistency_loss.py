"""FRNet的特征一致性损失。

该损失确保frustum特征和point特征之间的一致性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class FeatureConsistencyLoss(nn.Module):
    """Frustum特征和Point特征之间的特征一致性损失。
    
    该损失鼓励投影到点的frustum特征与原始点特征之间的一致性，
    确保两种表示方式良好对齐。
    
    Args:
        loss_weight (float): 一致性损失的权重。默认为0.1。
        loss_type (str): 一致性损失的类型。选项: 'l2', 'cosine', 'both'。
            默认为'both'。
        temperature (float): 余弦相似度的温度参数。默认为0.1。
    """
    
    def __init__(self,
                 loss_weight: float = 0.1,
                 loss_type: str = 'both',
                 temperature: float = 0.1) -> None:
        super(FeatureConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.temperature = temperature
        
        assert loss_type in ['l2', 'cosine', 'both'], \
            f'loss_type must be one of ["l2", "cosine", "both"], but got {loss_type}'
    
    def forward(self, frustum_feats: Tensor, point_feats: Tensor) -> Tensor:
        """前向传播函数。
        
        Args:
            frustum_feats (Tensor): 投影到点的frustum特征。
                形状: [N, C]
            point_feats (Tensor): 原始点特征。形状: [N, C]
            
        Returns:
            Tensor: 一致性损失值。
        """
        if frustum_feats.shape != point_feats.shape:
            # 如果形状不匹配，进行插值或投影
            if frustum_feats.shape[1] != point_feats.shape[1]:
                # 如果需要，投影到相同维度
                if frustum_feats.shape[1] < point_feats.shape[1]:
                    frustum_feats = F.linear(
                        frustum_feats, 
                        torch.eye(frustum_feats.shape[1], point_feats.shape[1], 
                                 device=frustum_feats.device))
                else:
                    point_feats = F.linear(
                        point_feats,
                        torch.eye(point_feats.shape[1], frustum_feats.shape[1],
                                 device=point_feats.device))
        
        loss = 0.0
        
        if self.loss_type in ['l2', 'both']:
            # L2一致性损失
            l2_loss = F.mse_loss(frustum_feats, point_feats, reduction='mean')
            loss = loss + l2_loss
        
        if self.loss_type in ['cosine', 'both']:
            # 余弦相似度损失
            frustum_feats_norm = F.normalize(frustum_feats, p=2, dim=1)
            point_feats_norm = F.normalize(point_feats, p=2, dim=1)
            cosine_sim = (frustum_feats_norm * point_feats_norm).sum(dim=1)
            cosine_loss = 1.0 - cosine_sim.mean()
            loss = loss + cosine_loss
        
        return self.loss_weight * loss

