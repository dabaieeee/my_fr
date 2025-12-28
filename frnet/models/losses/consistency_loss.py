# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from torch import Tensor
from typing import Optional


@MODELS.register_module()
class FeatureLevelConsistencyLoss(nn.Module):
    """Feature-level Consistency Loss.
    
    Constrains features from point/frustum/voxel branches to be consistent
    for the same physical point.
    
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_type (str): Type of consistency loss. Options: 'mse', 'cosine', 'kl'.
            Defaults to 'mse'.
    """
    
    def __init__(self, 
                 loss_weight: float = 1.0,
                 loss_type: str = 'mse') -> None:
        super(FeatureLevelConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        
    def forward(self, 
                feat1: Tensor, 
                feat2: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """Forward function.
        
        Args:
            feat1 (Tensor): Features from first branch [N, C].
            feat2 (Tensor): Features from second branch [N, C].
            mask (Tensor, optional): Valid mask [N].
            
        Returns:
            Tensor: Loss tensor.
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(feat1, feat2, reduction='none')
            loss = loss.mean(dim=-1)  # [N]
        elif self.loss_type == 'cosine':
            # Cosine similarity loss (1 - cosine_similarity)
            feat1_norm = F.normalize(feat1, p=2, dim=-1)
            feat2_norm = F.normalize(feat2, p=2, dim=-1)
            loss = 1.0 - (feat1_norm * feat2_norm).sum(dim=-1)  # [N]
        elif self.loss_type == 'kl':
            # KL divergence (treating features as distributions)
            feat1_prob = F.softmax(feat1, dim=-1)
            feat2_log_prob = F.log_softmax(feat2, dim=-1)
            loss = F.kl_div(feat2_log_prob, feat1_prob, reduction='none').sum(dim=-1)  # [N]
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-7)
        else:
            loss = loss.mean()
        
        return self.loss_weight * loss


@MODELS.register_module()
class PredictionConsistencyLoss(nn.Module):
    """Point-level Prediction Consistency Loss.
    
    Constrains prediction distributions from different branches to be consistent
    for the same physical point.
    
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_type (str): Type of consistency loss. Options: 'kl', 'js', 'ce'.
            Defaults to 'kl'.
    """
    
    def __init__(self, 
                 loss_weight: float = 1.0,
                 loss_type: str = 'kl') -> None:
        super(PredictionConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        
    def forward(self, 
                pred1: Tensor, 
                pred2: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """Forward function.
        
        Args:
            pred1 (Tensor): Predictions from first branch [N, num_classes].
            pred2 (Tensor): Predictions from second branch [N, num_classes].
            mask (Tensor, optional): Valid mask [N].
            
        Returns:
            Tensor: Loss tensor.
        """
        if self.loss_type == 'kl':
            # KL divergence: KL(pred1 || pred2)
            pred1_prob = F.softmax(pred1, dim=-1)
            pred2_log_prob = F.log_softmax(pred2, dim=-1)
            loss = F.kl_div(pred2_log_prob, pred1_prob, reduction='none').sum(dim=-1)  # [N]
        elif self.loss_type == 'js':
            # Jensen-Shannon divergence (symmetric)
            pred1_prob = F.softmax(pred1, dim=-1)
            pred2_prob = F.softmax(pred2, dim=-1)
            m = 0.5 * (pred1_prob + pred2_prob)
            pred1_log_prob = F.log_softmax(pred1, dim=-1)
            pred2_log_prob = F.log_softmax(pred2, dim=-1)
            m_log = torch.log(m + 1e-7)
            loss = 0.5 * (F.kl_div(m_log, pred1_prob, reduction='none').sum(dim=-1) +
                         F.kl_div(m_log, pred2_prob, reduction='none').sum(dim=-1))  # [N]
        elif self.loss_type == 'ce':
            # Cross-entropy (use pred1 as pseudo label)
            pred1_label = pred1.argmax(dim=-1)
            loss = F.cross_entropy(pred2, pred1_label, reduction='none')  # [N]
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-7)
        else:
            loss = loss.mean()
        
        return self.loss_weight * loss

