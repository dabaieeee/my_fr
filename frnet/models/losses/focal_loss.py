# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal loss focuses learning on hard examples by down-weighting
    easy examples. This is particularly useful for point cloud segmentation
    where class distribution is highly imbalanced.
    
    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float): Weighting factor for rare class. Defaults to 0.25.
        gamma (float): Focusing parameter. Defaults to 2.0.
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_weight (float): Loss weight. Defaults to 1.0.
        ignore_index (int): Index to ignore. Defaults to 255.
    """
    
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 ignore_index: int = 255) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            pred (Tensor): Prediction logits [N, C] or [B, C, H, W].
            target (Tensor): Ground truth labels [N] or [B, H, W].
            
        Returns:
            Tensor: Loss tensor.
        """
        # Flatten if needed
        if pred.dim() == 4:
            # [B, C, H, W] -> [B*H*W, C]
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
            target = target.view(-1)
        
        # Get valid mask (ignore index)
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Compute p_t (probability of true class)
        p = F.softmax(pred, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return self.loss_weight * focal_loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * focal_loss.sum()
        else:
            return self.loss_weight * focal_loss

