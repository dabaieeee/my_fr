# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation.
    
    Dice loss is particularly effective for imbalanced datasets as it
    directly optimizes the Dice coefficient, which is a common evaluation
    metric for segmentation tasks.
    
    Formula:
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        Loss = 1 - Dice
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero.
            Defaults to 1.0.
        reduction (str): Reduction method. Defaults to 'mean'.
        loss_weight (float): Loss weight. Defaults to 1.0.
        ignore_index (int): Index to ignore. Defaults to 255.
    """
    
    def __init__(self,
                 smooth: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 ignore_index: int = 255) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth
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
        
        # Get probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # One-hot encode target
        num_classes = pred_probs.shape[1]
        target_one_hot = F.one_hot(target, num_classes).float()
        
        # Compute Dice loss per class
        dice_losses = []
        for c in range(num_classes):
            pred_c = pred_probs[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1.0 - dice)
        
        dice_loss = torch.stack(dice_losses)
        
        if self.reduction == 'mean':
            return self.loss_weight * dice_loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * dice_loss.sum()
        else:
            return self.loss_weight * dice_loss

