from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class SemanticEncoder(nn.Module):
    """Semantic Encoder for context-aware semantic feature extraction.
    
    This encoder uses FFE (Frustum Feature Encoder) to extract semantic
    context features. It focuses on:
    - Context understanding
    - Class relationships
    - Long-range dependencies
    - Scene-level semantics
    
    Args:
        ffe_config (dict): Configuration for FrustumFeatureEncoder.
            This will be used to build the FFE module.
    """

    def __init__(self,
                 ffe_config: ConfigType) -> None:
        super(SemanticEncoder, self).__init__()
        
        from mmdet3d.registry import MODELS as MODELS_REGISTRY
        # Build FFE module
        self.ffe = MODELS_REGISTRY.build(ffe_config)

    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass of Semantic Encoder.
        
        Args:
            voxel_dict (dict): Dictionary containing point cloud data.
                
        Returns:
            dict: Updated voxel_dict with semantic features:
                - 'sem_point_feats': Semantic point features [N, C_sem]
                - 'sem_voxel_feats': Semantic frustum features [M, C_sem]
                - 'sem_voxel_coors': Frustum coordinates [M, 4]
        """
        # Use FFE to extract semantic features
        voxel_dict = self.ffe(voxel_dict)
        
        # Rename outputs for semantic path
        if 'point_feats' in voxel_dict:
            # Get the last point features from FFE
            if isinstance(voxel_dict['point_feats'], list):
                voxel_dict['sem_point_feats'] = voxel_dict['point_feats'][-1]
            else:
                voxel_dict['sem_point_feats'] = voxel_dict['point_feats']
        
        if 'voxel_feats' in voxel_dict:
            voxel_dict['sem_voxel_feats'] = voxel_dict['voxel_feats']
        
        if 'voxel_coors' in voxel_dict:
            voxel_dict['sem_voxel_coors'] = voxel_dict['voxel_coors']
        
        return voxel_dict

