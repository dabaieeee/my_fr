from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class SemanticEncoder(nn.Module):
    """语义编码器，用于上下文感知的语义特征提取。
    
    该编码器使用FFE（视锥特征编码器）来提取语义上下文特征。它专注于：
    - 上下文理解
    - 类别关系
    - 长程依赖
    - 场景级语义
    
    Args:
        ffe_config (dict): FrustumFeatureEncoder的配置。
            将用于构建FFE模块。
    """

    def __init__(self,
                 ffe_config: ConfigType) -> None:
        super(SemanticEncoder, self).__init__()
        
        from mmdet3d.registry import MODELS as MODELS_REGISTRY
        # 构建FFE模块
        self.ffe = MODELS_REGISTRY.build(ffe_config)

    def forward(self, voxel_dict: dict) -> dict:
        """语义编码器的前向传播。
        
        Args:
            voxel_dict (dict): 包含点云数据的字典。
                
        Returns:
            dict: 更新后的voxel_dict，包含语义特征：
                - 'sem_point_feats': 语义点特征 [N, C_sem]
                - 'sem_voxel_feats': 语义视锥特征 [M, C_sem]
                - 'sem_voxel_coors': 视锥坐标 [M, 4]
        """
        # 使用FFE提取语义特征
        voxel_dict = self.ffe(voxel_dict)
        
        # 为语义路径重命名输出
        if 'point_feats' in voxel_dict:
            # 从FFE获取最后的点特征
            if isinstance(voxel_dict['point_feats'], list):
                voxel_dict['sem_point_feats'] = voxel_dict['point_feats'][-1]
            else:
                voxel_dict['sem_point_feats'] = voxel_dict['point_feats']
        
        if 'voxel_feats' in voxel_dict:
            voxel_dict['sem_voxel_feats'] = voxel_dict['voxel_feats']
        
        if 'voxel_coors' in voxel_dict:
            voxel_dict['sem_voxel_coors'] = voxel_dict['voxel_coors']
        
        return voxel_dict

