from typing import Dict

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor


@MODELS.register_module()
class DualPathFRNet(EncoderDecoder3D):
    """双通路FRNet分割器，具有几何-语义解耦。
    
    该分割器实现了双通路架构：
    1. 几何路径：提取结构保持的几何特征
    2. 语义路径：使用FFE提取上下文感知的语义特征
    3. 交叉门控融合：自适应地融合几何和语义特征
    
    FPFM（视锥-点融合模块）在骨干网络中保留。
    
    Args:
        geometry_encoder (dict): GeometryEncoder的配置。
        semantic_encoder (dict): SemanticEncoder的配置（使用FFE）。
        backbone (dict): DualPathFRNetBackbone的配置。
        decode_head (dict): 解码头的配置。
        neck (dict, optional): 颈部的配置。默认为 None。
        auxiliary_head (dict, optional): 辅助头的配置。默认为 None。
        train_cfg (dict, optional): 训练配置。默认为 None。
        test_cfg (dict, optional): 测试配置。默认为 None。
        data_preprocessor (dict, optional): 数据预处理器配置。默认为 None。
        init_cfg (dict, optional): 权重初始化配置。默认为 None。
    """

    def __init__(self,
                 geometry_encoder: ConfigType,
                 semantic_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(DualPathFRNet, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # 几何编码器：结构保持
        self.geometry_encoder = MODELS.build(geometry_encoder)
        
        # 语义编码器：上下文感知（使用FFE）
        self.semantic_encoder = MODELS.build(semantic_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """使用双通路架构从点中提取特征。
        
        Args:
            batch_inputs_dict (dict): 包含'voxels'键的输入字典。
            
        Returns:
            dict: 特征字典，包含：
                - 'voxel_feats': 融合后的视锥特征
                - 'point_feats_backbone': 融合后的点特征
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        
        # 几何路径：提取几何特征
        geo_voxel_dict = voxel_dict.copy()
        geo_voxel_dict = self.geometry_encoder(geo_voxel_dict)
        
        # 语义路径：提取语义特征（使用FFE）
        sem_voxel_dict = voxel_dict.copy()
        sem_voxel_dict = self.semantic_encoder(sem_voxel_dict)
        
        # 为骨干网络合并两条路径
        combined_voxel_dict = {
            'geo_voxel_feats': geo_voxel_dict['geo_voxel_feats'],
            'geo_voxel_coors': geo_voxel_dict['geo_voxel_coors'],
            'geo_point_feats': geo_voxel_dict['geo_point_feats'],
            'sem_voxel_feats': sem_voxel_dict['sem_voxel_feats'],
            'sem_voxel_coors': sem_voxel_dict['sem_voxel_coors'],
            'sem_point_feats': sem_voxel_dict['sem_point_feats'],
            'coors': voxel_dict['coors'],
        }
        
        # 骨干网络：交叉门控融合 + FPFM
        combined_voxel_dict = self.backbone(combined_voxel_dict)
        
        if self.with_neck:
            combined_voxel_dict = self.neck(combined_voxel_dict)
        
        return combined_voxel_dict

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """从一批输入和数据样本计算损失。

        Args:
            batch_inputs_dict (dict): 输入样本字典，包含'points'
                和'imgs'键。
            batch_data_samples (List[:obj:`Det3DDataSample`]): det3d数据
                样本。通常包含诸如`metainfo`和`gt_pts_seg`的信息。

        Returns:
            Dict[str, Tensor]: 损失组件的字典。
        """

        # 使用骨干网络提取特征
        voxel_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                      batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)
        return losses

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """简单测试单个场景。

        Args:
            batch_inputs_dict (dict): 输入样本字典，包含'points'
                和'imgs'键。
            batch_data_samples (List[:obj:`Det3DDataSample`]): det3d数据
                样本。通常包含诸如`metainfo`和`gt_pts_seg`的信息。
            rescale (bool): 是否转换回原始点数。
                将用于基于体素化的分割器。
                默认为 True。

        Returns:
            List[:obj:`Det3DDataSample`]: 输入点的分割结果。
            每个Det3DDataSample通常包含：

            - ``pred_pts_seg`` (PointData): 3D语义分割的预测。
            - ``pts_seg_logits`` (PointData): 归一化前的3D语义分割预测logits。
        """
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict,
                                                   batch_input_metas,
                                                   self.test_cfg)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> dict:
        """网络前向传播过程。

        Args:
            batch_inputs_dict (dict): 输入样本字典，包含'points'
                和'imgs'键。
            batch_data_samples (List[:obj:`Det3DDataSample`]): det3d数据
                样本。通常包含诸如`metainfo`和`gt_pts_seg`的信息。

        Returns:
            dict: 模型的前向输出，不进行任何后处理。
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)

