from typing import Dict

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor


@MODELS.register_module()
class DualPathFRNet(EncoderDecoder3D):
    """Dual-Path FRNet Segmentor with Geometry-Semantic Decoupling.
    
    This segmentor implements a dual-path architecture:
    1. Geometry Path: Extracts structure-preserving geometric features
    2. Semantic Path: Extracts context-aware semantic features using FFE
    3. Cross-Gated Fusion: Adaptively fuses geometry and semantic features
    
    The FPFM (Frustum-Point Fusion Module) is preserved in the backbone.
    
    Args:
        geometry_encoder (dict): Config for GeometryEncoder.
        semantic_encoder (dict): Config for SemanticEncoder (uses FFE).
        backbone (dict): Config for DualPathFRNetBackbone.
        decode_head (dict): Config for decode head.
        neck (dict, optional): Config for neck. Defaults to None.
        auxiliary_head (dict, optional): Config for auxiliary head. Defaults to None.
        train_cfg (dict, optional): Training config. Defaults to None.
        test_cfg (dict, optional): Testing config. Defaults to None.
        data_preprocessor (dict, optional): Data preprocessor config. Defaults to None.
        init_cfg (dict, optional): Weight initialization config. Defaults to None.
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

        # Geometry encoder: structure-preserving
        self.geometry_encoder = MODELS.build(geometry_encoder)
        
        # Semantic encoder: context-aware (uses FFE)
        self.semantic_encoder = MODELS.build(semantic_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from points using dual-path architecture.
        
        Args:
            batch_inputs_dict (dict): Input dictionary containing 'voxels' key.
            
        Returns:
            dict: Feature dictionary with:
                - 'voxel_feats': Fused frustum features
                - 'point_feats_backbone': Fused point features
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        
        # Geometry Path: Extract geometric features
        geo_voxel_dict = voxel_dict.copy()
        geo_voxel_dict = self.geometry_encoder(geo_voxel_dict)
        
        # Semantic Path: Extract semantic features (uses FFE)
        sem_voxel_dict = voxel_dict.copy()
        sem_voxel_dict = self.semantic_encoder(sem_voxel_dict)
        
        # Combine both paths for backbone
        combined_voxel_dict = {
            'geo_voxel_feats': geo_voxel_dict['geo_voxel_feats'],
            'geo_voxel_coors': geo_voxel_dict['geo_voxel_coors'],
            'geo_point_feats': geo_voxel_dict['geo_point_feats'],
            'sem_voxel_feats': sem_voxel_dict['sem_voxel_feats'],
            'sem_voxel_coors': sem_voxel_dict['sem_voxel_coors'],
            'sem_point_feats': sem_voxel_dict['sem_point_feats'],
            'coors': voxel_dict['coors'],
        }
        
        # Backbone: Cross-Gated Fusion + FPFM
        combined_voxel_dict = self.backbone(combined_voxel_dict)
        
        if self.with_neck:
            combined_voxel_dict = self.neck(combined_voxel_dict)
        
        return combined_voxel_dict

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
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
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
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
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict: Forward output of model without any post-processes.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)

