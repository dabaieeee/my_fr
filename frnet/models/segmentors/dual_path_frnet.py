from typing import Dict

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor


@MODELS.register_module()
class DualPathFRNet(EncoderDecoder3D):
    """Dual-Path FRNet with Geometry-Semantic Decoupling.
    
    This segmentor implements a structure-level decoupling approach:
    - Geometry Path: Processes geometric features independently
    - Semantic Path: Uses FFE and FPFM for context-aware processing
    - Cross-Gated Fusion: Adaptively combines both paths
    
    Args:
        voxel_encoder (dict): Config for FFE (FrustumFeatureEncoder).
        geometry_encoder (dict): Config for GeometryEncoder.
        backbone (dict): Config for DualPathBackbone.
        decode_head (dict): Config for decode head.
        neck (dict, optional): Config for neck. Defaults to None.
        auxiliary_head (dict, optional): Config for auxiliary head. Defaults to None.
        train_cfg (dict, optional): Training config. Defaults to None.
        test_cfg (dict, optional): Testing config. Defaults to None.
        data_preprocessor (dict, optional): Data preprocessor config. Defaults to None.
        voxel_3d_encoder (dict, optional): Config for 3D voxel encoder. Defaults to None.
        use_multi_scale_voxel (bool): Whether to use multi-scale voxel encoder.
            Defaults to False.
        multi_scale_voxel_config (dict, optional): Config for multi-scale voxel encoder.
            Defaults to None.
        init_cfg (dict, optional): Weight initialization config. Defaults to None.
    """
    
    def __init__(self,
                 voxel_encoder: ConfigType,  # FFE
                 geometry_encoder: ConfigType,  # GeometryEncoder
                 backbone: ConfigType,  # DualPathBackbone
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 voxel_3d_encoder: OptConfigType = None,
                 use_multi_scale_voxel: bool = False,
                 multi_scale_voxel_config: OptConfigType = None,
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
        
        # FFE (FrustumFeatureEncoder) - 用于语义路径
        self.voxel_encoder = MODELS.build(voxel_encoder)
        
        # Geometry Encoder - 用于几何路径
        self.geometry_encoder = MODELS.build(geometry_encoder)
        
        # 3D voxel encoder (可选)
        self.voxel_3d_encoder = None
        if use_multi_scale_voxel and multi_scale_voxel_config is not None:
            self.voxel_3d_encoder = MODELS.build(multi_scale_voxel_config)
        elif voxel_3d_encoder is not None:
            self.voxel_3d_encoder = MODELS.build(voxel_3d_encoder)
    
    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from points using dual-path architecture.
        
        Args:
            batch_inputs_dict (dict): Input dictionary containing 'voxels'.
            
        Returns:
            dict: Feature dictionary with fused geometry-semantic features.
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        
        # ========== Geometry Path ==========
        # 首先提取几何特征（只使用xyz，不混合语义信息）
        geo_voxel_dict = {
            'voxels': voxel_dict['voxels'],
            'coors': voxel_dict['coors']
        }
        geo_voxel_dict = self.geometry_encoder(geo_voxel_dict)
        voxel_dict['geometry_feats'] = geo_voxel_dict['geometry_feats']
        voxel_dict['geometry_point_feats'] = geo_voxel_dict.get('geometry_point_feats', [])
        
        # ========== Semantic Path: FFE ==========
        # FFE提取frustum特征（保留原有功能）
        voxel_dict = self.voxel_encoder(voxel_dict)
        
        # ========== 3D Voxel Encoder (可选) ==========
        if self.voxel_3d_encoder is not None:
            voxel_dict_3d = batch_inputs_dict['voxels'].copy()
            voxel_dict_3d = self.voxel_3d_encoder(voxel_dict_3d)
            voxel_dict['voxel_3d_feats'] = voxel_dict_3d['voxel_3d_feats']
            voxel_dict['voxel_3d_coors'] = voxel_dict_3d['voxel_3d_coors']
            voxel_dict['voxel_shape'] = voxel_dict_3d['voxel_shape']
            voxel_dict['voxel_3d_sparse'] = voxel_dict_3d.get('voxel_3d_sparse', True)
            if 'voxels' not in voxel_dict:
                voxel_dict['voxels'] = batch_inputs_dict['voxels']['voxels']
        
        # ========== Dual-Path Backbone ==========
        # Backbone会处理：
        # 1. Geometry Path: 使用已提取的几何特征
        # 2. Semantic Path: 使用FFE和FPFM提取语义特征
        # 3. Cross-Gated Fusion: 融合两个路径
        voxel_dict = self.backbone(voxel_dict)
        
        if self.with_neck:
            voxel_dict = self.neck(voxel_dict)
        
        return voxel_dict
    
    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            batch_inputs_dict (dict): Input sample dict.
            batch_data_samples (List[:obj:`Det3DDataSample`]): Data samples.
            
        Returns:
            Dict[str, Tensor]: Dictionary of loss components.
        """
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
            batch_inputs_dict (dict): Input sample dict.
            batch_data_samples (List[:obj:`Det3DDataSample`]): Data samples.
            rescale (bool): Whether to rescale to original number of points.
                Defaults to True.
                
        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results.
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
            batch_inputs_dict (dict): Input sample dict.
            batch_data_samples (List[:obj:`Det3DDataSample`], optional): Data samples.
            
        Returns:
            dict: Forward output without post-processing.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)

