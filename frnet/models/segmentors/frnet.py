from typing import Dict, Optional

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer



@MODELS.register_module()
class FRNet(EncoderDecoder3D):
    """Frustum-Range Segmentor.

    Args:
        voxel_encoder (dict or :obj:`ConfigDict`): The config for the voxel
            encoder of segmentor.
        backbone (dict or :obj:`ConfigDict`): The config for the backbone of
            segmentor.
        decode_head (dict or :obj:`ConfigDict`): The config for the decode head
            of segmentor.
        neck (dict or :obj:`ConfigDict`, optional): The config for the neck of
            segmentor. Defaults to None.
        auxiliary_head (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (dict or :obj:`ConfigDict`, optional): The config for
            training. Defaults to None.
        test_cfg (dict or :obj:`ConfigDict`, optional): The config for testing.
            Defaults to None.
        data_preprocessor (dict or :obj:`ConfigDict`, optional): The
            pre-process config of :class:`BaseDataPreprocessor`.
            Defaults to None.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`],
            optional): The weight initialized config for :class:`BaseModule`.
            Defaults to None.
    """

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 voxel_3d_encoder: OptConfigType = None,
                 # Feature-level consistency parameters
                 use_feature_consistency: bool = False,
                 feature_consistency_loss: OptConfigType = None,
                 feature_consistency_stages: list = [1, 2, 3],
                 feature_consistency_weight: float = 0.1,
                 # Prediction-level consistency parameters
                 use_prediction_consistency: bool = False,
                 prediction_consistency_loss: OptConfigType = None,
                 prediction_consistency_weight: float = 0.1,
                 # Offset network parameters
                 frustum_offset_range: int = 3,
                 voxel_offset_range: int = 2,
                 offset_reg_weight: float = 0.01,
                 init_cfg: OptMultiConfig = None) -> None:
        super(FRNet, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # ffe (frustum feature encoder)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        
        # 3D voxel encoder (体素编码器)
        self.voxel_3d_encoder = None
        if voxel_3d_encoder is not None:
            self.voxel_3d_encoder = MODELS.build(voxel_3d_encoder)
        
        # Feature-level consistency
        self.use_feature_consistency = use_feature_consistency
        self.feature_consistency_stages = feature_consistency_stages
        self.feature_consistency_weight = feature_consistency_weight
        
        if self.use_feature_consistency:
            if feature_consistency_loss is None:
                feature_consistency_loss = dict(
                    type='FeatureLevelConsistencyLoss',
                    loss_weight=1.0,
                    loss_type='mse'
                )
            self.feature_consistency_loss = MODELS.build(feature_consistency_loss)
        else:
            self.feature_consistency_loss = None
        
        # Prediction-level consistency
        self.use_prediction_consistency = use_prediction_consistency
        self.prediction_consistency_weight = prediction_consistency_weight
        
        if self.use_prediction_consistency:
            if prediction_consistency_loss is None:
                prediction_consistency_loss = dict(
                    type='PredictionConsistencyLoss',
                    loss_weight=1.0,
                    loss_type='kl'
                )
            self.prediction_consistency_loss = MODELS.build(prediction_consistency_loss)
        else:
            self.prediction_consistency_loss = None
        
        # Offset networks (only created when needed)
        self.frustum_offset_range = frustum_offset_range
        self.voxel_offset_range = voxel_offset_range
        self.offset_reg_weight = offset_reg_weight
        
        # Import offset networks
        from ..modules.offset_networks import FrustumOffsetNetwork, VoxelOffsetNetwork
        
        # Create offset networks for each stage (only if feature consistency is enabled)
        self.frustum_offset_nets = nn.ModuleDict()
        self.voxel_offset_nets = nn.ModuleDict()
        
        # Only create offset networks if feature consistency is enabled
        if self.use_feature_consistency and len(feature_consistency_stages) > 0:
            # Get backbone output channels from config
            if hasattr(backbone, 'out_channels'):
                out_channels = backbone['out_channels']
            else:
                out_channels = (128, 128, 128, 128)  # Default
            
            # Get voxel encoder output channels
            voxel_3d_out_channels = None
            if self.voxel_3d_encoder is not None and hasattr(voxel_3d_encoder, 'feat_channels'):
                voxel_3d_out_channels = voxel_3d_encoder['feat_channels'][-1]  # Last channel
            elif self.voxel_3d_encoder is not None:
                voxel_3d_out_channels = 256  # Default
            
            for stage_idx in feature_consistency_stages:
                if stage_idx < len(out_channels):
                    channels = out_channels[stage_idx]
                    
                    # Frustum offset network
                    self.frustum_offset_nets[f'stage_{stage_idx}'] = FrustumOffsetNetwork(
                        in_channels=channels,
                        offset_range=frustum_offset_range,
                        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
                        act_cfg=dict(type='HSwish', inplace=True)
                    )
                    
                    # Voxel offset network (if voxel encoder exists)
                    if self.voxel_3d_encoder is not None and voxel_3d_out_channels is not None:
                        # Create voxel offset network
                        voxel_offset_net = VoxelOffsetNetwork(
                            in_channels=voxel_3d_out_channels,  # Use voxel encoder output channels
                            offset_range=voxel_offset_range,
                            norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
                            act_cfg=dict(type='ReLU', inplace=True)
                        )
                        self.voxel_offset_nets[f'stage_{stage_idx}'] = voxel_offset_net
                        
                        # Create projection layer if dimensions don't match
                        if voxel_3d_out_channels != channels:
                            proj_layer = nn.Sequential(
                                nn.Linear(voxel_3d_out_channels, channels, bias=False),
                                build_norm_layer(dict(type='SyncBN', eps=1e-3, momentum=0.01), channels)[1],
                                nn.ReLU(inplace=True)
                            )
                            self.voxel_offset_nets[f'stage_{stage_idx}_proj'] = proj_layer
        
        # Note: Prediction offset networks are not needed for prediction consistency
        # because predictions can be directly compared without spatial alignment
        # (predictions are already at the same spatial resolution)

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels'].copy()
        voxel_dict = self.voxel_encoder(voxel_dict)  # FFE 提取 frustum 特征
        
        # 提取3D体素特征
        if self.voxel_3d_encoder is not None:
            voxel_dict_3d = batch_inputs_dict['voxels'].copy()
            voxel_dict_3d = self.voxel_3d_encoder(voxel_dict_3d)  # 3D体素编码器提取特征
            voxel_dict['voxel_3d_feats'] = voxel_dict_3d['voxel_3d_feats']
            voxel_dict['voxel_3d_coors'] = voxel_dict_3d['voxel_3d_coors']
            voxel_dict['voxel_shape'] = voxel_dict_3d['voxel_shape']
            voxel_dict['voxel_3d_sparse'] = voxel_dict_3d.get('voxel_3d_sparse', True)
            # 保留原始点云信息，用于稀疏体素特征映射
            if 'voxels' not in voxel_dict:
                voxel_dict['voxels'] = batch_inputs_dict['voxels']['voxels']
        
        voxel_dict = self.backbone(voxel_dict)  # Backbone 进行层次化双向融合
        if self.with_neck:
            voxel_dict = self.neck(voxel_dict)
        return voxel_dict

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
        
        # Compute consistency losses if enabled
        if self.use_feature_consistency or self.use_prediction_consistency:
            consistency_losses = self._compute_consistency_losses(voxel_dict)
        else:
            consistency_losses = dict()
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                      batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)
        
        # Add consistency losses
        losses.update(consistency_losses)
        
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
    
    def _compute_consistency_losses(self, voxel_dict: dict) -> Dict[str, Tensor]:
        """Compute feature-level and prediction-level consistency losses.
        
        Args:
            voxel_dict (dict): Dictionary containing features from all branches.
            
        Returns:
            Dict[str, Tensor]: Dictionary of consistency losses.
        """
        losses = dict()
        offset_reg_losses = []
        
        # Get necessary data
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1
        
        # Feature-level consistency at intermediate stages
        if self.use_feature_consistency and 'stage_features' in voxel_dict:
            stage_features = voxel_dict['stage_features']
            
            for stage_idx in self.feature_consistency_stages:
                if stage_idx not in stage_features:
                    continue
                
                stage_data = stage_features[stage_idx]
                point_feats = stage_data['point_feats']  # [N, C]
                frustum_feats = stage_data['frustum_feats']  # [B, C, H, W]
                stride = stage_data['stride']
                
                # Point-Frustum consistency
                if f'stage_{stage_idx}' in self.frustum_offset_nets:
                    frustum_offset_net = self.frustum_offset_nets[f'stage_{stage_idx}']
                    
                    # Apply offset network to align frustum features to points
                    aligned_frustum_feats, offset_reg = frustum_offset_net(
                        frustum_feats, pts_coors, stride
                    )  # [N, C]
                    
                    offset_reg_losses.append(offset_reg)
                    
                    # Compute feature consistency loss
                    if self.feature_consistency_loss is not None:
                        feat_loss = self.feature_consistency_loss(
                            point_feats, aligned_frustum_feats
                        )
                        losses[f'loss_feat_consist_pf_s{stage_idx}'] = \
                            feat_loss * self.feature_consistency_weight
                
                # Point-Voxel consistency (if voxel branch exists)
                if (self.voxel_3d_encoder is not None and 
                    'voxel_3d_feats' in voxel_dict and
                    f'stage_{stage_idx}' in self.voxel_offset_nets):
                    
                    voxel_offset_net = self.voxel_offset_nets[f'stage_{stage_idx}']
                    
                    # Get voxel features (simplified - using global features)
                    # In practice, you need to properly extract stage-specific voxel features
                    voxel_3d_feats = voxel_dict.get('voxel_3d_feats')
                    voxel_3d_coors = voxel_dict.get('voxel_3d_coors')
                    voxel_shape = voxel_dict.get('voxel_shape')
                    
                    if voxel_3d_feats is not None and voxel_3d_coors is not None:
                        # Apply offset network to align voxel features to points
                        aligned_voxel_feats, offset_reg = voxel_offset_net(
                            voxel_3d_feats, voxel_3d_coors, pts_coors, 
                            voxel_shape, stride
                        )  # [N, C_voxel]
                        
                        offset_reg_losses.append(offset_reg)
                        
                        # Project voxel features to match point feature dimensions if needed
                        proj_key = f'stage_{stage_idx}_proj'
                        if proj_key in self.voxel_offset_nets:
                            aligned_voxel_feats = self.voxel_offset_nets[proj_key](aligned_voxel_feats)
                        
                        # Compute feature consistency loss
                        if self.feature_consistency_loss is not None:
                            feat_loss = self.feature_consistency_loss(
                                point_feats, aligned_voxel_feats
                            )
                            losses[f'loss_feat_consist_pv_s{stage_idx}'] = \
                                feat_loss * self.feature_consistency_weight
        
        # Prediction-level consistency at head outputs
        if self.use_prediction_consistency and 'seg_logit' in voxel_dict:
            point_pred = voxel_dict['seg_logit']  # [N, num_classes]
            
            # Get frustum predictions from auxiliary heads
            # Note: Auxiliary heads are already forwarded in loss() method,
            # but we need to forward them again here to get predictions for consistency loss
            if self.with_auxiliary_head:
                for aux_idx, aux_head in enumerate(self.auxiliary_head):
                    # Forward the auxiliary head to get predictions
                    # This is safe because forward() is idempotent for auxiliary heads
                    aux_voxel_dict = aux_head(voxel_dict)
                    if 'seg_logit' in aux_voxel_dict:
                        frustum_pred = aux_voxel_dict['seg_logit']  # [B, num_classes, H, W]
                        
                        # Map frustum predictions to point predictions using point coordinates
                        frustum_pred_permuted = frustum_pred.permute(0, 2, 3, 1).contiguous()
                        frustum_point_pred = frustum_pred_permuted[
                            pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2]
                        ]  # [N, num_classes]
                        
                        # Compute prediction consistency loss
                        if self.prediction_consistency_loss is not None:
                            pred_loss = self.prediction_consistency_loss(
                                point_pred, frustum_point_pred
                            )
                            losses[f'loss_pred_consist_pf_{aux_idx}'] = \
                                pred_loss * self.prediction_consistency_weight
        
        # Add offset regularization losses
        if len(offset_reg_losses) > 0:
            total_offset_reg = sum(offset_reg_losses) / len(offset_reg_losses)
            losses['loss_offset_reg'] = total_offset_reg * self.offset_reg_weight
        
        return losses
