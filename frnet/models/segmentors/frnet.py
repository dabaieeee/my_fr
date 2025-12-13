from typing import Dict

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor



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
                 use_multi_scale_voxel: bool = False,
                 multi_scale_voxel_config: OptConfigType = None,
                 diffusion_refiner: OptConfigType = None,
                 diffusion_point_refiner: OptConfigType = None,
                 diffusion_loss_weight: float = 0.1,
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
        # 如果use_multi_scale_voxel=True，优先使用多尺度体素编码器
        self.voxel_3d_encoder = None
        if use_multi_scale_voxel and multi_scale_voxel_config is not None:
            # 使用多尺度体素编码器
            self.voxel_3d_encoder = MODELS.build(multi_scale_voxel_config)
        elif voxel_3d_encoder is not None:
            # 使用单尺度体素编码器（默认）
            self.voxel_3d_encoder = MODELS.build(voxel_3d_encoder)
        
        # Diffusion特征精炼器（可选）
        self.diffusion_refiner = None
        if diffusion_refiner is not None:
            self.diffusion_refiner = MODELS.build(diffusion_refiner)
        
        self.diffusion_point_refiner = None
        if diffusion_point_refiner is not None:
            self.diffusion_point_refiner = MODELS.build(diffusion_point_refiner)
        
        self.diffusion_loss_weight = diffusion_loss_weight

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
        
        # Diffusion特征精炼（在backbone之后）
        if self.diffusion_refiner is not None:
            # 对frustum特征进行refinement
            frustum_feats = voxel_dict['voxel_feats'][0]  # [B, C, H, W]
            if self.training:
                # 训练模式：计算diffusion loss
                _, _, diffusion_loss = self.diffusion_refiner(frustum_feats, training=True)
                voxel_dict['diffusion_loss'] = diffusion_loss
            else:
                # 推理模式：refine特征
                refined_frustum_feats = self.diffusion_refiner(frustum_feats, training=False)
                voxel_dict['voxel_feats'][0] = refined_frustum_feats
        
        if self.diffusion_point_refiner is not None:
            # 对point特征进行refinement
            point_feats = voxel_dict['point_feats_backbone'][0]  # [N, C]
            if self.training:
                _, _, diffusion_point_loss = self.diffusion_point_refiner(point_feats, training=True)
                voxel_dict['diffusion_point_loss'] = diffusion_point_loss
            else:
                refined_point_feats = self.diffusion_point_refiner(point_feats, training=False)
                voxel_dict['point_feats_backbone'][0] = refined_point_feats
        
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
        losses = dict()
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                      batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)
        
        # 添加diffusion loss
        if 'diffusion_loss' in voxel_dict:
            losses['loss_diffusion'] = self.diffusion_loss_weight * voxel_dict['diffusion_loss']
        if 'diffusion_point_loss' in voxel_dict:
            losses['loss_diffusion_point'] = self.diffusion_loss_weight * voxel_dict['diffusion_point_loss']
        
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
