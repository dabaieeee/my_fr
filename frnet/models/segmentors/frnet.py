from typing import Dict

import torch.nn.functional as F
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
                 imitation_loss_cfg: OptConfigType = None,
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
        self.imitation_loss_cfg = imitation_loss_cfg or {}
        
        # 3D voxel encoder (体素编码器)
        # 如果use_multi_scale_voxel=True，优先使用多尺度体素编码器
        self.voxel_3d_encoder = None
        if use_multi_scale_voxel and multi_scale_voxel_config is not None:
            # 使用多尺度体素编码器
            self.voxel_3d_encoder = MODELS.build(multi_scale_voxel_config)
        elif voxel_3d_encoder is not None:
            # 使用单尺度体素编码器（默认）
            self.voxel_3d_encoder = MODELS.build(voxel_3d_encoder)

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
        losses: Dict[str, Tensor] = {}

        # -------- 主分支分割 --------
        voxel_dict = self.decode_head(voxel_dict)
        main_seg_logit = voxel_dict.get('seg_logit')
        if main_seg_logit is not None:
            voxel_dict['main_seg_logit'] = main_seg_logit
        losses.update(
            self.decode_head.loss(voxel_dict, batch_data_samples,
                                  self.train_cfg))

        # -------- 辅助分支（多尺度 frustum 等）--------
        frustum_logits = []
        if self.with_auxiliary_head:
            for idx, aux_head in enumerate(self.auxiliary_head):
                voxel_dict = aux_head(voxel_dict)
                aux_loss = aux_head.loss(voxel_dict, batch_data_samples,
                                         self.train_cfg)
                # 标注来源，方便日志区分
                losses.update({f'aux{idx}.{k}': v for k, v in aux_loss.items()})

                # 仅对开启模仿的辅助头记录 logits
                if getattr(aux_head, 'enable_imitation', False):
                    aux_pts_logit = self._gather_point_logits(voxel_dict)
                    if aux_pts_logit is not None:
                        frustum_logits.append((idx, aux_pts_logit))

        # -------- 模仿损失（XMUDa 风格）--------
        if getattr(self, 'imitation_loss_cfg', None):
            lambda_im = self.imitation_loss_cfg.get('weight', 0.0)
            temperature = self.imitation_loss_cfg.get('temperature', 1.0)
            if lambda_im > 0 and main_seg_logit is not None and frustum_logits:
                for idx, aux_logit in frustum_logits:
                    # 对齐两侧长度，避免批间填充不一致
                    min_len = min(main_seg_logit.shape[0], aux_logit.shape[0])
                    if min_len == 0:
                        continue
                    main_cut = main_seg_logit[:min_len]
                    aux_cut = aux_logit[:min_len]
                    loss_im = self._symmetric_kl(main_cut, aux_cut,
                                                 temperature)
                    losses[f'loss_imitation_aux{idx}'] = loss_im * lambda_im

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

    # ==================== XMUDa 风格模仿损失实现 ==================== #
    def _gather_point_logits(self, voxel_dict: dict) -> Tensor:
        """将当前 voxel_dict 中的 seg_logit 对齐到点级别."""
        if 'seg_logit' not in voxel_dict or 'coors' not in voxel_dict:
            return None
        seg_logit = voxel_dict['seg_logit']
        coors = voxel_dict['coors']
        # 主分支已是点级别 logits，直接返回
        if seg_logit.dim() == 2:
            return seg_logit
        # 辅助 frustum 头：B x C x H x W
        if seg_logit.dim() == 4:
            seg_logit = seg_logit.permute(0, 2, 3, 1).contiguous()
            return seg_logit[coors[:, 0], coors[:, 1], coors[:, 2]]
        return None

    def _symmetric_kl(self, p: Tensor, q: Tensor,
                      temperature: float = 1.0) -> Tensor:
        """对称 KL (p||q + q||p) / 2."""
        p = p / temperature
        q = q / temperature
        kl_pq = F.kl_div(
            F.log_softmax(p, dim=1),
            F.softmax(q.detach(), dim=1),
            reduction='batchmean')
        kl_qp = F.kl_div(
            F.log_softmax(q, dim=1),
            F.softmax(p.detach(), dim=1),
            reduction='batchmean')
        return 0.5 * (kl_pq + kl_qp)
