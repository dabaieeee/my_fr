# Enhanced FRNet configuration with adaptive fusion and new losses
# This is an example configuration showing how to use the new features

model = dict(
    type='FRNet',
    data_preprocessor=dict(type='FrustumRangePreprocessor'),
    voxel_encoder=dict(
        type='FrustumFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256, 256),
        with_distance=True,
        with_cluster_center=True,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        with_pre_norm=True,
        feat_compression=16),
    # 3D体素编码器配置
    voxel_3d_encoder=dict(
        type='VoxelFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256),
        voxel_size=(0.2, 0.2, 0.2),
        point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True),
        use_sparse=True),
    backbone=dict(
        type='FRNetBackbone',
        in_channels=16,
        point_in_channels=384,
        depth=34,
        stem_channels=128,
        num_stages=4,
        out_channels=(128, 128, 128, 128),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        fuse_channels=(256, 128),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        point_norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='HSwish', inplace=True),
        voxel_mid_fusion_indices=(-1, 2),
        voxel_3d_channels=256,
        # ========== 新增：启用自适应特征融合 ==========
        use_adaptive_fusion=True,  # 设置为True启用自适应融合，False使用门控融合
    ),
    decode_head=dict(
        type='FRHead',
        in_channels=128,
        middle_channels=(128, 256, 128, 64),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        channels=64,
        dropout_ratio=0,
        # ========== 新增：使用Focal Loss进行困难样本挖掘 ==========
        # 方式1：完全替换交叉熵损失
        loss_ce=dict(
            type='FocalLoss',
            alpha=0.25,  # 类别权重
            gamma=2.0,  # 聚焦参数，值越大越关注困难样本
            reduction='mean',
            ignore_index=255,
            loss_weight=1.0),
        # 方式2：混合使用Focal Loss和交叉熵损失（推荐）
        # loss_ce=dict(
        #     type='mmdet.CrossEntropyLoss',
        #     use_sigmoid=False,
        #     class_weight=None,
        #     loss_weight=0.8),  # 降低交叉熵权重
        # loss_focal=dict(
        #     type='FocalLoss',
        #     alpha=0.25,
        #     gamma=2.0,
        #     reduction='mean',
        #     ignore_index=255,
        #     loss_weight=0.2),  # 添加Focal Loss
        conv_seg_kernel_size=1))

# ========== 训练配置中的损失函数增强 ==========
# 如果需要使用特征一致性损失，需要在segmentor中添加
# 这需要在segmentor的loss函数中实现，可以参考以下方式：
# 
# 在segmentor的loss函数中：
# 1. 提取frustum特征和point特征
# 2. 计算一致性损失
# 3. 将一致性损失添加到总损失中
#
# 示例代码（需要在segmentor中实现）：
# consistency_loss = dict(
#     type='FeatureConsistencyLoss',
#     loss_weight=0.1,  # 一致性损失权重，建议0.05-0.2
#     loss_type='both',  # 'l2', 'cosine', 或 'both'
#     temperature=0.1)
#
# 在loss函数中使用：
# from mmdet3d.registry import MODELS
# consistency_loss_fn = MODELS.build(consistency_loss)
# loss_consistency = consistency_loss_fn(frustum_feats, point_feats)
# losses['loss_consistency'] = loss_consistency

