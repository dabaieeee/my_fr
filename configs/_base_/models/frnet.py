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
    # 3D体素编码器（可选，使用稀疏模式以减少显存占用）
    voxel_3d_encoder=dict(
        type='VoxelFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256),
        voxel_size=(0.2, 0.2, 0.2),  # 增大体素尺寸以减少显存占用
        point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True),
        use_sparse=True),  # 使用稀疏模式，避免创建密集3D网格
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
        voxel_3d_channels=256),  # 体素特征通道数，需要与voxel_3d_encoder的输出通道匹配
    decode_head=dict(
        type='FRHead',
        in_channels=128,
        middle_channels=(128, 256, 128, 64),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        channels=64,
        dropout_ratio=0,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        conv_seg_kernel_size=1),
    # Feature-level Consistency 配置
    use_feature_consistency=False,  # 是否启用特征级一致性约束
    feature_consistency_loss=dict(
        type='FeatureLevelConsistencyLoss',
        loss_weight=1.0,
        loss_type='mse'),  # 'mse', 'cosine', 'kl'
    feature_consistency_stages=[1, 2, 3],  # 在哪些stage应用特征一致性
    feature_consistency_weight=0.1,  # 特征一致性损失的权重
    # Prediction-level Consistency 配置
    use_prediction_consistency=False,  # 是否启用预测级一致性约束
    prediction_consistency_loss=dict(
        type='PredictionConsistencyLoss',
        loss_weight=1.0,
        loss_type='kl'),  # 'kl', 'js', 'ce'
    prediction_consistency_weight=0.1,  # 预测一致性损失的权重
    # Offset Network 配置
    frustum_offset_range=3,  # Frustum分支的偏移范围（像素）
    voxel_offset_range=2,  # Voxel分支的偏移范围（体素）
    offset_reg_weight=0.01)  # 偏移正则化损失的权重
