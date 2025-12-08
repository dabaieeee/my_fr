# FRNet配置 - 使用多尺度体素编码器
# 此配置文件展示了如何使用多尺度体素特征提取

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
    # 使用多尺度体素编码器
    # 多尺度体素编码器通过并行处理多个不同分辨率的体素网格，实现对点云的多尺度特征提取
    # 优势：能够同时捕获细节（高分辨率）和全局结构（低分辨率）信息
    voxel_3d_encoder=dict(
        type='MultiScaleVoxelFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256),
        voxel_sizes=((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4)),  # 三种不同分辨率
        # voxel_sizes=((0.4, 0.4, 0.4), (0.6, 0.6, 0.6), (0.8, 0.8, 0.8)),  # 三种不同分辨率
        point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True),
        fusion_method='concat',  # 融合方法：'concat'（拼接融合）或 'attention'（注意力融合）
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
        conv_seg_kernel_size=1))

