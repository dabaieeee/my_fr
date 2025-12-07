# Fast-FRNet配置：使用更小的模型深度和通道数以实现更快的推理速度
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
    # Fast-FRNet可以不使用3D体素编码器以进一步提升速度
    # 如果需要使用，可以取消下面的注释
    # voxel_3d_encoder=dict(
    #     type='VoxelFeatureEncoder',
    #     in_channels=4,
    #     feat_channels=(64, 128, 256),
    #     voxel_size=(0.2, 0.2, 0.2),
    #     point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
    #     norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
    #     act_cfg=dict(type='ReLU', inplace=True),
    #     use_sparse=True),
    backbone=dict(
        type='FRNetBackbone',
        in_channels=16,
        point_in_channels=384,
        depth=18,  # 使用ResNet-18而不是ResNet-34，减少参数量和计算量
        stem_channels=128,
        num_stages=4,
        out_channels=(128, 128, 128, 128),  # 可以进一步减小通道数以提升速度
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        fuse_channels=(256, 128),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        point_norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='HSwish', inplace=True)),
        # voxel_3d_channels=256),  # 如果使用体素编码器，取消注释
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

