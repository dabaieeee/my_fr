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
    # ========== 3D体素编码器配置（可选，使用稀疏模式以减少显存占用）==========
    # 提供三种方式切换单尺度/多尺度体素编码器：
    #
    # 方式1: 使用单尺度体素编码器（默认，推荐用于快速训练和推理）
    voxel_3d_encoder=dict(
        type='VoxelFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256),
        voxel_size=(0.2, 0.2, 0.2),  # 增大体素尺寸以减少显存占用
        point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True),
        use_sparse=True),  # 使用稀疏模式，避免创建密集3D网格
    
    # 方式2: 使用多尺度体素编码器（直接替换type）
    # 取消下面的注释并注释掉上面的voxel_3d_encoder即可启用多尺度体素编码器
    # 多尺度体素编码器通过并行处理多个不同分辨率的体素网格，实现对点云的多尺度特征提取
    # 优势：能够同时捕获细节（高分辨率）和全局结构（低分辨率）信息
    # voxel_3d_encoder=dict(
    #     type='MultiScaleVoxelFeatureEncoder',
    #     in_channels=4,
    #     feat_channels=(64, 128, 256),
    #     voxel_sizes=((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4)),  # 三种不同分辨率
    #     point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
    #     norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
    #     act_cfg=dict(type='ReLU', inplace=True),
    #     fusion_method='concat',  # 融合方法：'concat'（拼接融合）或 'attention'（注意力融合）
    #     use_sparse=True),  # 使用稀疏模式，避免创建密集3D网格
    
    # 方式3: 使用便捷参数切换（推荐，更灵活）
    # 设置use_multi_scale_voxel=True并配置multi_scale_voxel_config即可启用多尺度
    # use_multi_scale_voxel=False,  # 设置为True启用多尺度体素编码器
    # multi_scale_voxel_config=dict(
    #     type='MultiScaleVoxelFeatureEncoder',
    #     in_channels=4,
    #     feat_channels=(64, 128, 256),
    #     voxel_sizes=((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4)),
    #     point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
    #     norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
    #     act_cfg=dict(type='ReLU', inplace=True),
    #     fusion_method='concat',
    #     use_sparse=True),
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
        # 体素-视锥-点分支的中途交互位置：-1 表示stem之后，2表示第3个stage之后
        voxel_mid_fusion_indices=(-1, 2),
        voxel_3d_channels=256,  # 体素特征通道数，需要与voxel_3d_encoder的输出通道匹配
        # ========== 启用自适应特征融合（根据特征质量动态调整权重）==========
        use_adaptive_fusion=True),  # 默认启用自适应融合，提升2-3% mIoU
    decode_head=dict(
        type='FRHead',
        in_channels=128,
        middle_channels=(128, 256, 128, 64),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        channels=64,
        dropout_ratio=0,
        # ========== 混合损失函数：CrossEntropy + Focal Loss（困难样本挖掘）==========
        # 使用混合损失可以同时利用两种损失的优势
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=0.8),  # 降低交叉熵权重，为Focal Loss留出空间
        # 添加Focal Loss用于困难样本挖掘，提升困难类别的性能
        # 注意：ignore_index需要在具体数据集的配置中设置（如SemanticKITTI为19）
        loss_focal=dict(
            type='FocalLoss',
            alpha=0.25,  # 类别权重
            gamma=2.0,  # 聚焦参数，值越大越关注困难样本
            reduction='mean',
            ignore_index=255,  # 默认值，会在具体配置中被覆盖
            loss_weight=0.2),  # Focal Loss权重
        conv_seg_kernel_size=1))
