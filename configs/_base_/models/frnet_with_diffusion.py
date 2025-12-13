# FRNet with Diffusion Integration Example
# 这是一个示例配置文件，展示如何集成diffusion模块

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
    
    # 3D体素编码器
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
        voxel_3d_channels=256),
    
    # ========== Diffusion模块配置 ==========
    # 方案1: 特征增强型Diffusion（在backbone之后）
    diffusion_refiner=dict(
        type='DiffusionFeatureRefiner',
        in_channels=128,  # 与backbone输出通道匹配
        refiner_type='frustum',  # 'frustum' 或 'point'
        num_timesteps=1000,  # 扩散步数
        beta_schedule='cosine',  # 'linear' 或 'cosine'
        time_emb_dim=128,
        base_channels=64,
        num_res_blocks=2,
        use_ddim=True,  # 使用DDIM加速推理
        ddim_steps=50,  # DDIM采样步数（推理时）
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True)),
    
    # 可选：点特征refiner
    # diffusion_point_refiner=dict(
    #     type='DiffusionFeatureRefiner',
    #     in_channels=128,
    #     refiner_type='point',
    #     num_timesteps=1000,
    #     beta_schedule='cosine',
    #     time_emb_dim=128,
    #     base_channels=64,
    #     num_res_blocks=2,
    #     use_ddim=True,
    #     ddim_steps=50,
    #     norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    #     act_cfg=dict(type='ReLU', inplace=True)),
    
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

