model = dict(
    type='DualPathFRNet',
    data_preprocessor=dict(type='FrustumRangePreprocessor'),
    
    # Geometry Encoder: Structure-preserving geometric features
    geometry_encoder=dict(
        type='GeometryEncoder',
        in_channels=3,  # xyz only
        feat_channels=[64, 128, 128],
        with_normals=True,  # Compute surface normals
        with_curvature=True,  # Compute curvature features
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        k_neighbors=10),  # Number of neighbors for normal/curvature computation
    
    # Semantic Encoder: Context-aware semantic features (uses FFE)
    semantic_encoder=dict(
        type='SemanticEncoder',
        ffe_config=dict(
            type='FrustumFeatureEncoder',
            in_channels=4,  # xyz + intensity
            feat_channels=(64, 128, 256, 256),
            with_distance=True,
            with_cluster_center=True,
            norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
            with_pre_norm=True,
            feat_compression=16)),
    
    # Dual-Path Backbone with Cross-Gated Fusion and FPFM
    backbone=dict(
        type='DualPathFRNetBackbone',
        geo_channels=128,  # Output channels from geometry encoder
        sem_channels=16,  # Output channels from semantic encoder (after compression)
        output_shape=(64, 2048),  # Range image shape [H, W]
        depth=34,  # ResNet depth
        stem_channels=128,
        num_stages=4,
        out_channels=(128, 128, 128, 128),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        fuse_channels=(256, 128),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        point_norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='HSwish', inplace=True),
        use_cross_gated_fusion=True,  # Enable cross-gated fusion
        fusion_channels=128),  # Channels after fusion
    
    # Decode Head
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

