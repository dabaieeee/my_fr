_base_ = [
    '../_base_/datasets/semantickitti_seg.py', '../_base_/schedules/onecycle-150k.py',
    '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['frnet.datasets', 'frnet.datasets.transforms', 'frnet.models'],
    allow_failed_imports=False)

model = dict(
    type='DualPathFRNet',
    data_preprocessor=dict(
        type='FrustumRangePreprocessor',
        H=64, W=512, fov_up=3.0, fov_down=-25.0, ignore_index=19),
    
    # ========== FFE (FrustumFeatureEncoder) - 用于语义路径 ==========
    voxel_encoder=dict(
        type='FrustumFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256, 256),
        with_distance=True,
        with_cluster_center=True,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        with_pre_norm=True,
        feat_compression=16),
    
    # ========== Geometry Encoder - 用于几何路径 ==========
    geometry_encoder=dict(
        type='GeometryEncoder',
        in_channels=3,  # xyz only
        feat_channels=(32, 64, 128),  # 较小的通道数，保持局部性
        with_normals=True,
        with_curvature=True,
        with_distance=True,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        k_neighbors=10),
    
    # ========== Dual-Path Backbone ==========
    backbone=dict(
        type='DualPathBackbone',
        output_shape=(64, 512),
        
        # Geometry Encoder配置（与上面的geometry_encoder配置保持一致）
        geometry_encoder=dict(
            type='GeometryEncoder',
            in_channels=3,
            feat_channels=(32, 64, 128),
            with_normals=True,
            with_curvature=True,
            with_distance=True,
            norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
            k_neighbors=10),
        
        # Semantic Backbone配置（保留FFE和FPFM）
        semantic_backbone=dict(
            type='FRNetBackbone',
            in_channels=16,  # FFE输出通道数
            point_in_channels=384,  # FFE point特征通道数
            depth=34,
            stem_channels=128,
            num_stages=4,
            out_channels=(128, 128, 128, 128),
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            fuse_channels=(256, 128),  # 语义路径输出通道数
            norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
            point_norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
            act_cfg=dict(type='HSwish', inplace=True),
            voxel_3d_channels=None),  # 可选：如果使用3D体素编码器
        
        # Cross-Gated Fusion配置
        fusion_cfg=dict(
            type='CrossGatedFusion',
            geo_channels=128,  # Geometry Encoder输出通道数
            sem_channels=128,  # Semantic Backbone输出通道数（fuse_channels[-1]）
            out_channels=128,  # 融合后输出通道数
            fusion_type='point',  # 点级别融合
            norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
            act_cfg=dict(type='Sigmoid'))),
    
    # ========== Decode Head ==========
    decode_head=dict(
        type='FRHead',
        in_channels=128,
        middle_channels=(128, 256, 128, 64),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        channels=64,
        dropout_ratio=0,
        num_classes=20,
        ignore_index=19,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        conv_seg_kernel_size=1),
    
    # ========== Auxiliary Heads (可选) ==========
    auxiliary_head=[
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=2),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=3),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=4),
    ])

