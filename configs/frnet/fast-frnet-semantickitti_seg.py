_base_ = [
    '../_base_/datasets/semantickitti_seg.py', '../_base_/models/fast_frnet.py',
    '../_base_/schedules/onecycle-50k.py', '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['frnet.datasets', 'frnet.datasets.transforms', 'frnet.models'],
    allow_failed_imports=False)

model = dict(
    data_preprocessor=dict(
        H=32, W=360, fov_up=3.0, fov_down=-25.0, ignore_index=19),  # Fast-FRNet统一使用32×360分辨率
    backbone=dict(output_shape=(32, 360)),
    decode_head=dict(num_classes=20, ignore_index=19),
    # Fast-FRNet可以使用更少的auxiliary head以提升速度
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
        dict(
            type='BEVDistillHead',
            point_channels=128,
            frustum_channels=128,
            voxel_channels=256,  # 若未启用体素编码器，会自动跳过教师视角
            bev_channels=96,
            loss_l1_weight=0.4,
            loss_frustum_weight=0.4,
            loss_nce_weight=0.05,
            temperature=0.2,
            with_frustum_view=True,
            detach_teacher=True,
            num_classes=1,
            channels=96,
            conv_seg_kernel_size=1,
            ignore_index=19),
    ])

