_base_ = [
    '../_base_/datasets/semantickitti_seg.py', '../_base_/models/frnet.py',
    '../_base_/schedules/onecycle-150k.py', '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['frnet.datasets', 'frnet.datasets.transforms', 'frnet.models'],
    allow_failed_imports=False)

model = dict(
    data_preprocessor=dict(
        H=64, W=512, fov_up=3.0, fov_down=-25.0, ignore_index=19),
    backbone=dict(output_shape=(64, 512)),
    decode_head=dict(num_classes=20, ignore_index=19),
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
    ],
    # ============ Feature-level Consistency 配置 ============
    # 启用特征级一致性约束
    use_feature_consistency=True,
    feature_consistency_loss=dict(
        type='FeatureLevelConsistencyLoss',
        loss_weight=1.0,
        loss_type='cosine'),  # 可选: 'mse', 'cosine', 'kl'
    feature_consistency_stages=[1, 2, 3],  # 在 stage 1, 2, 3 应用
    feature_consistency_weight=0.1,  # 特征一致性损失权重
    
    # ============ Prediction-level Consistency 配置 ============
    # 启用预测级一致性约束
    use_prediction_consistency=True,
    prediction_consistency_loss=dict(
        type='PredictionConsistencyLoss',
        loss_weight=1.0,
        loss_type='kl'),  # 可选: 'kl', 'js', 'ce'
    prediction_consistency_weight=0.1,  # 预测一致性损失权重
    
    # ============ Offset Network 配置 ============
    frustum_offset_range=3,  # Frustum 偏移范围（像素）
    voxel_offset_range=2,    # Voxel 偏移范围（体素）
    offset_reg_weight=0.01,  # 偏移正则化损失权重
)

# ============ 使用说明 ============
# 1. 仅启用特征级一致性:
#    use_feature_consistency=True, use_prediction_consistency=False
#
# 2. 仅启用预测级一致性:
#    use_feature_consistency=False, use_prediction_consistency=True
#
# 3. 同时启用两种一致性:
#    use_feature_consistency=True, use_prediction_consistency=True
#
# 4. 仅学习特征对齐（不参与损失）:
#    use_feature_consistency=False, use_prediction_consistency=False
#
# 5. 命令行覆盖配置示例:
#    PORT=44171 CUDA_VISIBLE_DEVICES=4,5 bash dist_train.sh \
#        configs/frnet/frnet-semantickitti_seg_with_consistency.py 2 \
#        --cfg-options model.feature_consistency_weight=0.15 \
#                      model.prediction_consistency_weight=0.05

