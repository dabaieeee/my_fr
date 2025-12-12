# 带课程学习的FRNet配置文件
# 该配置展示了如何在训练中启用课程学习策略

_base_ = [
    '../_base_/datasets/semantickitti_seg.py', 
    '../_base_/models/frnet.py',
    '../_base_/schedules/onecycle-50k.py', 
    '../_base_/default_runtime.py'
]

# 导入自定义模块（包括课程学习Hook）
custom_imports = dict(
    imports=[
        'frnet.datasets', 
        'frnet.datasets.transforms', 
        'frnet.models',
        'frnet.hooks'  # 导入课程学习Hook
    ],
    allow_failed_imports=False)

# 模型配置：启用课程学习
model = dict(
    data_preprocessor=dict(
        H=64, W=512, fov_up=3.0, fov_down=-25.0, ignore_index=19),
    backbone=dict(output_shape=(64, 512)),
    decode_head=dict(
        num_classes=20, 
        ignore_index=19,
        use_curriculum_learning=True  # 启用课程学习
    ))

# 添加课程学习Hook到default_hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=1000, 
        save_best='miou', 
        save_last=True, 
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'),
    # ========== 课程学习Hook配置 ==========
    curriculum_learning=dict(
        type='CurriculumLearningHook',
        strategy='adaptive',  # 课程学习策略: 'loss_based', 'epoch_based', 'adaptive'
        start_epoch=0,  # 开始应用课程学习的轮次
        end_epoch=None,  # 完全应用课程学习的轮次（None表示使用总训练轮次）
        warmup_iters=1000,  # 预热迭代次数，在预热期间不使用课程学习
        difficulty_threshold=0.5,  # 困难样本的损失阈值（相对于batch均值的标准差）
        min_weight=0.1,  # 困难样本的最小权重（训练初期）
        max_weight=1.0,  # 困难样本的最大权重（训练后期）
        schedule_type='linear',  # 调度类型: 'linear', 'cosine', 'exponential'
        update_interval=100,  # 更新课程学习参数的间隔（迭代次数）
        log_interval=500  # 记录课程学习信息的间隔
    ))

# 课程学习策略说明：
# 1. 'loss_based': 基于损失值的课程学习，根据每个样本的损失值动态调整权重
# 2. 'epoch_based': 基于训练轮次的课程学习，按照固定的轮次进度调整
# 3. 'adaptive': 自适应课程学习，结合损失趋势和训练轮次，推荐使用
#
# 调度类型说明：
# - 'linear': 线性增加困难样本权重
# - 'cosine': 余弦曲线增加（开始慢，中间快，结束慢）
# - 'exponential': 指数增加（开始慢，后期快）
#
# 参数调优建议：
# - difficulty_threshold: 根据数据集调整，值越大，越少的样本被标记为困难样本
# - min_weight: 训练初期困难样本的权重，建议0.1-0.3
# - max_weight: 训练后期困难样本的权重，通常为1.0
# - warmup_iters: 建议设置为总迭代次数的5-10%

