default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 断点续训
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=1000, save_best='miou', save_last=True, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# DDP 设置：启用未用参数检测，避免新增蒸馏/辅助头在极端 batch 下触发报错
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

log_level = 'INFO'
load_from = None
resume = False
