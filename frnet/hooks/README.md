# 课程学习（Curriculum Learning）使用说明

## 概述

课程学习是一种训练策略，通过从简单样本到困难样本的渐进式学习，提高模型的训练效果和泛化能力。本实现针对三维点云语义分割任务，支持多种课程学习策略。

## 功能特点

1. **多种课程学习策略**：
   - `loss_based`: 基于损失值的课程学习
   - `epoch_based`: 基于训练轮次的课程学习
   - `adaptive`: 自适应课程学习（推荐）

2. **灵活的调度方式**：
   - `linear`: 线性增加困难样本权重
   - `cosine`: 余弦曲线调度
   - `exponential`: 指数调度

3. **样本级别的权重调整**：
   - 自动识别困难样本和简单样本
   - 动态调整样本权重
   - 支持预热期设置

## 使用方法

### 1. 在配置文件中启用课程学习

```python
# 在模型配置中启用课程学习
model = dict(
    decode_head=dict(
        use_curriculum_learning=True  # 启用课程学习
    ))

# 在default_hooks中添加课程学习Hook
default_hooks = dict(
    # ... 其他hooks ...
    curriculum_learning=dict(
        type='CurriculumLearningHook',
        strategy='adaptive',  # 推荐使用自适应策略
        start_epoch=0,
        end_epoch=None,  # None表示使用总训练轮次
        warmup_iters=1000,
        difficulty_threshold=0.5,
        min_weight=0.1,
        max_weight=1.0,
        schedule_type='linear',
        update_interval=100,
        log_interval=500
    ))
```

### 2. 导入自定义模块

确保在配置文件中导入hooks模块：

```python
custom_imports = dict(
    imports=[
        'frnet.datasets', 
        'frnet.datasets.transforms', 
        'frnet.models',
        'frnet.hooks'  # 导入课程学习Hook
    ],
    allow_failed_imports=False)
```

### 3. 运行训练

使用标准训练命令：

```bash
python train.py configs/frnet/frnet-semantickitti_seg-curriculum.py
```

## 参数说明

### 核心参数

- **strategy** (str): 课程学习策略
  - `'loss_based'`: 完全基于损失值判断样本难度
  - `'epoch_based'`: 基于训练轮次线性调整
  - `'adaptive'`: 结合损失趋势和训练轮次（推荐）

- **difficulty_threshold** (float): 困难样本的损失阈值
  - 值越大，越少的样本被标记为困难样本
  - 建议范围：0.3-0.7
  - 默认：0.5

- **min_weight** (float): 困难样本的最小权重
  - 训练初期困难样本的权重
  - 建议范围：0.1-0.3
  - 默认：0.1

- **max_weight** (float): 困难样本的最大权重
  - 训练后期困难样本的权重
  - 通常设置为1.0
  - 默认：1.0

### 调度参数

- **schedule_type** (str): 权重增加的调度方式
  - `'linear'`: 线性增加（推荐，稳定）
  - `'cosine'`: 余弦曲线（开始慢，中间快，结束慢）
  - `'exponential'`: 指数增加（开始慢，后期快）

- **start_epoch** (int): 开始应用课程学习的轮次
  - 默认：0（从训练开始就应用）

- **end_epoch** (int): 完全应用课程学习的轮次
  - None表示使用总训练轮次
  - 建议设置为总训练轮次的80-90%

- **warmup_iters** (int): 预热迭代次数
  - 在预热期间不使用课程学习
  - 建议设置为总迭代次数的5-10%

### 更新和日志参数

- **update_interval** (int): 更新课程学习参数的间隔
  - 默认：100（每100次迭代更新一次）

- **log_interval** (int): 记录课程学习信息的间隔
  - 默认：500（每500次迭代记录一次）

## 工作原理

1. **样本难度评估**：
   - 计算每个点的损失值
   - 根据损失值的分布判断样本难度
   - 损失值高于阈值的样本被标记为困难样本

2. **权重计算**：
   - 简单样本：权重始终为1.0
   - 困难样本：权重从`min_weight`逐渐增加到`max_weight`
   - 权重增加速度由`curriculum_progress`和`schedule_type`控制

3. **损失计算**：
   - 使用加权损失：`weighted_loss = (loss_per_point * sample_weights).mean()`
   - 在训练初期，困难样本的贡献较小
   - 随着训练进行，困难样本的贡献逐渐增加

## 调优建议

1. **首次使用**：
   - 使用默认参数（`strategy='adaptive'`, `schedule_type='linear'`）
   - 观察训练日志中的课程学习进度

2. **如果训练不稳定**：
   - 增加`warmup_iters`
   - 降低`difficulty_threshold`
   - 使用`schedule_type='cosine'`

3. **如果收敛太慢**：
   - 降低`min_weight`（更早关注困难样本）
   - 使用`schedule_type='exponential'`
   - 减少`warmup_iters`

4. **如果过拟合**：
   - 增加`difficulty_threshold`（减少困难样本数量）
   - 延长课程学习周期（增加`end_epoch`）

## 示例配置文件

参考 `configs/frnet/frnet-semantickitti_seg-curriculum.py` 获取完整示例。

## 注意事项

1. 课程学习会增加一定的计算开销（主要是样本权重计算）
2. 建议在训练初期使用课程学习，后期可以关闭
3. 不同数据集可能需要调整`difficulty_threshold`
4. 课程学习的效果可能因数据集和模型而异，需要实验验证

## 参考文献

- Curriculum Learning: [Bengio et al., 2009]
- Self-Paced Learning: [Kumar et al., 2010]

