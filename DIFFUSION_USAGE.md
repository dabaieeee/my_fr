# Diffusion模块使用指南

## 一、快速开始

### 1.1 基本配置

在配置文件中添加diffusion模块：

```python
# 在 model 配置中添加
model = dict(
    type='FRNet',
    # ... 其他配置 ...
    
    # 添加diffusion refiner
    diffusion_refiner=dict(
        type='DiffusionFeatureRefiner',
        in_channels=128,  # 与backbone输出通道匹配
        refiner_type='frustum',  # 'frustum' 或 'point'
        num_timesteps=1000,
        beta_schedule='cosine',
        use_ddim=True,
        ddim_steps=50,  # 推理时使用50步（比1000步快很多）
        diffusion_loss_weight=0.1,  # diffusion loss权重
    ),
)
```

### 1.2 训练

训练时会自动计算diffusion loss并加入到总loss中：

```bash
python train.py configs/frnet/frnet-semantickitti_seg.py
```

训练时的loss包括：
- `loss_ce`: 分割交叉熵损失
- `loss_diffusion`: Diffusion损失（如果启用）

### 1.3 推理

推理时会自动使用DDIM采样refine特征，无需额外配置。

## 二、配置选项

### 2.1 DiffusionFeatureRefiner参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_channels` | int | - | 输入特征通道数（需与backbone输出匹配） |
| `refiner_type` | str | 'frustum' | 'frustum'（2D）或'point'（1D） |
| `num_timesteps` | int | 1000 | 扩散步数（训练时） |
| `beta_schedule` | str | 'cosine' | 噪声调度：'linear'或'cosine' |
| `time_emb_dim` | int | 128 | 时间embedding维度 |
| `base_channels` | int | 64 | U-Net基础通道数 |
| `num_res_blocks` | int | 2 | 残差块数量 |
| `use_ddim` | bool | True | 是否使用DDIM加速推理 |
| `ddim_steps` | int | 50 | DDIM采样步数（推理时） |

### 2.2 FRNet参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `diffusion_refiner` | dict | None | Frustum特征refiner配置 |
| `diffusion_point_refiner` | dict | None | Point特征refiner配置 |
| `diffusion_loss_weight` | float | 0.1 | Diffusion loss权重 |

## 三、使用示例

### 示例1：只对Frustum特征使用Diffusion

```python
model = dict(
    type='FRNet',
    # ... 其他配置 ...
    diffusion_refiner=dict(
        type='DiffusionFeatureRefiner',
        in_channels=128,
        refiner_type='frustum',
        num_timesteps=1000,
        beta_schedule='cosine',
        use_ddim=True,
        ddim_steps=50,
    ),
    diffusion_loss_weight=0.1,
)
```

### 示例2：同时对Frustum和Point特征使用Diffusion

```python
model = dict(
    type='FRNet',
    # ... 其他配置 ...
    diffusion_refiner=dict(
        type='DiffusionFeatureRefiner',
        in_channels=128,
        refiner_type='frustum',
        num_timesteps=1000,
        beta_schedule='cosine',
        use_ddim=True,
        ddim_steps=50,
    ),
    diffusion_point_refiner=dict(
        type='DiffusionFeatureRefiner',
        in_channels=128,
        refiner_type='point',
        num_timesteps=1000,
        beta_schedule='cosine',
        use_ddim=True,
        ddim_steps=50,
    ),
    diffusion_loss_weight=0.1,
)
```

### 示例3：完整配置文件

参考 `configs/_base_/models/frnet_with_diffusion.py`

## 四、训练策略

### 4.1 两阶段训练（推荐）

**阶段1：冻结backbone，只训练diffusion**
```python
# 在训练脚本中冻结backbone参数
for name, param in model.backbone.named_parameters():
    param.requires_grad = False
```

**阶段2：端到端微调**
```python
# 解冻所有参数
for param in model.parameters():
    param.requires_grad = True
```

### 4.2 损失权重调整

根据训练情况调整`diffusion_loss_weight`：
- 初期：0.1-0.2（让模型先学习基本分割）
- 中期：0.2-0.5（逐步增加diffusion影响）
- 后期：0.1-0.2（保持稳定）

## 五、性能优化

### 5.1 推理加速

- 使用DDIM：`use_ddim=True`
- 减少采样步数：`ddim_steps=50`（可尝试20-100）
- 使用混合精度：`fp16=True`

### 5.2 显存优化

- 减少`base_channels`：64 → 32
- 减少`num_res_blocks`：2 → 1
- 使用梯度检查点（如果支持）

## 六、常见问题

### Q1: Diffusion loss很大，导致训练不稳定？

**A**: 降低`diffusion_loss_weight`，从0.1降到0.05或0.01。

### Q2: 推理速度太慢？

**A**: 
- 确保`use_ddim=True`
- 减少`ddim_steps`（如从50降到20）
- 只在关键阶段使用diffusion（如只在最后一个stage）

### Q3: 显存不足？

**A**:
- 减少`base_channels`
- 减少`num_res_blocks`
- 使用更小的batch size

### Q4: 效果不明显？

**A**:
- 增加`num_timesteps`（如1000 → 2000）
- 调整`beta_schedule`（尝试'cosine'）
- 增加`diffusion_loss_weight`
- 检查特征通道数是否匹配

## 七、进阶使用

### 7.1 自定义U-Net结构

修改`diffusion_refiner.py`中的`_build_frustum_unet()`方法，自定义U-Net结构。

### 7.2 条件Diffusion

可以基于ground truth或辅助信息进行条件生成，修改`DiffusionFeatureRefiner`的forward方法。

### 7.3 多尺度Diffusion

在不同stage应用不同强度的diffusion，在backbone中集成。

## 八、实验建议

1. **基线对比**：先训练不带diffusion的模型作为基线
2. **渐进集成**：先只对frustum特征使用，验证效果后再考虑point特征
3. **超参数搜索**：系统性地搜索`diffusion_loss_weight`、`ddim_steps`等参数
4. **消融实验**：验证diffusion在不同位置的效果

## 九、参考实现

- 基础代码：`frnet/models/diffusion/`
- 配置文件：`configs/_base_/models/frnet_with_diffusion.py`
- 集成代码：`frnet/models/segmentors/frnet.py`

