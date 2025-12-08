# 多尺度体素编码器使用指南

## 概述

多尺度体素特征提取模块通过并行处理多个不同分辨率的体素网格，实现对点云的多尺度特征提取。该模块支持三种不同分辨率的体素化，能够同时捕获细节（高分辨率）和全局结构（低分辨率）信息。

## 切换方式

### 方式1: 直接修改配置文件（最简单）

在 `configs/_base_/models/frnet.py` 中，将 `voxel_3d_encoder` 的 `type` 从 `VoxelFeatureEncoder` 改为 `MultiScaleVoxelFeatureEncoder`：

```python
# 单尺度（默认）
voxel_3d_encoder=dict(
    type='VoxelFeatureEncoder',
    ...
)

# 多尺度（修改type即可）
voxel_3d_encoder=dict(
    type='MultiScaleVoxelFeatureEncoder',
    in_channels=4,
    feat_channels=(64, 128, 256),
    voxel_sizes=((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4)),  # 三种分辨率
    point_cloud_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
    norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
    act_cfg=dict(type='ReLU', inplace=True),
    fusion_method='concat',  # 或 'attention'
    use_sparse=True),
```

### 方式2: 使用独立的多尺度配置文件

使用 `configs/_base_/models/frnet_multiscale.py` 作为基础配置，该文件已经配置好了多尺度体素编码器。

### 方式3: 通过命令行参数切换

使用 `--cfg-options` 参数在运行时切换：

```bash
# 切换到多尺度体素编码器
python train.py configs/xxx.py \
    --cfg-options model.voxel_3d_encoder.type=MultiScaleVoxelFeatureEncoder \
    model.voxel_3d_encoder.voxel_sizes="[(0.1,0.1,0.1),(0.2,0.2,0.2),(0.4,0.4,0.4)]" \
    model.voxel_3d_encoder.fusion_method=concat

# 切换回单尺度体素编码器
python train.py configs/xxx.py \
    --cfg-options model.voxel_3d_encoder.type=VoxelFeatureEncoder \
    model.voxel_3d_encoder.voxel_size="(0.2,0.2,0.2)"
```

## 参数说明

### MultiScaleVoxelFeatureEncoder 参数

- `in_channels` (int): 输入特征通道数，默认为4
- `feat_channels` (Sequence[int]): 每个MLP层的特征通道数，默认为(64, 128, 256)
- `voxel_sizes` (Sequence[Sequence[float]]): 多个体素尺寸，每个为[x, y, z]
  - 默认: `((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4))`
  - 建议使用三种不同分辨率，例如：细粒度(0.1m)、中等(0.2m)、粗粒度(0.4m)
- `point_cloud_range` (Sequence[float]): 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
- `fusion_method` (str): 多尺度特征融合方法
  - `'concat'`: 拼接后通过MLP融合（默认，计算效率高）
  - `'attention'`: 使用注意力机制加权融合（可能获得更好的性能）
- `use_sparse` (bool): 是否使用稀疏模式，默认为True（推荐，减少显存占用）

## 性能对比

- **单尺度体素编码器**: 训练速度快，显存占用低，适合快速实验
- **多尺度体素编码器**: 能够捕获更丰富的多尺度特征，可能提升模型性能，但计算量和显存占用会增加

## 注意事项

1. 使用多尺度体素编码器时，确保 `backbone` 中的 `voxel_3d_channels` 参数与 `feat_channels[-1]` 匹配
2. 多尺度体素编码器会增加计算量，建议在显存充足的情况下使用
3. 可以通过调整 `voxel_sizes` 来平衡性能和计算成本
4. `fusion_method='attention'` 可能获得更好的性能，但会增加模型参数量

## 示例

完整的多尺度配置示例请参考 `configs/_base_/models/frnet_multiscale.py`。

