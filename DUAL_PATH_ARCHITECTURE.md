# Dual-Path FRNet: Geometry-Semantic Decoupling Architecture

## 概述

本文档介绍了基于FRNet的双路径架构创新，实现了**结构级解耦**的几何-语义双路径设计。

## 核心创新点

### 1. 问题动机

现有方法（包括原始FRNet）在backbone最开始就把几何信息和语义上下文混合在一起，这破坏了几何结构的稳定性：

- ❌ **过早混合** = 法向量/平面结构被语义噪声污染
- ❌ **远距离稀疏区域**的几何被"上下文猜测"覆盖
- ❌ 几何是点云的主导信息，语义是"建立在几何之上的高层认知"

### 2. 解决方案：双路径架构

```
Input Point Cloud P = {xi, fi}
        │
        ├─────────────────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ Geometry Encoder │    │ Semantic Encoder │
│ (structure-pres) │    │ (context-aware)  │
└──────────────────┘    └──────────────────┘
        │                         │
        │ F_geo                   │ F_sem
        │                         │
        └──────────┬──────────────┘
                   ▼
        ┌──────────────────────┐
        │ Cross-Gated Fusion    │
        └──────────────────────┘
                   │
                   ▼
            Prediction Head
```

### 3. 架构组件

#### 3.1 Geometry Path（几何路径）

**目标**：稳定、独立地建模几何结构

**输入**：
- xyz坐标
- normals（法向量）
- curvature（曲率）
- distance（距离）

**特点**：
- ✅ 局部感受野（小范围）
- ✅ 不使用Transformer（避免全局混合）
- ✅ 对旋转、稀疏鲁棒
- ✅ 语义未显式编码

**实现**：`GeometryEncoder` (`frnet/models/voxel_encoders/geometry_encoder.py`)

#### 3.2 Semantic Path（语义路径）

**目标**：学习类别关系、场景上下文、远程依赖

**特点**：
- ✅ 保留FFE（FrustumFeatureEncoder）
- ✅ 保留FPFM（Frustum-Point Fusion Module）
- ✅ 大感受野、多尺度
- ✅ 可以使用frustum/voxel/transformer/CNN

**实现**：使用原有的`FRNetBackbone`，保留FFE和FPFM功能

#### 3.3 Cross-Gated Fusion（交叉门控融合）

**公式**：
```
g = σ(W_g [F_geo; F_sem])
F_fused = g ⊙ F_geo + (1 - g) ⊙ F_sem
```

**物理意义**：
- `g → 1`：信任几何（近距离、密集区域）
- `g → 0`：信任语义（远距离、稀疏区域）

**实现**：`CrossGatedFusion` (`frnet/models/voxel_encoders/cross_gated_fusion.py`)

## 文件结构

```
frnet/models/
├── voxel_encoders/
│   ├── geometry_encoder.py          # 几何编码器
│   ├── cross_gated_fusion.py         # 交叉门控融合模块
│   ├── frustum_encoder.py            # FFE（保留）
│   └── ...
├── backbones/
│   ├── dual_path_backbone.py         # 双路径backbone
│   ├── frnet_backbone.py            # 原始backbone（用于语义路径）
│   └── ...
└── segmentors/
    ├── dual_path_frnet.py           # 双路径segmentor
    └── frnet.py                      # 原始segmentor
```

## 使用方法

### 1. 配置文件

使用配置文件：`configs/frnet/dual-path-frnet-semantickitti_seg.py`

### 2. 训练

```bash
python train.py configs/frnet/dual-path-frnet-semantickitti_seg.py
```

### 3. 测试

```bash
python test.py configs/frnet/dual-path-frnet-semantickitti_seg.py \
    work_dirs/dual-path-frnet-semantickitti_seg/iter_xxx.pth
```

## 配置说明

### Geometry Encoder配置

```python
geometry_encoder=dict(
    type='GeometryEncoder',
    in_channels=3,              # xyz only
    feat_channels=(32, 64, 128), # 较小的通道数，保持局部性
    with_normals=True,          # 计算法向量
    with_curvature=True,        # 计算曲率
    with_distance=True,         # 计算距离
    norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
    k_neighbors=10),            # 计算法向量/曲率的近邻点数
```

### Cross-Gated Fusion配置

```python
fusion_cfg=dict(
    type='CrossGatedFusion',
    geo_channels=128,          # Geometry Encoder输出通道数
    sem_channels=128,          # Semantic Backbone输出通道数
    out_channels=128,          # 融合后输出通道数
    fusion_type='point',       # 'point' 或 'pixel'
    norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
    act_cfg=dict(type='Sigmoid')),
```

## 与原始FRNet的区别

| 特性 | 原始FRNet | Dual-Path FRNet |
|------|----------|----------------|
| 几何处理 | 与语义混合 | 独立几何路径 |
| 语义处理 | FFE + FPFM | FFE + FPFM（保留） |
| 融合方式 | 直接拼接/相加 | 交叉门控融合 |
| 结构解耦 | ❌ | ✅ |
| 几何稳定性 | 一般 | 高 |

## 优势

1. **结构级解耦**：几何和语义在结构层面分离，而非仅通过loss
2. **自适应融合**：门控机制根据区域特性自适应选择几何/语义信息
3. **保留原有优势**：FFE和FPFM完全保留，兼容性好
4. **易于扩展**：可以独立改进几何或语义路径

## 未来改进方向

1. **Curriculum Learning**：逐步引入语义信息（从纯几何到混合）
2. **多尺度几何特征**：在不同尺度提取几何特征
3. **几何-语义一致性约束**：添加几何和语义特征的一致性loss
4. **动态门控**：根据点云密度动态调整门控权重

## 引用

如果使用本架构，请引用原始FRNet论文和本实现。

