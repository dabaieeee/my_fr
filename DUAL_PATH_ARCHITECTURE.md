# 双通路FRNet架构说明

## 概述

本实现基于FRNet论文，引入了**几何-语义双通路架构**，实现了结构级解耦（而非loss级），通过Cross-Gated Fusion自适应融合几何和语义特征。

## 核心创新点

### 1. 结构级解耦
- **几何路径 (Geometry Path)**: 只处理几何信息（xyz, normals, curvature），使用小感受野，保持几何结构
- **语义路径 (Semantic Path)**: 处理语义上下文，使用FFE提取特征，支持大感受野和场景理解

### 2. Cross-Gated Fusion
自适应门控融合机制：
```
g = σ(W_g [F_geo; F_sem])
F_fused = g ⊙ F_geo + (1 - g) ⊙ F_sem
```
- `g → 1`: 信任几何特征（近距离、密集区域）
- `g → 0`: 信任语义特征（远距离、稀疏区域）

### 3. 保留FFE和FPFM模块
- **FFE (Frustum Feature Encoder)**: 保留在语义路径中
- **FPFM (Frustum-Point Fusion Module)**: 保留在backbone中，用于层次化双向融合

## 架构总览

```
Input Point Cloud
    │
    ├─── Geometry Encoder ────┐
    │   (structure-preserving) │
    │                          │
    └─── Semantic Encoder ─────┤
        (context-aware, FFE)    │
                                │
                    Cross-Gated Fusion
                                │
                    DualPathFRNetBackbone
                    (with FPFM)
                                │
                    Prediction Head
```

## 模块说明

### 1. GeometryEncoder (`geometry_encoder.py`)
- **输入**: xyz坐标（可选：normals, curvature）
- **输出**: 几何特征 [N, C_geo]
- **特点**: 
  - 小感受野（k-NN，k=10）
  - 只处理局部几何结构
  - 不包含语义信息

### 2. SemanticEncoder (`semantic_encoder.py`)
- **输入**: xyz + intensity等原始特征
- **输出**: 语义特征 [N, C_sem]
- **特点**:
  - 使用FFE提取特征
  - 支持大感受野
  - 关注场景上下文和类别关系

### 3. CrossGatedFusion (`cross_gated_fusion.py`)
- **功能**: 自适应融合几何和语义特征
- **支持**: 2D range image特征和1D point特征
- **机制**: 门控网络学习融合权重

### 4. DualPathFRNetBackbone (`dual_path_frnet_backbone.py`)
- **功能**: 整合双通路架构
- **包含**: 
  - 几何路径的stem层
  - 语义路径的stem层
  - Cross-Gated Fusion模块
  - FPFM模块（层次化双向融合）

### 5. DualPathFRNet (`dual_path_frnet.py`)
- **功能**: 完整的segmentor
- **流程**: 
  1. 分别通过GeometryEncoder和SemanticEncoder提取特征
  2. 在backbone中进行Cross-Gated Fusion
  3. 通过FPFM进行层次化融合
  4. 输出最终预测

## 使用方法

### 配置文件示例

参考 `configs/_base_/models/dual_path_frnet.py`:

```python
model = dict(
    type='DualPathFRNet',
    geometry_encoder=dict(
        type='GeometryEncoder',
        in_channels=3,
        feat_channels=[64, 128, 128],
        with_normals=True,
        with_curvature=True,
        k_neighbors=10),
    semantic_encoder=dict(
        type='SemanticEncoder',
        ffe_config=dict(
            type='FrustumFeatureEncoder',
            in_channels=4,
            feat_channels=(64, 128, 256, 256),
            with_distance=True,
            with_cluster_center=True,
            feat_compression=16)),
    backbone=dict(
        type='DualPathFRNetBackbone',
        geo_channels=128,
        sem_channels=16,
        use_cross_gated_fusion=True,
        ...),
    ...)
```

### 训练

```bash
python train.py configs/frnet/dual-path-frnet-semantickitti_seg.py
```

## 与原始FRNet的区别

| 特性 | 原始FRNet | 双通路FRNet |
|------|----------|------------|
| 特征提取 | 单一FFE | Geometry + Semantic双路径 |
| 融合方式 | 直接concat/sum | Cross-Gated Fusion |
| 几何保护 | 无显式保护 | 独立几何路径 |
| FFE位置 | 直接使用 | 在语义路径中使用 |
| FPFM位置 | Backbone中 | Backbone中（保留） |

## 优势

1. **结构级解耦**: 几何和语义信息在编码阶段分离，避免早期混合导致的几何结构破坏
2. **自适应融合**: Cross-Gated Fusion根据区域特性自适应选择几何或语义特征
3. **保留优势模块**: FFE和FPFM模块得到保留，维持原有优势
4. **可解释性**: 门控权重可以可视化，理解模型在不同区域的决策

## 注意事项

1. **计算开销**: 双通路架构会增加一定的计算开销，但通过特征压缩可以控制
2. **超参数**: `k_neighbors`影响几何特征质量，建议根据点云密度调整
3. **特征维度**: 确保`geo_channels`和`sem_channels`与encoder输出匹配

## 未来改进方向

1. **Curriculum Learning**: 在训练过程中逐步调整融合权重
2. **多尺度几何特征**: 在几何路径中引入多尺度特征
3. **注意力增强**: 在Cross-Gated Fusion中加入注意力机制

