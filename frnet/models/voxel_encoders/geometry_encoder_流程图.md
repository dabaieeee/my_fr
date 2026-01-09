# GeometryEncoder 运行流程图解

## 整体架构图

```mermaid
graph TB
    A[输入: voxel_dict] --> B[提取 xyz 坐标]
    B --> C{是否启用法向量?}
    C -->|是| D[compute_normals<br/>计算法向量]
    C -->|否| E[geo_features = xyz]
    D --> E
    E --> F{是否启用曲率?}
    F -->|是| G[compute_curvature<br/>计算曲率特征]
    F -->|否| H[拼接所有特征]
    G --> H
    H --> I[MLP特征提取]
    I --> J[点级别特征<br/>geo_point_feats]
    J --> K[聚合到视锥级别<br/>scatter_max]
    K --> L[输出: voxel_dict<br/>包含点特征和视锥特征]
    
    style A fill:#e1f5ff
    style L fill:#d4edda
    style D fill:#fff3cd
    style G fill:#fff3cd
    style I fill:#f8d7da
```

## 详细流程说明

### 1. 初始化阶段 (__init__)

```mermaid
graph LR
    A[输入参数] --> B[计算实际输入通道数]
    B --> C{with_normals?}
    C -->|是| D[+3通道]
    C -->|否| E[保持原通道]
    D --> F{with_curvature?}
    E --> F
    F -->|是| G[+3通道]
    F -->|否| H[构建MLP层]
    G --> H
    H --> I[geo_layers<br/>MLP模块列表]
    
    style A fill:#e1f5ff
    style I fill:#d4edda
```

**示例**：
- 如果 `in_channels=3`, `with_normals=True`, `with_curvature=True`
- 则 `actual_in_channels = 3 + 3 + 3 = 9`
- MLP层：`[9 → 64 → 128 → 128]`

### 2. 法向量计算流程 (compute_normals)

```mermaid
graph TB
    A[输入点云 points<br/>N个点, 每个点3D坐标] --> B[计算成对距离矩阵<br/>torch.cdist]
    B --> C[对每个点找k个最近邻<br/>k-NN搜索]
    C --> D[对每个点循环处理]
    D --> E[获取邻居点<br/>排除自身]
    E --> F[计算协方差矩阵<br/>Cov = X^T X / N]
    F --> G[特征值分解<br/>eigenvals, eigenvecs]
    G --> H[取最小特征值对应的特征向量<br/>作为法向量]
    H --> I[方向校正<br/>确保指向原点]
    I --> J{特征值分解失败?}
    J -->|是| K[备用方案:<br/>叉积法计算法向量]
    J -->|否| L[输出法向量<br/>N×3]
    K --> L
    
    style A fill:#e1f5ff
    style L fill:#d4edda
    style F fill:#fff3cd
    style G fill:#f8d7da
```

**关键步骤**：
1. **k-NN搜索**：对每个点找最近的k个邻居（默认k=10）
2. **协方差矩阵**：计算局部邻域的协方差矩阵
3. **PCA分析**：通过特征值分解找到主方向
4. **法向量**：最小特征值对应的特征向量就是法向量

### 3. 曲率计算流程 (compute_curvature)

```mermaid
graph TB
    A[输入点云 points] --> B[k-NN搜索<br/>找最近邻]
    B --> C[计算协方差矩阵]
    C --> D[特征值分解<br/>λ1 ≥ λ2 ≥ λ3]
    D --> E[归一化特征值]
    E --> F[计算三个曲率特征]
    F --> G[线性度<br/>λ1-λ2/λ1]
    F --> H[平面度<br/>λ2-λ3/λ1]
    F --> I[球面度<br/>λ3/λ1]
    G --> J[输出曲率特征<br/>N×3]
    H --> J
    I --> J
    
    style A fill:#e1f5ff
    style J fill:#d4edda
    style F fill:#fff3cd
```

**曲率特征含义**：
- **线性度 (Linearity)**：接近1表示点沿一条线分布（如电线）
- **平面度 (Planarity)**：接近1表示点在一个平面上（如墙面）
- **球面度 (Sphericity)**：接近1表示点呈球形分布（如球体）

### 4. 前向传播完整流程 (forward)

```mermaid
graph TB
    A[voxel_dict输入<br/>voxels: N×C<br/>coors: N×4] --> B[提取xyz坐标<br/>features[:, :3]]
    B --> C[初始化特征列表<br/>geo_features = [xyz]]
    C --> D{with_normals?}
    D -->|是| E[compute_normals<br/>输出: N×3]
    D -->|否| F[跳过]
    E --> G[添加到特征列表]
    F --> H{with_curvature?}
    G --> H
    H -->|是| I[compute_curvature<br/>输出: N×3]
    H -->|否| J[跳过]
    I --> K[添加到特征列表]
    J --> L[拼接所有特征<br/>torch.cat]
    K --> L
    L --> M[MLP特征提取<br/>通过geo_layers]
    M --> N[点级别特征<br/>geo_point_feats: N×C_geo]
    N --> O[聚合到视锥级别<br/>scatter_max]
    O --> P[视锥级别特征<br/>geo_voxel_feats: M×C_geo]
    P --> Q[更新voxel_dict<br/>返回结果]
    
    style A fill:#e1f5ff
    style Q fill:#d4edda
    style M fill:#f8d7da
    style O fill:#fff3cd
```

### 5. 数据维度变化

```mermaid
graph LR
    A[输入<br/>voxels: N×C<br/>C通常≥3] --> B[xyz: N×3]
    B --> C[法向量: N×3<br/>可选]
    B --> D[曲率: N×3<br/>可选]
    C --> E[拼接特征<br/>N×9<br/>如果都启用]
    D --> E
    E --> F[MLP Layer 1<br/>N×9 → N×64]
    F --> G[MLP Layer 2<br/>N×64 → N×128]
    G --> H[MLP Layer 3<br/>N×128 → N×128]
    H --> I[点特征<br/>geo_point_feats<br/>N×128]
    I --> J[聚合<br/>scatter_max]
    J --> K[视锥特征<br/>geo_voxel_feats<br/>M×128]
    
    style A fill:#e1f5ff
    style K fill:#d4edda
    style E fill:#fff3cd
    style I fill:#f8d7da
```

## 关键算法详解

### k-NN搜索和协方差计算

```python
# 伪代码示例
for each point p_i:
    # 1. 找k个最近邻
    neighbors = k_nearest_neighbors(p_i, k=10)
    
    # 2. 计算局部坐标系
    centered = neighbors - p_i  # [k, 3]
    
    # 3. 协方差矩阵
    cov = (centered.T @ centered) / k  # [3, 3]
    
    # 4. 特征值分解
    eigenvals, eigenvecs = eigendecomposition(cov)
    
    # 5. 法向量 = 最小特征值对应的特征向量
    normal = eigenvecs[:, min_eigenval_index]
    
    # 6. 曲率特征 = 基于特征值的几何描述
    curvature = compute_curvature_features(eigenvals)
```

### 特征聚合 (scatter_max)

```mermaid
graph LR
    A[点特征<br/>N×C<br/>每个点一个特征] --> B[按视锥坐标分组<br/>coors]
    B --> C[同一视锥内的点<br/>取最大值]
    C --> D[视锥特征<br/>M×C<br/>M个视锥]
    
    style A fill:#e1f5ff
    style D fill:#d4edda
    style C fill:#fff3cd
```

**为什么用max pooling？**
- 保持几何结构的显著性
- 避免平均化导致的细节丢失
- 适合几何特征（如法向量、曲率）

## 运行示例

假设输入：
- `N = 1000` 个点
- `C = 4` (xyz + intensity)
- `with_normals = True`
- `with_curvature = True`
- `k_neighbors = 10`

**处理流程**：

1. **输入**：`voxels: [1000, 4]`, `coors: [1000, 4]`

2. **提取xyz**：`xyz: [1000, 3]`

3. **计算法向量**：
   - 对每个点找10个最近邻
   - 计算1000个协方差矩阵 [3×3]
   - 特征值分解得到法向量
   - 输出：`normals: [1000, 3]`

4. **计算曲率**：
   - 同样使用k-NN
   - 基于特征值计算线性度、平面度、球面度
   - 输出：`curvature: [1000, 3]`

5. **特征拼接**：
   - `geo_input = concat([xyz, normals, curvature])`
   - 输出：`[1000, 9]`

6. **MLP处理**：
   - Layer 1: `[1000, 9] → [1000, 64]`
   - Layer 2: `[1000, 64] → [1000, 128]`
   - Layer 3: `[1000, 128] → [1000, 128]`

7. **点级别输出**：`geo_point_feats: [1000, 128]`

8. **视锥级别聚合**：
   - 假设有 `M = 500` 个视锥
   - 每个视锥内的点特征取最大值
   - 输出：`geo_voxel_feats: [500, 128]`

## 设计特点

1. **小感受野**：k=10，只关注局部几何结构
2. **结构保持**：使用max pooling而非mean pooling
3. **几何专注**：只处理几何信息，不涉及语义
4. **可配置**：可以选择性启用法向量和曲率

## 输出说明

返回的 `voxel_dict` 包含：
- `geo_point_feats`: 点级别的几何特征 [N, C_geo]
- `geo_voxel_feats`: 视锥级别的几何特征 [M, C_geo]
- `geo_voxel_coors`: 视锥坐标 [M, 4]

这些特征将用于后续的交叉门控融合模块。

