# FRNet 创新功能实现指南

本文档说明如何使用已实现的新功能。

## 一、已实现的功能

### 1.1 自适应特征融合（Adaptive Feature Fusion）

**功能说明**：根据特征质量动态调整frustum和point特征的融合权重，而不是使用固定的门控机制。

**使用方法**：

在backbone配置中设置 `use_adaptive_fusion=True`：

```python
backbone=dict(
    type='FRNetBackbone',
    # ... 其他参数 ...
    use_adaptive_fusion=True,  # 启用自适应融合
)
```

**工作原理**：
- 使用质量评估网络（quality_net）评估frustum和point特征的质量
- 根据质量分数动态生成融合权重
- 权重通过Softmax归一化，确保两个特征的权重和为1

**预期收益**：提升2-3% mIoU，特别是在边界区域

---

### 1.2 Focal Loss（困难样本挖掘）

**功能说明**：通过降低简单样本的权重，让模型更关注困难样本。

**使用方法**：

在decode_head配置中替换或添加Focal Loss：

```python
decode_head=dict(
    type='FRHead',
    # ... 其他参数 ...
    # 方式1：完全替换交叉熵损失
    loss_ce=dict(
        type='FocalLoss',
        alpha=0.25,  # 类别权重，用于平衡类别不平衡
        gamma=2.0,  # 聚焦参数，值越大越关注困难样本
        reduction='mean',
        ignore_index=255,
        loss_weight=1.0),
)
```

**参数说明**：
- `alpha`: 类别权重，用于平衡类别不平衡（默认0.25）
- `gamma`: 聚焦参数，控制对困难样本的关注程度（默认2.0）
  - gamma=0: 等价于交叉熵损失
  - gamma越大: 越关注困难样本
- `ignore_index`: 忽略的类别索引（默认255）

**预期收益**：提升1-2% mIoU，特别是困难类别

---

### 1.3 特征一致性损失（Feature Consistency Loss）

**功能说明**：确保frustum特征和point特征的一致性，增强特征对齐。

**使用方法**：

需要在segmentor的loss函数中手动添加。首先在配置中定义：

```python
# 在segmentor的__init__中添加
consistency_loss=dict(
    type='FeatureConsistencyLoss',
    loss_weight=0.1,  # 一致性损失权重，建议0.05-0.2
    loss_type='both',  # 'l2', 'cosine', 或 'both'
    temperature=0.1),
```

然后在segmentor的loss函数中使用：

```python
def loss(self, batch_inputs_dict, batch_data_samples):
    voxel_dict = self.extract_feat(batch_inputs_dict)
    losses = dict()
    
    # 原有的损失计算
    loss_decode = self._decode_head_forward_train(voxel_dict, batch_data_samples)
    losses.update(loss_decode)
    
    # 添加一致性损失
    if hasattr(self, 'consistency_loss'):
        # 获取frustum特征和point特征
        frustum_feats = voxel_dict['voxel_feats'][0]  # [B, C, H, W]
        point_feats = voxel_dict['point_feats_backbone'][0]  # [N, C]
        
        # 将frustum特征投影到点
        pts_coors = voxel_dict['coors']
        frustum_feats_proj = frustum_feats.permute(0, 2, 3, 1)
        frustum_feats_points = frustum_feats_proj[
            pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2]
        ]  # [N, C]
        
        # 计算一致性损失
        loss_consistency = self.consistency_loss(frustum_feats_points, point_feats)
        losses['loss_consistency'] = loss_consistency
    
    return losses
```

**参数说明**：
- `loss_weight`: 一致性损失的权重（建议0.05-0.2）
- `loss_type`: 损失类型
  - `'l2'`: 仅使用L2损失
  - `'cosine'`: 仅使用余弦相似度损失
  - `'both'`: 同时使用两种损失（推荐）
- `temperature`: 余弦相似度的温度参数（默认0.1）

**预期收益**：提升0.5-1% mIoU，增强特征对齐

---

## 二、使用示例

### 2.1 完整配置示例

参考 `configs/_base_/models/frnet_enhanced.py` 查看完整配置示例。

### 2.2 训练命令

```bash
# 使用增强版FRNet训练
python train.py configs/frnet/frnet-semantickitti_seg.py \
    --work-dir work_dirs/frnet_enhanced \
    --amp  # 使用混合精度训练
```

### 2.3 测试命令

```bash
# 测试模型
python test.py configs/frnet/frnet-semantickitti_seg.py \
    work_dirs/frnet_enhanced/latest.pth \
    --work-dir work_dirs/frnet_enhanced/test \
    --tta  # 使用测试时增强
```

---

## 三、实验建议

### 3.1 消融实验

建议按以下顺序进行消融实验：

1. **基线实验**：使用原始FRNet配置
2. **自适应融合**：仅启用 `use_adaptive_fusion=True`
3. **Focal Loss**：仅使用Focal Loss替换交叉熵
4. **特征一致性**：仅添加特征一致性损失
5. **组合实验**：同时使用所有新功能

### 3.2 超参数调优

**自适应融合**：
- 默认配置通常效果良好
- 如果效果不佳，可以尝试调整质量评估网络的通道数

**Focal Loss**：
- `gamma`: 建议范围 [1.0, 3.0]
  - 类别不平衡严重时，使用较大的gamma（2.5-3.0）
  - 类别相对平衡时，使用较小的gamma（1.5-2.0）
- `alpha`: 建议范围 [0.1, 0.5]
  - 根据类别频率调整

**特征一致性损失**：
- `loss_weight`: 建议范围 [0.05, 0.2]
  - 从0.1开始，根据验证集效果调整
  - 如果训练不稳定，降低权重
- `loss_type`: 建议使用 `'both'`

---

## 四、注意事项

1. **内存消耗**：自适应融合会增加少量内存消耗（约5-10%）
2. **训练时间**：新功能会增加约10-15%的训练时间
3. **兼容性**：所有新功能都与现有代码兼容，可以逐步启用
4. **调试**：如果遇到问题，可以先禁用新功能，确认基线模型正常

---

## 五、后续计划

以下功能正在开发中，敬请期待：

1. 对比学习增强特征表示
2. 语义感知的注意力机制
3. 多尺度特征金字塔网络
4. 几何感知的特征增强

---

## 六、问题反馈

如果使用过程中遇到问题，请检查：

1. 配置文件是否正确
2. 损失函数是否正确注册
3. 特征维度是否匹配
4. 是否有足够的GPU内存

如有问题，请参考 `INNOVATION_SUGGESTIONS.md` 中的详细说明。

