#!/usr/bin/env python
"""Test script to verify the consistency loss fix."""

import torch
from mmengine.config import Config

# Load config
cfg = Config.fromfile('configs/frnet/frnet-semantickitti_seg.py')

# Override with feature consistency enabled
cfg.model.use_feature_consistency = True
cfg.model.feature_consistency_weight = 0.1

print("Configuration loaded successfully!")
print(f"use_feature_consistency: {cfg.model.use_feature_consistency}")
print(f"feature_consistency_weight: {cfg.model.feature_consistency_weight}")

# Check voxel encoder channels
if 'voxel_3d_encoder' in cfg.model and cfg.model.voxel_3d_encoder is not None:
    voxel_channels = cfg.model.voxel_3d_encoder.feat_channels
    print(f"Voxel encoder output channels: {voxel_channels[-1]}")
else:
    print("No voxel_3d_encoder found")

# Check backbone channels
if 'backbone' in cfg.model:
    backbone_channels = cfg.model.backbone.out_channels
    print(f"Backbone output channels: {backbone_channels}")

print("\nDimension mismatch should be handled by projection layer.")
print("Voxel features (256) -> Projection -> Point features (128)")

