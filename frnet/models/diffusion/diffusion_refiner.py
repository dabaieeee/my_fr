"""Diffusion特征精炼模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from mmengine.model import BaseModule
from typing import Optional, Tuple

from .diffusion_utils import DiffusionScheduler, add_noise, sample_noise


class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码（用于时间步embedding）"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: 时间步 [B]
        Returns:
            embedding: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """时间步embedding模块"""
    
    def __init__(self, time_dim: int, embed_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(time)


class ResidualBlock(nn.Module):
    """残差块（用于U-Net）"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='ReLU')):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.block2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)  # 在最后添加时间embedding后再激活
        
        self.act = build_activation_layer(act_cfg)
        
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()
            
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
        """
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]  # [B, C, 1, 1]
        h = h + time_emb
        h = self.block2(h)
        h = self.act(h)
        return h + self.res_conv(x)


@MODELS.register_module()
class DiffusionFeatureRefiner(BaseModule):
    """
    使用Diffusion过程refine特征
    
    支持两种模式：
    1. frustum模式：处理2D range image特征 [B, C, H, W]
    2. point模式：处理点特征 [N, C]
    """
    
    def __init__(self,
                 in_channels: int,
                 refiner_type: str = 'frustum',  # 'frustum' or 'point'
                 num_timesteps: int = 1000,
                 beta_schedule: str = 'cosine',
                 time_emb_dim: int = 128,
                 base_channels: int = 64,
                 num_res_blocks: int = 2,
                 use_ddim: bool = True,
                 ddim_steps: int = 50,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        
        self.refiner_type = refiner_type
        self.num_timesteps = num_timesteps
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps if use_ddim else num_timesteps
        
        # 创建噪声调度器
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            schedule_type=beta_schedule
        )
        
        # 时间embedding
        self.time_embedding = TimeEmbedding(time_emb_dim, time_emb_dim)
        
        if refiner_type == 'frustum':
            # 2D U-Net for frustum features
            self._build_frustum_unet(in_channels, base_channels, num_res_blocks, 
                                    time_emb_dim, norm_cfg, act_cfg)
        elif refiner_type == 'point':
            # PointNet++ style for point features
            self._build_point_unet(in_channels, base_channels, num_res_blocks,
                                  time_emb_dim, norm_cfg, act_cfg)
        else:
            raise ValueError(f"Unknown refiner_type: {refiner_type}")
    
    def _build_frustum_unet(self, in_channels, base_channels, num_res_blocks,
                           time_emb_dim, norm_cfg, act_cfg):
        """构建2D U-Net用于frustum特征"""
        # 简化的U-Net结构
        # Encoder
        self.encoder = nn.ModuleList([
            ResidualBlock(in_channels, base_channels, time_emb_dim, norm_cfg, act_cfg),
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim, norm_cfg, act_cfg),
        ])
        
        # Middle
        self.middle = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, norm_cfg, act_cfg)
        
        # Decoder
        self.decoder = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels, time_emb_dim, norm_cfg, act_cfg),
            ResidualBlock(base_channels * 2, in_channels, time_emb_dim, norm_cfg, act_cfg),
        ])
        
        # 输出层（预测噪声）
        self.output_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
    
    def _build_point_unet(self, in_channels, base_channels, num_res_blocks,
                         time_emb_dim, norm_cfg, act_cfg):
        """构建PointNet++风格的网络用于点特征"""
        # 简化的MLP结构
        layers = []
        ch = in_channels
        for i in range(num_res_blocks):
            layers.append(nn.Linear(ch, base_channels))
            layers.append(build_norm_layer(dict(type='BN1d'), base_channels)[1])
            layers.append(build_activation_layer(act_cfg))
            ch = base_channels
        
        self.point_mlp = nn.Sequential(*layers)
        
        # 时间embedding融合
        self.time_fusion = nn.Linear(time_emb_dim, base_channels)
        
        # 输出层
        self.output_mlp = nn.Sequential(
            nn.Linear(base_channels + in_channels, base_channels),
            build_activation_layer(act_cfg),
            nn.Linear(base_channels, in_channels)
        )
    
    def forward(self, 
                features: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            features: 
                - frustum模式: [B, C, H, W]
                - point模式: [N, C]
            training: 是否为训练模式
            
        Returns:
            训练模式: (predicted_noise, true_noise, loss)
            推理模式: refined_features
        """
        device = features.device
        self.scheduler.to(device)
        
        if training:
            return self._forward_train(features)
        else:
            return self._forward_inference(features)
    
    def _forward_train(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练前向传播"""
        batch_size = features.shape[0] if self.refiner_type == 'frustum' else 1
        
        # 采样时间步（每个样本独立采样）
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=features.device)
        
        # 采样噪声
        noise = sample_noise(features.shape, features.device)
        
        # 添加噪声
        noisy_features = add_noise(features, noise, t, self.scheduler)
        
        # 时间embedding
        time_emb = self.time_embedding(t)
        
        # 预测噪声
        predicted_noise = self._predict_noise(noisy_features, time_emb)
        
        # 计算损失（MSE loss）
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        return predicted_noise, noise, loss
    
    def _forward_inference(self, features: torch.Tensor) -> torch.Tensor:
        """推理前向传播（使用DDIM采样）"""
        if self.use_ddim:
            return self._ddim_sample(features)
        else:
            return self._ddpm_sample(features)
    
    def _predict_noise(self, noisy_features: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """预测噪声"""
        if self.refiner_type == 'frustum':
            # 2D U-Net
            # Encoder
            x = noisy_features
            skip_connections = []
            for layer in self.encoder:
                x = layer(x, time_emb)
                skip_connections.append(x)
                x = F.avg_pool2d(x, 2)  # Downsample
            
            # Middle
            x = self.middle(x, time_emb)
            
            # Decoder
            for i, layer in enumerate(self.decoder):
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                skip = skip_connections[-(i+1)]
                x = torch.cat([x, skip], dim=1)
                x = layer(x, time_emb)
            
            predicted_noise = self.output_conv(x)
            
        else:  # point
            # Point MLP
            h = self.point_mlp(noisy_features)
            time_emb_expanded = self.time_fusion(time_emb).unsqueeze(0).expand_as(h)
            h = h + time_emb_expanded
            h = torch.cat([h, noisy_features], dim=-1)
            predicted_noise = self.output_mlp(h)
        
        return predicted_noise
    
    def _ddim_sample(self, features: torch.Tensor) -> torch.Tensor:
        """DDIM采样（快速推理）"""
        shape = features.shape
        batch_size = shape[0] if self.refiner_type == 'frustum' else 1
        
        # 初始化：从原始特征添加少量噪声开始（而不是完全随机）
        # 这样可以更快收敛
        x = features + 0.1 * sample_noise(shape, features.device)
        
        # DDIM采样步数
        step_size = self.num_timesteps // self.ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=features.device, dtype=torch.long)
            time_emb = self.time_embedding(t_tensor)
            
            # 预测噪声
            predicted_noise = self._predict_noise(x, time_emb)
            
            # DDIM更新
            alpha_t = self.scheduler.alphas_cumprod[t].to(features.device)
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.scheduler.alphas_cumprod[t_prev].to(features.device)
            else:
                alpha_t_prev = torch.tensor(1.0, device=features.device)
            
            # 预测x0
            sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            pred_x0 = sqrt_recip_alpha_t * (x - sqrt_one_minus_alpha_t * predicted_noise)
            
            # 更新x（DDIM公式）
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1.0 - alpha_t_prev)
            dir_xt = sqrt_one_minus_alpha_t_prev * predicted_noise
            x = sqrt_alpha_t_prev * pred_x0 + dir_xt
        
        # 最后与原始特征融合（可选，保留原始信息）
        refined_features = 0.7 * x + 0.3 * features
        
        return refined_features
    
    def _ddpm_sample(self, features: torch.Tensor) -> torch.Tensor:
        """DDPM采样（完整采样，较慢）"""
        # 实现完整的DDPM采样过程
        # 这里简化处理，实际使用时建议使用DDIM
        return self._ddim_sample(features)

