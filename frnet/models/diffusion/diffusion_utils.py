"""Diffusion模型的工具函数"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class DiffusionScheduler:
    """Diffusion噪声调度器"""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule_type: str = 'linear'):
        """
        Args:
            num_timesteps: 扩散步数
            beta_start: 初始beta值
            beta_end: 最终beta值
            schedule_type: 调度类型 ('linear' or 'cosine')
        """
        self.num_timesteps = num_timesteps
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 用于DDIM采样
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def to(self, device):
        """将调度器参数移到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """余弦噪声调度"""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sample_noise(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """采样噪声"""
    return torch.randn(shape, device=device)


def add_noise(x: torch.Tensor, 
              noise: torch.Tensor, 
              t: torch.Tensor,
              scheduler: DiffusionScheduler) -> torch.Tensor:
    """
    根据时间步t添加噪声到输入x
    
    Args:
        x: 原始特征 [B, C, H, W] 或 [N, C]
        noise: 噪声 [B, C, H, W] 或 [N, C]
        t: 时间步 [B]
        scheduler: 扩散调度器
        
    Returns:
        noisy_x: 添加噪声后的特征
    """
    sqrt_alphas_cumprod_t = scheduler.sqrt_alphas_cumprod[t].view(-1, *([1] * (x.dim() - 1)))
    sqrt_one_minus_alphas_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x.dim() - 1)))
    
    noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_x


def extract_coef_at_t(scheduler: DiffusionScheduler, t: torch.Tensor, coef: str) -> torch.Tensor:
    """提取指定时间步的系数"""
    if coef == 'sqrt_alphas_cumprod':
        return scheduler.sqrt_alphas_cumprod[t]
    elif coef == 'sqrt_one_minus_alphas_cumprod':
        return scheduler.sqrt_one_minus_alphas_cumprod[t]
    elif coef == 'alphas_cumprod':
        return scheduler.alphas_cumprod[t]
    else:
        raise ValueError(f"Unknown coefficient: {coef}")

