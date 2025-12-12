"""课程学习Hook，用于动态调整训练难度。"""
import math
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from typing import Optional, Dict, Any


@HOOKS.register_module()
class CurriculumLearningHook(Hook):
    """课程学习Hook，用于在训练过程中动态调整难度。
    
    该Hook实现了基于训练进度的课程学习策略：
    1. 在训练初期，降低困难样本的权重
    2. 随着训练进行，逐步增加困难样本的权重
    3. 可以动态调整数据增强强度
    
    Args:
        strategy (str): 课程学习策略，可选 'loss_based', 'epoch_based', 'adaptive'
            - 'loss_based': 基于损失值的课程学习
            - 'epoch_based': 基于训练轮次的课程学习
            - 'adaptive': 自适应课程学习（结合损失和轮次）
        start_epoch (int): 开始应用课程学习的轮次，默认0
        end_epoch (int): 完全应用课程学习的轮次，默认总训练轮次
        warmup_iters (int): 预热迭代次数，在预热期间不使用课程学习
        difficulty_threshold (float): 困难样本的损失阈值，默认0.5
        min_weight (float): 困难样本的最小权重，默认0.1
        max_weight (float): 困难样本的最大权重，默认1.0
        schedule_type (str): 调度类型，可选 'linear', 'cosine', 'exponential'
        update_interval (int): 更新课程学习参数的间隔（迭代次数），默认100
        log_interval (int): 记录课程学习信息的间隔，默认500
    """
    
    def __init__(self,
                 strategy: str = 'adaptive',
                 start_epoch: int = 0,
                 end_epoch: Optional[int] = None,
                 warmup_iters: int = 0,
                 difficulty_threshold: float = 0.5,
                 min_weight: float = 0.1,
                 max_weight: float = 1.0,
                 schedule_type: str = 'linear',
                 update_interval: int = 100,
                 log_interval: int = 500):
        super().__init__()
        self.strategy = strategy
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.warmup_iters = warmup_iters
        self.difficulty_threshold = difficulty_threshold
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.schedule_type = schedule_type
        self.update_interval = update_interval
        self.log_interval = log_interval
        
        # 内部状态
        self.current_epoch = 0
        self.current_iter = 0
        self.curriculum_progress = 0.0  # 0.0 到 1.0，表示课程学习进度
        self.sample_weights = None  # 样本权重，用于损失计算
        self.loss_history = []  # 用于自适应策略的损失历史
        
    def before_train(self, runner) -> None:
        """训练开始前的初始化。"""
        if self.end_epoch is None:
            # 如果没有指定结束轮次，使用总训练轮次
            self.end_epoch = runner.max_epochs if hasattr(runner, 'max_epochs') else 100
        
        # 初始化模型中的课程学习参数
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
            
        if hasattr(model, 'decode_head'):
            # 在decode_head中注册课程学习参数
            if not hasattr(model.decode_head, 'curriculum_learning'):
                model.decode_head.curriculum_learning = {
                    'enabled': True,
                    'sample_weights': None,
                    'difficulty_threshold': self.difficulty_threshold,
                    'min_weight': self.min_weight,
                    'max_weight': self.max_weight
                }
        
        runner.logger.info(
            f'课程学习Hook已初始化: strategy={self.strategy}, '
            f'start_epoch={self.start_epoch}, end_epoch={self.end_epoch}'
        )
    
    def before_train_iter(self, runner, batch_idx: int, data_batch: Dict[str, Any] = None) -> None:
        """每次迭代前的处理。"""
        self.current_iter = runner.iter
        
        # 更新课程学习进度
        self._update_curriculum_progress(runner)
        
        # 定期更新课程学习参数
        if self.current_iter % self.update_interval == 0:
            self._update_curriculum_params(runner)
    
    def after_train_iter(self, runner, batch_idx: int, data_batch: Dict[str, Any] = None, outputs: Dict[str, Any] = None) -> None:
        """每次迭代后的处理。"""
        # 记录损失历史（用于自适应策略）
        if self.strategy == 'adaptive' and outputs is not None:
            if 'loss' in outputs:
                # 提取主要损失值
                if isinstance(outputs['loss'], torch.Tensor):
                    loss_value = outputs['loss'].item()
                elif isinstance(outputs['loss'], dict):
                    # 取第一个损失值
                    loss_value = list(outputs['loss'].values())[0].item() if outputs['loss'] else 0.0
                else:
                    loss_value = 0.0
                
                self.loss_history.append(loss_value)
                # 只保留最近1000个损失值
                if len(self.loss_history) > 1000:
                    self.loss_history = self.loss_history[-1000:]
        
        # 定期记录课程学习信息
        if self.current_iter % self.log_interval == 0:
            self._log_curriculum_info(runner)
    
    def before_train_epoch(self, runner) -> None:
        """每个训练轮次前的处理。"""
        self.current_epoch = runner.epoch
    
    def _update_curriculum_progress(self, runner) -> None:
        """更新课程学习进度。"""
        # 检查是否在预热期
        if self.current_iter < self.warmup_iters:
            self.curriculum_progress = 0.0
            return
        
        # 检查是否在课程学习时间范围内（epoch 制约束）
        if self.current_epoch < self.start_epoch:
            self.curriculum_progress = 0.0
            return
        if self.end_epoch is not None and self.current_epoch >= self.end_epoch:
            self.curriculum_progress = 1.0
            return

        # 判断是否为迭代制训练（IterBasedTrainLoop）
        epoch_based = getattr(getattr(runner, 'train_loop', None), '_epoch_based', True)
        if not epoch_based:
            # 按 iter 计算进度
            total_iters = max(getattr(runner, 'max_iters', 0) - self.warmup_iters, 1)
            current_iters = max(self.current_iter - self.warmup_iters, 0)
            iter_progress = min(1.0, max(0.0, current_iters / total_iters))

            if self.strategy == 'adaptive':
                if len(self.loss_history) >= 100:
                    recent_loss = sum(self.loss_history[-100:]) / 100
                    early_loss = sum(self.loss_history[:100]) / 100 if len(self.loss_history) >= 200 else recent_loss
                    if early_loss > 0 and recent_loss < early_loss * 0.8:
                        loss_factor = 1.2
                    elif recent_loss < early_loss * 0.9:
                        loss_factor = 1.1
                    else:
                        loss_factor = 1.0
                    self.curriculum_progress = min(1.0, iter_progress * loss_factor)
                else:
                    self.curriculum_progress = iter_progress
            else:
                # loss_based / epoch_based / default 在迭代制下使用 iter 进度
                self.curriculum_progress = iter_progress
        else:
            # epoch 制训练：按 epoch 进度
            total_epochs = self.end_epoch - self.start_epoch
            current_epochs = self.current_epoch - self.start_epoch
            epoch_progress = min(1.0, max(0.0, current_epochs / total_epochs))

            if self.strategy == 'adaptive':
                if len(self.loss_history) >= 100:
                    recent_loss = sum(self.loss_history[-100:]) / 100
                    early_loss = sum(self.loss_history[:100]) / 100 if len(self.loss_history) >= 200 else recent_loss
                    if early_loss > 0 and recent_loss < early_loss * 0.8:
                        loss_factor = 1.2  # 加快进度
                    elif recent_loss < early_loss * 0.9:
                        loss_factor = 1.1
                    else:
                        loss_factor = 1.0
                    self.curriculum_progress = min(1.0, epoch_progress * loss_factor)
                else:
                    self.curriculum_progress = epoch_progress
            else:
                # loss_based / epoch_based / default
                self.curriculum_progress = epoch_progress
        
        # 应用调度类型
        if self.schedule_type == 'cosine':
            self.curriculum_progress = (1 - math.cos(self.curriculum_progress * math.pi / 2))
        elif self.schedule_type == 'exponential':
            self.curriculum_progress = (math.exp(self.curriculum_progress * 2) - 1) / (math.exp(2) - 1)
        # linear 不需要额外处理
    
    def _update_curriculum_params(self, runner) -> None:
        """更新模型中的课程学习参数。"""
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
        
        if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'curriculum_learning'):
            model.decode_head.curriculum_learning['enabled'] = self.curriculum_progress > 0
            model.decode_head.curriculum_learning['curriculum_progress'] = self.curriculum_progress
            model.decode_head.curriculum_learning['difficulty_threshold'] = self.difficulty_threshold
            model.decode_head.curriculum_learning['min_weight'] = self.min_weight
            model.decode_head.curriculum_learning['max_weight'] = self.max_weight
    
    def _log_curriculum_info(self, runner) -> None:
        """记录课程学习信息。"""
        runner.logger.info(
            f'[课程学习] Epoch: {self.current_epoch}, Iter: {self.current_iter}, '
            f'Progress: {self.curriculum_progress:.4f}, '
            f'Strategy: {self.strategy}'
        )

