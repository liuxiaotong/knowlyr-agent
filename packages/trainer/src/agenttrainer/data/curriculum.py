"""Curriculum Learning 采样器 — 从简单到困难渐进式训练.

难度指标（按优先级）:
1. 轨迹步骤数（少 = 简单）
2. 最终 reward（高 = 简单/已解决）

训练过程:
- 初始阶段: 只使用最简单的 start_ratio 比例数据
- 随 epoch 推进: 线性增加使用数据量
- warmup_epochs 后: 使用全部数据
"""

from __future__ import annotations

import math
from typing import Any

from torch.utils.data import Sampler


class CurriculumSampler(Sampler[int]):
    """渐进式难度采样器.

    按难度排序数据集，逐步扩大使用范围。

    Args:
        difficulties: 每条样本的难度分数（越小越简单）
        num_epochs: 总训练 epoch 数
        start_ratio: 初始 epoch 使用的数据比例 (0.0~1.0)
        warmup_epochs: 几个 epoch 后使用全部数据
    """

    def __init__(
        self,
        difficulties: list[float],
        num_epochs: int,
        start_ratio: float = 0.3,
        warmup_epochs: int = 1,
    ) -> None:
        self.difficulties = difficulties
        self.num_epochs = num_epochs
        self.start_ratio = max(0.1, min(start_ratio, 1.0))
        self.warmup_epochs = max(1, warmup_epochs)
        self.current_epoch = 0

        # 按难度升序排列的索引
        self._sorted_indices = sorted(range(len(difficulties)), key=lambda i: difficulties[i])

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch（控制难度门槛）."""
        self.current_epoch = epoch

    def __iter__(self):
        """返回当前 epoch 应使用的样本索引."""
        n = len(self._sorted_indices)
        # 当前应使用的数据比例
        ratio = self._get_ratio()
        use_count = max(1, math.ceil(n * ratio))
        indices = self._sorted_indices[:use_count]

        # 在选定范围内随机打乱
        import random
        shuffled = list(indices)
        random.shuffle(shuffled)
        return iter(shuffled)

    def __len__(self) -> int:
        n = len(self._sorted_indices)
        ratio = self._get_ratio()
        return max(1, math.ceil(n * ratio))

    def _get_ratio(self) -> float:
        """计算当前 epoch 的数据使用比例."""
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        # 线性插值: start_ratio → 1.0
        progress = self.current_epoch / self.warmup_epochs
        return self.start_ratio + (1.0 - self.start_ratio) * progress


def compute_difficulties(
    records: list[dict[str, Any]],
    step_weight: float = 1.0,
    reward_weight: float = -1.0,
) -> list[float]:
    """计算每条样本的难度分数.

    难度 = step_weight * 步骤数 + reward_weight * reward
    (reward 取负是因为高 reward = 简单 = 低难度)

    Args:
        records: SFT/GRPO 样本列表
        step_weight: 步骤数的权重
        reward_weight: reward 的权重（默认 -1.0，高 reward 降低难度）

    Returns:
        每条样本的难度分数列表
    """
    difficulties = []
    for rec in records:
        # 步骤数
        n_steps = rec.get("metadata", {}).get("total_steps", 1)
        if "steps" in rec:
            n_steps = len(rec["steps"])

        # reward
        reward = rec.get("reward", 0.0)

        difficulty = step_weight * n_steps + reward_weight * reward
        difficulties.append(difficulty)

    return difficulties
