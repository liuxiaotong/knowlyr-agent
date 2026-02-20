"""GDI (Global Desirability Index) — 轨迹全局质量排名.

借鉴 EvoMap 的四维加权排名思路，适配 knowlyr-agent 轨迹场景：
- intrinsic  (0.35): 内在质量，来自 reward.total_score
- utility    (0.30): 训练效用，被 SFT/DPO 导出引用的次数
- feedback   (0.20): 效果反馈，训练后模型在相似任务上的表现提升
- freshness  (0.15): 时效性，随时间衰减
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# 默认权重
DEFAULT_WEIGHTS: dict[str, float] = {
    "intrinsic": 0.35,
    "utility": 0.30,
    "feedback": 0.20,
    "freshness": 0.15,
}

# freshness 衰减系数: score = 1 / (1 + days * DECAY_RATE)
# DECAY_RATE=0.05 → 30天后 freshness≈0.4, 90天后≈0.18
FRESHNESS_DECAY_RATE = 0.05


@dataclass
class GDIScore:
    """单条轨迹的 GDI 评分."""

    intrinsic: float = 0.0
    utility: float = 0.0
    feedback: float = 0.5
    freshness: float = 1.0
    total: float = 0.0


class GDIScorer:
    """计算和排名轨迹的 GDI 分数.

    Usage::

        scorer = GDIScorer()
        score = scorer.score(reward=0.82, export_count=5, created_at=1708300000)
        print(score.total)  # 加权总分
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 0.01:
            msg = f"权重之和必须为 1.0，当前 {total_w:.3f}"
            raise ValueError(msg)

    def score(
        self,
        *,
        reward: float = 0.0,
        export_count: int = 0,
        max_export_count: int = 100,
        feedback_score: float = 0.5,
        created_at: float | None = None,
    ) -> GDIScore:
        """计算单条轨迹的 GDI 分数.

        Args:
            reward: reward 总分 [0, 1]，直接作为 intrinsic.
            export_count: 被导出/引用次数.
            max_export_count: 归一化用的最大引用次数.
            feedback_score: 训练效果反馈 [0, 1]，初期无数据时默认 0.5.
            created_at: 创建时间戳 (epoch seconds)，None 时视为当前.

        Returns:
            GDIScore: 四维评分 + 加权总分.
        """
        intrinsic = max(0.0, min(1.0, reward))

        # utility: 对数归一化，避免引用次数线性爆炸
        if export_count <= 0:
            utility = 0.0
        else:
            import math
            utility = min(1.0, math.log1p(export_count) / math.log1p(max_export_count))

        feedback = max(0.0, min(1.0, feedback_score))

        # freshness: 时间衰减
        if created_at is None:
            freshness = 1.0
        else:
            days = max(0.0, (time.time() - created_at) / 86400)
            freshness = 1.0 / (1.0 + days * FRESHNESS_DECAY_RATE)

        total = (
            self.weights["intrinsic"] * intrinsic
            + self.weights["utility"] * utility
            + self.weights["feedback"] * feedback
            + self.weights["freshness"] * freshness
        )

        return GDIScore(
            intrinsic=round(intrinsic, 4),
            utility=round(utility, 4),
            feedback=round(feedback, 4),
            freshness=round(freshness, 4),
            total=round(total, 4),
        )

    def rank(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对一批轨迹计算 GDI 并按总分降序排名.

        Args:
            items: 每项包含 score() 所需的关键字参数.
                例: [{"reward": 0.8, "export_count": 3, "created_at": 170830...}, ...]

        Returns:
            按 GDI total 降序排列的列表，每项增加 "gdi" 字段.
        """
        scored = []
        for item in items:
            gdi = self.score(
                reward=item.get("reward", 0.0),
                export_count=item.get("export_count", 0),
                max_export_count=item.get("max_export_count", 100),
                feedback_score=item.get("feedback_score", 0.5),
                created_at=item.get("created_at"),
            )
            scored.append({**item, "gdi": gdi})

        scored.sort(key=lambda x: x["gdi"].total, reverse=True)
        return scored
