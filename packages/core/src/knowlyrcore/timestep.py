"""TimeStep — Gymnasium 风格的环境返回值.

对应 Gymnasium step() 返回的 (obs, reward, terminated, truncated, info) 五元组，
打包为 dataclass 更 Pythonic。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeStep:
    """Agent 环境单步返回值.

    Attributes:
        observation: 环境反馈（工具输出 / 页面内容 / 状态描述）
        reward: 即时 reward（默认 0.0，由 RewardWrapper 注入）
        terminated: 任务是否完成（成功或放弃）
        truncated: 是否被截断（超时 / 超步数）
        info: 附加信息（exit_code, step_count, tokens 等）
    """

    observation: str = ""
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """是否结束（terminated 或 truncated）."""
        return self.terminated or self.truncated
