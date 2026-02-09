"""AgentEnv — Gymnasium 风格 Agent 环境协议.

借鉴 Gymnasium Env + BrowserGym + AgentGym 的核心模式：
- reset() → TimeStep
- step(action) → TimeStep
- close()
- Wrapper 可组合
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from knowlyrcore.timestep import TimeStep


class AgentEnv(ABC):
    """Agent 环境协议.

    所有环境（Docker 沙箱、浏览器、API mock 等）都实现此接口。
    action 格式统一为 ``{"tool": "...", "params": {...}}``。

    Usage::

        env = SandboxEnv(config)
        ts = env.reset(task=my_task)
        while not ts.done:
            action = agent.act(ts.observation)
            ts = env.step(action)
        env.close()
    """

    metadata: dict[str, Any] = {}
    domain: str = "coding"

    @abstractmethod
    def reset(
        self,
        *,
        task: Any | None = None,
        seed: int | None = None,
    ) -> TimeStep:
        """重置环境到初始状态.

        Args:
            task: 任务信息 (TaskInfo 或兼容 dict)
            seed: 随机种子（用于可复现性）

        Returns:
            初始 TimeStep
        """
        ...

    @abstractmethod
    def step(self, action: dict[str, Any]) -> TimeStep:
        """执行一步动作.

        Args:
            action: 动作字典，格式 {"tool": "...", "params": {...}}

        Returns:
            执行后的 TimeStep
        """
        ...

    def close(self) -> None:
        """清理资源（默认空实现）."""

    @property
    def available_tools(self) -> list[str]:
        """当前可用的工具/动作名列表."""
        return []

    @property
    def unwrapped(self) -> AgentEnv:
        """返回最内层非 Wrapper 环境."""
        return self

    def __enter__(self) -> AgentEnv:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} domain={self.domain!r}>"


class EnvWrapper(AgentEnv):
    """环境包装器基类 — 对应 gymnasium.Wrapper.

    透传所有方法到内部环境，子类只需覆盖想要修改的方法。

    Usage::

        class MyWrapper(EnvWrapper):
            def step(self, action):
                ts = self.env.step(action)
                ts.reward = compute_reward(ts)
                return ts
    """

    def __init__(self, env: AgentEnv):
        self.env = env

    def reset(self, **kwargs: Any) -> TimeStep:
        """透传 reset."""
        return self.env.reset(**kwargs)

    def step(self, action: dict[str, Any]) -> TimeStep:
        """透传 step."""
        return self.env.step(action)

    def close(self) -> None:
        """透传 close."""
        self.env.close()

    @property
    def available_tools(self) -> list[str]:
        """透传 available_tools."""
        return self.env.available_tools

    @property
    def unwrapped(self) -> AgentEnv:
        """递归返回最内层环境."""
        return self.env.unwrapped

    @property
    def domain(self) -> str:  # type: ignore[override]
        """透传 domain."""
        return self.env.domain

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.env!r})>"
