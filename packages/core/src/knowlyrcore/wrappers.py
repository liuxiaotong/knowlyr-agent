"""内置 Wrapper 套件 — 可组合的环境变换.

借鉴 Gymnasium Wrapper 模式，提供常用的环境包装器。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

from knowlyrcore.env import EnvWrapper, AgentEnv
from knowlyrcore.timestep import TimeStep


class MaxStepsWrapper(EnvWrapper):
    """限制最大步数 — 超过后自动 truncate.

    Usage::

        env = MaxStepsWrapper(env, max_steps=50)
    """

    def __init__(self, env: AgentEnv, max_steps: int = 30):
        super().__init__(env)
        self._max_steps = max_steps
        self._step_count = 0

    def reset(self, **kwargs: Any) -> TimeStep:
        """重置步数计数器."""
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action: dict[str, Any]) -> TimeStep:
        """执行一步，超过 max_steps 设置 truncated."""
        ts = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self._max_steps and not ts.terminated:
            ts.truncated = True
        return ts

    @property
    def step_count(self) -> int:
        """当前步数."""
        return self._step_count


class TimeoutWrapper(EnvWrapper):
    """限制 wall-clock 时间 — 超过后自动 truncate.

    Usage::

        env = TimeoutWrapper(env, timeout_seconds=300)
    """

    def __init__(self, env: AgentEnv, timeout_seconds: float = 300.0):
        super().__init__(env)
        self._timeout = timeout_seconds
        self._start_time: float = 0.0

    def reset(self, **kwargs: Any) -> TimeStep:
        """重置计时器."""
        self._start_time = time.monotonic()
        return self.env.reset(**kwargs)

    def step(self, action: dict[str, Any]) -> TimeStep:
        """执行一步，超时设置 truncated."""
        ts = self.env.step(action)
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self._timeout and not ts.terminated:
            ts.truncated = True
            ts.info["timeout"] = True
            ts.info["elapsed_seconds"] = elapsed
        return ts

    @property
    def elapsed(self) -> float:
        """已用时间 (秒)."""
        return time.monotonic() - self._start_time


class RewardWrapper(EnvWrapper):
    """每步注入 reward — 调用用户提供的 reward 函数.

    reward_fn 签名: (steps: list[dict], current_action: dict) -> float

    Usage::

        def my_reward(steps, action):
            return 1.0 if action.get("tool") != "think" else 0.0

        env = RewardWrapper(env, reward_fn=my_reward)
    """

    def __init__(
        self,
        env: AgentEnv,
        reward_fn: Callable[[list[dict[str, Any]], dict[str, Any]], float],
    ):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._steps: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> TimeStep:
        """重置步骤历史."""
        self._steps = []
        return self.env.reset(**kwargs)

    def step(self, action: dict[str, Any]) -> TimeStep:
        """执行一步，注入 reward."""
        ts = self.env.step(action)
        step_record = {
            "tool": action.get("tool", ""),
            "params": action.get("params", {}),
            "output": ts.observation,
        }
        self._steps.append(step_record)
        ts.reward = self._reward_fn(self._steps, action)
        return ts


class RecorderWrapper(EnvWrapper):
    """自动录制轨迹 — 每步记录 action + observation.

    录制结果为 list[dict]，可直接传给 RewardEngine 或导出。

    Usage::

        env = RecorderWrapper(env)
        ts = env.reset(task=my_task)
        while not ts.done:
            ts = env.step(action)
        trajectory = env.get_trajectory()
    """

    def __init__(self, env: AgentEnv, agent_name: str = "", model_name: str = ""):
        super().__init__(env)
        self._agent_name = agent_name
        self._model_name = model_name
        self._task: Any = None
        self._steps: list[dict[str, Any]] = []
        self._start_time: str = ""
        self._outcome: dict[str, Any] = {}

    def reset(self, **kwargs: Any) -> TimeStep:
        """重置录制状态."""
        self._task = kwargs.get("task")
        self._steps = []
        self._start_time = datetime.now().isoformat()
        self._outcome = {}
        return self.env.reset(**kwargs)

    def step(self, action: dict[str, Any]) -> TimeStep:
        """记录一步."""
        ts = self.env.step(action)
        step_record = {
            "step_id": len(self._steps),
            "thought": action.get("thought", ""),
            "tool": action.get("tool", ""),
            "params": action.get("params", {}),
            "output": ts.observation,
            "exit_code": ts.info.get("exit_code", 0),
            "reward": ts.reward,
            "timestamp": datetime.now().isoformat(),
        }
        self._steps.append(step_record)

        if ts.terminated or ts.truncated:
            self._outcome = {
                "success": ts.terminated and not ts.truncated,
                "total_steps": len(self._steps),
                "terminated": ts.terminated,
                "truncated": ts.truncated,
            }

        return ts

    def get_trajectory(self) -> dict[str, Any]:
        """获取录制的轨迹.

        Returns:
            轨迹字典，包含 task/agent/model/steps/outcome/metadata
        """
        task_id = ""
        task_desc = ""
        if self._task is not None:
            task_id = getattr(self._task, "task_id", "")
            task_desc = getattr(self._task, "description", "")

        return {
            "task": {
                "task_id": task_id,
                "description": task_desc,
            },
            "agent": self._agent_name,
            "model": self._model_name,
            "steps": list(self._steps),
            "outcome": dict(self._outcome),
            "metadata": {
                "start_time": self._start_time,
                "domain": self.env.domain,
            },
        }
