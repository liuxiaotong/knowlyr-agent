"""SandboxEnv — 将 Sandbox 包装为 Gymnasium 风格 AgentEnv.

Usage::

    from agentsandbox.env import SandboxEnv
    from knowlyrcore import TaskInfo

    env = SandboxEnv()
    ts = env.reset(task=TaskInfo(repo="owner/repo", base_commit="abc123"))
    while not ts.done:
        action = {"tool": "shell", "params": {"command": "pytest tests/"}}
        ts = env.step(action)
    env.close()
"""

from __future__ import annotations

import logging
from typing import Any

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.sandbox import Sandbox
from agentsandbox.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class SandboxEnv(AgentEnv):
    """Docker 沙箱环境 — AgentEnv 适配器.

    将现有 Sandbox 的 execute_tool() 接口适配为 Gymnasium 风格
    reset() / step() / close()。

    Attributes:
        domain: 环境领域，默认 "coding"
        metadata: 环境元信息
    """

    domain = "coding"
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: SandboxConfig | None = None,
        task_config: TaskConfig | None = None,
        max_steps: int = 30,
    ):
        """初始化.

        Args:
            config: Docker 沙箱配置
            task_config: 任务配置（也可在 reset 时通过 task 传入）
            max_steps: 最大步数（超过后 truncated）
        """
        self._config = config or SandboxConfig()
        self._task_config = task_config
        self._max_steps = max_steps
        self._sandbox: Sandbox | None = None
        self._step_count = 0
        self._history: list[dict[str, Any]] = []

    def reset(self, *, task: Any | None = None, seed: int | None = None) -> TimeStep:
        """重置沙箱环境.

        如果传入 task (TaskInfo)，会从中提取 repo / base_commit
        构造 TaskConfig。

        Args:
            task: TaskInfo 或兼容对象
            seed: 未使用（保留接口兼容）

        Returns:
            初始 TimeStep
        """
        # 从 TaskInfo 构造 TaskConfig
        if task is not None:
            repo = getattr(task, "repo", "") or ""
            base_commit = getattr(task, "base_commit", "") or ""
            test_command = getattr(task, "test_command", "") or ""
            description = getattr(task, "description", "") or ""
            domain = getattr(task, "domain", "coding") or "coding"
            self._task_config = TaskConfig(
                repo_url=repo,
                base_commit=base_commit,
                test_command=test_command,
                description=description,
                domain=domain,
            )

        # 重置或创建沙箱
        if self._sandbox is not None:
            self._sandbox.reset()
        else:
            if self._task_config is None:
                self._task_config = TaskConfig()
            self._sandbox = Sandbox.create(self._config, self._task_config)

        self._step_count = 0
        self._history = []

        logger.info("SandboxEnv 就绪: container=%s", self._sandbox.container_id)
        return TimeStep(
            observation="沙箱就绪",
            info={
                "container_id": self._sandbox.container_id,
                "max_steps": self._max_steps,
            },
        )

    def step(self, action: dict[str, Any]) -> TimeStep:
        """执行一步动作.

        Args:
            action: {"tool": "shell", "params": {"command": "ls"}}

        Returns:
            TimeStep
        """
        if self._sandbox is None:
            return TimeStep(
                observation="",
                terminated=True,
                info={"error": "沙箱未启动，请先调用 reset()"},
            )

        tool = action.get("tool", "")
        params = action.get("params", {})

        result = self._sandbox.execute_tool(tool, params)
        self._step_count += 1

        self._history.append({
            "step": self._step_count,
            "tool": tool,
            "params": params,
            "output": result.output,
            "exit_code": result.exit_code,
        })

        terminated = tool in ("submit", "finish")
        truncated = self._step_count >= self._max_steps

        return TimeStep(
            observation=result.output,
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
            info={
                "exit_code": result.exit_code,
                "error": result.error,
                "step": self._step_count,
                "tool": tool,
            },
        )

    def close(self) -> None:
        """关闭沙箱，清理 Docker 资源."""
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None
            logger.info("SandboxEnv 已关闭")

    @property
    def available_tools(self) -> list[str]:
        """返回沙箱中可用的工具列表."""
        return list(TOOL_REGISTRY.keys())

    @property
    def history(self) -> list[dict[str, Any]]:
        """当前 episode 的执行历史."""
        return list(self._history)
