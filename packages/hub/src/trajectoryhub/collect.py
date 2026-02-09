"""轨迹收集 — 在环境中运行 agent 并收集轨迹.

借鉴 AgentGym 的轨迹收集 + SWE-Gym 的训练数据生产模式。

Usage::

    from trajectoryhub.collect import collect

    def my_agent(observation: str) -> dict:
        return {"tool": "bash", "params": {"command": "ls"}}

    trajectories = collect("knowlyr/sandbox", agent=my_agent, n_episodes=5)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from knowlyrcore.registry import make as _make_env
    from knowlyrcore.wrappers import MaxStepsWrapper, RecorderWrapper

    _HAS_CORE_ENV = True
except ImportError:
    _HAS_CORE_ENV = False


def collect(
    env: Any,
    agent: Callable[[str], dict[str, Any]],
    n_episodes: int = 1,
    max_steps: int = 30,
    agent_name: str = "",
    model_name: str = "",
    task: Any = None,
) -> list[dict[str, Any]]:
    """在环境中运行 agent，收集轨迹.

    Args:
        env: AgentEnv 实例或环境 ID 字符串 (如 "knowlyr/sandbox")
        agent: agent 函数，签名 (observation: str) -> action dict
        n_episodes: 收集轮数
        max_steps: 每轮最大步数
        agent_name: Agent 名称（记录在轨迹中）
        model_name: 模型名称（记录在轨迹中）
        task: 任务信息（传给 env.reset()）

    Returns:
        轨迹字典列表，每个包含 task/agent/model/steps/outcome/metadata

    Raises:
        RuntimeError: knowlyr-core 未安装
    """
    if not _HAS_CORE_ENV:
        raise RuntimeError(
            "轨迹收集需要 knowlyr-core >= 0.1.0: pip install knowlyr-core"
        )

    # 解析 env：字符串 → make()，实例 → 直接用
    if isinstance(env, str):
        env = _make_env(env)

    # 包装：MaxSteps + Recorder
    env = MaxStepsWrapper(env, max_steps=max_steps)
    env = RecorderWrapper(env, agent_name=agent_name, model_name=model_name)

    trajectories: list[dict[str, Any]] = []

    for episode in range(n_episodes):
        ts = env.reset(task=task)
        logger.info("Episode %d/%d 开始", episode + 1, n_episodes)

        while not ts.done:
            action = agent(ts.observation)
            ts = env.step(action)

        traj = env.get_trajectory()
        trajectories.append(traj)
        logger.info(
            "Episode %d/%d 完成: %d 步, success=%s",
            episode + 1,
            n_episodes,
            len(traj["steps"]),
            traj["outcome"].get("success"),
        )

    env.close()
    logger.info("收集完成: %d 条轨迹", len(trajectories))
    return trajectories
