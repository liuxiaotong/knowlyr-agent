"""轨迹收集 — 在环境中运行 agent 并收集轨迹.

借鉴 AgentGym 的轨迹收集 + SWE-Gym 的训练数据生产模式。

Usage::

    from trajectoryhub.collect import collect

    def my_agent(observation: str) -> dict:
        return {"tool": "bash", "params": {"command": "ls"}}

    trajectories = collect("knowlyr/sandbox", agent=my_agent, n_episodes=5)

    # 带 reward 的收集
    def my_reward(steps, action):
        return 0.5

    trajectories = collect("knowlyr/sandbox", agent=my_agent, reward_fn=my_reward)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from knowlyrcore.registry import make as _make_env
    from knowlyrcore.wrappers import MaxStepsWrapper, RecorderWrapper, RewardWrapper

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
    reward_fn: Callable[[list[dict[str, Any]], dict[str, Any]], float] | None = None,
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
        reward_fn: 可选的 reward 函数，签名 (steps, action) -> float。
                   如果提供，每步会通过 RewardWrapper 注入 reward。

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

    # 包装顺序: MaxSteps → (Reward) → Recorder
    # Recorder 在最外层，才能记录到 RewardWrapper 注入的 reward 值
    env = MaxStepsWrapper(env, max_steps=max_steps)
    if reward_fn is not None:
        env = RewardWrapper(env, reward_fn=reward_fn)
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


def make_reward_fn(
    domain: str = "coding",
    model_name: str = "",
    model_weight: float = 0.5,
) -> Callable[[list[dict[str, Any]], dict[str, Any]], float]:
    """创建 RewardEngine 适配的 reward_fn.

    将轨迹级 RewardEngine 适配为步骤级 (steps, action) -> float。
    model_name 为空时只用规则层（快速、无网络调用）。

    Args:
        domain: 领域标识 (coding/conversation/engineering/advisory)
        model_name: LLM judge 模型名 (如 "gpt-4o-mini")，空则只用规则层
        model_weight: 模型层权重 (0.0~1.0)

    Returns:
        reward_fn: (steps, action) -> float

    Raises:
        RuntimeError: knowlyr-reward 未安装

    Example::

        reward_fn = make_reward_fn(domain="conversation")
        trajectories = collect(env, agent=my_agent, reward_fn=reward_fn)
    """
    try:
        from agentreward.config import RewardConfig
        from agentreward.reward import RewardEngine
    except ImportError:
        raise RuntimeError(
            "make_reward_fn 需要 knowlyr-reward >= 0.1.0: pip install knowlyr-reward"
        )

    config = RewardConfig(
        domain=domain,
        model_name=model_name,
        model_weight=model_weight if model_name else 0.0,
    )
    engine = RewardEngine(config=config)

    def reward_fn(steps: list[dict[str, Any]], action: dict[str, Any]) -> float:
        """每步返回规则层即时评分."""
        partial_traj = {
            "steps": steps,
            "outcome": {"success": False},
        }
        result = engine.score(partial_traj)
        if result.step_rewards:
            return result.step_rewards[-1].total_score
        return 0.0

    return reward_fn
