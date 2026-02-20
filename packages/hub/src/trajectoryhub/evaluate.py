"""Agent 评估桥接 — Hub 层便捷入口.

转发到 agenttrainer.eval，提供 Hub 层的一站式评估体验。

Usage::

    from trajectoryhub.evaluate import evaluate_agent, compare_agents

    # 评估单个 agent
    results = evaluate_agent(agent_fn=my_agent, env_id="knowlyr/conversation")
    print(f"成功率: {results['success_rate']:.1%}")

    # 对比多个 agent
    comparison = compare_agents(
        agents={"baseline": agent_a, "finetuned": agent_b},
        env_id="knowlyr/conversation",
    )

    # 评测结果回写到 CAS（更新 GDI feedback 维度）
    from trajectoryhub.evaluate import write_back_feedback
    write_back_feedback("data/index.sqlite", eval_results)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def evaluate_agent(
    *,
    agent_fn: Callable[[str], dict[str, Any]] | None = None,
    model_path: str = "",
    env_id: str = "",
    env: Any = None,
    n_episodes: int = 10,
    max_steps: int = 30,
    system_prompt: str = "",
    reward_fn: Callable[[list[dict], dict], float] | None = None,
    tasks: list[Any] | None = None,
) -> dict[str, Any]:
    """评估 agent 在环境中的表现.

    Hub 层便捷入口，转发到 agenttrainer.eval.agent_eval.evaluate_agent()。

    Args:
        agent_fn: agent 函数 (observation: str) -> action dict
        model_path: 模型 checkpoint 路径 (与 agent_fn 二选一)
        env_id: 环境 ID
        env: 环境实例 (与 env_id 二选一)
        n_episodes: 评估轮数
        max_steps: 每轮最大步数
        system_prompt: system prompt (model_path 模式)
        reward_fn: 可选的 reward 函数
        tasks: 任务列表

    Returns:
        评估结果字典 (success_rate, avg_reward, reward_distribution 等)

    Raises:
        RuntimeError: knowlyr-trainer 未安装
    """
    try:
        from agenttrainer.eval.agent_eval import evaluate_agent as _evaluate
    except ImportError:
        raise RuntimeError(
            "evaluate_agent 需要 knowlyr-trainer >= 0.1.0: pip install knowlyr-trainer"
        )

    return _evaluate(
        agent_fn=agent_fn,
        model_path=model_path,
        env_id=env_id,
        env=env,
        n_episodes=n_episodes,
        max_steps=max_steps,
        system_prompt=system_prompt,
        reward_fn=reward_fn,
        tasks=tasks,
    )


def compare_agents(
    *,
    agents: dict[str, Callable[[str], dict[str, Any]]],
    env_id: str = "",
    env: Any = None,
    n_episodes: int = 10,
    max_steps: int = 30,
    reward_fn: Callable[[list[dict], dict], float] | None = None,
    tasks: list[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """对比多个 agent.

    Hub 层便捷入口，转发到 agenttrainer.eval.agent_eval.compare_agents()。

    Args:
        agents: {名称: agent_fn} 字典
        env_id: 环境 ID
        env: 环境实例
        n_episodes: 每个 agent 的评估轮数
        max_steps: 每轮最大步数
        reward_fn: 可选的 reward 函数
        tasks: 共享任务列表

    Returns:
        {agent_name: 评估结果} + _leaderboard + _comparisons

    Raises:
        RuntimeError: knowlyr-trainer 未安装
    """
    try:
        from agenttrainer.eval.agent_eval import compare_agents as _compare
    except ImportError:
        raise RuntimeError(
            "compare_agents 需要 knowlyr-trainer >= 0.1.0: pip install knowlyr-trainer"
        )

    return _compare(
        agents=agents,
        env_id=env_id,
        env=env,
        n_episodes=n_episodes,
        max_steps=max_steps,
        reward_fn=reward_fn,
        tasks=tasks,
    )


def write_back_feedback(
    store_path: str,
    feedback: dict[str, float],
) -> int:
    """将评测结果回写到 CAS，更新 GDI feedback 维度并重新计算排名.

    训练后用新模型跑相同任务，对比 reward 提升，将提升值作为 feedback 分数
    写回训练数据中对应轨迹的 GDI。这样「被用来训练且效果好的轨迹」GDI 会升高，
    形成正循环。

    Args:
        store_path: CAS SQLite 存储路径.
        feedback: {content_hash: feedback_score} 映射.
            feedback_score ∈ [0, 1]，0 = 训练后无提升，1 = 大幅提升.

    Returns:
        更新的轨迹数.

    Usage::

        # 训练后评测，计算每条训练数据的贡献
        feedback = {}
        for h in training_hashes:
            traj = store.get(h)
            old_reward = traj["reward"]
            new_reward = new_model_reward_on_same_task(traj["task_id"])
            improvement = max(0, min(1, (new_reward - old_reward) / max(old_reward, 0.01)))
            feedback[h] = improvement

        updated = write_back_feedback("data/index.sqlite", feedback)
        print(f"更新了 {updated} 条轨迹的 GDI")
    """
    from trajectoryhub.cas import CAStore
    from trajectoryhub.gdi import GDIScorer

    store = CAStore(store_path)
    scorer = GDIScorer()
    updated = 0

    try:
        gdi_batch: dict[str, float] = {}
        for content_hash, feedback_score in feedback.items():
            row = store.get(content_hash)
            if row is None:
                continue
            gdi = scorer.score(
                reward=row.get("reward", 0.0),
                export_count=row.get("export_count", 0),
                feedback_score=max(0.0, min(1.0, feedback_score)),
                created_at=row.get("created_at"),
            )
            gdi_batch[content_hash] = gdi.total
            updated += 1

        if gdi_batch:
            store.update_gdi_batch(gdi_batch)
            logger.info("GDI feedback 回写完成: %d 条轨迹更新", updated)
    finally:
        store.close()

    return updated
