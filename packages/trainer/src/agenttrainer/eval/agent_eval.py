"""Agent 级别评估 — 在环境中运行 agent 并计算成功率/reward 分布.

与 evaluator.py 的区别:
- evaluator.py: 训练指标（perplexity、token accuracy）
- agent_eval.py: 部署指标（成功率、平均 reward、步数分布）

Usage::

    from agenttrainer.eval.agent_eval import evaluate_agent

    # 评估 mock agent
    results = evaluate_agent(
        agent_fn=my_agent,
        env_id="knowlyr/conversation",
        n_episodes=20,
    )
    print(f"成功率: {results['success_rate']:.1%}")
    print(f"平均 reward: {results['avg_reward']:.3f}")

    # 评估训练后模型
    results = evaluate_agent(
        model_path="./checkpoints/step-1000",
        env_id="knowlyr/engineering",
        n_episodes=50,
        system_prompt="你是代码审查员",
    )

    # 对比多个模型
    comparison = compare_agents(
        agents={"baseline": agent_a, "finetuned": agent_b},
        env_id="knowlyr/conversation",
        n_episodes=30,
    )
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any, Callable

from agenttrainer.eval.stats import (
    confidence_interval,
    welch_t_test,
    bonferroni_correct,
)

logger = logging.getLogger(__name__)


def significance_test(
    data_a: list[float],
    data_b: list[float],
) -> dict[str, Any]:
    """Welch's t 检验 (向后兼容包装器).

    新代码应使用 agenttrainer.eval.stats.welch_t_test()。

    Args:
        data_a: 样本 A 的数据
        data_b: 样本 B 的数据

    Returns:
        {"t_statistic": float, "p_approx": str, "significant": bool, "effect_size": float}
    """
    result = welch_t_test(data_a, data_b)
    return {
        "t_statistic": result.statistic,
        "p_approx": result.p_value_approx,
        "significant": result.significant,
        "effect_size": result.effect_size,
    }


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

    提供 agent_fn 或 model_path 二选一:
    - agent_fn: 直接传入 agent 函数
    - model_path: 从 checkpoint 加载模型并创建 agent

    提供 env_id 或 env 二选一:
    - env_id: 通过 Registry make() 创建环境
    - env: 直接传入环境实例

    Args:
        agent_fn: agent 函数，签名 (observation: str) -> action dict
        model_path: HuggingFace 模型名或本地 checkpoint 路径
        env_id: 环境 ID (如 "knowlyr/conversation")
        env: AgentEnv 实例（与 env_id 二选一）
        n_episodes: 评估轮数
        max_steps: 每轮最大步数
        system_prompt: system prompt (model_path 模式下使用)
        reward_fn: 可选的 reward 函数
        tasks: 任务列表。如果提供，每轮使用对应任务；否则使用默认任务。
               如果 tasks 数量 < n_episodes，则循环使用。

    Returns:
        评估结果字典::

            {
                "success_rate": 0.75,
                "avg_reward": 0.62,
                "std_reward": 0.15,
                "avg_steps": 4.2,
                "std_steps": 1.8,
                "min_reward": 0.1,
                "max_reward": 0.95,
                "reward_distribution": {"<0.25": 2, "0.25-0.5": 3, ...},
                "n_episodes": 20,
                "episodes": [...],  # 每轮详细数据
            }

    Raises:
        ValueError: agent_fn 和 model_path 都未提供
        RuntimeError: 依赖包未安装
    """
    if agent_fn is None and not model_path:
        raise ValueError("必须提供 agent_fn 或 model_path")

    # 加载模型创建 agent
    if agent_fn is None:
        from agenttrainer.inference import AgentInference

        inference = AgentInference.from_pretrained(
            model_path, temperature=0.0, max_new_tokens=512,
        )
        agent_fn = inference.create_agent(system_prompt=system_prompt)

    # 创建环境
    if env is None:
        if not env_id:
            raise ValueError("必须提供 env_id 或 env")
        try:
            from knowlyrcore.registry import make as _make_env
        except ImportError:
            raise RuntimeError("需要 knowlyr-core >= 0.1.0: pip install knowlyr-core")
        env = _make_env(env_id)

    # 收集轨迹
    try:
        from trajectoryhub.collect import collect
    except ImportError:
        raise RuntimeError("需要 knowlyr-hub >= 0.1.0: pip install knowlyr-hub")

    episodes: list[dict[str, Any]] = []

    for i in range(n_episodes):
        # 确定本轮任务
        task = None
        if tasks:
            task = tasks[i % len(tasks)]

        # 每轮需要独立的 agent 闭包（重置历史）
        if model_path:
            # model_path 模式: 每轮重新创建 agent 以重置对话历史
            episode_agent = agent_fn  # 已在上面创建
        else:
            episode_agent = agent_fn

        # 用 collect 收集单轮轨迹
        trajs = collect(
            env,
            agent=episode_agent,
            n_episodes=1,
            max_steps=max_steps,
            reward_fn=reward_fn,
            task=task,
        )

        if trajs:
            traj = trajs[0]
            traj_reward = sum(
                s.get("reward", 0.0) for s in traj.get("steps", [])
            )
            episode_data = {
                "episode": i,
                "success": traj.get("outcome", {}).get("success", False),
                "total_reward": traj_reward,
                "n_steps": len(traj.get("steps", [])),
                "outcome": traj.get("outcome", {}),
            }
            episodes.append(episode_data)

            logger.info(
                "Episode %d/%d: success=%s, reward=%.3f, steps=%d",
                i + 1, n_episodes,
                episode_data["success"],
                episode_data["total_reward"],
                episode_data["n_steps"],
            )

    env.close()

    return _compute_stats(episodes, n_episodes)


def compare_agents(
    *,
    agents: dict[str, Callable[[str], dict[str, Any]]],
    env_id: str = "",
    env: Any = None,
    n_episodes: int = 10,
    max_steps: int = 30,
    reward_fn: Callable[[list[dict], dict], float] | None = None,
    tasks: list[Any] | None = None,
    multiple_testing_correction: bool = True,
) -> dict[str, dict[str, Any]]:
    """对比多个 agent 在同一环境上的表现.

    Args:
        agents: {名称: agent_fn} 字典
        env_id: 环境 ID
        env: 环境实例（每个 agent 会独立创建）
        n_episodes: 每个 agent 的评估轮数
        max_steps: 每轮最大步数
        reward_fn: 可选的 reward 函数
        tasks: 共享任务列表（确保公平对比）
        multiple_testing_correction: 是否应用 Bonferroni 校正 (>2 agents 时)

    Returns:
        {agent_name: evaluate_agent 结果} 字典，额外包含::

            {
                "agent_a": { ... evaluate_agent 结果 ... },
                "agent_b": { ... evaluate_agent 结果 ... },
                "_leaderboard": [           # 按 avg_reward 降序排行
                    {"rank": 1, "agent": "agent_a", "success_rate": 0.8,
                     "avg_reward": 0.72, "avg_steps": 4.5},
                    {"rank": 2, "agent": "agent_b", ...},
                ],
                "_comparisons": {           # 两两 Welch's t 检验
                    "agent_a_vs_agent_b": {
                        "t_statistic": 2.31, "p_approx": "p<0.05",
                        "significant": True, "effect_size": 0.65,
                    },
                },
                "_corrected": {             # Bonferroni 校正 (>2 agents)
                    "agent_a_vs_agent_b": True,
                    "agent_a_vs_agent_c": False,
                },
            }
    """
    results: dict[str, dict[str, Any]] = {}

    for name, agent_fn in agents.items():
        logger.info("评估 agent: %s", name)
        results[name] = evaluate_agent(
            agent_fn=agent_fn,
            env_id=env_id,
            env=env,
            n_episodes=n_episodes,
            max_steps=max_steps,
            reward_fn=reward_fn,
            tasks=tasks,
        )

    if len(results) <= 1:
        return results

    # Leaderboard (按 avg_reward 降序)
    agent_names = [k for k in results if not k.startswith("_")]
    ranked = sorted(agent_names, key=lambda k: results[k]["avg_reward"], reverse=True)
    results["_leaderboard"] = [  # type: ignore[assignment]
        {
            "rank": i + 1,
            "agent": name,
            "success_rate": results[name]["success_rate"],
            "avg_reward": results[name]["avg_reward"],
            "avg_steps": results[name]["avg_steps"],
        }
        for i, name in enumerate(ranked)
    ]

    best_agent = ranked[0]
    logger.info(
        "最佳 agent: %s (success=%.1f%%, reward=%.3f)",
        best_agent,
        results[best_agent]["success_rate"] * 100,
        results[best_agent]["avg_reward"],
    )

    # 两两显著性检验 (reward)
    comparisons: dict[str, dict[str, Any]] = {}
    p_values_list: list[str] = []
    comparison_keys: list[str] = []

    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a, b = agent_names[i], agent_names[j]
            rewards_a = [e["total_reward"] for e in results[a]["episodes"]]
            rewards_b = [e["total_reward"] for e in results[b]["episodes"]]
            test_result = significance_test(rewards_a, rewards_b)
            key = f"{a}_vs_{b}"
            comparisons[key] = test_result
            p_values_list.append(test_result["p_approx"])
            comparison_keys.append(key)
            logger.info(
                "  %s vs %s: t=%.3f, %s, effect_size=%.3f",
                a, b,
                test_result["t_statistic"],
                test_result["p_approx"],
                test_result["effect_size"],
            )

    results["_comparisons"] = comparisons  # type: ignore[assignment]

    # Bonferroni 校正 (>2 agents 时)
    if multiple_testing_correction and len(agent_names) > 2 and p_values_list:
        corrected = bonferroni_correct(p_values_list)
        corrected_dict = dict(zip(comparison_keys, corrected))
        results["_corrected"] = corrected_dict  # type: ignore[assignment]
        logger.info(
            "多重比较校正 (Bonferroni): %d/%d 对比显著 (α=0.05)",
            sum(corrected),
            len(corrected),
        )

    return results


def _compute_stats(
    episodes: list[dict[str, Any]],
    n_episodes: int,
) -> dict[str, Any]:
    """计算评估统计指标."""
    if not episodes:
        return {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "std_reward": 0.0,
            "avg_steps": 0.0,
            "std_steps": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "reward_distribution": {},
            "n_episodes": n_episodes,
            "episodes": [],
        }

    rewards = [e["total_reward"] for e in episodes]
    step_counts = [e["n_steps"] for e in episodes]
    successes = sum(1 for e in episodes if e["success"])

    # Reward 分布 (4 个区间)
    distribution = {"<0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, ">=0.75": 0}
    for r in rewards:
        if r < 0.25:
            distribution["<0.25"] += 1
        elif r < 0.5:
            distribution["0.25-0.5"] += 1
        elif r < 0.75:
            distribution["0.5-0.75"] += 1
        else:
            distribution[">=0.75"] += 1

    # 置信区间
    reward_ci = confidence_interval(rewards)
    steps_ci = confidence_interval([float(s) for s in step_counts])

    # 成功率 CI (Wilson score interval 简化版)
    n = len(episodes)
    p = successes / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    success_ci = (max(0.0, center - spread), min(1.0, center + spread))

    return {
        "success_rate": successes / max(len(episodes), 1),
        "success_rate_ci": success_ci,
        "avg_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_ci": reward_ci,
        "avg_steps": statistics.mean(step_counts),
        "std_steps": statistics.stdev(step_counts) if len(step_counts) > 1 else 0.0,
        "steps_ci": steps_ci,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "reward_distribution": distribution,
        "n_episodes": n_episodes,
        "episodes": episodes,
    }
