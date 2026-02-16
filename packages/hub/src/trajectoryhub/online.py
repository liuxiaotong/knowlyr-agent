"""在线训练循环 — Model → Collect → Reward → Train 全闭环.

将训练后的模型封装为 agent，在环境中收集轨迹，
计算 reward，导出训练数据，再用 SFTTrainer 微调。重复此过程。

Usage::

    from trajectoryhub.online import online_training_loop

    # 用默认组件（需要 GPU + 模型）
    results = online_training_loop(
        model_path="Qwen/Qwen2.5-Coder-7B",
        env_id="knowlyr/engineering",
        domain="engineering",
    )

    # 自定义组件（dependency injection）
    results = online_training_loop(
        agent_factory=my_agent_factory,
        env_factory=lambda: my_env,
        reward_fn=my_reward,
        skip_training=True,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 类型别名
AgentFn = Callable[[str], dict[str, Any]]
AgentFactory = Callable[[int, str], AgentFn]
EnvFactory = Callable[[], Any]
RewardFn = Callable[[list[dict[str, Any]], dict[str, Any]], float]


# 评估回调类型: (iteration, agent_fn, checkpoint_dir) -> eval_result dict
EvalFn = Callable[[int, AgentFn, str], dict[str, Any]]


@dataclass
class IterationStats:
    """单次迭代的统计结果.

    Attributes:
        iteration: 迭代序号 (0-indexed)
        n_episodes: 收集的轨迹数
        success_rate: 收集阶段成功率 (0.0~1.0)
        avg_reward: 收集阶段平均 reward
        avg_steps: 收集阶段平均步数
        min_reward: 收集阶段最低 reward
        max_reward: 收集阶段最高 reward
        eval_success_rate: 评估阶段成功率 (eval_episodes > 0 时填充)
        eval_avg_reward: 评估阶段平均 reward
        eval_n_episodes: 评估轮数
    """

    iteration: int = 0
    n_episodes: int = 0
    success_rate: float = 0.0
    avg_reward: float = 0.0
    avg_steps: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    # 评估指标 (eval_fn 或 eval_episodes 启用时填充)
    eval_success_rate: float | None = None
    eval_avg_reward: float | None = None
    eval_n_episodes: int | None = None


def online_training_loop(
    *,
    # Agent: 提供 factory 或 model_path (自动创建)
    agent_factory: AgentFactory | None = None,
    model_path: str = "",
    # 环境: 提供 factory 或 env_id (自动创建)
    env_factory: EnvFactory | None = None,
    env_id: str = "",
    # Reward: 提供 fn 或 domain (自动创建)
    reward_fn: RewardFn | None = None,
    domain: str = "conversation",
    # 循环控制
    n_iterations: int = 3,
    n_episodes: int = 10,
    max_steps: int = 20,
    output_dir: str = "./output/online_loop",
    patience: int = 0,
    # 训练控制
    sft_overrides: dict[str, Any] | None = None,
    skip_training: bool = False,
    # 评估控制
    eval_fn: EvalFn | None = None,
    eval_episodes: int = 0,
) -> list[IterationStats]:
    """在线训练循环主函数.

    每次 iteration 执行:
    1. 创建 agent (从 checkpoint 或自定义 factory)
    2. 创建环境 + reward
    3. collect() 收集轨迹
    4. 导出 SFT 训练数据
    5. SFTTrainer 微调
    6. 评估 (eval_fn 或 eval_episodes)
    7. 检查早停

    组件注入优先级:
    - agent_factory > model_path (自动创建 inference agent)
    - env_factory > env_id (自动 make())
    - reward_fn > domain (自动 make_reward_fn())

    Args:
        agent_factory: Agent 工厂函数 (iteration, checkpoint_dir) -> agent_fn.
            iteration 是当前迭代序号, checkpoint_dir 是最新 checkpoint 路径。
        model_path: HuggingFace 模型名或本地 checkpoint (agent_factory 为空时使用)
        env_factory: 环境工厂函数 () -> AgentEnv 实例
        env_id: 环境 ID (env_factory 为空时使用)
        reward_fn: 步骤级 reward 函数 (steps, action) -> float
        domain: 领域 (reward_fn 为空时用于创建默认 reward)
        n_iterations: 训练循环次数
        n_episodes: 每次循环收集的轨迹数
        max_steps: 每条轨迹最大步数
        output_dir: 输出目录
        patience: 早停耐心值 (0 = 不使用早停)
        sft_overrides: SFTConfig 覆盖参数 (如 num_epochs, batch_size 等)
        skip_training: 跳过训练步骤 (用于测试或纯收集)
        eval_fn: 自定义评估回调 (iteration, agent_fn, checkpoint) -> result dict.
            result 需包含 success_rate 和 avg_reward 键。
        eval_episodes: 内置评估轮数。> 0 时在每轮训练后用 evaluate_agent() 评估。
            eval_fn 优先级高于 eval_episodes。

    Returns:
        每次迭代的统计结果列表
    """
    from trajectoryhub.collect import collect, make_reward_fn
    from trajectoryhub.exporter import DatasetExporter

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[IterationStats] = []
    best_reward = float("-inf")
    patience_counter = 0

    for iteration in range(n_iterations):
        iter_dir = out / f"iter-{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Iteration %d/%d", iteration + 1, n_iterations)
        logger.info("=" * 60)

        # ── 确定 checkpoint 路径 ──
        if iteration == 0:
            checkpoint = model_path
        else:
            checkpoint = str(out / f"iter-{iteration - 1}" / "final")

        # ── Step 1: 创建 agent ──
        if agent_factory is not None:
            agent = agent_factory(iteration, checkpoint)
        else:
            agent = _default_agent_factory(checkpoint, domain)

        # ── Step 2: 创建环境 + reward ──
        if env_factory is not None:
            env = env_factory()
        else:
            from knowlyrcore.registry import make
            env = make(env_id)

        if reward_fn is None:
            actual_reward_fn = make_reward_fn(domain=domain)
        else:
            actual_reward_fn = reward_fn

        # ── Step 3: 收集轨迹 ──
        logger.info("收集轨迹: %d episodes, max_steps=%d", n_episodes, max_steps)
        trajectories = collect(
            env,
            agent=agent,
            n_episodes=n_episodes,
            max_steps=max_steps,
            agent_name=f"online-agent-iter{iteration}",
            model_name=model_path or "custom",
            reward_fn=actual_reward_fn,
        )

        # ── Step 4: 统计 ──
        stats = _compute_stats(iteration, trajectories)
        results.append(stats)

        logger.info(
            "轨迹统计: success=%.1f%%, avg_reward=%.3f, avg_steps=%.1f",
            stats.success_rate * 100, stats.avg_reward, stats.avg_steps,
        )

        # ── 早停检查 ──
        if patience > 0:
            if stats.avg_reward > best_reward:
                best_reward = stats.avg_reward
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    "早停计数: %d/%d (best_reward=%.3f)",
                    patience_counter, patience, best_reward,
                )
            if patience_counter >= patience:
                logger.info(
                    "早停触发: 连续 %d 轮 avg_reward 未改善 (best=%.3f)",
                    patience, best_reward,
                )
                break

        # ── Step 5: 导出训练数据 ──
        traj_file = iter_dir / "trajectories.jsonl"
        with open(traj_file, "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj, ensure_ascii=False, default=str) + "\n")

        exporter = DatasetExporter(str(traj_file))
        sft_file = iter_dir / "sft_train.jsonl"
        export_result = exporter.export_sft(str(sft_file))
        logger.info("SFT 导出: %d 条记录 → %s", export_result.total_records, sft_file)

        # ── Step 6: 训练 ──
        if skip_training:
            logger.info("跳过训练 (skip_training=True)")
        else:
            if not sft_file.exists() or export_result.total_records == 0:
                logger.warning("无训练数据，跳过本轮训练")
                continue

            _run_sft_training(
                model_path=checkpoint,
                train_file=str(sft_file),
                output_dir=str(iter_dir),
                overrides=sft_overrides,
            )
            logger.info("SFT 训练完成 → %s", iter_dir / "final")

        # ── Step 7: 评估 ──
        eval_result = _run_evaluation(
            eval_fn=eval_fn,
            eval_episodes=eval_episodes,
            iteration=iteration,
            agent_factory=agent_factory,
            checkpoint=checkpoint,
            env_factory=env_factory,
            env_id=env_id,
            reward_fn=actual_reward_fn,
            max_steps=max_steps,
            domain=domain,
        )
        if eval_result is not None:
            stats.eval_success_rate = eval_result.get("success_rate")
            stats.eval_avg_reward = eval_result.get("avg_reward")
            stats.eval_n_episodes = eval_result.get("n_episodes", eval_episodes)
            logger.info(
                "评估: success=%.1f%%, avg_reward=%.3f (%d episodes)",
                (stats.eval_success_rate or 0) * 100,
                stats.eval_avg_reward or 0,
                stats.eval_n_episodes or 0,
            )

        # ── Step 8: 保存迭代结果 ──
        stats_file = iter_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(asdict(stats), f, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    logger.info("训练循环完成: %d 轮迭代", len(results))
    for s in results:
        logger.info(
            "  Iter %d: success=%.1f%%, reward=%.3f, steps=%.1f",
            s.iteration, s.success_rate * 100, s.avg_reward, s.avg_steps,
        )

    summary_file = out / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in results], f, ensure_ascii=False, indent=2)
    logger.info("汇总已保存: %s", summary_file)

    return results


# ── 内部工具函数 ────────────────────────────────────────────────


def _default_agent_factory(checkpoint: str, domain: str) -> AgentFn:
    """从 checkpoint 创建默认 agent."""
    from trajectoryhub.inference import create_model_agent
    return create_model_agent(
        checkpoint,
        system_prompt=f"你是一个专业的{domain}领域 AI 助手。",
    )


def _compute_stats(iteration: int, trajectories: list[dict[str, Any]]) -> IterationStats:
    """从轨迹列表计算统计."""
    rewards: list[float] = []
    step_counts: list[int] = []
    successes = 0

    for traj in trajectories:
        traj_reward = sum(s.get("reward", 0.0) for s in traj.get("steps", []))
        rewards.append(traj_reward)
        step_counts.append(len(traj.get("steps", [])))
        if traj.get("outcome", {}).get("success"):
            successes += 1

    return IterationStats(
        iteration=iteration,
        n_episodes=len(trajectories),
        success_rate=successes / max(len(trajectories), 1),
        avg_reward=sum(rewards) / max(len(rewards), 1),
        avg_steps=sum(step_counts) / max(len(step_counts), 1),
        min_reward=min(rewards) if rewards else 0.0,
        max_reward=max(rewards) if rewards else 0.0,
    )


def _run_evaluation(
    *,
    eval_fn: EvalFn | None,
    eval_episodes: int,
    iteration: int,
    agent_factory: AgentFactory | None,
    checkpoint: str,
    env_factory: EnvFactory | None,
    env_id: str,
    reward_fn: RewardFn | None,
    max_steps: int,
    domain: str,
) -> dict[str, Any] | None:
    """运行评估 (如果启用).

    优先使用 eval_fn，否则用内置 evaluate_agent()。
    """
    if eval_fn is not None:
        # 自定义评估: 重建 agent 传给回调
        if agent_factory is not None:
            agent = agent_factory(iteration, checkpoint)
        else:
            agent = _default_agent_factory(checkpoint, domain)
        return eval_fn(iteration, agent, checkpoint)

    if eval_episodes > 0:
        # 内置评估: 用 evaluate_agent()
        try:
            from trajectoryhub.evaluate import evaluate_agent
        except RuntimeError:
            logger.warning("评估依赖未安装，跳过评估")
            return None

        if agent_factory is not None:
            agent = agent_factory(iteration, checkpoint)
        else:
            agent = _default_agent_factory(checkpoint, domain)

        if env_factory is not None:
            env = env_factory()
        else:
            from knowlyrcore.registry import make
            env = make(env_id)

        return evaluate_agent(
            agent_fn=agent,
            env=env,
            n_episodes=eval_episodes,
            max_steps=max_steps,
            reward_fn=reward_fn,
        )

    return None


def _run_sft_training(
    model_path: str,
    train_file: str,
    output_dir: str,
    overrides: dict[str, Any] | None = None,
) -> None:
    """执行 SFT 训练."""
    from agenttrainer.config import SFTConfig
    from agenttrainer.trainers.sft import SFTTrainer

    defaults = {
        "model_name_or_path": model_path,
        "train_file": train_file,
        "output_dir": output_dir,
        "num_epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "agent_format": True,
        "mask_observations": True,
        "save_steps": 0,
    }
    if overrides:
        defaults.update(overrides)

    config = SFTConfig(**defaults)
    trainer = SFTTrainer(config)
    trainer.train()
