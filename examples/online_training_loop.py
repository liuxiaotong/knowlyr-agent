"""在线训练循环 — 打通 Model → Collect → Reward → Train 全闭环.

将训练后的模型通过 AgentInference 封装为 agent，在环境中收集轨迹，
计算 reward，导出训练数据，再用 SFTTrainer 微调模型。重复此过程。

Usage::

    # 快速体验（用 mock 环境 + mock agent，无需 GPU）
    python examples/online_training_loop.py --mock

    # 真实训练（需要 GPU + 模型）
    python examples/online_training_loop.py \
        --model_path Qwen/Qwen2.5-Coder-7B \
        --env knowlyr/engineering \
        --n_episodes 20 \
        --n_iterations 3

依赖::

    pip install knowlyr-core knowlyr-sandbox knowlyr-hub knowlyr-reward knowlyr-trainer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

# 本地包
from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep
from knowlyrcore.registry import register, make
from trajectoryhub.collect import collect, make_reward_fn
from trajectoryhub.exporter import DatasetExporter

logger = logging.getLogger(__name__)


# ── Mock 组件（用于无 GPU 体验）──────────────────────────────────


class _MockEnv(AgentEnv):
    """Mock 环境 — 模拟对话交互."""

    domain = "conversation"

    def __init__(self) -> None:
        self._step_count = 0
        self._task_msg = ""

    def reset(self, *, task: Any = None, seed: int | None = None) -> TimeStep:
        self._step_count = 0
        self._task_msg = task or "请帮我分析一下代码质量"
        return TimeStep(observation=self._task_msg)

    def step(self, action: dict) -> TimeStep:
        self._step_count += 1
        tool = action.get("tool", "think")

        if tool == "respond":
            return TimeStep(
                observation=f"用户收到回复: {action.get('params', {}).get('message', '')}",
                terminated=True,
                info={"success": True, "steps": self._step_count},
            )

        return TimeStep(
            observation=f"工具 {tool} 执行完成 (步骤 {self._step_count})",
            info={"step": self._step_count},
        )

    @property
    def available_tools(self) -> list[str]:
        return ["think", "read_file", "grep", "respond"]


def _mock_agent_factory(iteration: int = 0) -> Callable[[str], dict[str, Any]]:
    """创建 mock agent — 模拟模型推理.

    每次迭代 agent 变得"更聪明"（更少步骤完成任务）。
    """
    call_count = 0
    max_steps_before_respond = max(3 - iteration, 1)

    def agent(observation: str) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1

        if call_count >= max_steps_before_respond:
            call_count = 0
            return {"tool": "respond", "params": {"message": "分析完成，代码质量良好"}}

        tools = ["think", "read_file", "grep"]
        tool = tools[(call_count - 1) % len(tools)]
        return {"tool": tool, "params": {"input": f"分析中... (步骤 {call_count})"}}

    return agent


def _mock_reward_fn(steps: list[dict[str, Any]], action: dict[str, Any]) -> float:
    """Mock reward 函数 — 步数越少得分越高."""
    tool = action.get("tool", "")
    if tool == "respond":
        # 回复时给最终 reward: 步骤越少越好
        step_count = len(steps) + 1
        return max(1.0 - (step_count - 1) * 0.2, 0.1)
    return 0.05  # 中间步骤给小分


# ── 核心循环 ──────────────────────────────────────────────────────


def online_training_loop(
    *,
    model_path: str = "",
    env_id: str = "knowlyr/conversation",
    n_iterations: int = 3,
    n_episodes: int = 10,
    max_steps: int = 20,
    output_dir: str = "./output/online_loop",
    domain: str = "conversation",
    mock: bool = False,
    patience: int = 0,
) -> list[dict[str, Any]]:
    """在线训练循环主函数.

    每个 iteration 执行:
    1. 封装模型为 agent（或使用 mock agent）
    2. collect() 收集轨迹 + 计算 reward
    3. 导出为 SFT 训练数据
    4. SFTTrainer 微调（或 mock 训练）
    5. 评估并记录指标
    6. 检查早停条件

    Args:
        model_path: HuggingFace 模型名或本地 checkpoint 路径
        env_id: 环境 ID
        n_iterations: 训练循环次数
        n_episodes: 每次循环收集的轨迹数
        max_steps: 每条轨迹最大步数
        output_dir: 输出目录
        domain: 领域 (conversation/engineering/advisory)
        mock: 使用 mock 组件（无需 GPU）
        patience: 早停耐心值。连续 patience 轮 avg_reward 未改善时停止。
                  0 = 不使用早停。

    Returns:
        每次迭代的统计结果列表
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    iteration_results: list[dict[str, Any]] = []
    best_reward = float("-inf")
    patience_counter = 0

    for iteration in range(n_iterations):
        iter_dir = out / f"iter-{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Iteration %d/%d", iteration + 1, n_iterations)
        logger.info("=" * 60)

        # ── Step 1: 创建 agent ──
        if mock:
            agent = _mock_agent_factory(iteration)
            logger.info("使用 mock agent (iteration=%d)", iteration)
        else:
            from agenttrainer.inference import AgentInference

            checkpoint = model_path if iteration == 0 else str(
                out / f"iter-{iteration - 1}" / "final"
            )
            logger.info("从 checkpoint 加载: %s", checkpoint)
            inference = AgentInference.from_pretrained(
                checkpoint, temperature=0.7, max_new_tokens=512,
            )
            agent = inference.create_agent(
                system_prompt=f"你是一个专业的{domain}领域 AI 助手。",
            )

        # ── Step 2: 创建环境 + reward ──
        if mock:
            env = _MockEnv()
            reward_fn = _mock_reward_fn
        else:
            env = make(env_id)
            reward_fn = make_reward_fn(domain=domain)

        # ── Step 3: 收集轨迹 ──
        logger.info("收集轨迹: %d episodes, max_steps=%d", n_episodes, max_steps)

        trajectories = collect(
            env,
            agent=agent,
            n_episodes=n_episodes,
            max_steps=max_steps,
            agent_name=f"online-agent-iter{iteration}",
            model_name=model_path or "mock",
            reward_fn=reward_fn,
        )

        # ── Step 4: 统计 ──
        rewards = []
        step_counts = []
        successes = 0

        for traj in trajectories:
            traj_reward = sum(s.get("reward", 0.0) for s in traj.get("steps", []))
            rewards.append(traj_reward)
            step_counts.append(len(traj.get("steps", [])))
            if traj.get("outcome", {}).get("success"):
                successes += 1

        stats = {
            "iteration": iteration,
            "n_episodes": len(trajectories),
            "success_rate": successes / max(len(trajectories), 1),
            "avg_reward": sum(rewards) / max(len(rewards), 1),
            "avg_steps": sum(step_counts) / max(len(step_counts), 1),
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }
        iteration_results.append(stats)

        logger.info(
            "轨迹统计: success=%.1f%%, avg_reward=%.3f, avg_steps=%.1f",
            stats["success_rate"] * 100,
            stats["avg_reward"],
            stats["avg_steps"],
        )

        # ── 早停检查 ──
        if patience > 0:
            if stats["avg_reward"] > best_reward:
                best_reward = stats["avg_reward"]
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

        # ── Step 6: 训练（最后一轮可选跳过）──
        if mock:
            logger.info("Mock 训练完成 (跳过真实训练)")
        else:
            if not sft_file.exists() or export_result.total_records == 0:
                logger.warning("无训练数据，跳过本轮训练")
                continue

            from agenttrainer.config import SFTConfig
            from agenttrainer.trainers.sft import SFTTrainer

            train_config = SFTConfig(
                model_name_or_path=(
                    model_path if iteration == 0
                    else str(out / f"iter-{iteration - 1}" / "final")
                ),
                train_file=str(sft_file),
                output_dir=str(iter_dir),
                num_epochs=1,
                batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=1e-5,
                agent_format=True,
                mask_observations=True,
                save_steps=0,  # 只保存最终模型
            )

            trainer = SFTTrainer(train_config)
            trainer.train()
            logger.info("SFT 训练完成 → %s", iter_dir / "final")

        # ── Step 7: 保存迭代结果 ──
        stats_file = iter_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("训练循环完成! %d 轮迭代", n_iterations)
    logger.info("=" * 60)

    for stats in iteration_results:
        logger.info(
            "  Iter %d: success=%.1f%%, reward=%.3f, steps=%.1f",
            stats["iteration"],
            stats["success_rate"] * 100,
            stats["avg_reward"],
            stats["avg_steps"],
        )

    # 保存汇总
    summary_file = out / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(iteration_results, f, ensure_ascii=False, indent=2)
    logger.info("汇总已保存: %s", summary_file)

    return iteration_results


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        description="在线训练循环: Model → Collect → Reward → Train",
    )
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen2.5-Coder-7B",
        help="HuggingFace 模型名或本地 checkpoint",
    )
    parser.add_argument(
        "--env", type=str, default="knowlyr/conversation",
        help="环境 ID (knowlyr/conversation, knowlyr/engineering, ...)",
    )
    parser.add_argument(
        "--domain", type=str, default="conversation",
        help="领域 (conversation/engineering/advisory)",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=3,
        help="训练循环次数",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10,
        help="每轮收集的轨迹数",
    )
    parser.add_argument(
        "--max_steps", type=int, default=20,
        help="每条轨迹最大步数",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/online_loop",
        help="输出目录",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="使用 mock 组件（无需 GPU，快速体验）",
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="早停耐心值: 连续 N 轮无改善则停止 (0=不使用早停)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    online_training_loop(
        model_path=args.model_path,
        env_id=args.env,
        n_iterations=args.n_iterations,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        domain=args.domain,
        mock=args.mock,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
