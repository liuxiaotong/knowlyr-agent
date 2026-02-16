"""在线训练循环示例 — 使用 mock 组件快速体验全闭环.

Usage::

    # 快速体验（mock 环境 + mock agent，无需 GPU）
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
import logging
from typing import Any, Callable

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep
from trajectoryhub.online import online_training_loop


# ── Mock 组件（用于无 GPU 体验）──────────────────────────────────


class MockEnv(AgentEnv):
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


def mock_agent_factory(iteration: int, checkpoint: str) -> Callable[[str], dict[str, Any]]:
    """Mock agent 工厂 — 每次迭代 agent 变得"更聪明"."""
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


def mock_reward_fn(steps: list[dict[str, Any]], action: dict[str, Any]) -> float:
    """Mock reward — 步数越少得分越高."""
    tool = action.get("tool", "")
    if tool == "respond":
        step_count = len(steps) + 1
        return max(1.0 - (step_count - 1) * 0.2, 0.1)
    return 0.05


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
    parser.add_argument("--env", type=str, default="knowlyr/conversation", help="环境 ID")
    parser.add_argument("--domain", type=str, default="conversation", help="领域")
    parser.add_argument("--n_iterations", type=int, default=3, help="训练循环次数")
    parser.add_argument("--n_episodes", type=int, default=10, help="每轮收集的轨迹数")
    parser.add_argument("--max_steps", type=int, default=20, help="每条轨迹最大步数")
    parser.add_argument("--output_dir", type=str, default="./output/online_loop", help="输出目录")
    parser.add_argument("--mock", action="store_true", help="使用 mock 组件（无需 GPU）")
    parser.add_argument("--patience", type=int, default=0, help="早停耐心值 (0=不使用)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.mock:
        # Mock 模式: 使用 mock 组件，跳过训练
        online_training_loop(
            agent_factory=mock_agent_factory,
            env_factory=MockEnv,
            reward_fn=mock_reward_fn,
            n_iterations=args.n_iterations,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            patience=args.patience,
            skip_training=True,
        )
    else:
        # 真实模式: 使用 model_path + env_id + domain
        online_training_loop(
            model_path=args.model_path,
            env_id=args.env,
            domain=args.domain,
            n_iterations=args.n_iterations,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
