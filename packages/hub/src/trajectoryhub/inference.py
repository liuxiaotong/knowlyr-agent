"""推理工具 — 从训练后模型创建 collect() 兼容的 agent 函数.

转发到 agenttrainer.inference，是 Hub 层的便捷入口。

Usage::

    from trajectoryhub import collect, create_model_agent

    agent = create_model_agent("./checkpoints/step-1000", system_prompt="你是代码审查员")
    trajectories = collect("knowlyr/engineering", agent=agent, n_episodes=10)
"""

from __future__ import annotations

from typing import Any, Callable


def create_model_agent(
    model_path: str,
    *,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> Callable[[str], dict[str, Any]]:
    """从训练后模型创建 agent 函数.

    Args:
        model_path: HuggingFace 模型名或本地 checkpoint 路径
        system_prompt: system prompt (如员工角色描述)
        temperature: 采样温度
        max_new_tokens: 最大生成 token 数

    Returns:
        agent_fn: (observation: str) -> {"tool": ..., "params": {...}}

    Raises:
        RuntimeError: knowlyr-trainer 未安装
    """
    try:
        from agenttrainer.inference import AgentInference
    except ImportError:
        raise RuntimeError(
            "create_model_agent 需要 knowlyr-trainer >= 0.1.0: pip install knowlyr-trainer"
        )

    inference = AgentInference.from_pretrained(
        model_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return inference.create_agent(system_prompt=system_prompt)
