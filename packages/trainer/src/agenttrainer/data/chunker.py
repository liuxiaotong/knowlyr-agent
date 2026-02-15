"""长轨迹分块 — 将超过 max_length 的 agent 轨迹拆分为多个训练样本.

策略:
- 按步骤边界拆分（不在步骤中间断开）
- 每个 chunk 包含任务描述 + 当前步骤段
- 可选: 包含前一个 chunk 最后 N 步作为上下文重叠
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from agenttrainer.data.agent_format import AgentStep, build_agent_messages


@dataclass
class TrajectoryChunk:
    """分块后的轨迹片段."""

    steps: list[AgentStep]
    step_rewards: list[float]
    chunk_index: int
    total_chunks: int
    # 上下文步骤（来自前一个 chunk，不参与 loss）
    context_steps: list[AgentStep]


def chunk_trajectory(
    instruction: str,
    input_text: str,
    steps: list[AgentStep],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    overlap_steps: int = 1,
) -> list[TrajectoryChunk]:
    """将长轨迹按步骤边界拆分为多个 chunk.

    Args:
        instruction: 任务指令
        input_text: 任务输入
        steps: 完整步骤列表
        tokenizer: HF tokenizer（用于估计 token 长度）
        max_length: 每个 chunk 的最大 token 长度
        overlap_steps: chunk 之间重叠的步骤数

    Returns:
        TrajectoryChunk 列表，每个 chunk 包含一段步骤
    """
    if not steps:
        return []

    # 如果整个轨迹不超过 max_length，直接返回单个 chunk
    full_messages = build_agent_messages(instruction, input_text, steps)
    full_len = _estimate_token_length(tokenizer, full_messages)
    if full_len <= max_length:
        return [
            TrajectoryChunk(
                steps=steps,
                step_rewards=[s.reward for s in steps],
                chunk_index=0,
                total_chunks=1,
                context_steps=[],
            )
        ]

    # 估计 prompt（instruction + input）占用的 token 数
    prompt_messages = [{"role": "user", "content": instruction + ("\n\n" + input_text if input_text else "")}]
    prompt_len = _estimate_token_length(tokenizer, prompt_messages)

    # 可用于步骤的 token 预算
    step_budget = max_length - prompt_len - 50  # 留 50 token 余量
    if step_budget <= 0:
        step_budget = max_length // 2  # prompt 太长时的回退策略

    # 贪心拆分: 逐步累加，超过预算时切分
    chunks: list[TrajectoryChunk] = []
    current_steps: list[AgentStep] = []
    current_len = 0

    for step in steps:
        step_len = _estimate_step_length(tokenizer, step)

        if current_steps and current_len + step_len > step_budget:
            # 当前 chunk 已满，保存并开始新 chunk
            context = _get_context_steps(chunks, overlap_steps)
            chunks.append(
                TrajectoryChunk(
                    steps=list(current_steps),
                    step_rewards=[s.reward for s in current_steps],
                    chunk_index=len(chunks),
                    total_chunks=0,  # 最后统一设置
                    context_steps=context,
                )
            )
            current_steps = []
            current_len = 0

        current_steps.append(step)
        current_len += step_len

    # 最后一个 chunk
    if current_steps:
        context = _get_context_steps(chunks, overlap_steps)
        chunks.append(
            TrajectoryChunk(
                steps=list(current_steps),
                step_rewards=[s.reward for s in current_steps],
                chunk_index=len(chunks),
                total_chunks=0,
                context_steps=context,
            )
        )

    # 设置 total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


def _get_context_steps(
    existing_chunks: list[TrajectoryChunk],
    overlap_steps: int,
) -> list[AgentStep]:
    """获取上一个 chunk 最后 N 步作为上下文."""
    if not existing_chunks or overlap_steps <= 0:
        return []
    last_chunk = existing_chunks[-1]
    return last_chunk.steps[-overlap_steps:]


def _estimate_token_length(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
) -> int:
    """估计 messages 的 token 长度."""
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    return len(tokenizer.encode(text))


def _estimate_step_length(tokenizer: PreTrainedTokenizer, step: AgentStep) -> int:
    """估计单步的 token 长度."""
    parts = []
    if step.thought:
        parts.append(f"Thought: {step.thought}")
    if step.action:
        parts.append(f"Action: {step.action}")
    if step.observation:
        parts.append(f"Observation: {step.observation}")
    text = "\n".join(parts)
    return len(tokenizer.encode(text))
