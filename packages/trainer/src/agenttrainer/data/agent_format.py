"""Agent 轨迹格式化 — 多轮对话 + 观察遮蔽 + 步骤加权.

核心思想:
- Agent 轨迹不是平文本，而是多轮 (thought+action) → observation 交互
- 模型只需要学习生成 thought + action（assistant turns）
- 环境观察（observation）不参与 loss 计算（labels = -100）
- 可选: 使用步骤级 process reward 加权每步的 loss
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizer


@dataclass
class AgentStep:
    """Agent 轨迹中的单步."""

    thought: str = ""
    action: str = ""
    observation: str = ""
    reward: float = 0.0


def parse_trajectory(response: str) -> list[AgentStep]:
    """将平文本轨迹解析为结构化步骤.

    支持的格式::

        Step 1:
        Thought: Let me read the file
        Action: read_file /sort.py
        Observation: def sort(arr): ...

        Step 2:
        Thought: Fix the bug
        Action: edit_file /sort.py
        Observation: File edited
    """
    steps: list[AgentStep] = []

    # 按 "Step N:" 分割
    step_pattern = re.compile(r"Step\s+\d+\s*:", re.IGNORECASE)
    parts = step_pattern.split(response)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        step = AgentStep()

        # 提取 Thought（到下一个标签或结尾）
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*(?:Action|Observation):|$)", part, re.DOTALL
        )
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(
            r"Action:\s*(.*?)(?=\n\s*(?:Thought|Observation):|$)", part, re.DOTALL
        )
        if action_match:
            step.action = action_match.group(1).strip()

        # 提取 Observation
        obs_match = re.search(
            r"Observation:\s*(.*?)(?=\n\s*(?:Thought|Action|Step)|\Z)", part, re.DOTALL
        )
        if obs_match:
            step.observation = obs_match.group(1).strip()

        if step.thought or step.action:
            steps.append(step)

    return steps


def parse_structured_steps(steps_data: list[dict[str, Any]]) -> list[AgentStep]:
    """从结构化 JSON 数据解析步骤.

    用于 hub exporter 的 ``steps`` 字段格式。
    """
    return [
        AgentStep(
            thought=s.get("thought", ""),
            action=s.get("action", ""),
            observation=s.get("observation", ""),
            reward=s.get("reward", 0.0),
        )
        for s in steps_data
    ]


def build_agent_messages(
    instruction: str,
    input_text: str,
    steps: list[AgentStep],
) -> list[dict[str, str]]:
    """构建 agent 多轮对话 messages.

    格式:
    - user: instruction + input (初始任务)
    - assistant: thought + action (步骤 1)
    - user: observation (步骤 1 环境反馈)
    - assistant: thought + action (步骤 2)
    - user: observation (步骤 2 环境反馈)
    - ...

    assistant turns 的 token 参与训练，user/observation turns 被遮蔽。
    """
    messages: list[dict[str, str]] = []

    # 初始任务描述
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    messages.append({"role": "user", "content": user_content})

    for step in steps:
        # assistant: thought + action
        assistant_parts = []
        if step.thought:
            assistant_parts.append(f"Thought: {step.thought}")
        if step.action:
            assistant_parts.append(f"Action: {step.action}")
        if assistant_parts:
            messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})

        # observation → user turn（环境反馈，不参与 loss）
        if step.observation:
            messages.append({"role": "user", "content": f"Observation: {step.observation}"})

    return messages


def format_agent_sft(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    input_text: str,
    steps: list[AgentStep],
    max_length: int = 2048,
    mask_observations: bool = True,
    step_rewards: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """格式化 agent SFT 样本 — 多轮对话 + 观察遮蔽 + 步骤加权.

    Args:
        tokenizer: HF tokenizer
        instruction: 任务指令
        input_text: 任务输入（可选）
        steps: 结构化步骤列表
        max_length: 最大 token 长度
        mask_observations: 是否遮蔽观察 token
        step_rewards: 每步的 process reward（长度须 == len(steps)）

    Returns:
        dict 包含 input_ids, labels, attention_mask, [step_weights]
    """
    messages = build_agent_messages(instruction, input_text, steps)

    # tokenize 完整对话
    full_ids = _tokenize_messages(tokenizer, messages, max_length)

    if mask_observations:
        # 只对 assistant turns 计算 loss
        labels = _build_agent_labels(tokenizer, messages, full_ids, max_length)
    else:
        # 不遮蔽: 只遮蔽初始 prompt
        labels = _build_prompt_only_labels(tokenizer, messages, full_ids, max_length)

    result: dict[str, torch.Tensor] = {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(full_ids),
    }

    # 步骤级 reward 权重
    if step_rewards is not None and len(step_rewards) > 0:
        weights = _build_step_weights(
            tokenizer, messages, full_ids, max_length, steps, step_rewards
        )
        result["step_weights"] = weights

    return result


# ── 内部辅助 ──────────────────────────────────────────────


def _tokenize_messages(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    max_length: int,
) -> torch.Tensor:
    """Tokenize 完整多轮对话."""
    if _has_chat_template(tokenizer):
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if ids.dim() == 2:
            ids = ids[0]
        return ids

    # 回退: 手动拼接
    text = _messages_to_text(messages)
    ids = tokenizer.encode(text, truncation=True, max_length=max_length, return_tensors="pt")
    if ids.dim() == 2:
        ids = ids[0]
    return ids


def _build_agent_labels(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    full_ids: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """构建 agent labels — 只对 assistant turns 计算 loss.

    使用逐步前缀 tokenize 来确定每条消息的 token 边界。
    """
    labels = torch.full_like(full_ids, -100)
    full_len = len(full_ids)

    # 逐步构建 messages 前缀，找到每条消息的 token 边界
    prev_len = 0
    for i in range(len(messages)):
        prefix = messages[: i + 1]
        prefix_ids = _tokenize_prefix(tokenizer, prefix, max_length)
        curr_len = min(len(prefix_ids), full_len)

        if messages[i]["role"] == "assistant":
            # assistant turn → 参与 loss
            start = max(prev_len, 0)
            end = curr_len
            if start < end:
                labels[start:end] = full_ids[start:end]

        prev_len = curr_len

    return labels


def _build_prompt_only_labels(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    full_ids: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """只遮蔽初始 user prompt，其余都参与 loss."""
    labels = full_ids.clone()

    # 初始 prompt 的 token 长度
    first_msg = messages[:1]
    prompt_ids = _tokenize_prefix(tokenizer, first_msg, max_length)
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels[:prompt_len] = -100

    return labels


def _build_step_weights(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    full_ids: torch.Tensor,
    max_length: int,
    steps: list[AgentStep],
    step_rewards: list[float],
) -> torch.Tensor:
    """构建步骤级权重 tensor — 每个 token 按所属步骤的 reward 加权.

    归一化方式: w_j = r_j / mean(r) 使权重均值为 1.0。
    """
    weights = torch.ones(len(full_ids), dtype=torch.float32)
    full_len = len(full_ids)

    # 计算归一化的步骤权重
    mean_reward = sum(step_rewards) / max(len(step_rewards), 1)
    if mean_reward <= 0:
        mean_reward = 1.0
    normalized = [r / mean_reward for r in step_rewards]

    # 找到每条消息的 token 边界
    prev_len = 0
    step_idx = 0
    for i in range(len(messages)):
        prefix = messages[: i + 1]
        prefix_ids = _tokenize_prefix(tokenizer, prefix, max_length)
        curr_len = min(len(prefix_ids), full_len)

        # assistant turn → 用对应步骤的 reward 加权
        if messages[i]["role"] == "assistant" and step_idx < len(normalized):
            weights[prev_len:curr_len] = normalized[step_idx]
            step_idx += 1

        prev_len = curr_len

    return weights


def _tokenize_prefix(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    max_length: int,
) -> torch.Tensor:
    """Tokenize 消息前缀（加 generation_prompt 以获得准确的前缀长度）."""
    if _has_chat_template(tokenizer):
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if ids.dim() == 2:
            ids = ids[0]
        return ids

    # 回退: 手动拼接 + "assistant: " 后缀
    text = _messages_to_text(messages) + "assistant: "
    ids = tokenizer.encode(text, truncation=True, max_length=max_length, return_tensors="pt")
    if ids.dim() == 2:
        ids = ids[0]
    return ids


def _has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    """检查 tokenizer 是否有可用的 chat template."""
    return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    """手动拼接 messages 为纯文本（无 chat template 时的回退）."""
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(parts) + "\n"
