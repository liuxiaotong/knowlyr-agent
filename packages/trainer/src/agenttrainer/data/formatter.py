"""Chat template 格式化 + label masking."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizer


def format_sft(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    input_text: str,
    response: str,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """格式化 SFT 样本.

    使用 tokenizer.apply_chat_template 构建对话，
    只对 assistant response 部分计算 loss（prompt 部分 labels = -100）。
    """
    # 构建 prompt (system + user)
    prompt_messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}" if input_text else instruction},
    ]
    # 完整对话 (含 assistant)
    full_messages = prompt_messages + [
        {"role": "assistant", "content": response},
    ]

    # tokenize 完整对话
    full_ids = _apply_chat_template(tokenizer, full_messages, max_length)

    # tokenize prompt（加 generation_prompt 以获得 assistant 前缀长度）
    prompt_ids = _apply_chat_template_prompt(tokenizer, prompt_messages, max_length)

    # 创建 labels: prompt 部分设为 -100
    labels = full_ids.clone()
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels[:prompt_len] = -100

    attention_mask = torch.ones_like(full_ids)

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def format_dpo(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """格式化 DPO 样本.

    分别 tokenize chosen 和 rejected，对 prompt 部分 labels = -100。
    """
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_ids = _apply_chat_template_prompt(tokenizer, prompt_messages, max_length)
    prompt_len = len(prompt_ids)

    # chosen
    chosen_messages = prompt_messages + [{"role": "assistant", "content": chosen}]
    chosen_ids = _apply_chat_template(tokenizer, chosen_messages, max_length)
    chosen_labels = chosen_ids.clone()
    chosen_labels[: min(prompt_len, len(chosen_ids))] = -100

    # rejected
    rejected_messages = prompt_messages + [{"role": "assistant", "content": rejected}]
    rejected_ids = _apply_chat_template(tokenizer, rejected_messages, max_length)
    rejected_labels = rejected_ids.clone()
    rejected_labels[: min(prompt_len, len(rejected_ids))] = -100

    return {
        "input_ids_chosen": chosen_ids,
        "labels_chosen": chosen_labels,
        "attention_mask_chosen": torch.ones_like(chosen_ids),
        "input_ids_rejected": rejected_ids,
        "labels_rejected": rejected_labels,
        "attention_mask_rejected": torch.ones_like(rejected_ids),
    }


def format_grpo(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """格式化 GRPO 单条轨迹."""
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_ids = _apply_chat_template_prompt(tokenizer, prompt_messages, max_length)
    prompt_len = len(prompt_ids)

    full_messages = prompt_messages + [{"role": "assistant", "content": response}]
    full_ids = _apply_chat_template(tokenizer, full_messages, max_length)

    labels = full_ids.clone()
    labels[: min(prompt_len, len(full_ids))] = -100

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(full_ids),
    }


# ── 内部辅助 ──────────────────────────────────────────────


def _to_1d_tensor(ids: Any) -> torch.Tensor:
    """将 tokenize 结果统一转为 1D LongTensor."""
    if isinstance(ids, torch.Tensor):
        return ids[0] if ids.dim() == 2 else ids
    if isinstance(ids, list):
        return torch.tensor(ids, dtype=torch.long)
    # BatchEncoding 等对象 → 取 input_ids
    if hasattr(ids, "input_ids"):
        return _to_1d_tensor(ids.input_ids)
    return torch.tensor(list(ids), dtype=torch.long)


def _apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    max_length: int,
) -> torch.Tensor:
    """使用 chat template tokenize 完整对话."""
    if _has_chat_template(tokenizer):
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )
        ids = _to_1d_tensor(ids)
        return ids[:max_length]

    # 回退: 手动拼接
    text = _messages_to_text(messages)
    ids = tokenizer.encode(text, truncation=True, max_length=max_length)
    return _to_1d_tensor(ids)


def _apply_chat_template_prompt(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    max_length: int,
) -> torch.Tensor:
    """Tokenize prompt 部分（加 generation_prompt 以包含 assistant 前缀）."""
    if _has_chat_template(tokenizer):
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        ids = _to_1d_tensor(ids)
        return ids[:max_length]

    # 回退
    text = _messages_to_text(messages) + "assistant: "
    ids = tokenizer.encode(text, truncation=True, max_length=max_length)
    return _to_1d_tensor(ids)


def _has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    """检查 tokenizer 是否有可用的 chat template."""
    return (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
    )


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """手动拼接 messages 为纯文本（无 chat template 时的回退）."""
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(parts) + "\n"
