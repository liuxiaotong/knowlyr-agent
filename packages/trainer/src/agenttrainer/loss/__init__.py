"""Loss 函数."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any


def compute_sequence_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """计算序列级 log probability.

    对 labels 非 -100 的 token 计算 log prob 并求和。

    Args:
        model: CausalLM 模型
        input_ids: (B, L)
        labels: (B, L)，prompt 部分为 -100
        attention_mask: (B, L)

    Returns:
        (B,) 每条序列的 sum log prob
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # (B, L-1, V)
    shift_labels = labels[:, 1:]  # (B, L-1)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # (B, L-1)

    # 只对 labels != -100 的位置累加
    mask = (shift_labels != -100).float()
    seq_log_probs = (token_log_probs * mask).sum(dim=-1)  # (B,)

    return seq_log_probs


def weighted_cross_entropy(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    step_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算加权交叉熵 loss.

    与标准 CE 相比，每个 token 的 loss 额外乘以 step_weights。
    当 step_weights 为 None 时，等价于标准 CE。

    Args:
        model: CausalLM 模型
        input_ids: (B, L)
        labels: (B, L)，prompt 部分为 -100
        attention_mask: (B, L)
        step_weights: (B, L) 每个 token 的权重，默认 1.0

    Returns:
        标量 loss
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()  # (B, L-1, V)
    shift_labels = labels[:, 1:].contiguous()  # (B, L-1)

    # per-token CE loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = shift_labels.view(-1)
    # 将 -100 替换为 0（CE 的 ignore_index 不适用于 reduction="none"）
    valid_mask = (flat_labels != -100).float()
    flat_labels_safe = flat_labels.clamp(min=0)
    token_losses = loss_fct(flat_logits, flat_labels_safe)  # (B*(L-1),)
    token_losses = token_losses * valid_mask  # 遮蔽 prompt 部分

    token_losses = token_losses.view(shift_labels.size())  # (B, L-1)

    # 应用步骤权重
    if step_weights is not None:
        shift_weights = step_weights[:, 1:].contiguous()  # (B, L-1)
        token_losses = token_losses * shift_weights

    # 均值（只在 valid token 上）
    num_valid = valid_mask.sum()
    if num_valid > 0:
        loss = token_losses.sum() / num_valid
    else:
        loss = token_losses.sum()

    return loss
