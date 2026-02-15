"""DPO (Direct Preference Optimization) loss 函数."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 DPO loss.

    DPO loss = -log(sigmoid(beta * (log(pi/ref)_chosen - log(pi/ref)_rejected)))

    Args:
        policy_chosen_logps: (B,) 策略模型对 chosen 的 log prob
        policy_rejected_logps: (B,) 策略模型对 rejected 的 log prob
        ref_chosen_logps: (B,) 参考模型对 chosen 的 log prob
        ref_rejected_logps: (B,) 参考模型对 rejected 的 log prob
        beta: KL 散度控制系数
        label_smoothing: 标签平滑

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)

    if label_smoothing > 0:
        # label smoothing: (1 - eps) * -log(sigmoid(x)) + eps * -log(sigmoid(-x))
        loss = (
            (1 - label_smoothing) * (-F.logsigmoid(logits))
            + label_smoothing * (-F.logsigmoid(-logits))
        ).mean()
    else:
        loss = -F.logsigmoid(logits).mean()

    # 隐式 reward（用于监控）
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards
