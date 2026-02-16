"""GRPO (Group Relative Policy Optimization) loss 函数.

参考 DeepSeek-R1 的 GRPO 算法:
- 组内按 reward 归一化计算 advantage
- PPO-style clipped objective
- 可选 KL penalty
- 支持步骤级 advantage 加权
"""

from __future__ import annotations

import torch


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算组内相对 advantage.

    A_i = (R_i - mean(R)) / (std(R) + eps)

    Args:
        rewards: (G,) 组内每条轨迹的 reward
        eps: 数值稳定性

    Returns:
        (G,) 归一化后的 advantages
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def compute_step_weighted_advantages(
    trajectory_advantages: torch.Tensor,
    step_rewards: list[list[float]],
    eps: float = 1e-8,
) -> list[torch.Tensor]:
    """计算步骤级加权 advantage.

    在轨迹级 advantage 基础上，用步骤级 reward 进一步加权:
    A_{i,j} = A_i * (r_{i,j} / mean_j(r_{i,j}))

    这样在一条好的轨迹中，好的步骤获得更多信用;
    在一条差的轨迹中，差的步骤受到更多惩罚。

    Args:
        trajectory_advantages: (G,) 轨迹级 advantages
        step_rewards: G 条轨迹的步骤 reward 列表
        eps: 数值稳定性

    Returns:
        G 个 tensor 列表，每个是 (S_i,) 的步骤级 advantage
    """
    result = []
    for i, rewards_per_step in enumerate(step_rewards):
        if not rewards_per_step:
            result.append(torch.tensor(
                [trajectory_advantages[i].item()],
                device=trajectory_advantages.device,
            ))
            continue

        step_r = torch.tensor(
            rewards_per_step, dtype=torch.float32,
            device=trajectory_advantages.device,
        )
        mean_r = step_r.mean()
        if mean_r.abs() < eps:
            # 步骤 reward 全为 0 或接近 0，不加权
            step_weights = torch.ones_like(step_r)
        else:
            step_weights = step_r / (mean_r + eps)

        step_adv = trajectory_advantages[i] * step_weights
        result.append(step_adv)

    return result


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 GRPO clipped objective.

    L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + kl_coef * KL

    Args:
        log_probs: (G,) 当前策略的 log prob
        old_log_probs: (G,) 旧策略的 log prob
        advantages: (G,) group-normalized advantages
        clip_epsilon: PPO clip 范围
        kl_coef: KL penalty 系数

    Returns:
        (loss, metrics_dict) - loss 标量 + 监控指标
    """
    # importance ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -surrogate.mean()

    # KL penalty (近似)
    kl = (old_log_probs - log_probs).mean()
    loss = policy_loss + kl_coef * kl

    metrics = {
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio.std().detach(),
        "advantages_mean": advantages.mean().detach(),
    }

    return loss, metrics
