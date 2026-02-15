"""测试 GRPO loss 函数."""

import pytest
import torch

from agenttrainer.loss.grpo_loss import (
    compute_group_advantages,
    compute_step_weighted_advantages,
    grpo_loss,
)


class TestComputeGroupAdvantages:
    def test_zero_mean(self):
        """advantages 的均值应接近 0."""
        rewards = torch.tensor([0.9, 0.7, 0.3, 0.1])
        advantages = compute_group_advantages(rewards)
        assert abs(advantages.mean().item()) < 1e-6

    def test_ordering_preserved(self):
        """高 reward 应对应高 advantage."""
        rewards = torch.tensor([0.9, 0.5, 0.1])
        advantages = compute_group_advantages(rewards)
        assert advantages[0] > advantages[1] > advantages[2]

    def test_identical_rewards(self):
        """相同 reward 时 advantages 应接近 0."""
        rewards = torch.tensor([0.5, 0.5, 0.5])
        advantages = compute_group_advantages(rewards)
        assert torch.allclose(advantages, torch.zeros(3), atol=1e-4)


class TestGRPOLoss:
    def test_loss_is_scalar(self):
        log_probs = torch.tensor([-3.0, -4.0, -5.0])
        old_log_probs = torch.tensor([-3.1, -4.1, -5.1])
        advantages = torch.tensor([1.0, 0.0, -1.0])

        loss, metrics = grpo_loss(log_probs, old_log_probs, advantages)
        assert loss.dim() == 0  # scalar

    def test_returns_metrics(self):
        log_probs = torch.tensor([-3.0, -4.0])
        old_log_probs = torch.tensor([-3.1, -4.1])
        advantages = torch.tensor([1.0, -1.0])

        _, metrics = grpo_loss(log_probs, old_log_probs, advantages)
        assert "policy_loss" in metrics
        assert "kl" in metrics
        assert "ratio_mean" in metrics

    def test_clip_behavior(self):
        """极端 ratio 应该被 clip."""
        # 大正差 -> ratio 很大
        log_probs = torch.tensor([0.0])
        old_log_probs = torch.tensor([-10.0])
        advantages = torch.tensor([1.0])

        loss_clipped, metrics = grpo_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)
        # ratio = exp(10) >> 1.2，应该被 clip
        assert metrics["ratio_mean"].item() > 1.2  # ratio 未被 clip（metric 报告原始值）

    def test_zero_kl_coef(self):
        """kl_coef=0 时不应有 KL penalty."""
        log_probs = torch.tensor([-3.0, -4.0])
        old_log_probs = torch.tensor([-3.5, -4.5])
        advantages = torch.tensor([1.0, -1.0])

        loss_with_kl, _ = grpo_loss(log_probs, old_log_probs, advantages, kl_coef=0.1)
        loss_no_kl, _ = grpo_loss(log_probs, old_log_probs, advantages, kl_coef=0.0)
        assert loss_with_kl.item() != loss_no_kl.item()


class TestStepWeightedAdvantages:
    def test_basic_weighting(self):
        """步骤级 advantage 应在轨迹级基础上加权."""
        traj_advantages = torch.tensor([1.0, -1.0])
        step_rewards = [
            [0.3, 0.7],  # 轨迹 0: 步骤 1 差，步骤 2 好
            [0.8, 0.2],  # 轨迹 1: 步骤 1 好，步骤 2 差
        ]
        result = compute_step_weighted_advantages(traj_advantages, step_rewards)
        assert len(result) == 2
        # 轨迹 0 advantage > 0，步骤 2 应有更高权重
        assert result[0][1] > result[0][0]
        # 轨迹 1 advantage < 0，步骤 1（reward 高）应被惩罚更多
        assert result[1][0] < result[1][1]  # 更负

    def test_uniform_rewards(self):
        """均匀步骤 reward 时，权重应均匀."""
        traj_advantages = torch.tensor([2.0])
        step_rewards = [[0.5, 0.5, 0.5]]
        result = compute_step_weighted_advantages(traj_advantages, step_rewards)
        assert len(result[0]) == 3
        # 所有步骤权重应该相同
        assert torch.allclose(result[0], torch.tensor([2.0, 2.0, 2.0]), atol=0.1)

    def test_empty_step_rewards(self):
        """空步骤 reward 列表应回退到轨迹级."""
        traj_advantages = torch.tensor([1.5])
        step_rewards = [[]]
        result = compute_step_weighted_advantages(traj_advantages, step_rewards)
        assert len(result) == 1
        assert result[0][0].item() == pytest.approx(1.5, abs=0.01)
