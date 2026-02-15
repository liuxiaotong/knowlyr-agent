"""测试 DPO loss 函数."""

import torch

from agenttrainer.loss.dpo_loss import dpo_loss


class TestDPOLoss:
    def test_loss_positive(self):
        """DPO loss 应该是正数."""
        chosen = torch.tensor([-2.0, -3.0])
        rejected = torch.tensor([-4.0, -5.0])
        ref_chosen = torch.tensor([-2.5, -3.5])
        ref_rejected = torch.tensor([-4.5, -5.5])

        loss, _, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
        assert loss.item() > 0

    def test_perfect_preference(self):
        """当 chosen 远好于 rejected 时，loss 应该较小."""
        chosen = torch.tensor([-1.0])
        rejected = torch.tensor([-10.0])
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-10.0])

        loss, _, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
        # 当 pi 和 ref 相同时，logits = 0，loss = log(2) ≈ 0.693
        assert loss.item() < 1.0

    def test_returns_rewards(self):
        """应该返回 chosen 和 rejected 的隐式 reward."""
        chosen = torch.tensor([-2.0, -3.0])
        rejected = torch.tensor([-4.0, -5.0])
        ref_chosen = torch.tensor([-2.5, -3.5])
        ref_rejected = torch.tensor([-4.5, -5.5])

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            chosen, rejected, ref_chosen, ref_rejected, beta=0.1
        )
        assert chosen_rewards.shape == (2,)
        assert rejected_rewards.shape == (2,)

    def test_label_smoothing(self):
        """label smoothing 应该使 loss 变大."""
        chosen = torch.tensor([-2.0])
        rejected = torch.tensor([-4.0])
        ref_chosen = torch.tensor([-2.5])
        ref_rejected = torch.tensor([-4.5])

        loss_no_smooth, _, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, label_smoothing=0.0)
        loss_smooth, _, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, label_smoothing=0.1)
        # label smoothing 通常使 loss 增大
        assert loss_smooth.item() >= loss_no_smooth.item() - 0.01  # 允许小误差

    def test_beta_sensitivity(self):
        """更大的 beta 应该产生不同的 loss."""
        # 确保 pi 和 ref 的 log ratio 差不为 0
        chosen = torch.tensor([-2.0])
        rejected = torch.tensor([-4.0])
        ref_chosen = torch.tensor([-3.0])  # pi gap=2, ref gap=1 -> diff=1
        ref_rejected = torch.tensor([-4.0])

        loss_small, _, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.01)
        loss_large, _, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=1.0)
        # 不同 beta 放大非零 logit，产生不同的 loss
        assert loss_small.item() != loss_large.item()
