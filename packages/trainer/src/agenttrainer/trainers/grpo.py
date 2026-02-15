"""GRPO (Group Relative Policy Optimization) Trainer.

Phase 1: 离线模式 - 读取预先分组的轨迹数据。
Phase 2: 在线模式 - 接入 vLLM 在线生成（预留接口）。
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from agenttrainer.config import GRPOConfig
from agenttrainer.data.reader import read_grpo_groups
from agenttrainer.data.formatter import format_grpo
from agenttrainer.data.collator import GRPOCollator
from agenttrainer.loss import compute_sequence_log_probs
from agenttrainer.loss.grpo_loss import compute_group_advantages, grpo_loss
from agenttrainer.models.loader import load_model
from agenttrainer.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)


class _GRPOGroupDataset(Dataset):
    """GRPO 分组数据集.

    每个 item 是一组轨迹（同一 task_id），包含 prompt + 多条 (response, reward)。
    """

    def __init__(self, groups: list[dict], tokenizer, max_length: int) -> None:
        self.groups = groups
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> list[dict]:
        """返回一组 tokenized 轨迹."""
        group = self.groups[idx]
        prompt = group.get("prompt", f"Solve task: {group.get('task_id', '')}")
        items = []
        for traj in group.get("trajectories", []):
            formatted = format_grpo(
                self.tokenizer,
                prompt=prompt,
                response=traj.get("response", ""),
                max_length=self.max_length,
            )
            formatted["reward"] = traj.get("reward", 0.0)
            items.append(formatted)
        return items


def _grpo_collate_fn(batch: list[list[dict]]) -> list[dict]:
    """GRPO collate: batch_size=1，每个 item 就是一个 group."""
    # batch 是 list[list[dict]]，取第一个 group
    return batch[0]


class GRPOTrainer(BaseTrainer):
    """GRPO 训练器 - 离线模式.

    读取预先分组的轨迹，计算组内 relative advantage，
    使用 PPO-style clipped objective 训练。

    Attributes:
        generate_fn: 在线生成函数（Phase 2），签名 (model, prompts) -> list[str]
    """

    def __init__(
        self,
        config: GRPOConfig,
        generate_fn: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.config: GRPOConfig = config
        self.generate_fn = generate_fn  # Phase 2 接口

    def _train_loop(self) -> None:
        # 加载模型
        model, tokenizer = load_model(
            self.config.model_name_or_path,
            tokenizer_name=self.config.tokenizer_name,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
        )
        model.to(self.device)

        # 参考模型（用于计算 old_log_probs 和 KL penalty）
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        logger.info("已创建冻结参考模型")

        # 加载分组数据
        groups = read_grpo_groups(self.config.train_file)
        dataset = _GRPOGroupDataset(groups, tokenizer, self.config.max_length)
        collator_fn = GRPOCollator(tokenizer, self.config.max_length)

        # 每个 batch 是一个 group
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=_grpo_collate_fn,
            num_workers=0,
        )

        # 优化器 & 调度器
        total_steps = len(loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, max(total_steps, 1))

        logger.info(
            "开始 GRPO 训练 (离线模式): %d 组, clip=%.2f, kl_coef=%.3f",
            len(dataset), self.config.clip_epsilon, self.config.kl_coef,
        )

        # 训练循环
        global_step = 0
        model.train()

        for epoch in range(self.config.num_epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", disable=False)

            for batch_idx, group_items in enumerate(pbar):
                if len(group_items) < 2:
                    continue  # 至少需要 2 条才能计算 advantage

                # 整理 group batch
                group_batch = collator_fn(group_items)
                group_batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in group_batch.items()
                }

                with self._maybe_autocast():
                    # 当前策略 log probs
                    log_probs = compute_sequence_log_probs(
                        model,
                        group_batch["input_ids"],
                        group_batch["labels"],
                        group_batch["attention_mask"],
                    )

                    # 参考模型 log probs (作为 old policy)
                    with torch.no_grad():
                        old_log_probs = compute_sequence_log_probs(
                            ref_model,
                            group_batch["input_ids"],
                            group_batch["labels"],
                            group_batch["attention_mask"],
                        )

                    # 计算 group advantages
                    rewards = group_batch["rewards"].to(self.device)
                    advantages = compute_group_advantages(rewards)

                    # GRPO loss
                    loss, metrics = grpo_loss(
                        log_probs,
                        old_log_probs,
                        advantages,
                        clip_epsilon=self.config.clip_epsilon,
                        kl_coef=self.config.kl_coef,
                    )
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.logging_steps == 0:
                        self._log(
                            {
                                "grpo_loss": loss.item() * self.config.gradient_accumulation_steps,
                                "mean_reward": rewards.mean().item(),
                                "kl": metrics["kl"].item(),
                                "ratio_mean": metrics["ratio_mean"].item(),
                                "lr": scheduler.get_last_lr()[0],
                            },
                            global_step,
                        )

                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        self._save(model, tokenizer, optimizer, scheduler, global_step)

                pbar.set_postfix(loss=f"{loss.item() * self.config.gradient_accumulation_steps:.4f}")

            logger.info("Epoch %d 完成", epoch + 1)

        self._save_final(model, tokenizer)
        logger.info("GRPO 训练完成")
