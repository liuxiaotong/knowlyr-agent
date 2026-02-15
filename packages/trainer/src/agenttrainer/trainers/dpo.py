"""DPO (Direct Preference Optimization) Trainer."""

from __future__ import annotations

import copy
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from agenttrainer.config import DPOConfig
from agenttrainer.data.reader import read_dpo
from agenttrainer.data.formatter import format_dpo
from agenttrainer.data.collator import DPOCollator
from agenttrainer.loss import compute_sequence_log_probs
from agenttrainer.loss.dpo_loss import dpo_loss
from agenttrainer.models.loader import load_model
from agenttrainer.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)


class _DPODataset(Dataset):
    """DPO 数据集."""

    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        return format_dpo(
            self.tokenizer,
            prompt=rec.get("prompt", ""),
            chosen=rec.get("chosen", ""),
            rejected=rec.get("rejected", ""),
            max_length=self.max_length,
        )


class DPOTrainer(BaseTrainer):
    """DPO 训练器 - 偏好学习."""

    def __init__(self, config: DPOConfig) -> None:
        super().__init__(config)
        self.config: DPOConfig = config

    def _train_loop(self) -> None:
        # 加载策略模型
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

        # 参考模型（冻结副本）
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        logger.info("已创建冻结参考模型")

        # 加载数据
        records = read_dpo(self.config.train_file)
        dataset = _DPODataset(records, tokenizer, self.config.max_length)
        collator = DPOCollator(tokenizer, self.config.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )

        # 优化器 & 调度器
        total_steps = (
            len(loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        )
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, total_steps)

        logger.info(
            "开始 DPO 训练: %d 条偏好对, beta=%.2f, %d total steps",
            len(dataset), self.config.beta, total_steps,
        )

        # 训练循环
        global_step = 0
        model.train()

        for epoch in range(self.config.num_epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", disable=False)

            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with self._maybe_autocast():
                    # 策略模型 log probs
                    policy_chosen_logps = compute_sequence_log_probs(
                        model,
                        batch["input_ids_chosen"],
                        batch["labels_chosen"],
                        batch["attention_mask_chosen"],
                    )
                    policy_rejected_logps = compute_sequence_log_probs(
                        model,
                        batch["input_ids_rejected"],
                        batch["labels_rejected"],
                        batch["attention_mask_rejected"],
                    )

                    # 参考模型 log probs
                    with torch.no_grad():
                        ref_chosen_logps = compute_sequence_log_probs(
                            ref_model,
                            batch["input_ids_chosen"],
                            batch["labels_chosen"],
                            batch["attention_mask_chosen"],
                        )
                        ref_rejected_logps = compute_sequence_log_probs(
                            ref_model,
                            batch["input_ids_rejected"],
                            batch["labels_rejected"],
                            batch["attention_mask_rejected"],
                        )

                    loss, chosen_rewards, rejected_rewards = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        beta=self.config.beta,
                        label_smoothing=self.config.label_smoothing,
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
                        acc = (chosen_rewards > rejected_rewards).float().mean().item()
                        self._log(
                            {
                                "dpo_loss": loss.item() * self.config.gradient_accumulation_steps,
                                "accuracy": acc,
                                "chosen_reward": chosen_rewards.mean().item(),
                                "rejected_reward": rejected_rewards.mean().item(),
                                "lr": scheduler.get_last_lr()[0],
                            },
                            global_step,
                        )

                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        self._save(model, tokenizer, optimizer, scheduler, global_step)

                pbar.set_postfix(loss=f"{loss.item() * self.config.gradient_accumulation_steps:.4f}")

            logger.info("Epoch %d 完成", epoch + 1)

        self._save_final(model, tokenizer)
        logger.info("DPO 训练完成")
