"""SFT (Supervised Fine-Tuning) Trainer.

支持两种模式:
1. 标准模式: 平文本轨迹 → 简单 prompt/response 格式
2. Agent 模式 (agent_format=True):
   - 多轮对话格式 (thought+action / observation)
   - 观察 token 遮蔽 (mask_observations)
   - 步骤级 reward 加权 (step_weighted_loss)
   - 长轨迹分块 (chunk_long_trajectories)
   - 课程学习 (curriculum)
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from agenttrainer.config import SFTConfig
from agenttrainer.data.reader import read_sft
from agenttrainer.data.formatter import format_sft
from agenttrainer.data.collator import SFTCollator
from agenttrainer.models.loader import load_model
from agenttrainer.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)


class _SFTDataset(Dataset):
    """标准 SFT 数据集."""

    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        return format_sft(
            self.tokenizer,
            instruction=rec.get("instruction", ""),
            input_text=rec.get("input", ""),
            response=rec.get("response", ""),
            max_length=self.max_length,
        )


class _AgentSFTDataset(Dataset):
    """Agent 模式 SFT 数据集 — 多轮对话 + 观察遮蔽 + 步骤加权."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int,
        mask_observations: bool = True,
        step_weighted_loss: bool = False,
    ) -> None:
        from agenttrainer.data.agent_format import (
            parse_trajectory,
            parse_structured_steps,
            format_agent_sft,
        )
        from agenttrainer.data.chunker import chunk_trajectory

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_observations = mask_observations
        self.step_weighted_loss = step_weighted_loss
        self._format_agent_sft = format_agent_sft

        # 预处理: 解析 + 分块
        self.items: list[dict] = []
        for rec in records:
            instruction = rec.get("instruction", "")
            input_text = rec.get("input", "")

            # 解析步骤（优先使用结构化格式，回退到文本解析）
            if "steps" in rec:
                steps = parse_structured_steps(rec["steps"])
            else:
                steps = parse_trajectory(rec.get("response", ""))

            if not steps:
                # 无法解析为步骤，回退到标准格式
                self.items.append({
                    "instruction": instruction,
                    "input": input_text,
                    "response": rec.get("response", ""),
                    "_fallback": True,
                })
                continue

            # 长轨迹分块
            chunks = chunk_trajectory(
                instruction, input_text, steps, tokenizer, max_length
            )

            for chunk in chunks:
                # 合并上下文步骤 + 当前步骤
                all_steps = chunk.context_steps + chunk.steps
                # 上下文步骤的 reward 不参与加权
                step_rewards = (
                    [0.0] * len(chunk.context_steps) + chunk.step_rewards
                    if step_weighted_loss else None
                )
                self.items.append({
                    "instruction": instruction,
                    "input": input_text,
                    "steps": all_steps,
                    "step_rewards": step_rewards,
                    "_fallback": False,
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        if item.get("_fallback"):
            return format_sft(
                self.tokenizer,
                instruction=item["instruction"],
                input_text=item["input"],
                response=item["response"],
                max_length=self.max_length,
            )

        return self._format_agent_sft(
            self.tokenizer,
            instruction=item["instruction"],
            input_text=item["input"],
            steps=item["steps"],
            max_length=self.max_length,
            mask_observations=self.mask_observations,
            step_rewards=item.get("step_rewards"),
        )


class SFTTrainer(BaseTrainer):
    """SFT 训练器 - 标准 Causal LM 微调.

    当 config.agent_format=True 时启用 agent 训练增强:
    - 多轮对话格式
    - 观察 token 遮蔽
    - 步骤级 reward 加权
    - 长轨迹分块
    - 课程学习
    """

    def __init__(self, config: SFTConfig) -> None:
        super().__init__(config)
        self.config: SFTConfig = config

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

        # 加载数据
        records = read_sft(self.config.train_file)

        if self.config.agent_format:
            dataset = _AgentSFTDataset(
                records, tokenizer, self.config.max_length,
                mask_observations=self.config.mask_observations,
                step_weighted_loss=self.config.step_weighted_loss,
            )
            logger.info("Agent 模式: %d 训练样本（含分块）", len(dataset))
        else:
            dataset = _SFTDataset(records, tokenizer, self.config.max_length)

        collator = SFTCollator(tokenizer, self.config.max_length)

        # Curriculum learning 采样器
        sampler = None
        if self.config.agent_format and self.config.curriculum:
            from agenttrainer.data.curriculum import CurriculumSampler, compute_difficulties
            difficulties = compute_difficulties(records)
            # 如果分块后样本数 > 原始数，扩展 difficulties
            if len(dataset) > len(difficulties):
                # 简单扩展: 复制最后一个值
                difficulties.extend(
                    [difficulties[-1] if difficulties else 0.0]
                    * (len(dataset) - len(difficulties))
                )
            elif len(dataset) < len(difficulties):
                difficulties = difficulties[:len(dataset)]
            sampler = CurriculumSampler(
                difficulties,
                num_epochs=self.config.num_epochs,
                start_ratio=self.config.curriculum_start_ratio,
                warmup_epochs=self.config.curriculum_warmup_epochs,
            )
            logger.info(
                "Curriculum learning: start_ratio=%.1f, warmup=%d epochs",
                self.config.curriculum_start_ratio, self.config.curriculum_warmup_epochs,
            )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collator,
            num_workers=0,
        )

        # 是否使用加权 loss
        use_weighted_loss = (
            self.config.agent_format and self.config.step_weighted_loss
        )

        # 优化器 & 调度器
        total_steps = (
            len(loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        )
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, max(total_steps, 1))

        logger.info(
            "开始 SFT 训练: %d 条数据, %d epochs, %d total steps%s",
            len(dataset), self.config.num_epochs, total_steps,
            " (agent 模式)" if self.config.agent_format else "",
        )

        # 训练循环
        global_step = 0
        model.train()

        for epoch in range(self.config.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", disable=False)

            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with self._maybe_autocast():
                    if use_weighted_loss and "step_weights" in batch:
                        from agenttrainer.loss import weighted_cross_entropy
                        loss = weighted_cross_entropy(
                            model,
                            batch["input_ids"],
                            batch["labels"],
                            batch["attention_mask"],
                            step_weights=batch["step_weights"],
                        )
                    else:
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        loss = outputs.loss

                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                epoch_loss += loss.item()

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
                                "loss": loss.item() * self.config.gradient_accumulation_steps,
                                "lr": scheduler.get_last_lr()[0],
                                "epoch": epoch + 1,
                            },
                            global_step,
                        )

                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        self._save(model, tokenizer, optimizer, scheduler, global_step)

                pbar.set_postfix(loss=f"{loss.item() * self.config.gradient_accumulation_steps:.4f}")

            avg_loss = epoch_loss / max(len(loader), 1)
            logger.info("Epoch %d 完成, avg_loss=%.4f", epoch + 1, avg_loss)

        # 保存最终模型
        self._save_final(model, tokenizer)
        logger.info("SFT 训练完成")
