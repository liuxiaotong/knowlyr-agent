"""DataCollator - padding 批次整理."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizer


@dataclass
class SFTCollator:
    """SFT 数据整理器 - 对 batch 内样本做 right-padding.

    支持可选的 step_weights 字段（用于步骤级 reward 加权）。
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"][:self.max_length] for item in batch]
        labels = [item["labels"][:self.max_length] for item in batch]

        input_ids = _pad_sequence(input_ids, self.tokenizer.pad_token_id)
        labels = _pad_sequence(labels, -100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # 步骤级权重（可选）
        if "step_weights" in batch[0]:
            step_weights = [item["step_weights"][:self.max_length] for item in batch]
            result["step_weights"] = _pad_sequence_float(step_weights, 0.0)

        return result


@dataclass
class DPOCollator:
    """DPO 数据整理器 - 分别 pad chosen 和 rejected."""

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id

        chosen_ids = [item["input_ids_chosen"][:self.max_length] for item in batch]
        chosen_labels = [item["labels_chosen"][:self.max_length] for item in batch]
        rejected_ids = [item["input_ids_rejected"][:self.max_length] for item in batch]
        rejected_labels = [item["labels_rejected"][:self.max_length] for item in batch]

        chosen_ids = _pad_sequence(chosen_ids, pad_id)
        chosen_labels = _pad_sequence(chosen_labels, -100)
        rejected_ids = _pad_sequence(rejected_ids, pad_id)
        rejected_labels = _pad_sequence(rejected_labels, -100)

        return {
            "input_ids_chosen": chosen_ids,
            "labels_chosen": chosen_labels,
            "attention_mask_chosen": (chosen_ids != pad_id).long(),
            "input_ids_rejected": rejected_ids,
            "labels_rejected": rejected_labels,
            "attention_mask_rejected": (rejected_ids != pad_id).long(),
        }


@dataclass
class GRPOCollator:
    """GRPO 数据整理器 - pad 组内所有轨迹."""

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """batch 是一组轨迹，每条有 input_ids, labels, attention_mask, reward."""
        input_ids = [item["input_ids"][:self.max_length] for item in batch]
        labels = [item["labels"][:self.max_length] for item in batch]
        rewards = torch.tensor([item["reward"] for item in batch], dtype=torch.float32)

        input_ids = _pad_sequence(input_ids, self.tokenizer.pad_token_id)
        labels = _pad_sequence(labels, -100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "rewards": rewards,
        }


# ── 内部辅助 ──────────────────────────────────────────────


def _pad_sequence(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Right-pad 一组 1D tensor 到相同长度."""
    max_len = max(t.size(0) for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        padded[i, : t.size(0)] = t
    return padded


def _pad_sequence_float(tensors: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    """Right-pad 一组 float 1D tensor 到相同长度."""
    max_len = max(t.size(0) for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.float32)
    for i, t in enumerate(tensors):
        padded[i, : t.size(0)] = t
    return padded
