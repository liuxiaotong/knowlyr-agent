"""Checkpoint 保存和恢复."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    output_dir: str,
) -> Path:
    """保存训练 checkpoint.

    包含模型权重 + 优化器状态 + scheduler 状态。
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型（兼容 DDP 和 PEFT）
    unwrapped = _unwrap_model(model)
    unwrapped.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # 保存训练状态
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "global_step": step,
        },
        ckpt_dir / "training_state.pt",
    )

    logger.info("保存 checkpoint: %s (step=%d)", ckpt_dir, step)
    return ckpt_dir


def save_final(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> Path:
    """保存最终模型."""
    out = Path(output_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)

    unwrapped = _unwrap_model(model)
    unwrapped.save_pretrained(out)
    tokenizer.save_pretrained(out)

    logger.info("保存最终模型: %s", out)
    return out


def load_training_state(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """加载训练状态（optimizer + scheduler + step）."""
    state_path = Path(checkpoint_dir) / "training_state.pt"
    if not state_path.exists():
        return None
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    logger.info("恢复训练状态: step=%d", state.get("global_step", 0))
    return state


def _unwrap_model(model: Any) -> Any:
    """解包 DDP / PEFT 包装."""
    if hasattr(model, "module"):
        return model.module
    return model
