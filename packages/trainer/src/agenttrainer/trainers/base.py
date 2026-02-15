"""BaseTrainer - 共享训练循环骨架."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.optim import AdamW
from transformers import get_scheduler

from agenttrainer.config import TrainConfig
from agenttrainer.utils.seed import set_seed
from agenttrainer.utils.logging import init_logging, log_metrics, finish_logging
from agenttrainer.utils.distributed import is_main_process
from agenttrainer.models.checkpoint import save_checkpoint, save_final

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """训练器基类.

    子类只需实现 ``_train_loop`` 方法。
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = self._resolve_device()
        self.rank = 0
        self.world_size = 1

    def train(self) -> None:
        """运行完整训练流程."""
        set_seed(self.config.seed)

        init_logging(
            wandb_project=self.config.wandb_project,
            wandb_run_name=self.config.wandb_run_name,
            config=self.config.model_dump(),
        )

        try:
            self._train_loop()
        finally:
            finish_logging()

    @abstractmethod
    def _train_loop(self) -> None:
        """子类实现具体的训练循环."""

    # ── 共享工具方法 ──────────────────────────────────────

    def _build_optimizer(self, model: Any) -> AdamW:
        """构建 AdamW 优化器."""
        no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
        params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(params, lr=self.config.learning_rate)

    def _build_scheduler(
        self,
        optimizer: AdamW,
        total_steps: int,
    ) -> Any:
        """构建学习率调度器."""
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        return get_scheduler(
            name=self.config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _log(self, metrics: dict[str, Any], step: int) -> None:
        """记录指标（仅主进程）."""
        if is_main_process():
            log_metrics(metrics, step)

    def _save(
        self,
        model: Any,
        tokenizer: Any,
        optimizer: Any,
        scheduler: Any,
        step: int,
    ) -> None:
        """保存 checkpoint（仅主进程）."""
        if is_main_process():
            save_checkpoint(model, tokenizer, optimizer, scheduler, step, self.config.output_dir)

    def _save_final(self, model: Any, tokenizer: Any) -> None:
        """保存最终模型（仅主进程）."""
        if is_main_process():
            save_final(model, tokenizer, self.config.output_dir)

    def _resolve_device(self) -> torch.device:
        """确定训练设备."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _maybe_autocast(self) -> torch.autocast:
        """混合精度 context manager."""
        if self.config.bf16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return _NullContext()


class _NullContext:
    """空 context manager."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        pass
