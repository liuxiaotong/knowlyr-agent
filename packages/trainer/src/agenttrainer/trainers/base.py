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
from agenttrainer.models.checkpoint import save_checkpoint, save_final, load_training_state

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
        # Early stopping / best model 追踪
        self._best_eval_loss: float = float("inf")
        self._patience_counter: int = 0
        self._should_stop: bool = False

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

    def _save_best(self, model: Any, tokenizer: Any) -> None:
        """保存验证 loss 最优的模型到 {output_dir}/best."""
        import os
        best_dir = os.path.join(self.config.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(best_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(best_dir)
        logger.info("Best model 已保存至 %s", best_dir)

    def _maybe_resume(
        self,
        optimizer: AdamW,
        scheduler: Any,
    ) -> int:
        """从 checkpoint 恢复训练状态.

        Args:
            optimizer: 已构建的优化器
            scheduler: 已构建的调度器

        Returns:
            恢复的 global_step (未恢复则返回 0)
        """
        ckpt_path = self.config.resume_from_checkpoint
        if not ckpt_path:
            return 0

        state = load_training_state(ckpt_path)
        if state is None:
            logger.warning("Checkpoint %s 不包含 training_state.pt，从头开始", ckpt_path)
            return 0

        if "optimizer" in state and state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state and state["scheduler"] is not None and scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])

        global_step = state.get("global_step", 0)
        logger.info("从 checkpoint 恢复: step=%d, path=%s", global_step, ckpt_path)
        return global_step

    def _eval_step(self, model: Any, ref_model: Any, batch: dict[str, Any]) -> float:
        """计算单 batch 的 eval loss.

        子类实现具体计算逻辑。SFT 忽略 ref_model (传 None)，
        DPO/GRPO 使用 ref_model 计算参考策略 log probs。

        Args:
            model: 当前策略模型
            ref_model: 参考模型（可能为 None）
            batch: 已移至 device 的 batch dict

        Returns:
            loss 标量值
        """
        raise NotImplementedError("子类需实现 _eval_step")

    def _run_eval(
        self,
        model: Any,
        ref_model: Any,
        eval_loader: Any,
        epoch: int,
        global_step: int,
        tokenizer: Any = None,
    ) -> float:
        """通用验证评估循环.

        在验证集上计算平均 loss 并记录。支持 early stopping 和 best model saving。

        Args:
            model: 当前策略模型
            ref_model: 参考模型（SFT 传 None）
            eval_loader: 验证 DataLoader
            epoch: 当前 epoch
            global_step: 当前训练步数
            tokenizer: 分词器（save_best_model 时需要）

        Returns:
            平均验证 loss
        """
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with self._maybe_autocast():
                    loss = self._eval_step(model, ref_model, batch)
                total_loss += loss
                n_batches += 1

        model.train()
        avg_loss = total_loss / max(n_batches, 1)
        self._log({"eval_loss": avg_loss, "epoch": epoch}, global_step)
        logger.info("Eval epoch %d: loss=%.4f", epoch, avg_loss)

        # Best model saving
        if avg_loss < self._best_eval_loss:
            self._best_eval_loss = avg_loss
            self._patience_counter = 0
            if self.config.save_best_model and is_main_process():
                self._save_best(model, tokenizer)
                logger.info("新最优模型: eval_loss=%.4f", avg_loss)
        else:
            self._patience_counter += 1

        # Early stopping 检查
        patience = self.config.early_stopping_patience
        if patience is not None and self._patience_counter >= patience:
            self._should_stop = True
            logger.info(
                "Early stopping: eval_loss 连续 %d 个 epoch 未改善 (best=%.4f)",
                patience, self._best_eval_loss,
            )

        return avg_loss

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
