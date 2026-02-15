"""日志工具 - stdout / wandb."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_wandb_run = None


def init_logging(
    *,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """初始化日志系统."""
    global _wandb_run

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

    if wandb_project:
        try:
            import wandb

            _wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config or {},
            )
            logger.info("wandb 已初始化: project=%s", wandb_project)
        except ImportError:
            logger.warning("wandb 未安装，跳过: pip install knowlyr-trainer[wandb]")


def log_metrics(metrics: dict[str, Any], step: int) -> None:
    """记录指标到 stdout + wandb."""
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    logger.info("step=%d %s", step, " ".join(parts))

    if _wandb_run is not None:
        import wandb

        wandb.log(metrics, step=step)


def finish_logging() -> None:
    """结束日志记录."""
    if _wandb_run is not None:
        import wandb

        wandb.finish()
