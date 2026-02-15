"""分布式训练工具 - DDP."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_ddp(backend: str = "nccl") -> tuple[int, int]:
    """初始化 DDP 进程组.

    Returns:
        (rank, world_size)
    """
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(rank)

    return rank, world_size


def cleanup_ddp() -> None:
    """清理 DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """判断是否为主进程."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
