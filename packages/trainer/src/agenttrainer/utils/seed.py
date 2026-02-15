"""可复现性工具."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
