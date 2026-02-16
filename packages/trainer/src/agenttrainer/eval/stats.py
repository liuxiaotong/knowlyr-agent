"""统计工具模块 — Bootstrap、非参数检验、多重比较校正.

不依赖 scipy，使用手动实现的统计方法。

提供:
- confidence_interval: t 分布近似置信区间
- bootstrap_ci: Bootstrap 百分位法置信区间
- welch_t_test: Welch's t 检验 (独立样本)
- mann_whitney_u: Mann-Whitney U 检验 (非参数)
- paired_t_test: 配对 t 检验
- bonferroni_correct: Bonferroni 多重比较校正
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StatTestResult:
    """统计检验结果.

    Attributes:
        test_name: 检验名称 (如 "welch_t", "mann_whitney_u", "paired_t")
        statistic: 检验统计量
        p_value_approx: p 值近似 (区间字符串)
        significant: 是否显著 (α=0.05)
        effect_size: 效应量 (Cohen's d 或 rank-biserial)
        details: 额外信息 (如自由度、样本量等)
    """

    test_name: str
    statistic: float
    p_value_approx: str
    significant: bool
    effect_size: float
    details: dict[str, Any] = field(default_factory=dict)


# ── 置信区间 ──────────────────────────────────────────────────────


def confidence_interval(
    data: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """计算均值的置信区间 (使用 t 分布近似).

    当 n ≥ 30 时 t 分布近似正态分布，使用 z 值；
    n < 30 时使用查表的 t 值近似。

    Args:
        data: 数据列表
        confidence: 置信水平 (默认 0.95)

    Returns:
        (lower, upper) 置信区间
    """
    n = len(data)
    if n < 2:
        mean = data[0] if data else 0.0
        return (mean, mean)

    mean = statistics.mean(data)
    std_err = statistics.stdev(data) / math.sqrt(n)

    # t 值近似 (95% CI)
    if confidence == 0.95:
        if n >= 30:
            t_val = 1.96
        elif n >= 15:
            t_val = 2.13
        elif n >= 10:
            t_val = 2.26
        elif n >= 5:
            t_val = 2.78
        else:
            t_val = 4.30  # n=2 的 t 值
    elif confidence == 0.99:
        t_val = 2.576 if n >= 30 else 3.50
    else:
        t_val = 1.96

    margin = t_val * std_err
    return (mean - margin, mean + margin)


def bootstrap_ci(
    data: list[float],
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    statistic_fn: Callable[[list[float]], float] | None = None,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap 置信区间 (百分位法).

    通过有放回重采样估计统计量的分布，适用于小样本或非正态分布数据。

    Args:
        data: 原始数据
        n_resamples: 重采样次数 (默认 10000)
        confidence: 置信水平 (默认 0.95)
        statistic_fn: 统计量函数，默认为 mean
        seed: 随机种子 (用于可复现性)

    Returns:
        (lower, upper) 置信区间
    """
    if not data:
        return (0.0, 0.0)
    if len(data) == 1:
        return (data[0], data[0])

    if statistic_fn is None:
        statistic_fn = statistics.mean

    rng = random.Random(seed)
    n = len(data)
    bootstrap_stats: list[float] = []

    for _ in range(n_resamples):
        resample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(statistic_fn(resample))

    bootstrap_stats.sort()
    alpha = 1 - confidence
    lower_idx = max(0, int(n_resamples * alpha / 2))
    upper_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))

    return (bootstrap_stats[lower_idx], bootstrap_stats[upper_idx])


# ── 参数检验 ──────────────────────────────────────────────────────


def _approx_p(abs_t: float) -> tuple[str, bool]:
    """根据 |t| 或 |z| 值近似 p 值和显著性."""
    if abs_t > 3.5:
        return "p<0.001", True
    elif abs_t > 2.5:
        return "p<0.01", True
    elif abs_t > 2.0:
        return "p<0.05", True
    elif abs_t > 1.5:
        return "p<0.15", False
    else:
        return "p>0.15", False


def welch_t_test(
    data_a: list[float],
    data_b: list[float],
) -> StatTestResult:
    """Welch's t 检验 (独立双样本).

    不依赖 scipy，手动计算 t 统计量和 Cohen's d。

    Args:
        data_a: 样本 A
        data_b: 样本 B

    Returns:
        StatTestResult 包含 t 统计量、p 值近似、Cohen's d
    """
    n_a, n_b = len(data_a), len(data_b)
    if n_a < 2 or n_b < 2:
        return StatTestResult(
            test_name="welch_t",
            statistic=0.0,
            p_value_approx="insufficient_data",
            significant=False,
            effect_size=0.0,
            details={"n_a": n_a, "n_b": n_b},
        )

    mean_a = statistics.mean(data_a)
    mean_b = statistics.mean(data_b)
    var_a = statistics.variance(data_a)
    var_b = statistics.variance(data_b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-10:
        return StatTestResult(
            test_name="welch_t",
            statistic=0.0,
            p_value_approx="identical",
            significant=False,
            effect_size=0.0,
            details={"n_a": n_a, "n_b": n_b},
        )

    t_stat = (mean_a - mean_b) / se

    # Cohen's d
    pooled_std = math.sqrt((var_a + var_b) / 2)
    effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 1e-10 else 0.0

    p_approx, significant = _approx_p(abs(t_stat))

    return StatTestResult(
        test_name="welch_t",
        statistic=round(t_stat, 4),
        p_value_approx=p_approx,
        significant=significant,
        effect_size=round(effect_size, 4),
        details={"n_a": n_a, "n_b": n_b},
    )


def paired_t_test(
    data_a: list[float],
    data_b: list[float],
) -> StatTestResult:
    """配对 t 检验.

    适用于同一任务集上两个 agent 的对比 (paired tasks 场景)。

    Args:
        data_a: 样本 A (与 B 一一对应)
        data_b: 样本 B

    Returns:
        StatTestResult 包含 t 统计量、p 值近似、Cohen's d_z

    Raises:
        ValueError: 样本长度不一致
    """
    if len(data_a) != len(data_b):
        raise ValueError(f"样本长度不一致: {len(data_a)} vs {len(data_b)}")

    n = len(data_a)
    if n < 2:
        return StatTestResult(
            test_name="paired_t",
            statistic=0.0,
            p_value_approx="insufficient_data",
            significant=False,
            effect_size=0.0,
            details={"n": n},
        )

    diffs = [a - b for a, b in zip(data_a, data_b)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)

    if std_diff < 1e-10:
        return StatTestResult(
            test_name="paired_t",
            statistic=0.0,
            p_value_approx="identical",
            significant=False,
            effect_size=0.0,
            details={"n": n, "mean_diff": round(mean_diff, 4)},
        )

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    d_z = abs(mean_diff / std_diff)
    p_approx, significant = _approx_p(abs(t_stat))

    return StatTestResult(
        test_name="paired_t",
        statistic=round(t_stat, 4),
        p_value_approx=p_approx,
        significant=significant,
        effect_size=round(d_z, 4),
        details={"n": n, "df": n - 1, "mean_diff": round(mean_diff, 4)},
    )


# ── 非参数检验 ────────────────────────────────────────────────────


def mann_whitney_u(
    data_a: list[float],
    data_b: list[float],
) -> StatTestResult:
    """Mann-Whitney U 检验 (非参数).

    适用于不满足正态性假设的双样本比较。
    使用正态近似计算 z 分数。

    Args:
        data_a: 样本 A
        data_b: 样本 B

    Returns:
        StatTestResult 包含 U 统计量、p 值近似、rank-biserial r
    """
    n_a, n_b = len(data_a), len(data_b)
    if n_a < 2 or n_b < 2:
        return StatTestResult(
            test_name="mann_whitney_u",
            statistic=0.0,
            p_value_approx="insufficient_data",
            significant=False,
            effect_size=0.0,
            details={"n_a": n_a, "n_b": n_b},
        )

    # 合并并打标签
    combined = [(val, "a") for val in data_a] + [(val, "b") for val in data_b]
    combined.sort(key=lambda x: x[0])

    # 计算秩 (处理并列: 平均秩)
    ranks_a: list[float] = []
    ranks_b: list[float] = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # 1-based
        for k in range(i, j):
            if combined[k][1] == "a":
                ranks_a.append(avg_rank)
            else:
                ranks_b.append(avg_rank)
        i = j

    # U 统计量
    r_a = sum(ranks_a)
    u_a = r_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    # 正态近似
    mu_u = n_a * n_b / 2
    sigma_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

    if sigma_u < 1e-10:
        z = 0.0
    else:
        z = abs(u - mu_u) / sigma_u

    p_approx, significant = _approx_p(z)

    # Rank-biserial correlation (effect size)
    r_rb = 1 - (2 * u) / (n_a * n_b) if n_a * n_b > 0 else 0.0

    return StatTestResult(
        test_name="mann_whitney_u",
        statistic=round(u, 4),
        p_value_approx=p_approx,
        significant=significant,
        effect_size=round(abs(r_rb), 4),
        details={"n_a": n_a, "n_b": n_b, "z_score": round(z, 4)},
    )


# ── 多重比较校正 ──────────────────────────────────────────────────


# p 值近似字符串 → 保守上界
_P_MAP: dict[str, float] = {
    "p<0.001": 0.001,
    "p<0.01": 0.01,
    "p<0.05": 0.05,
    "p<0.15": 0.15,
    "p>0.15": 0.20,
    "insufficient_data": 1.0,
    "identical": 1.0,
}


def bonferroni_correct(
    p_values: list[str],
    alpha: float = 0.05,
) -> list[bool]:
    """Bonferroni 多重比较校正.

    调整显著性水平为 α/k，其中 k 为比较次数。
    因为只有 p 值区间，采用保守策略 (使用上界判断)。

    Args:
        p_values: p 值近似列表 (如 ["p<0.01", "p<0.05"])
        alpha: 总体显著性水平 (默认 0.05)

    Returns:
        每个检验的校正后显著性列表
    """
    k = len(p_values)
    if k == 0:
        return []

    alpha_corrected = alpha / k

    return [_P_MAP.get(p, 0.20) < alpha_corrected for p in p_values]
