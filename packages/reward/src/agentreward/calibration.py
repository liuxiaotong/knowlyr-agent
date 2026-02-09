"""Reward calibration against human annotations.

Computes correlation metrics between automated reward scores and human judgments
to validate and tune the reward model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationResult:
    """Result of calibrating automated rewards against human scores.

    Attributes:
        pearson_r: Pearson correlation coefficient
        spearman_rho: Spearman rank correlation coefficient
        agreement_rate: Fraction of pairs where auto and human agree on ordering
        details: Additional metrics and per-item breakdowns
    """

    pearson_r: float = 0.0
    spearman_rho: float = 0.0
    agreement_rate: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pearson_r": round(self.pearson_r, 4),
            "spearman_rho": round(self.spearman_rho, 4),
            "agreement_rate": round(self.agreement_rate, 4),
            "details": self.details,
        }


def calibrate(
    reward_scores: list[float],
    human_scores: list[float],
) -> CalibrationResult:
    """Calibrate automated reward scores against human annotations.

    Computes Pearson r, Spearman rho, and pairwise agreement rate.
    Uses scipy if available, otherwise falls back to basic computation.

    Args:
        reward_scores: List of automated reward scores
        human_scores: List of corresponding human scores

    Returns:
        CalibrationResult with correlation metrics

    Raises:
        ValueError: If input lists have different lengths or are too short
    """
    if len(reward_scores) != len(human_scores):
        raise ValueError(
            f"Score lists must have same length: "
            f"reward={len(reward_scores)}, human={len(human_scores)}"
        )

    if len(reward_scores) < 2:
        raise ValueError("Need at least 2 score pairs for calibration")

    n = len(reward_scores)

    # Try scipy first for robust statistics
    try:
        from scipy import stats

        pearson_r, pearson_p = stats.pearsonr(reward_scores, human_scores)
        spearman_rho, spearman_p = stats.spearmanr(reward_scores, human_scores)

        details = {
            "pearson_p_value": round(pearson_p, 6),
            "spearman_p_value": round(spearman_p, 6),
            "n": n,
            "method": "scipy",
        }
    except ImportError:
        # Fallback: compute manually
        pearson_r = _pearson_r(reward_scores, human_scores)
        spearman_rho = _spearman_rho(reward_scores, human_scores)

        details = {
            "n": n,
            "method": "fallback",
        }

    # Compute pairwise agreement rate
    agreement_rate = _pairwise_agreement(reward_scores, human_scores)

    # Compute per-item error details
    errors = [abs(r - h) for r, h in zip(reward_scores, human_scores)]
    details["mean_absolute_error"] = round(sum(errors) / n, 4)
    details["max_error"] = round(max(errors), 4)
    details["min_error"] = round(min(errors), 4)

    return CalibrationResult(
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        agreement_rate=agreement_rate,
        details=details,
    )


# --- Fallback implementations ---


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient (fallback without scipy)."""
    n = len(x)
    if n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient (fallback without scipy)."""
    rank_x = _rank(x)
    rank_y = _rank(y)
    return _pearson_r(rank_x, rank_y)


def _rank(values: list[float]) -> list[float]:
    """Compute ranks for a list of values (average rank for ties)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        # Find all items with the same value (ties)
        j = i
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1

        # Average rank for ties (1-based ranking)
        avg_rank = (i + j + 1) / 2  # (i+1 + j) / 2 in 1-based
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank

        i = j

    return ranks


def _pairwise_agreement(x: list[float], y: list[float]) -> float:
    """Compute pairwise agreement rate.

    For all pairs (i, j), checks if x and y agree on which is larger.

    Returns:
        Fraction of concordant pairs
    """
    n = len(x)
    if n < 2:
        return 1.0

    concordant = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff_x = x[i] - x[j]
            diff_y = y[i] - y[j]

            if diff_x == 0 and diff_y == 0:
                # Both tied — concordant
                concordant += 1
            elif diff_x * diff_y > 0:
                # Same direction — concordant
                concordant += 1
            # else: discordant (opposite direction or one tied)

            total += 1

    return concordant / total if total > 0 else 1.0
