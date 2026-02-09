"""Preference pair construction for RLHF / DPO training.

Given multiple trajectories for the same task, builds preference pairs
by ranking them by reward score and pairing best vs worst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PreferencePair:
    """A preference pair: chosen trajectory is better than rejected.

    Attributes:
        task_id: Identifier of the task
        chosen_trajectory_id: ID of the preferred trajectory
        rejected_trajectory_id: ID of the less preferred trajectory
        chosen_reward: Reward score of the chosen trajectory
        rejected_reward: Reward score of the rejected trajectory
        rationale: Why chosen is preferred over rejected
    """

    task_id: str
    chosen_trajectory_id: str
    rejected_trajectory_id: str
    chosen_reward: float
    rejected_reward: float
    rationale: str = ""

    def margin(self) -> float:
        """Return the preference margin (score difference)."""
        return self.chosen_reward - self.rejected_reward

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "chosen_trajectory_id": self.chosen_trajectory_id,
            "rejected_trajectory_id": self.rejected_trajectory_id,
            "chosen_reward": round(self.chosen_reward, 4),
            "rejected_reward": round(self.rejected_reward, 4),
            "margin": round(self.margin(), 4),
            "rationale": self.rationale,
        }


def build_preferences(
    trajectories_by_task: dict[str, list[dict[str, Any]]],
    min_margin: float = 0.05,
) -> list[PreferencePair]:
    """Build preference pairs from trajectories grouped by task.

    For each task:
    1. Sort trajectories by reward score (descending)
    2. Pair best vs worst, second-best vs second-worst, etc.
    3. Only include pairs where the margin exceeds min_margin

    Args:
        trajectories_by_task: Dict mapping task_id to list of trajectory dicts.
            Each trajectory dict must have:
            - id: str (trajectory identifier)
            - reward: float (reward score)
            - (optional) summary: str (for generating rationale)
        min_margin: Minimum score difference to include a pair (default: 0.05)

    Returns:
        List of PreferencePair objects
    """
    pairs: list[PreferencePair] = []

    for task_id, trajectories in trajectories_by_task.items():
        if len(trajectories) < 2:
            continue

        # Sort by reward descending
        sorted_trajs = sorted(
            trajectories,
            key=lambda t: t.get("reward", 0.0),
            reverse=True,
        )

        # Pair from outside in: best vs worst, second vs second-worst, etc.
        n = len(sorted_trajs)
        left = 0
        right = n - 1

        while left < right:
            chosen = sorted_trajs[left]
            rejected = sorted_trajs[right]

            chosen_reward = chosen.get("reward", 0.0)
            rejected_reward = rejected.get("reward", 0.0)
            margin = chosen_reward - rejected_reward

            if margin >= min_margin:
                # Generate rationale from summaries if available
                rationale = _build_rationale(chosen, rejected, margin)

                pairs.append(
                    PreferencePair(
                        task_id=task_id,
                        chosen_trajectory_id=chosen.get("id", f"traj_{left}"),
                        rejected_trajectory_id=rejected.get("id", f"traj_{right}"),
                        chosen_reward=chosen_reward,
                        rejected_reward=rejected_reward,
                        rationale=rationale,
                    )
                )

            left += 1
            right -= 1

    return pairs


def _build_rationale(
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    margin: float,
) -> str:
    """Build a human-readable rationale for why chosen > rejected.

    Args:
        chosen: The preferred trajectory dict
        rejected: The less preferred trajectory dict
        margin: Score difference

    Returns:
        Rationale string
    """
    parts: list[str] = []

    # Score difference
    parts.append(
        f"chosen scored {chosen.get('reward', 0.0):.3f} vs "
        f"rejected {rejected.get('reward', 0.0):.3f} (margin={margin:.3f})"
    )

    # Step count comparison
    chosen_steps = chosen.get("step_count", 0)
    rejected_steps = rejected.get("step_count", 0)
    if chosen_steps and rejected_steps:
        if chosen_steps < rejected_steps:
            parts.append(
                f"chosen used fewer steps ({chosen_steps} vs {rejected_steps})"
            )
        elif chosen_steps > rejected_steps:
            parts.append(
                f"chosen used more steps ({chosen_steps} vs {rejected_steps}) "
                f"but achieved higher quality"
            )

    # Outcome comparison
    chosen_outcome = chosen.get("outcome_score", None)
    rejected_outcome = rejected.get("outcome_score", None)
    if chosen_outcome is not None and rejected_outcome is not None:
        if chosen_outcome > rejected_outcome:
            parts.append("chosen had better task outcome")
        elif chosen_outcome == rejected_outcome:
            parts.append("both had same outcome, but chosen had better process quality")

    # Summary-based rationale
    chosen_summary = chosen.get("summary", "")
    rejected_summary = rejected.get("summary", "")
    if chosen_summary:
        parts.append(f"chosen: {chosen_summary}")
    if rejected_summary:
        parts.append(f"rejected: {rejected_summary}")

    return "; ".join(parts)


def preferences_to_dicts(pairs: list[PreferencePair]) -> list[dict[str, Any]]:
    """Convert preference pairs to list of dicts for serialization."""
    return [p.to_dict() for p in pairs]


def preferences_summary(pairs: list[PreferencePair]) -> dict[str, Any]:
    """Generate summary statistics for preference pairs.

    Args:
        pairs: List of PreferencePair objects

    Returns:
        Summary dict with counts and statistics
    """
    if not pairs:
        return {
            "total_pairs": 0,
            "unique_tasks": 0,
            "avg_margin": 0.0,
            "min_margin": 0.0,
            "max_margin": 0.0,
        }

    margins = [p.margin() for p in pairs]
    task_ids = {p.task_id for p in pairs}

    return {
        "total_pairs": len(pairs),
        "unique_tasks": len(task_ids),
        "avg_margin": round(sum(margins) / len(margins), 4),
        "min_margin": round(min(margins), 4),
        "max_margin": round(max(margins), 4),
    }
