"""Core reward engine - combines rule-based and model-based scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentreward.config import RewardConfig
from agentreward.rubrics import RubricSet, get_default_rubric_set
from agentreward.rules import (
    check_redundancy,
    check_regression,
    check_info_utilization,
    check_outcome,
    check_efficiency,
)
from agentreward.judge import judge_trajectory, JudgeConfig

logger = logging.getLogger(__name__)


@dataclass
class StepReward:
    """Reward for a single step in a trajectory.

    Attributes:
        step_id: Identifier for the step (0-based index)
        rubric_scores: Per-rubric scores {rubric_id: score}
        total_score: Weighted total score for this step
        rationale: Optional explanation (from model layer)
    """

    step_id: int
    rubric_scores: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "rubric_scores": self.rubric_scores,
            "total_score": round(self.total_score, 4),
            "rationale": self.rationale,
        }


@dataclass
class TrajectoryReward:
    """Reward for an entire trajectory.

    Attributes:
        step_rewards: Per-step reward details
        total_score: Overall trajectory score
        outcome_score: Task completion score (binary or test-based)
        process_score: Process quality score (average of step scores)
    """

    step_rewards: list[StepReward] = field(default_factory=list)
    total_score: float = 0.0
    outcome_score: float = 0.0
    process_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_score": round(self.total_score, 4),
            "outcome_score": round(self.outcome_score, 4),
            "process_score": round(self.process_score, 4),
            "step_count": len(self.step_rewards),
            "step_rewards": [sr.to_dict() for sr in self.step_rewards],
        }


class RewardEngine:
    """Main reward computation engine.

    Combines rule-based and model-based scoring using the 3-layer architecture:
    1. Rule layer: deterministic checks (redundancy, regression, efficiency)
    2. Model layer: LLM-as-Judge for semantic evaluation
    3. Human layer: calibration against human annotations (via calibration module)

    Args:
        config: RewardConfig with weights and model settings
        rubric_set: Optional RubricSet override (defaults to built-in set)
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        rubric_set: RubricSet | None = None,
    ):
        self.config = config or RewardConfig()
        self.rubric_set = rubric_set or get_default_rubric_set()

    def score(self, trajectory: dict[str, Any]) -> TrajectoryReward:
        """Score a single trajectory.

        Args:
            trajectory: Dict with:
                - task: str (task description)
                - steps: list[dict] (each with tool, params, output)
                - outcome: dict (success, tests_passed, tests_total)
                - reference_steps: int (optional, expected step count)

        Returns:
            TrajectoryReward with detailed per-step scores
        """
        steps = trajectory.get("steps", [])
        outcome = trajectory.get("outcome", {})
        reference_steps = trajectory.get("reference_steps", len(steps))
        logger.debug("评分轨迹: %d 步, reference=%d", len(steps), reference_steps)

        if not steps:
            return TrajectoryReward(
                total_score=0.0,
                outcome_score=check_outcome(outcome),
                process_score=0.0,
            )

        # --- Layer 1: Rule-based scoring ---
        redundancy_scores = check_redundancy(steps)
        regression_scores = check_regression(steps)
        info_util_scores = check_info_utilization(steps)

        # Map rule scores to rubric IDs
        rule_scores_by_step: list[dict[str, float]] = []
        for i in range(len(steps)):
            step_rule_scores: dict[str, float] = {}

            # Map redundancy to non_redundancy rubric
            non_redundancy_rubric = self.rubric_set.get_by_id("non_redundancy")
            if non_redundancy_rubric and non_redundancy_rubric.evaluator == "rule":
                # Combine redundancy and regression: both penalize wasteful steps
                step_rule_scores["non_redundancy"] = min(
                    redundancy_scores[i], regression_scores[i]
                )

            # Map info utilization to info_utilization rubric
            info_rubric = self.rubric_set.get_by_id("info_utilization")
            if info_rubric and info_rubric.evaluator == "rule":
                step_rule_scores["info_utilization"] = info_util_scores[i]

            rule_scores_by_step.append(step_rule_scores)

        # --- Layer 2: Model-based scoring (LLM-as-Judge) ---
        model_rubrics = self.rubric_set.get_model_rubrics()
        model_scores_by_step: list[dict[str, float]] = []

        if model_rubrics and self.config.model_weight > 0:
            judge_config = JudgeConfig(
                model=self.config.model_name,
                provider=self.config.provider,
                temperature=self.config.temperature,
            )
            judgments = judge_trajectory(
                trajectory=trajectory,
                rubrics=model_rubrics,
                config=judge_config,
            )
            for judgment in judgments:
                model_scores_by_step.append(judgment.rubric_scores)
        else:
            # No model scoring: fill with neutral scores
            for _ in steps:
                model_scores_by_step.append({r.id: 0.5 for r in model_rubrics})

        # --- Combine scores ---
        step_rewards: list[StepReward] = []
        for i in range(len(steps)):
            # Merge all rubric scores for this step
            all_scores: dict[str, float] = {}
            all_scores.update(rule_scores_by_step[i])
            all_scores.update(model_scores_by_step[i])

            # Compute weighted total
            weighted_sum = 0.0
            weight_total = 0.0
            for rubric in self.rubric_set.rubrics:
                if rubric.id in all_scores:
                    weighted_sum += rubric.weight * all_scores[rubric.id]
                    weight_total += rubric.weight

            total = weighted_sum / weight_total if weight_total > 0 else 0.0

            step_rewards.append(
                StepReward(
                    step_id=i,
                    rubric_scores=all_scores,
                    total_score=total,
                )
            )

        # --- Compute trajectory-level scores ---
        outcome_score = check_outcome(outcome)
        process_score = (
            sum(sr.total_score for sr in step_rewards) / len(step_rewards)
            if step_rewards
            else 0.0
        )

        # Efficiency bonus/penalty
        efficiency = check_efficiency(steps, reference_steps)

        # Total = weighted combination of outcome and process, adjusted by efficiency
        total_score = (
            0.4 * outcome_score
            + 0.5 * process_score
            + 0.1 * efficiency
        )

        logger.info("评分完成: total=%.4f (outcome=%.4f, process=%.4f, efficiency=%.4f)",
                    total_score, outcome_score, process_score, efficiency)
        return TrajectoryReward(
            step_rewards=step_rewards,
            total_score=total_score,
            outcome_score=outcome_score,
            process_score=process_score,
        )

    def score_batch(
        self, trajectories: list[dict[str, Any]]
    ) -> list[TrajectoryReward]:
        """Score multiple trajectories.

        Args:
            trajectories: List of trajectory dicts

        Returns:
            List of TrajectoryReward, one per trajectory
        """
        return [self.score(t) for t in trajectories]
