"""Tests for pipeline configuration and basic setup."""

from trajectoryhub.config import (
    AgentConfig,
    PipelineConfig,
    RewardConfig,
    SandboxConfig,
    TaskSource,
)
from trajectoryhub.pipeline import Pipeline, PipelineResult, Trajectory
from trajectoryhub.tasks import TaskInfo


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.output_dir == "./output"
        assert config.parallel_workers == 1
        assert config.checkpoint_interval == 10
        assert len(config.agents) == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            task_source=TaskSource(path="tasks.jsonl"),
            agents=[
                AgentConfig(framework="openhands", model="claude-sonnet-4-20250514"),
                AgentConfig(framework="sweagent", model="gpt-4o"),
            ],
            output_dir="./custom_output",
            parallel_workers=4,
        )
        assert config.task_source.path == "tasks.jsonl"
        assert len(config.agents) == 2
        assert config.agents[0].framework == "openhands"
        assert config.agents[1].model == "gpt-4o"
        assert config.output_dir == "./custom_output"
        assert config.parallel_workers == 4

    def test_agent_config_defaults(self):
        """Test default agent configuration."""
        agent = AgentConfig()
        assert agent.framework == "openhands"
        assert agent.max_steps == 30
        assert agent.temperature == 0.0

    def test_sandbox_config_defaults(self):
        """Test default sandbox configuration."""
        sandbox = SandboxConfig()
        assert sandbox.timeout == 600
        assert sandbox.memory_limit == "4g"
        assert sandbox.network_enabled is False

    def test_reward_config_defaults(self):
        """Test default reward configuration."""
        reward = RewardConfig()
        assert reward.enable_rule_layer is True
        assert reward.enable_model_layer is False
        assert len(reward.rubric_weights) == 5
        assert abs(sum(reward.rubric_weights.values()) - 1.0) < 0.01


class TestPipeline:
    """Tests for Pipeline basic setup."""

    def test_pipeline_creation(self):
        """Test pipeline can be created with config."""
        config = PipelineConfig()
        pipeline = Pipeline(config)
        assert pipeline.config == config

    def test_run_single_returns_trajectory(self):
        """Test run_single returns a Trajectory."""
        config = PipelineConfig()
        pipeline = Pipeline(config)
        task = TaskInfo(task_id="test-001", description="Test task")
        agent_config = AgentConfig()

        trajectory = pipeline.run_single(task, agent_config)

        assert isinstance(trajectory, Trajectory)
        assert trajectory.task_id == "test-001"
        assert trajectory.agent_framework == "openhands"

    def test_pipeline_result_dataclass(self):
        """Test PipelineResult has correct fields."""
        result = PipelineResult(
            total_tasks=10,
            completed=8,
            failed=2,
            trajectories_path="/tmp/traj.jsonl",
        )
        assert result.total_tasks == 10
        assert result.completed == 8
        assert result.failed == 2
        assert result.trajectories_path == "/tmp/traj.jsonl"


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_default_trajectory(self):
        """Test default trajectory values."""
        traj = Trajectory()
        assert traj.task_id == ""
        assert traj.steps == []
        assert traj.reward == 0.0
        assert traj.success is False

    def test_custom_trajectory(self):
        """Test trajectory with custom values."""
        traj = Trajectory(
            task_id="task-001",
            agent_framework="openhands",
            agent_model="claude-sonnet-4-20250514",
            steps=[{"action": "file_read", "path": "/test.py"}],
            total_steps=1,
            success=True,
            reward=0.85,
            step_rewards=[0.85],
        )
        assert traj.task_id == "task-001"
        assert len(traj.steps) == 1
        assert traj.reward == 0.85
        assert traj.success is True
