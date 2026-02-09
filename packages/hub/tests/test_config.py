"""测试 Hub 配置模型."""

from trajectoryhub.config import (
    AgentConfig,
    PipelineConfig,
    RewardConfig,
    SandboxConfig,
    TaskSource,
)


class TestTaskSource:
    """测试任务来源配置."""

    def test_defaults(self):
        """默认值."""
        ts = TaskSource()
        assert ts.path is None
        assert ts.source_type == "jsonl"
        assert ts.split == "test"
        assert ts.limit is None
        assert ts.filters == {}

    def test_custom(self):
        """自定义配置."""
        ts = TaskSource(
            path="/data/tasks.jsonl",
            source_type="swebench",
            dataset_name="princeton-nlp/SWE-bench",
            limit=50,
            filters={"language": "python"},
        )
        assert ts.path == "/data/tasks.jsonl"
        assert ts.source_type == "swebench"
        assert ts.limit == 50
        assert ts.filters["language"] == "python"


class TestAgentConfig:
    """测试 Agent 配置."""

    def test_defaults(self):
        """默认值."""
        ac = AgentConfig()
        assert ac.framework == "openhands"
        assert ac.model == "claude-sonnet-4-20250514"
        assert ac.max_steps == 30
        assert ac.temperature == 0.0

    def test_custom(self):
        """自定义 Agent."""
        ac = AgentConfig(framework="sweagent", model="gpt-4o", max_steps=50)
        assert ac.framework == "sweagent"
        assert ac.model == "gpt-4o"
        assert ac.max_steps == 50


class TestSandboxConfig:
    """测试沙箱配置."""

    def test_defaults(self):
        """默认值."""
        sc = SandboxConfig()
        assert sc.image == "knowlyr/sandbox:latest"
        assert sc.timeout == 600
        assert sc.memory_limit == "4g"
        assert sc.network_enabled is False


class TestRewardConfig:
    """测试 Reward 配置."""

    def test_defaults(self):
        """默认值."""
        rc = RewardConfig()
        assert rc.enable_rule_layer is True
        assert rc.enable_model_layer is False
        assert "goal_progress" in rc.rubric_weights

    def test_custom_weights(self):
        """自定义权重."""
        rc = RewardConfig(rubric_weights={"custom": 1.0})
        assert rc.rubric_weights == {"custom": 1.0}


class TestPipelineConfig:
    """测试 Pipeline 全局配置."""

    def test_defaults(self):
        """默认配置."""
        pc = PipelineConfig()
        assert pc.output_dir == "./output"
        assert pc.parallel_workers == 1
        assert pc.checkpoint_interval == 10
        assert len(pc.agents) == 1

    def test_custom(self):
        """自定义 Pipeline."""
        pc = PipelineConfig(
            task_source=TaskSource(path="/tasks.jsonl"),
            agents=[
                AgentConfig(framework="openhands"),
                AgentConfig(framework="sweagent", model="gpt-4o"),
            ],
            output_dir="/data/output",
            parallel_workers=4,
        )
        assert pc.task_source.path == "/tasks.jsonl"
        assert len(pc.agents) == 2
        assert pc.agents[1].model == "gpt-4o"
        assert pc.parallel_workers == 4

    def test_nested_defaults(self):
        """嵌套配置的默认值应正确."""
        pc = PipelineConfig()
        assert pc.sandbox_config.timeout == 600
        assert pc.reward_config.enable_rule_layer is True
        assert pc.task_source.source_type == "jsonl"

    def test_model_dump(self):
        """应能序列化."""
        pc = PipelineConfig(output_dir="/test")
        d = pc.model_dump()
        assert d["output_dir"] == "/test"
        assert "agents" in d
        assert "sandbox_config" in d
