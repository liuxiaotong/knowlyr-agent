"""Tests for sandbox configuration and tool result."""

from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.tools import ToolResult


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()
        assert config.image == "python:3.11-slim"
        assert config.timeout == 300
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.work_dir == "/workspace"
        assert config.network_enabled is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SandboxConfig(
            image="node:18-slim",
            timeout=600,
            memory_limit="2g",
            cpu_limit=2.0,
            work_dir="/app",
            network_enabled=True,
        )
        assert config.image == "node:18-slim"
        assert config.timeout == 600
        assert config.memory_limit == "2g"
        assert config.cpu_limit == 2.0
        assert config.work_dir == "/app"
        assert config.network_enabled is True

    def test_env_vars(self):
        """Test environment variables."""
        config = SandboxConfig(env_vars={"PYTHONPATH": "/workspace"})
        assert config.env_vars["PYTHONPATH"] == "/workspace"


class TestTaskConfig:
    """Tests for TaskConfig."""

    def test_default_config(self):
        """Test default task configuration."""
        config = TaskConfig()
        assert config.repo_url == ""
        assert config.base_commit == ""
        assert config.language == "python"

    def test_validate_empty(self):
        """Test validation with empty config."""
        config = TaskConfig()
        errors = config.validate()
        assert len(errors) == 2
        assert "repo_url" in errors[0]
        assert "base_commit" in errors[1]

    def test_validate_valid(self):
        """Test validation with valid config."""
        config = TaskConfig(
            repo_url="https://github.com/user/repo",
            base_commit="abc123",
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_full_config(self):
        """Test full task configuration."""
        config = TaskConfig(
            repo_url="https://github.com/user/repo",
            base_commit="abc123",
            test_command="pytest tests/",
            language="python",
            setup_commands=["pip install -e ."],
            description="Fix bug in parser",
        )
        assert config.test_command == "pytest tests/"
        assert len(config.setup_commands) == 1


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(output="hello world", exit_code=0)
        assert result.success is True
        assert result.output == "hello world"
        assert result.exit_code == 0
        assert result.error is None

    def test_failure_result(self):
        """Test failed tool result."""
        result = ToolResult(output="", exit_code=1, error="command not found")
        assert result.success is False
        assert result.exit_code == 1
        assert result.error == "command not found"

    def test_error_without_exit_code(self):
        """Test result with error but zero exit code."""
        result = ToolResult(output="partial", exit_code=0, error="timeout")
        assert result.success is False

    def test_default_result(self):
        """Test default tool result."""
        result = ToolResult()
        assert result.success is True
        assert result.output == ""
        assert result.exit_code == 0
