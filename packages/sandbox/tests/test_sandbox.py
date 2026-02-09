"""Tests for sandbox configuration, tool result, and Docker operations."""

from unittest.mock import MagicMock, patch

from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.replay import (
    Trajectory,
    TrajectoryStep,
    replay_trajectory,
)
from agentsandbox.sandbox import Sandbox, _LABEL
from agentsandbox.tools import (
    TOOL_REGISTRY,
    ToolResult,
    _exec_in_container,
    file_read,
    file_write,
    git,
    search,
    shell,
)


# ---------------------------------------------------------------------------
# Config & ToolResult 基础测试
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        config = SandboxConfig()
        assert config.image == "python:3.11-slim"
        assert config.timeout == 300
        assert config.memory_limit == "512m"
        assert config.cpu_limit == 1.0
        assert config.work_dir == "/workspace"
        assert config.network_enabled is False

    def test_custom_config(self):
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
        config = SandboxConfig(env_vars={"PYTHONPATH": "/workspace"})
        assert config.env_vars["PYTHONPATH"] == "/workspace"


class TestTaskConfig:
    """Tests for TaskConfig."""

    def test_default_config(self):
        config = TaskConfig()
        assert config.repo_url == ""
        assert config.base_commit == ""
        assert config.language == "python"
        assert config.domain == "coding"

    def test_validate_empty(self):
        """coding 领域空配置应报 2 个错误."""
        config = TaskConfig()
        errors = config.validate()
        assert len(errors) == 2
        assert "repo_url" in errors[0]
        assert "base_commit" in errors[1]

    def test_validate_valid(self):
        config = TaskConfig(
            repo_url="https://github.com/user/repo",
            base_commit="abc123",
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_non_coding_no_repo(self):
        """非 coding 领域不要求 repo_url/base_commit."""
        config = TaskConfig(domain="browser")
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_non_coding_with_description(self):
        """非 coding 领域只需 description 即可."""
        config = TaskConfig(domain="data_analysis", description="分析 CSV 数据")
        errors = config.validate()
        assert len(errors) == 0

    def test_full_config(self):
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
        result = ToolResult(output="hello world", exit_code=0)
        assert result.success is True
        assert result.output == "hello world"
        assert result.exit_code == 0
        assert result.error is None

    def test_failure_result(self):
        result = ToolResult(output="", exit_code=1, error="command not found")
        assert result.success is False
        assert result.exit_code == 1
        assert result.error == "command not found"

    def test_error_without_exit_code(self):
        result = ToolResult(output="partial", exit_code=0, error="timeout")
        assert result.success is False

    def test_default_result(self):
        result = ToolResult()
        assert result.success is True
        assert result.output == ""
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Helper: mock 容器的 exec_run 返回值
# ---------------------------------------------------------------------------


def _mock_exec_result(stdout: str = "", stderr: str = "", exit_code: int = 0):
    """创建 mock 的 container.exec_run() 返回值."""
    result = MagicMock()
    result.exit_code = exit_code
    result.output = (
        stdout.encode("utf-8") if stdout else None,
        stderr.encode("utf-8") if stderr else None,
    )
    return result


# ---------------------------------------------------------------------------
# _exec_in_container 测试
# ---------------------------------------------------------------------------


class TestExecInContainer:
    """Tests for _exec_in_container helper."""

    def test_success(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="ok\n")

        result = _exec_in_container(container, "echo ok")
        assert result.success is True
        assert result.output == "ok\n"
        container.exec_run.assert_called_once_with(
            ["bash", "-c", "echo ok"], demux=True, workdir=None
        )

    def test_with_work_dir(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="")

        _exec_in_container(container, "ls", work_dir="/workspace")
        container.exec_run.assert_called_once_with(
            ["bash", "-c", "ls"], demux=True, workdir="/workspace"
        )

    def test_stderr(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stderr="not found", exit_code=1)

        result = _exec_in_container(container, "cat /nonexistent")
        assert result.success is False
        assert result.exit_code == 1
        assert result.error == "not found"

    def test_exception(self):
        container = MagicMock()
        container.exec_run.side_effect = RuntimeError("Docker error")

        result = _exec_in_container(container, "cmd")
        assert result.success is False
        assert result.exit_code == 1
        assert "Docker error" in result.error


# ---------------------------------------------------------------------------
# 工具函数测试
# ---------------------------------------------------------------------------


class TestFileRead:
    """Tests for file_read tool."""

    def test_read_whole_file(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="line1\nline2\n")

        result = file_read(container, path="/workspace/test.py")
        assert result.output == "line1\nline2\n"
        cmd = container.exec_run.call_args[0][0]
        assert cmd == ["bash", "-c", "cat /workspace/test.py"]

    def test_read_range(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="line5\n")

        result = file_read(container, path="/f.py", start_line=5, end_line=10)
        assert result.success is True
        cmd = container.exec_run.call_args[0][0]
        assert "sed -n '5,10p'" in cmd[2]

    def test_read_from_start_line(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="tail\n")

        file_read(container, path="/f.py", start_line=3)
        cmd = container.exec_run.call_args[0][0]
        assert "sed -n '3,$p'" in cmd[2]


class TestFileWrite:
    """Tests for file_write tool."""

    def test_write(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result()

        result = file_write(container, path="/workspace/out.txt", content="hello")
        assert result.success is True
        assert "已写入" in result.output
        container.put_archive.assert_called_once()

    def test_write_creates_dirs(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result()

        file_write(container, path="/workspace/sub/dir/out.txt", content="data")
        # 应该先调用 mkdir -p
        mkdir_call = container.exec_run.call_args_list[0]
        assert "mkdir -p" in mkdir_call[0][0][2]


class TestShell:
    """Tests for shell tool."""

    def test_shell_command(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="result\n")

        result = shell(container, command="python -m pytest")
        assert result.output == "result\n"


class TestSearch:
    """Tests for search tool."""

    def test_search_with_results(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="file.py:10:match\n")

        result = search(container, pattern="match")
        assert result.success is True
        assert "file.py" in result.output

    def test_search_no_results(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(exit_code=1)

        result = search(container, pattern="notfound")
        # grep 返回 1 表示无匹配，不是错误
        assert result.exit_code == 0
        assert result.error is None

    def test_search_with_file_pattern(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="m\n")

        search(container, pattern="def", file_pattern="*.py")
        cmd = container.exec_run.call_args[0][0]
        assert "--include='*.py'" in cmd[2]


class TestGit:
    """Tests for git tool."""

    def test_git_status(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result(stdout="clean\n")

        result = git(container, subcommand="status")
        cmd = container.exec_run.call_args[0][0]
        assert cmd == ["bash", "-c", "git status"]
        assert result.output == "clean\n"

    def test_git_with_args(self):
        container = MagicMock()
        container.exec_run.return_value = _mock_exec_result()

        git(container, subcommand="diff", args="HEAD~1")
        cmd = container.exec_run.call_args[0][0]
        assert cmd == ["bash", "-c", "git diff HEAD~1"]


class TestToolRegistry:
    """Tests for TOOL_REGISTRY."""

    def test_all_tools_registered(self):
        assert set(TOOL_REGISTRY.keys()) == {"file_read", "file_write", "shell", "search", "git"}

    def test_registry_has_function(self):
        for name, entry in TOOL_REGISTRY.items():
            assert "function" in entry, f"{name} 缺少 function"
            assert callable(entry["function"]), f"{name} function 不可调用"


# ---------------------------------------------------------------------------
# Sandbox 类测试 (mock Docker)
# ---------------------------------------------------------------------------


class TestSandboxCreate:
    """Tests for Sandbox.create()."""

    @patch("agentsandbox.sandbox.docker")
    def test_create_basic(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client.containers.create.return_value = mock_container
        # git clone 和 checkout 成功
        mock_container.exec_run.return_value = _mock_exec_result()

        config = SandboxConfig()
        task = TaskConfig(repo_url="https://github.com/user/repo", base_commit="deadbeef")

        sandbox = Sandbox.create(config, task)
        assert sandbox.container_id == "abc123"
        assert sandbox.is_running is True
        mock_client.containers.create.assert_called_once()
        mock_container.start.assert_called_once()

    @patch("agentsandbox.sandbox.docker")
    def test_create_with_setup_commands(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.short_id = "xyz789"
        mock_client.containers.create.return_value = mock_container
        mock_container.exec_run.return_value = _mock_exec_result()

        config = SandboxConfig()
        task = TaskConfig(
            repo_url="https://github.com/user/repo",
            base_commit="abc",
            setup_commands=["pip install -e .", "npm install"],
        )

        sandbox = Sandbox.create(config, task)
        assert sandbox.container_id == "xyz789"
        # 应该执行: git clone, git checkout, 2 个 setup 命令
        assert mock_container.exec_run.call_count == 4

    @patch("agentsandbox.sandbox.docker")
    def test_create_git_clone_failure(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.short_id = "fail"
        mock_client.containers.create.return_value = mock_container
        mock_container.exec_run.return_value = _mock_exec_result(
            stderr="fatal: repo not found", exit_code=128
        )

        config = SandboxConfig()
        task = TaskConfig(repo_url="https://github.com/bad/repo", base_commit="abc")

        try:
            Sandbox.create(config, task)
            assert False, "应该抛出 RuntimeError"
        except RuntimeError as e:
            assert "git clone" in str(e)

    @patch("agentsandbox.sandbox.docker")
    def test_create_container_labels(self, mock_docker):
        """确认容器创建时带有标签."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.short_id = "lbl"
        mock_client.containers.create.return_value = mock_container
        mock_container.exec_run.return_value = _mock_exec_result()

        config = SandboxConfig()
        task = TaskConfig(repo_url="https://github.com/u/r", base_commit="abc")
        Sandbox.create(config, task)

        call_kwargs = mock_client.containers.create.call_args[1]
        assert call_kwargs["labels"] == {_LABEL: "true"}


class TestSandboxExecuteTool:
    """Tests for Sandbox.execute_tool()."""

    def test_execute_known_tool(self):
        sandbox = Sandbox(SandboxConfig())
        sandbox._container = MagicMock()
        sandbox._container.exec_run.return_value = _mock_exec_result(stdout="content\n")

        result = sandbox.execute_tool("file_read", {"path": "/workspace/f.py"})
        assert result.success is True
        assert result.output == "content\n"

    def test_execute_unknown_tool(self):
        sandbox = Sandbox(SandboxConfig())
        sandbox._container = MagicMock()

        result = sandbox.execute_tool("unknown_tool", {})
        assert result.success is False
        assert "未知工具" in result.error

    def test_execute_without_container(self):
        sandbox = Sandbox(SandboxConfig())

        result = sandbox.execute_tool("shell", {"command": "ls"})
        assert result.success is False
        assert "未启动" in result.error


class TestSandboxReset:
    """Tests for Sandbox.reset()."""

    def test_reset_with_base_commit(self):
        sandbox = Sandbox(SandboxConfig())
        sandbox.task = TaskConfig(base_commit="abc123")
        sandbox._container = MagicMock()
        sandbox._container.exec_run.return_value = _mock_exec_result()

        sandbox.reset()
        calls = sandbox._container.exec_run.call_args_list
        assert len(calls) == 2
        # git checkout abc123
        assert "abc123" in calls[0][0][0][2]
        # git clean -fdx
        assert "clean" in calls[1][0][0][2]

    def test_reset_without_container(self):
        sandbox = Sandbox(SandboxConfig())
        try:
            sandbox.reset()
            assert False, "应该抛出 RuntimeError"
        except RuntimeError:
            pass


class TestSandboxSnapshot:
    """Tests for Sandbox.snapshot()."""

    def test_snapshot(self):
        sandbox = Sandbox(SandboxConfig())
        sandbox._container = MagicMock()
        sandbox._container_id = "snap"
        mock_image = MagicMock()
        mock_image.short_id = "sha256:abc"
        sandbox._container.commit.return_value = mock_image

        result = sandbox.snapshot()
        assert result == "sha256:abc"
        assert len(sandbox._snapshots) == 1


class TestSandboxClose:
    """Tests for Sandbox.close()."""

    def test_close(self):
        sandbox = Sandbox(SandboxConfig())
        mock_container = MagicMock()
        sandbox._container = mock_container
        sandbox._container_id = "cid"
        sandbox._client = MagicMock()
        sandbox._snapshots = ["img1", "img2"]

        sandbox.close()
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert sandbox._container is None
        assert sandbox._container_id is None
        # 快照镜像应该被清理
        assert sandbox._client.images.remove.call_count == 2

    def test_context_manager(self):
        sandbox = Sandbox(SandboxConfig())
        sandbox._container = MagicMock()
        sandbox._container_id = "ctx"
        sandbox._client = MagicMock()

        with sandbox:
            pass

        sandbox._container is None


# ---------------------------------------------------------------------------
# Replay 测试
# ---------------------------------------------------------------------------


class TestTrajectory:
    """Tests for Trajectory data classes."""

    def test_from_dict(self):
        data = {
            "steps": [
                {"tool_name": "shell", "params": {"command": "ls"}, "expected_output": "a.py\n"},
                {"tool_name": "file_read", "params": {"path": "/f.py"}},
            ],
            "metadata": {"agent": "test"},
        }
        traj = Trajectory.from_dict(data)
        assert len(traj.steps) == 2
        assert traj.steps[0].tool_name == "shell"
        assert traj.steps[0].expected_output == "a.py\n"
        assert traj.steps[1].expected_output is None
        assert traj.metadata["agent"] == "test"

    def test_to_dict(self):
        traj = Trajectory(
            steps=[TrajectoryStep(tool_name="shell", params={"command": "ls"})],
            metadata={"key": "value"},
        )
        d = traj.to_dict()
        assert len(d["steps"]) == 1
        assert d["steps"][0]["tool_name"] == "shell"
        assert d["metadata"]["key"] == "value"

    def test_roundtrip(self):
        data = {
            "steps": [
                {"tool_name": "git", "params": {"subcommand": "status"}, "expected_output": "ok"},
            ],
            "metadata": {},
        }
        traj = Trajectory.from_dict(data)
        assert Trajectory.from_dict(traj.to_dict()).steps[0].tool_name == "git"


class TestReplayTrajectory:
    """Tests for replay_trajectory()."""

    def test_replay_success(self):
        sandbox = MagicMock(spec=Sandbox)
        sandbox.execute_tool.return_value = ToolResult(output="ok\n", exit_code=0)

        trajectory = Trajectory(
            steps=[
                TrajectoryStep(tool_name="shell", params={"command": "ls"}, expected_output="ok\n"),
                TrajectoryStep(tool_name="shell", params={"command": "pwd"}),
            ]
        )

        result = replay_trajectory(sandbox, trajectory)
        assert result.success is True
        assert result.total_steps == 2
        assert result.completed_steps == 2
        assert result.divergence_step == -1

    def test_replay_divergence(self):
        sandbox = MagicMock(spec=Sandbox)
        sandbox.execute_tool.return_value = ToolResult(output="actual\n", exit_code=0)

        trajectory = Trajectory(
            steps=[
                TrajectoryStep(
                    tool_name="shell",
                    params={"command": "ls"},
                    expected_output="expected\n",
                ),
            ]
        )

        result = replay_trajectory(sandbox, trajectory)
        assert result.success is False
        assert result.divergence_step == 0
        assert result.details[0]["diverged"] is True

    def test_replay_failure(self):
        sandbox = MagicMock(spec=Sandbox)
        sandbox.execute_tool.return_value = ToolResult(output="", exit_code=1, error="fail")

        trajectory = Trajectory(
            steps=[TrajectoryStep(tool_name="shell", params={"command": "bad"})],
        )

        result = replay_trajectory(sandbox, trajectory)
        assert result.success is False
        assert result.completed_steps == 1

    def test_replay_empty(self):
        sandbox = MagicMock(spec=Sandbox)
        trajectory = Trajectory(steps=[])

        result = replay_trajectory(sandbox, trajectory)
        assert result.success is True
        assert result.total_steps == 0
        assert result.completed_steps == 0
