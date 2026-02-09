"""测试 SandboxEnv — Gymnasium 风格环境适配器."""

from unittest.mock import MagicMock, patch

from knowlyrcore.env import AgentEnv, EnvWrapper
from knowlyrcore.timestep import TimeStep
from knowlyrcore import ToolResult

from agentsandbox.env import SandboxEnv
from agentsandbox.config import SandboxConfig, TaskConfig


# ── 基础测试 ──────────────────────────────────────────────────────


class TestSandboxEnvInterface:
    """验证 SandboxEnv 实现 AgentEnv 协议."""

    def test_is_agent_env(self):
        assert issubclass(SandboxEnv, AgentEnv)

    def test_domain(self):
        env = SandboxEnv()
        assert env.domain == "coding"

    def test_available_tools(self):
        env = SandboxEnv()
        tools = env.available_tools
        assert "shell" in tools
        assert "file_read" in tools
        assert "file_write" in tools
        assert "search" in tools
        assert "git" in tools

    def test_init_defaults(self):
        env = SandboxEnv()
        assert env._max_steps == 30
        assert env._sandbox is None

    def test_init_custom(self):
        config = SandboxConfig(image="ubuntu:22.04", timeout=600)
        env = SandboxEnv(config=config, max_steps=50)
        assert env._config.image == "ubuntu:22.04"
        assert env._max_steps == 50


# ── Mock 测试 (不需要 Docker) ─────────────────────────────────────


class TestSandboxEnvMocked:
    """使用 mock 测试 reset/step/close."""

    @patch("agentsandbox.env.Sandbox")
    def test_reset(self, MockSandbox):
        """reset 应创建沙箱并返回 TimeStep."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"))
        ts = env.reset()

        assert isinstance(ts, TimeStep)
        assert ts.observation == "沙箱就绪"
        assert ts.done is False
        assert "container_id" in ts.info
        MockSandbox.create.assert_called_once()

    @patch("agentsandbox.env.Sandbox")
    def test_reset_with_task(self, MockSandbox):
        """reset(task=...) 应从 TaskInfo 构造 TaskConfig."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        MockSandbox.create.return_value = mock_sandbox

        task = MagicMock()
        task.repo = "owner/repo"
        task.base_commit = "def456"
        task.test_command = "pytest"
        task.description = "Fix bug"
        task.domain = "coding"

        env = SandboxEnv()
        ts = env.reset(task=task)

        assert ts.observation == "沙箱就绪"
        assert env._task_config.repo_url == "owner/repo"
        assert env._task_config.base_commit == "def456"

    @patch("agentsandbox.env.Sandbox")
    def test_step(self, MockSandbox):
        """step 应调用 execute_tool 并返回 TimeStep."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        mock_sandbox.execute_tool.return_value = ToolResult(
            output="hello.py\n", exit_code=0,
        )
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"))
        env.reset()
        ts = env.step({"tool": "shell", "params": {"command": "ls"}})

        assert ts.observation == "hello.py\n"
        assert ts.reward == 0.0
        assert ts.terminated is False
        assert ts.truncated is False
        assert ts.info["exit_code"] == 0
        assert ts.info["step"] == 1
        mock_sandbox.execute_tool.assert_called_once_with("shell", {"command": "ls"})

    @patch("agentsandbox.env.Sandbox")
    def test_step_terminated(self, MockSandbox):
        """submit 工具应设置 terminated=True."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        mock_sandbox.execute_tool.return_value = ToolResult(
            output="submitted", exit_code=0,
        )
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"))
        env.reset()
        ts = env.step({"tool": "submit", "params": {}})

        assert ts.terminated is True

    @patch("agentsandbox.env.Sandbox")
    def test_step_truncated(self, MockSandbox):
        """超过 max_steps 应设置 truncated=True."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        mock_sandbox.execute_tool.return_value = ToolResult(
            output="ok", exit_code=0,
        )
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"), max_steps=2)
        env.reset()
        env.step({"tool": "shell", "params": {"command": "ls"}})
        ts = env.step({"tool": "shell", "params": {"command": "cat f.py"}})

        assert ts.truncated is True

    @patch("agentsandbox.env.Sandbox")
    def test_close(self, MockSandbox):
        """close 应关闭底层沙箱."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"))
        env.reset()
        env.close()

        mock_sandbox.close.assert_called_once()
        assert env._sandbox is None

    @patch("agentsandbox.env.Sandbox")
    def test_history(self, MockSandbox):
        """history 应记录所有步骤."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        mock_sandbox.execute_tool.return_value = ToolResult(
            output="ok", exit_code=0,
        )
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"))
        env.reset()
        env.step({"tool": "shell", "params": {"command": "ls"}})
        env.step({"tool": "file_read", "params": {"path": "f.py"}})

        assert len(env.history) == 2
        assert env.history[0]["tool"] == "shell"
        assert env.history[1]["tool"] == "file_read"

    @patch("agentsandbox.env.Sandbox")
    def test_context_manager(self, MockSandbox):
        """with 语句应自动 close."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        MockSandbox.create.return_value = mock_sandbox

        with SandboxEnv(task_config=TaskConfig(domain="coding")) as env:
            env.reset()
        mock_sandbox.close.assert_called_once()

    def test_step_without_reset(self):
        """未 reset 时 step 应返回 terminated TimeStep."""
        env = SandboxEnv()
        ts = env.step({"tool": "shell", "params": {"command": "ls"}})
        assert ts.terminated is True
        assert "error" in ts.info

    @patch("agentsandbox.env.Sandbox")
    def test_wrapper_compatible(self, MockSandbox):
        """SandboxEnv 应与 EnvWrapper 兼容."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"
        mock_sandbox.execute_tool.return_value = ToolResult(
            output="ok", exit_code=0,
        )
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"))
        wrapped = EnvWrapper(env)

        ts = wrapped.reset()
        assert ts.observation == "沙箱就绪"
        assert wrapped.unwrapped is env

    @patch("agentsandbox.env.Sandbox")
    def test_full_episode(self, MockSandbox):
        """完整 episode 流程."""
        mock_sandbox = MagicMock()
        mock_sandbox.container_id = "abc123"

        call_count = [0]

        def fake_execute(tool, params):
            call_count[0] += 1
            if tool == "submit":
                return ToolResult(output="submitted", exit_code=0)
            return ToolResult(output=f"step-{call_count[0]}", exit_code=0)

        mock_sandbox.execute_tool.side_effect = fake_execute
        MockSandbox.create.return_value = mock_sandbox

        env = SandboxEnv(task_config=TaskConfig(domain="coding"))
        ts = env.reset()
        steps = 0
        while not ts.done:
            if steps >= 3:
                ts = env.step({"tool": "submit", "params": {}})
            else:
                ts = env.step({"tool": "shell", "params": {"command": "ls"}})
            steps += 1

        assert steps == 4
        assert ts.terminated is True
        assert len(env.history) == 4
