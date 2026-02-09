"""测试 Recorder CLI 命令."""

import json

from click.testing import CliRunner

from agentrecorder.cli import main


def _write_openhands_log(path):
    """写最小的 OpenHands JSONL 日志."""
    events = [
        {
            "id": 0, "source": "user", "action": "message",
            "args": {"content": "修复 bug", "task_id": "oh-cli"},
            "timestamp": "2026-01-01T10:00:00Z",
        },
        {
            "id": 1, "source": "agent", "action": "run",
            "args": {"command": "pytest"},
            "thought": "跑测试",
            "timestamp": "2026-01-01T10:01:00Z",
            "extras": {"model": "claude-sonnet-4-20250514", "agent_class": "CodeActAgent"},
        },
        {
            "id": 2, "observation": "run",
            "content": "1 passed",
            "extras": {"exit_code": 0},
        },
        {
            "id": 3, "action": "finish",
            "args": {"outputs": {"test_result": {
                "success": True, "tests_passed": 1, "tests_failed": 0,
            }}},
        },
    ]
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _write_sweagent_log(path):
    """写最小的 SWE-agent JSON 日志."""
    data = {
        "problem_statement": "修复 Django bug",
        "history": [
            [{"action": "bash", "command": "ls", "thought": "看看"}, "file.py\n"],
            [{"action": "submit"}, "Submitted"],
        ],
        "info": {
            "instance_id": "swe-001",
            "model_name": "gpt-4o",
            "exit_status": "submitted",
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)


class TestConvertCommand:
    """测试 convert 命令."""

    def test_convert_openhands(self, tmp_path):
        """转换 OpenHands 日志."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        runner = CliRunner()
        result = runner.invoke(main, ["convert", str(log_path), "-f", "openhands"])

        assert result.exit_code == 0
        assert "oh-cli" in result.output or "转换" in result.output

    def test_convert_sweagent(self, tmp_path):
        """转换 SWE-agent 日志."""
        log_path = tmp_path / "traj.json"
        _write_sweagent_log(log_path)

        runner = CliRunner()
        result = runner.invoke(main, ["convert", str(log_path), "-f", "swe-agent"])

        assert result.exit_code == 0

    def test_convert_with_output(self, tmp_path):
        """使用 -o 输出到文件."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)
        out_path = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(main, [
            "convert", str(log_path), "-f", "openhands", "-o", str(out_path),
        ])

        assert result.exit_code == 0
        assert "转换成功" in result.output


class TestValidateCommand:
    """测试 validate 命令."""

    def test_validate_openhands(self, tmp_path):
        """验证 OpenHands 日志."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(log_path), "-f", "openhands"])

        assert result.exit_code == 0
        assert "有效" in result.output

    def test_validate_auto_detect(self, tmp_path):
        """自动检测框架."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(log_path)])

        assert result.exit_code == 0
        assert "匹配框架" in result.output

    def test_validate_wrong_format(self, tmp_path):
        """格式不匹配应报错."""
        log_path = tmp_path / "bad.jsonl"
        with open(log_path, "w") as f:
            f.write("not a valid log\n")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(log_path), "-f", "openhands"])

        assert result.exit_code != 0


class TestSchemaCommand:
    """测试 schema 命令."""

    def test_schema_output(self):
        """schema 命令应输出内容."""
        runner = CliRunner()
        result = runner.invoke(main, ["schema"])

        assert result.exit_code == 0
        assert len(result.output) > 0


class TestBatchCommand:
    """测试 batch 命令."""

    def test_batch_sweagent(self, tmp_path):
        """批量转换 SWE-agent 日志."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for i in range(2):
            _write_sweagent_log(log_dir / f"traj_{i}.json")

        out_path = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(main, [
            "batch", str(log_dir),
            "-f", "swe-agent",
            "-o", str(out_path),
            "-p", "*.json",
        ])

        assert result.exit_code == 0
        assert "批量转换完成" in result.output
        assert "转换文件数: 2" in result.output
