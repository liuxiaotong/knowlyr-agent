"""测试 CLI process / process-batch 命令."""

import json

from click.testing import CliRunner

from trajectoryhub.cli import main


def _write_openhands_log(path):
    """写最小的 OpenHands JSONL 日志."""
    events = [
        {
            "id": 0, "source": "user", "action": "message",
            "args": {"content": "修复 bug", "task_id": "cli-test"},
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


def _write_sweagent_log(path, task_desc="修复 Django bug"):
    """写最小的 SWE-agent JSON 日志."""
    data = {
        "problem_statement": task_desc,
        "history": [
            [{"action": "bash", "command": "ls", "thought": "看看文件"}, "file.py\n"],
            [{"action": "submit"}, "Submitted"],
        ],
        "info": {
            "instance_id": "swe-test-001",
            "model_name": "gpt-4o",
            "repo": "django/django",
            "base_commit": "abc123",
            "exit_status": "submitted",
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)


class TestProcessCommand:
    """测试 process 命令."""

    def test_process_openhands(self, tmp_path):
        """处理 OpenHands 日志."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        runner = CliRunner()
        result = runner.invoke(main, [
            "process", str(log_path),
            "-f", "openhands",
            "-o", str(tmp_path / "out"),
        ])

        assert result.exit_code == 0
        assert "处理完成" in result.output
        assert "cli-test" in result.output
        assert "Reward" in result.output

    def test_process_sweagent(self, tmp_path):
        """处理 SWE-agent 日志."""
        log_path = tmp_path / "traj.json"
        _write_sweagent_log(log_path)

        runner = CliRunner()
        result = runner.invoke(main, [
            "process", str(log_path),
            "-f", "sweagent",
            "-o", str(tmp_path / "out"),
        ])

        assert result.exit_code == 0
        assert "处理完成" in result.output
        assert "swe-test-001" in result.output

    def test_process_with_save(self, tmp_path):
        """使用 --save 应写入 trajectories.jsonl."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(main, [
            "process", str(log_path),
            "-f", "openhands",
            "-o", str(out_dir),
            "--save",
        ])

        assert result.exit_code == 0
        assert "已保存" in result.output

        traj_file = out_dir / "trajectories.jsonl"
        assert traj_file.exists()

        with open(traj_file) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == 1
        assert records[0]["task_id"] == "cli-test"


class TestProcessBatchCommand:
    """测试 process-batch 命令."""

    def test_batch_sweagent(self, tmp_path):
        """批量处理 SWE-agent 日志."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for i in range(3):
            _write_sweagent_log(log_dir / f"traj_{i}.json", f"任务 {i}")

        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(main, [
            "process-batch", str(log_dir),
            "-f", "sweagent",
            "-o", str(out_dir),
            "-p", "*.json",
        ])

        assert result.exit_code == 0
        assert "批量处理完成" in result.output
        assert "轨迹数: 3" in result.output

        traj_file = out_dir / "trajectories.jsonl"
        assert traj_file.exists()

    def test_batch_empty_dir(self, tmp_path):
        """空目录应提示无文件."""
        log_dir = tmp_path / "empty"
        log_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            "process-batch", str(log_dir),
            "-f", "sweagent",
            "-o", str(tmp_path / "out"),
        ])

        assert result.exit_code == 0
        assert "没有找到" in result.output
