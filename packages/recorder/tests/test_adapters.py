"""适配器单元测试."""

import json

import pytest

from agentrecorder.adapters import OpenHandsAdapter, SWEAgentAdapter


# ── OpenHands 测试数据 ──────────────────────────────────────────────


def _write_openhands_log(path, events):
    """写入 OpenHands JSONL 日志."""
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _make_openhands_events():
    """构造一个完整的 OpenHands 日志事件序列."""
    return [
        # 用户任务描述
        {
            "id": 0,
            "source": "user",
            "action": "message",
            "args": {"content": "修复 sort.py 的 off-by-one 错误", "task_id": "oh-001"},
            "timestamp": "2026-01-01T10:00:00Z",
        },
        # Agent 读文件
        {
            "id": 1,
            "source": "agent",
            "action": "read",
            "args": {"path": "/workspace/sort.py"},
            "thought": "先看看代码",
            "timestamp": "2026-01-01T10:01:00Z",
            "extras": {"model": "claude-sonnet-4-20250514", "agent_class": "CodeActAgent"},
        },
        {
            "id": 2,
            "observation": "read",
            "content": "def sort(lst):\n    pass\n",
            "extras": {},
        },
        # Agent 执行命令
        {
            "id": 3,
            "source": "agent",
            "action": "run",
            "args": {"command": "pytest tests/"},
            "thought": "运行测试",
            "timestamp": "2026-01-01T10:02:00Z",
            "extras": {"token_count": 100},
        },
        {
            "id": 4,
            "observation": "run",
            "content": "1 passed in 0.02s",
            "extras": {"exit_code": 0},
        },
        # Agent 编辑文件
        {
            "id": 5,
            "source": "agent",
            "action": "edit",
            "args": {"path": "/workspace/sort.py", "old_str": "pass", "new_str": "return sorted(lst)"},
            "thought": "修复排序",
            "timestamp": "2026-01-01T10:03:00Z",
        },
        {
            "id": 6,
            "observation": "edit",
            "content": "File edited successfully",
            "extras": {},
        },
        # 完成
        {
            "id": 7,
            "action": "finish",
            "args": {"outputs": {"test_result": {"success": True, "tests_passed": 3, "tests_failed": 0}}},
        },
    ]


# ── SWE-agent 测试数据 ─────────────────────────────────────────────


def _write_sweagent_log(path, data):
    """写入 SWE-agent JSON 日志."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _make_sweagent_data():
    """构造一个完整的 SWE-agent 轨迹数据."""
    return {
        "environment": "swe_main",
        "problem_statement": "修复 Django model 的 N+1 查询问题",
        "history": [
            [
                {"action": "bash", "command": "find . -name models.py", "thought": "找到模型文件"},
                "./app/models.py\n",
            ],
            [
                {"action": "open", "path": "./app/models.py"},
                "class User(Model):\n    name = CharField()\n",
            ],
            [
                {"action": "edit", "path": "./app/models.py", "old_str": "name", "new_str": "username"},
                "File edited successfully",
            ],
            [
                {"action": "submit"},
                "Submitted",
            ],
        ],
        "info": {
            "instance_id": "django__django-12345",
            "model_name": "gpt-4o",
            "repo": "django/django",
            "base_commit": "abc123",
            "exit_status": "submitted",
            "model_stats": {"tokens_sent": 5000, "tokens_received": 2000},
        },
    }


# ── OpenHandsAdapter 测试 ──────────────────────────────────────────


class TestOpenHandsValidate:
    """验证 OpenHands 日志格式检测."""

    def test_valid_jsonl(self, tmp_path):
        """合法的 OpenHands JSONL 应通过验证."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, [{"action": "message", "source": "user"}])
        assert OpenHandsAdapter().validate(str(path)) is True

    def test_invalid_extension(self, tmp_path):
        """非 .jsonl/.json 文件应失败."""
        path = tmp_path / "log.txt"
        path.write_text('{"action": "run"}')
        assert OpenHandsAdapter().validate(str(path)) is False

    def test_empty_file(self, tmp_path):
        """空文件应失败."""
        path = tmp_path / "log.jsonl"
        path.write_text("")
        assert OpenHandsAdapter().validate(str(path)) is False

    def test_no_action_field(self, tmp_path):
        """没有 action/observation 字段应失败."""
        path = tmp_path / "log.jsonl"
        path.write_text('{"type": "other"}\n')
        assert OpenHandsAdapter().validate(str(path)) is False

    def test_nonexistent_file(self):
        """不存在的文件应失败."""
        assert OpenHandsAdapter().validate("/nonexistent/log.jsonl") is False


class TestOpenHandsParse:
    """验证 OpenHands 日志解析."""

    def test_full_parse(self, tmp_path):
        """完整日志应正确解析为 Trajectory."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        adapter = OpenHandsAdapter()
        traj = adapter.parse(str(path))

        assert traj.agent == "openhands/CodeActAgent"
        assert traj.model == "claude-sonnet-4-20250514"
        assert traj.task.task_id == "oh-001"
        assert "off-by-one" in traj.task.description
        assert len(traj.steps) == 3  # read + run + edit

    def test_step_mapping(self, tmp_path):
        """action 类型应正确映射到工具名."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        traj = OpenHandsAdapter().parse(str(path))
        tool_names = [s.tool_call.name for s in traj.steps]
        assert tool_names == ["read_file", "bash", "edit_file"]

    def test_tool_result(self, tmp_path):
        """observation 应正确映射到 ToolResult."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        traj = OpenHandsAdapter().parse(str(path))
        # bash 命令的结果
        bash_step = traj.steps[1]
        assert bash_step.tool_result.output == "1 passed in 0.02s"
        assert bash_step.tool_result.exit_code == 0
        assert bash_step.tool_result.success is True

    def test_outcome(self, tmp_path):
        """finish 事件应正确设置 outcome."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        traj = OpenHandsAdapter().parse(str(path))
        assert traj.outcome.success is True
        assert traj.outcome.tests_passed == 3
        assert traj.outcome.tests_failed == 0
        assert traj.outcome.total_steps == 3

    def test_thought_preserved(self, tmp_path):
        """agent 的 thought 应保留."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        traj = OpenHandsAdapter().parse(str(path))
        assert traj.steps[0].thought == "先看看代码"
        assert traj.steps[1].thought == "运行测试"

    def test_token_count(self, tmp_path):
        """token_count 应从 extras 中提取."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        traj = OpenHandsAdapter().parse(str(path))
        assert traj.steps[1].token_count == 100
        assert traj.steps[0].token_count is None  # 没有 token_count

    def test_skip_user_messages(self, tmp_path):
        """用户消息不应生成 Step."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        traj = OpenHandsAdapter().parse(str(path))
        # 只有 3 个 agent 操作，不含用户消息
        assert len(traj.steps) == 3

    def test_empty_log(self, tmp_path):
        """空日志应返回空步骤."""
        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, [])

        traj = OpenHandsAdapter().parse(str(path))
        assert len(traj.steps) == 0
        assert traj.outcome.success is False

    def test_metadata_extraction(self, tmp_path):
        """extras 中的元数据应被提取."""
        path = tmp_path / "log.jsonl"
        events = _make_openhands_events()
        events[1]["extras"]["instance_id"] = "swebench-123"
        events[1]["extras"]["max_iterations"] = 30
        _write_openhands_log(path, events)

        traj = OpenHandsAdapter().parse(str(path))
        assert traj.metadata["instance_id"] == "swebench-123"
        assert traj.metadata["max_iterations"] == 30
        assert traj.metadata["agent_class"] == "CodeActAgent"

    def test_error_observation(self, tmp_path):
        """含 error 的 observation 应设置 ToolResult.error."""
        path = tmp_path / "log.jsonl"
        events = [
            {"id": 1, "source": "agent", "action": "run", "args": {"command": "bad"},
             "thought": "试试", "timestamp": "2026-01-01T10:00:00Z"},
            {"id": 2, "observation": "run", "content": "command not found",
             "extras": {"exit_code": 127, "error": "command not found"}},
        ]
        _write_openhands_log(path, events)

        traj = OpenHandsAdapter().parse(str(path))
        assert traj.steps[0].tool_result.exit_code == 127
        assert traj.steps[0].tool_result.error == "command not found"
        assert traj.steps[0].tool_result.success is False


# ── SWEAgentAdapter 测试 ───────────────────────────────────────────


class TestSWEAgentValidate:
    """验证 SWE-agent 日志格式检测."""

    def test_valid_json(self, tmp_path):
        """合法的 SWE-agent JSON 应通过验证."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())
        assert SWEAgentAdapter().validate(str(path)) is True

    def test_invalid_extension(self, tmp_path):
        """非 .json 文件应失败."""
        path = tmp_path / "traj.jsonl"
        path.write_text('{"history": [], "info": {}}')
        assert SWEAgentAdapter().validate(str(path)) is False

    def test_missing_fields(self, tmp_path):
        """缺少 history/info 字段应失败."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, {"data": "other"})
        assert SWEAgentAdapter().validate(str(path)) is False

    def test_nonexistent_file(self):
        """不存在的文件应失败."""
        assert SWEAgentAdapter().validate("/nonexistent/traj.json") is False


class TestSWEAgentParse:
    """验证 SWE-agent 日志解析."""

    def test_full_parse(self, tmp_path):
        """完整日志应正确解析为 Trajectory."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        traj = SWEAgentAdapter().parse(str(path))

        assert traj.agent == "swe-agent"
        assert traj.model == "gpt-4o"
        assert traj.task.task_id == "django__django-12345"
        assert "N+1" in traj.task.description
        assert len(traj.steps) == 4  # bash + open + edit + submit

    def test_task_info(self, tmp_path):
        """task 字段应从 info 中正确提取."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        traj = SWEAgentAdapter().parse(str(path))
        assert traj.task.repo == "django/django"
        assert traj.task.base_commit == "abc123"

    def test_step_mapping(self, tmp_path):
        """action 类型应正确映射到工具名."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        traj = SWEAgentAdapter().parse(str(path))
        tool_names = [s.tool_call.name for s in traj.steps]
        assert tool_names == ["bash", "read_file", "edit_file", "submit"]

    def test_outcome_submitted(self, tmp_path):
        """exit_status=submitted 应设为成功."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        traj = SWEAgentAdapter().parse(str(path))
        assert traj.outcome.success is True
        assert traj.outcome.total_steps == 4

    def test_outcome_failed(self, tmp_path):
        """非 submitted 的 exit_status 应设为失败."""
        path = tmp_path / "traj.json"
        data = _make_sweagent_data()
        data["info"]["exit_status"] = "error"
        _write_sweagent_log(path, data)

        traj = SWEAgentAdapter().parse(str(path))
        assert traj.outcome.success is False

    def test_token_stats(self, tmp_path):
        """model_stats 应正确计算 total_tokens."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        traj = SWEAgentAdapter().parse(str(path))
        assert traj.outcome.total_tokens == 7000  # 5000 + 2000

    def test_thought_preserved(self, tmp_path):
        """action 中的 thought 应保留."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        traj = SWEAgentAdapter().parse(str(path))
        assert traj.steps[0].thought == "找到模型文件"

    def test_metadata_extraction(self, tmp_path):
        """元数据应从 info 中正确提取."""
        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        traj = SWEAgentAdapter().parse(str(path))
        assert traj.metadata["instance_id"] == "django__django-12345"
        assert traj.metadata["exit_status"] == "submitted"
        assert traj.metadata["environment"] == "swe_main"
        assert traj.metadata["model_stats"]["tokens_sent"] == 5000

    def test_empty_history(self, tmp_path):
        """空 history 应返回空步骤."""
        path = tmp_path / "traj.json"
        data = _make_sweagent_data()
        data["history"] = []
        _write_sweagent_log(path, data)

        traj = SWEAgentAdapter().parse(str(path))
        assert len(traj.steps) == 0

    def test_string_action(self, tmp_path):
        """纯字符串 action 应作为 bash 命令处理."""
        path = tmp_path / "traj.json"
        data = _make_sweagent_data()
        data["history"] = [["ls -la", "file1\nfile2\n"]]
        _write_sweagent_log(path, data)

        traj = SWEAgentAdapter().parse(str(path))
        assert len(traj.steps) == 1
        assert traj.steps[0].tool_call.name == "bash"
        assert traj.steps[0].tool_call.parameters == {"command": "ls -la"}

    def test_skip_think_actions(self, tmp_path):
        """think action 应被跳过."""
        path = tmp_path / "traj.json"
        data = _make_sweagent_data()
        data["history"] = [
            [{"action": "think", "output": "让我想想..."}, ""],
            [{"action": "bash", "command": "ls"}, "file.py\n"],
        ]
        _write_sweagent_log(path, data)

        traj = SWEAgentAdapter().parse(str(path))
        assert len(traj.steps) == 1
        assert traj.steps[0].tool_call.name == "bash"


# ── Recorder + Adapter 集成 ────────────────────────────────────────


class TestRecorderWithAdapters:
    """验证 Recorder 与适配器的集成."""

    def test_recorder_convert_openhands(self, tmp_path):
        """Recorder.convert 应使用 OpenHands 适配器."""
        from agentrecorder import Recorder

        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        recorder = Recorder(OpenHandsAdapter())
        traj = recorder.convert(str(path))
        assert len(traj.steps) == 3
        assert traj.agent == "openhands/CodeActAgent"

    def test_recorder_convert_sweagent(self, tmp_path):
        """Recorder.convert 应使用 SWE-agent 适配器."""
        from agentrecorder import Recorder

        path = tmp_path / "traj.json"
        _write_sweagent_log(path, _make_sweagent_data())

        recorder = Recorder(SWEAgentAdapter())
        traj = recorder.convert(str(path))
        assert len(traj.steps) == 4
        assert traj.agent == "swe-agent"

    def test_recorder_batch_convert(self, tmp_path):
        """Recorder.convert_batch 应批量转换."""
        from agentrecorder import Recorder

        for i in range(3):
            path = tmp_path / f"traj_{i}.json"
            _write_sweagent_log(path, _make_sweagent_data())

        recorder = Recorder(SWEAgentAdapter())
        trajectories = recorder.convert_batch(str(tmp_path), "*.json")
        assert len(trajectories) == 3

    def test_recorder_invalid_format(self, tmp_path):
        """格式不匹配应抛出 ValueError."""
        from agentrecorder import Recorder

        path = tmp_path / "bad.jsonl"
        path.write_text('{"type": "not_openhands"}\n')

        recorder = Recorder(OpenHandsAdapter())
        with pytest.raises(ValueError, match="不匹配"):
            recorder.convert(str(path))

    def test_recorder_file_not_found(self):
        """文件不存在应抛出 FileNotFoundError."""
        from agentrecorder import Recorder

        recorder = Recorder(OpenHandsAdapter())
        with pytest.raises(FileNotFoundError):
            recorder.convert("/nonexistent/log.jsonl")

    def test_trajectory_roundtrip(self, tmp_path):
        """解析后的 Trajectory 应可序列化/反序列化."""
        from agentrecorder.schema import Trajectory

        path = tmp_path / "log.jsonl"
        _write_openhands_log(path, _make_openhands_events())

        traj = OpenHandsAdapter().parse(str(path))

        # 序列化
        out_path = tmp_path / "output.jsonl"
        traj.to_jsonl(str(out_path))

        # 反序列化
        loaded = Trajectory.from_jsonl(str(out_path))
        assert len(loaded) == 1
        assert loaded[0].task.task_id == traj.task.task_id
        assert len(loaded[0].steps) == len(traj.steps)
