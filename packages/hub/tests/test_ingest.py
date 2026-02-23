"""CrewIngestor 增量拉取测试."""

import json

import pytest

from trajectoryhub.cas import CAStore
from trajectoryhub.ingest import CrewIngestor


def _make_crew_trajectory(employee="backend-engineer", task_id="t1", output="done"):
    """构造一条 crew 格式轨迹."""
    return {
        "task": {"task_id": task_id, "description": "test task", "domain": "crew"},
        "agent": f"crew/{employee}",
        "model": "claude-sonnet-4-6",
        "steps": [
            {
                "step_id": 1,
                "thought": "分析需求",
                "tool_call": {"name": "Read", "parameters": {"file_path": "/a.py"}},
                "tool_result": {"output": output, "exit_code": 0},
            }
        ],
        "outcome": {"success": True, "total_steps": 1, "total_tokens": 100},
        "metadata": {
            "employee": employee,
            "channel": "claude-code",
            "source_session": "abc-123",
        },
    }


def _write_jsonl(path, trajectories):
    """写入 JSONL 文件."""
    with open(path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


class TestCrewIngestor:
    """CrewIngestor 冒烟测试."""

    @pytest.fixture()
    def store(self, tmp_path):
        s = CAStore(tmp_path / "data" / "cas.sqlite")
        yield s
        s.close()

    def test_ingest_basic(self, store, tmp_path):
        """基本拉取：一条轨迹入库."""
        source = tmp_path / "trajectories.jsonl"
        _write_jsonl(source, [_make_crew_trajectory()])

        ingestor = CrewIngestor(store)
        result = ingestor.ingest(source)

        assert result.ingested == 1
        assert result.errors == 0
        assert store.count() == 1

        # 验证元数据
        trajs = store.list(limit=1)
        assert len(trajs) == 1
        t = trajs[0]
        assert t["employee"] == "backend-engineer"
        assert t["source"] == "claude-code"
        assert t["domain"] == "crew"
        assert t["agent_framework"] == "crew"

    def test_ingest_incremental(self, store, tmp_path):
        """增量拉取：第二次只拉新数据."""
        source = tmp_path / "trajectories.jsonl"

        # 第一次：1 条
        _write_jsonl(source, [_make_crew_trajectory(output="v1")])
        ingestor = CrewIngestor(store)
        r1 = ingestor.ingest(source)
        assert r1.ingested == 1

        # 追加 1 条
        with open(source, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(_make_crew_trajectory(output="v2"), ensure_ascii=False)
                + "\n"
            )

        # 第二次：只拉新增
        r2 = ingestor.ingest(source)
        assert r2.ingested == 1  # 只有新追加的 1 条
        assert store.count() == 2

    def test_ingest_empty_file(self, store, tmp_path):
        """空文件不报错."""
        source = tmp_path / "empty.jsonl"
        source.write_text("")

        ingestor = CrewIngestor(store)
        result = ingestor.ingest(source)
        assert result.ingested == 0
        assert result.errors == 0

    def test_ingest_bad_json(self, store, tmp_path):
        """损坏的 JSON 行计入 errors."""
        source = tmp_path / "bad.jsonl"
        with open(source, "w") as f:
            f.write('{"valid": this is broken}\n')
            f.write(json.dumps(_make_crew_trajectory()) + "\n")

        ingestor = CrewIngestor(store)
        result = ingestor.ingest(source)
        assert result.errors == 1
        assert result.ingested == 1

    def test_ingest_nonexistent_file(self, store):
        """不存在的文件返回空结果."""
        ingestor = CrewIngestor(store)
        result = ingestor.ingest("/nonexistent/path.jsonl")
        assert result.ingested == 0
        assert result.errors == 0

    def test_ingest_file_truncated(self, store, tmp_path):
        """文件缩小时（截断/重建），从头拉取."""
        source = tmp_path / "trajectories.jsonl"

        # 第一次：写多条
        _write_jsonl(
            source,
            [_make_crew_trajectory(output=f"v{i}") for i in range(5)],
        )
        ingestor = CrewIngestor(store)
        r1 = ingestor.ingest(source)
        assert r1.ingested == 5

        # 文件被截断重建（更小了）
        _write_jsonl(source, [_make_crew_trajectory(output="new")])
        r2 = ingestor.ingest(source)
        # 由于文件变小，应从头开始
        assert r2.ingested == 1

    def test_ingest_multiple_employees(self, store, tmp_path):
        """多个员工的轨迹都能正确入库."""
        source = tmp_path / "trajectories.jsonl"
        _write_jsonl(
            source,
            [
                _make_crew_trajectory(employee="backend-engineer", output="be"),
                _make_crew_trajectory(employee="frontend-engineer", output="fe"),
                _make_crew_trajectory(employee="ceo-assistant", output="ceo"),
            ],
        )

        ingestor = CrewIngestor(store)
        result = ingestor.ingest(source)
        assert result.ingested == 3

        # 按 employee 确认
        trajs = store.list(limit=10, order_by="created_at")
        employees = {t["employee"] for t in trajs}
        assert "backend-engineer" in employees
        assert "frontend-engineer" in employees
        assert "ceo-assistant" in employees
