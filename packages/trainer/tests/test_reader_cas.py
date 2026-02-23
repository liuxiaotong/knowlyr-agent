"""测试 read_from_cas - CAS 直读训练数据 (Phase 3)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agenttrainer.data.reader import read_from_cas, _steps_to_text


def _create_cas_db(db_path: Path, trajectories: list[dict]) -> str:
    """创建一个模拟 CAS SQLite 数据库并写入测试数据."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trajectories (
            content_hash TEXT PRIMARY KEY,
            task_id      TEXT NOT NULL,
            agent_framework TEXT DEFAULT '',
            agent_model  TEXT DEFAULT '',
            total_steps  INTEGER DEFAULT 0,
            success      INTEGER DEFAULT 0,
            reward       REAL DEFAULT 0.0,
            gdi_score    REAL DEFAULT 0.0,
            export_count INTEGER DEFAULT 0,
            created_at   REAL NOT NULL,
            data         TEXT NOT NULL,
            employee     TEXT DEFAULT '',
            source       TEXT DEFAULT '',
            domain       TEXT DEFAULT ''
        );
    """)

    for i, t in enumerate(trajectories):
        conn.execute(
            """INSERT INTO trajectories
               (content_hash, task_id, agent_framework, agent_model,
                total_steps, success, reward, gdi_score, created_at, data,
                employee, source, domain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                t.get("content_hash", f"hash_{i:04d}"),
                t.get("task_id", f"task-{i}"),
                t.get("agent_framework", "crew"),
                t.get("agent_model", "test-model"),
                t.get("total_steps", len(t.get("steps", []))),
                1 if t.get("success", True) else 0,
                t.get("reward", 0.5),
                t.get("gdi_score", 0.5),
                t.get("created_at", 1000000.0 + i),
                json.dumps({
                    "steps": t.get("steps", []),
                    "metadata": t.get("metadata", {}),
                }),
                t.get("employee", ""),
                t.get("source", ""),
                t.get("domain", ""),
            ),
        )
    conn.commit()
    conn.close()
    return str(db_path)


class TestStepsToText:
    """_steps_to_text 辅助函数测试."""

    def test_standard_format(self):
        steps = [
            {"thought": "Read file", "action": "file_read /a.py", "observation": "content"},
        ]
        text = _steps_to_text(steps)
        assert "Step 1:" in text
        assert "Read file" in text
        assert "file_read /a.py" in text
        assert "content" in text

    def test_hub_format(self):
        steps = [
            {"tool": "Bash", "params": {"command": "ls"}, "output": "files"},
        ]
        text = _steps_to_text(steps)
        assert "Bash" in text
        assert "files" in text

    def test_crew_format(self):
        steps = [
            {
                "tool_call": {"name": "Read", "parameters": {"path": "/x"}},
                "tool_result": {"output": "data"},
            },
        ]
        text = _steps_to_text(steps)
        assert "Read" in text
        assert "data" in text

    def test_empty_steps(self):
        assert _steps_to_text([]) == ""


class TestReadFromCasSFT:
    """read_from_cas(format='sft') 测试."""

    def test_basic_sft(self, tmp_path):
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {
                    "task_id": "t1",
                    "success": True,
                    "reward": 0.8,
                    "steps": [{"action": "read /a.py", "observation": "content"}],
                    "metadata": {"task_description": "Fix bug"},
                },
                {
                    "task_id": "t2",
                    "success": True,
                    "reward": 0.6,
                    "steps": [{"action": "write /b.py", "observation": "done"}],
                    "metadata": {"task_description": "Add feature"},
                },
            ],
        )
        records = read_from_cas(db_path, format="sft")
        assert len(records) == 2
        assert records[0]["task_id"] == "t1"  # higher reward first
        assert records[0]["reward"] == 0.8
        assert "instruction" in records[0]
        assert "response" in records[0]
        assert "Fix bug" in records[0]["instruction"]

    def test_sft_filters_failed(self, tmp_path):
        """SFT 只导出成功的轨迹."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"task_id": "t1", "success": True, "reward": 0.8,
                 "steps": [{"action": "a"}]},
                {"task_id": "t2", "success": False, "reward": 0.3,
                 "steps": [{"action": "b"}]},
            ],
        )
        records = read_from_cas(db_path, format="sft")
        assert len(records) == 1
        assert records[0]["task_id"] == "t1"

    def test_min_reward_filter(self, tmp_path):
        """min_reward 应过滤低分轨迹."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"task_id": "t1", "success": True, "reward": 0.9,
                 "steps": [{"action": "a"}]},
                {"task_id": "t2", "success": True, "reward": 0.4,
                 "steps": [{"action": "b"}]},
                {"task_id": "t3", "success": True, "reward": 0.2,
                 "steps": [{"action": "c"}]},
            ],
        )
        records = read_from_cas(db_path, format="sft", min_reward=0.5)
        assert len(records) == 1
        assert records[0]["task_id"] == "t1"

    def test_employee_filter(self, tmp_path):
        """employee 筛选."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"task_id": "t1", "success": True, "reward": 0.8,
                 "employee": "backend-engineer", "steps": [{"action": "a"}]},
                {"task_id": "t2", "success": True, "reward": 0.7,
                 "employee": "frontend-engineer", "steps": [{"action": "b"}]},
            ],
        )
        records = read_from_cas(db_path, format="sft", employee="backend-engineer")
        assert len(records) == 1
        assert records[0]["metadata"]["employee"] == "backend-engineer"

    def test_domain_filter(self, tmp_path):
        """domain 筛选."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"task_id": "t1", "success": True, "reward": 0.8,
                 "domain": "engineering", "steps": [{"action": "a"}]},
                {"task_id": "t2", "success": True, "reward": 0.7,
                 "domain": "support", "steps": [{"action": "b"}]},
            ],
        )
        records = read_from_cas(db_path, format="sft", domain="engineering")
        assert len(records) == 1

    def test_empty_db(self, tmp_path):
        """空数据库应返回空列表."""
        db_path = _create_cas_db(tmp_path / "cas.sqlite", [])
        records = read_from_cas(db_path, format="sft")
        assert records == []

    def test_limit(self, tmp_path):
        """limit 应限制返回条数."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"task_id": f"t{i}", "success": True, "reward": 0.5 + i * 0.01,
                 "steps": [{"action": f"a{i}"}]}
                for i in range(10)
            ],
        )
        records = read_from_cas(db_path, format="sft", limit=3)
        assert len(records) == 3


class TestReadFromCasDPO:
    """read_from_cas(format='dpo') 测试."""

    def test_basic_dpo(self, tmp_path):
        """同 task_id 的轨迹应配对."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {
                    "content_hash": "h1", "task_id": "t1",
                    "success": True, "reward": 0.9,
                    "steps": [{"action": "good_action"}],
                    "metadata": {"task_description": "Fix bug"},
                },
                {
                    "content_hash": "h2", "task_id": "t1",
                    "success": True, "reward": 0.3,
                    "steps": [{"action": "bad_action"}],
                    "metadata": {"task_description": "Fix bug"},
                },
            ],
        )
        records = read_from_cas(db_path, format="dpo")
        assert len(records) == 1  # 1 pair
        assert records[0]["task_id"] == "t1"
        assert records[0]["reward_margin"] == pytest.approx(0.6, abs=0.01)
        assert "chosen" in records[0]
        assert "rejected" in records[0]
        assert "good_action" in records[0]["chosen"]
        assert "bad_action" in records[0]["rejected"]

    def test_dpo_needs_multiple_trajectories(self, tmp_path):
        """只有 1 条轨迹的任务不应生成 DPO 对."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"task_id": "t1", "success": True, "reward": 0.9,
                 "steps": [{"action": "a"}]},
            ],
        )
        records = read_from_cas(db_path, format="dpo")
        assert len(records) == 0

    def test_dpo_multiple_pairs(self, tmp_path):
        """3 条同 task 轨迹应生成 2 个 DPO 对（相邻配对）."""
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"content_hash": "h1", "task_id": "t1", "success": True, "reward": 0.9,
                 "steps": [{"action": "a"}]},
                {"content_hash": "h2", "task_id": "t1", "success": True, "reward": 0.6,
                 "steps": [{"action": "b"}]},
                {"content_hash": "h3", "task_id": "t1", "success": True, "reward": 0.3,
                 "steps": [{"action": "c"}]},
            ],
        )
        records = read_from_cas(db_path, format="dpo")
        assert len(records) == 2
        # 第一对: 0.9 vs 0.6
        assert records[0]["reward_margin"] == pytest.approx(0.3, abs=0.01)
        # 第二对: 0.6 vs 0.3
        assert records[1]["reward_margin"] == pytest.approx(0.3, abs=0.01)


class TestReadFromCasGRPO:
    """read_from_cas(format='grpo') 测试."""

    def test_basic_grpo(self, tmp_path):
        db_path = _create_cas_db(
            tmp_path / "cas.sqlite",
            [
                {"content_hash": "h1", "task_id": "t1", "reward": 0.9,
                 "steps": [{"action": "a"}],
                 "metadata": {"task_description": "Fix bug"}},
                {"content_hash": "h2", "task_id": "t1", "reward": 0.5,
                 "steps": [{"action": "b"}]},
                {"content_hash": "h3", "task_id": "t2", "reward": 0.7,
                 "steps": [{"action": "c"}]},
            ],
        )
        records = read_from_cas(db_path, format="grpo")
        assert len(records) == 1  # 只有 t1 有 >=2 条
        assert records[0]["task_id"] == "t1"
        assert len(records[0]["trajectories"]) == 2

    def test_invalid_format(self, tmp_path):
        db_path = _create_cas_db(tmp_path / "cas.sqlite", [])
        with pytest.raises(ValueError, match="不支持的格式"):
            read_from_cas(db_path, format="unknown")
