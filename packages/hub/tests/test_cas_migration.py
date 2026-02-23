"""CAS 表扩展和迁移测试."""

import sqlite3

import pytest

from trajectoryhub.cas import CAStore
from trajectoryhub.pipeline import Trajectory


class TestCASMigration:
    """测试 Phase 1 新增的 employee/source/domain 列."""

    @pytest.fixture()
    def store(self, tmp_path):
        s = CAStore(tmp_path / "test.sqlite")
        yield s
        s.close()

    def _make_traj(self, output="code", employee="", source="", domain=""):
        metadata = {}
        if employee:
            metadata["employee"] = employee
        if source:
            metadata["source"] = source
        if domain:
            metadata["domain"] = domain
        return Trajectory(
            task_id="t1",
            agent_framework="crew",
            agent_model="claude-sonnet-4-6",
            steps=[{"tool": "read", "params": {"path": "/a.py"}, "output": output}],
            total_steps=1,
            success=True,
            reward=0.8,
            metadata=metadata,
        )

    def test_new_columns_exist(self, store):
        """新建数据库应有 employee/source/domain 列."""
        cursor = store._conn.execute("PRAGMA table_info(trajectories)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "employee" in columns
        assert "source" in columns
        assert "domain" in columns

    def test_put_with_metadata(self, store):
        """put() 应将 metadata 中的 employee/source/domain 写入新列."""
        traj = self._make_traj(
            employee="backend-engineer",
            source="claude-code",
            domain="crew",
        )
        h = store.put(traj)
        result = store.get(h)
        assert result is not None
        assert result["employee"] == "backend-engineer"
        assert result["source"] == "claude-code"
        assert result["domain"] == "crew"

    def test_put_without_metadata(self, store):
        """不传 metadata 时新列应为空字符串（向后兼容）."""
        traj = Trajectory(
            task_id="t2",
            agent_framework="test",
            agent_model="gpt-4o",
            steps=[{"tool": "x", "params": {}, "output": "y"}],
            total_steps=1,
            success=True,
            reward=0.5,
        )
        h = store.put(traj)
        result = store.get(h)
        assert result["employee"] == ""
        assert result["source"] == ""
        assert result["domain"] == ""

    def test_migrate_existing_db(self, tmp_path):
        """模拟旧数据库（没有新列），打开时自动迁移."""
        db_path = tmp_path / "old.sqlite"

        # 1. 创建旧版表（不含新列）
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE trajectories (
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
                data         TEXT NOT NULL
            );
        """)
        # 插入一条旧数据
        import json
        import time

        conn.execute(
            """INSERT INTO trajectories
               (content_hash, task_id, created_at, data)
               VALUES (?, ?, ?, ?)""",
            ("abcd1234abcd1234", "old_task", time.time(), json.dumps({"steps": []})),
        )
        conn.commit()
        conn.close()

        # 2. 用 CAStore 打开旧库 → 自动迁移
        store = CAStore(db_path)
        cursor = store._conn.execute("PRAGMA table_info(trajectories)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "employee" in columns
        assert "source" in columns
        assert "domain" in columns

        # 3. 旧数据仍可读取
        row = store._conn.execute(
            "SELECT employee, source, domain FROM trajectories WHERE content_hash = ?",
            ("abcd1234abcd1234",),
        ).fetchone()
        assert row is not None
        assert row[0] == ""  # 默认值
        assert row[1] == ""
        assert row[2] == ""

        store.close()

    def test_indexes_created(self, store):
        """应创建 employee 和 domain 索引."""
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        assert "idx_employee" in indexes
        assert "idx_domain" in indexes
