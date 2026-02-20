"""CAS (Content-Addressable Store) — 轨迹内容寻址存储.

SHA-256 哈希做轨迹 ID，SQLite 做索引。
- 去重: 相同操作序列只存一份
- 防篡改: content_hash 由步骤内容决定
- 轻量: SQLite 零外部依赖
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# content_hash 取 SHA-256 前 16 位 (64 bits, 碰撞概率极低)
HASH_LENGTH = 16


def content_hash(steps: list[dict[str, Any]]) -> str:
    """对轨迹步骤做确定性序列化后取 SHA-256 前 16 位.

    只取 tool + params + output，不含 thought（思考链每次不同）和时间戳，
    这样相同操作序列 = 相同哈希 = 去重。

    Args:
        steps: 轨迹步骤列表.

    Returns:
        16 字符的十六进制哈希.
    """
    canonical = json.dumps(
        [
            {
                "tool": s.get("tool", ""),
                "params": s.get("params", {}),
                "output": s.get("output", ""),
            }
            for s in steps
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_LENGTH]


_SCHEMA = """
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
    data         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_id ON trajectories(task_id);
CREATE INDEX IF NOT EXISTS idx_reward ON trajectories(reward DESC);
CREATE INDEX IF NOT EXISTS idx_gdi ON trajectories(gdi_score DESC);
"""


class CAStore:
    """SQLite-backed 内容寻址轨迹存储.

    Usage::

        store = CAStore(Path("data/index.sqlite"))
        h = store.put(trajectory)          # 存储，返回 content_hash
        traj = store.get(h)                # 检索
        store.update_gdi(h, 0.72)          # 更新 GDI
        store.increment_export(h)          # 引用计数 +1
        results = store.list(limit=20)     # 按 GDI 排名
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """关闭数据库连接."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def put(self, trajectory) -> str:
        """存储轨迹，返回 content_hash.

        已存在则跳过数据写入，只更新 reward（取较高值）。

        Args:
            trajectory: hub Trajectory dataclass.

        Returns:
            content_hash (16 字符十六进制).
        """
        h = content_hash(trajectory.steps)
        now = time.time()

        existing = self._conn.execute(
            "SELECT reward FROM trajectories WHERE content_hash = ?", (h,)
        ).fetchone()

        if existing is not None:
            # 已存在，更新 reward（取较高值）
            if trajectory.reward > existing["reward"]:
                self._conn.execute(
                    "UPDATE trajectories SET reward = ? WHERE content_hash = ?",
                    (trajectory.reward, h),
                )
                self._conn.commit()
            logger.debug("轨迹已存在，跳过: %s", h)
            return h

        data = json.dumps(
            {
                "steps": trajectory.steps,
                "step_rewards": getattr(trajectory, "step_rewards", []),
                "duration_seconds": getattr(trajectory, "duration_seconds", 0.0),
                "metadata": getattr(trajectory, "metadata", {}),
            },
            ensure_ascii=False,
        )

        self._conn.execute(
            """INSERT INTO trajectories
               (content_hash, task_id, agent_framework, agent_model,
                total_steps, success, reward, created_at, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                h,
                trajectory.task_id,
                trajectory.agent_framework,
                trajectory.agent_model,
                trajectory.total_steps,
                1 if trajectory.success else 0,
                trajectory.reward,
                now,
                data,
            ),
        )
        self._conn.commit()
        logger.debug("轨迹已存储: %s (task=%s)", h, trajectory.task_id)
        return h

    def get(self, hash_id: str) -> dict[str, Any] | None:
        """按 content_hash 检索轨迹.

        Returns:
            包含所有字段的 dict，或 None.
        """
        row = self._conn.execute(
            "SELECT * FROM trajectories WHERE content_hash = ?", (hash_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list(
        self,
        *,
        task_id: str | None = None,
        order_by: str = "gdi_score",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """列出轨迹，支持过滤和排序.

        Args:
            task_id: 按任务 ID 过滤.
            order_by: 排序字段 (gdi_score / reward / created_at).
            limit: 返回条数.
            offset: 跳过条数.

        Returns:
            轨迹 dict 列表.
        """
        allowed_order = {"gdi_score", "reward", "created_at", "export_count"}
        if order_by not in allowed_order:
            order_by = "gdi_score"

        sql = "SELECT * FROM trajectories"
        params: list[Any] = []

        if task_id:
            sql += " WHERE task_id = ?"
            params.append(task_id)

        sql += f" ORDER BY {order_by} DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self, task_id: str | None = None) -> int:
        """返回轨迹总数."""
        if task_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE task_id = ?", (task_id,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()
        return row[0]

    def update_gdi(self, hash_id: str, gdi_score: float) -> None:
        """更新轨迹的 GDI 分数."""
        self._conn.execute(
            "UPDATE trajectories SET gdi_score = ? WHERE content_hash = ?",
            (gdi_score, hash_id),
        )
        self._conn.commit()

    def increment_export(self, hash_id: str) -> None:
        """引用计数 +1（被 SFT/DPO 导出时调用）."""
        self._conn.execute(
            "UPDATE trajectories SET export_count = export_count + 1 WHERE content_hash = ?",
            (hash_id,),
        )
        self._conn.commit()

    def update_gdi_batch(self, scores: dict[str, float]) -> None:
        """批量更新 GDI 分数.

        Args:
            scores: {content_hash: gdi_score} 映射.
        """
        with self._conn:
            self._conn.executemany(
                "UPDATE trajectories SET gdi_score = ? WHERE content_hash = ?",
                [(score, h) for h, score in scores.items()],
            )

    def stats(self) -> dict[str, Any]:
        """返回存储统计信息."""
        total = self.count()
        row = self._conn.execute(
            """SELECT
                AVG(reward) as avg_reward,
                AVG(gdi_score) as avg_gdi,
                SUM(export_count) as total_exports,
                COUNT(DISTINCT task_id) as unique_tasks
               FROM trajectories"""
        ).fetchone()
        return {
            "total_trajectories": total,
            "unique_tasks": row["unique_tasks"] or 0,
            "avg_reward": round(row["avg_reward"] or 0, 4),
            "avg_gdi": round(row["avg_gdi"] or 0, 4),
            "total_exports": row["total_exports"] or 0,
        }

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """将 sqlite3.Row 转为 dict，解析 data JSON."""
        d = dict(row)
        d["success"] = bool(d["success"])
        if "data" in d:
            try:
                extra = json.loads(d.pop("data"))
                d.update(extra)
            except (json.JSONDecodeError, TypeError):
                pass
        return d
