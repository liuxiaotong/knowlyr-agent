"""CrewIngestor — 从 Crew trajectories.jsonl 增量拉取轨迹入 CAS.

从 crew 的 trajectories.jsonl 文件读取轨迹数据，转换为 hub Trajectory 格式后
存入 CAS。支持增量拉取（记录文件 offset，不重复导入）。

用法::

    from trajectoryhub.ingest import CrewIngestor

    ingestor = CrewIngestor(cas_store)
    result = ingestor.ingest("~/.crew/trajectories/trajectories.jsonl")
    print(f"新增 {result.ingested} 条，跳过 {result.skipped} 条")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajectoryhub.cas import CAStore
from trajectoryhub.pipeline import Trajectory

logger = logging.getLogger(__name__)

# 进度文件名（放在 CAS 数据库同目录）
_CURSOR_FILENAME = ".ingest_cursor.json"


@dataclass
class IngestResult:
    """拉取结果统计."""

    source_path: str
    ingested: int = 0
    skipped: int = 0
    errors: int = 0
    last_offset: int = 0


class CrewIngestor:
    """从 Crew trajectories.jsonl 增量拉取轨迹入 CAS.

    增量策略：记录文件字节 offset，每次从上次位置继续读。
    """

    def __init__(self, store: CAStore) -> None:
        self.store = store
        self._cursor_path = store.db_path.parent / _CURSOR_FILENAME

    def ingest(self, source_path: str | Path) -> IngestResult:
        """从 JSONL 文件增量拉取轨迹.

        Args:
            source_path: crew trajectories.jsonl 文件路径.

        Returns:
            IngestResult 包含统计信息.
        """
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            logger.error("源文件不存在: %s", source)
            return IngestResult(source_path=str(source))

        cursor = self._load_cursor()
        source_key = str(source)
        last_offset = cursor.get(source_key, 0)

        file_size = source.stat().st_size
        if last_offset > file_size:
            # 文件被截断/重建了，从头开始
            logger.warning("文件大小(%d) < 上次 offset(%d)，从头拉取", file_size, last_offset)
            last_offset = 0

        result = IngestResult(source_path=str(source), last_offset=last_offset)

        with open(source, "r", encoding="utf-8") as f:
            f.seek(last_offset)
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("JSON 解析错误 offset=%d: %s", f.tell(), e)
                    result.errors += 1
                    continue

                try:
                    trajectory = self._convert(raw)
                    self.store.put(trajectory)
                    result.ingested += 1
                except Exception as e:
                    logger.warning("转换/存储失败: %s", e)
                    result.errors += 1

            new_offset = f.tell()

        # 保存游标
        result.last_offset = new_offset
        cursor[source_key] = new_offset
        self._save_cursor(cursor)

        logger.info(
            "拉取完成: source=%s, ingested=%d, skipped=%d, errors=%d",
            source.name,
            result.ingested,
            result.skipped,
            result.errors,
        )
        return result

    def _convert(self, raw: dict[str, Any]) -> Trajectory:
        """将 crew 轨迹 dict 转换为 hub Trajectory dataclass.

        crew 轨迹格式::

            {
                "task": {"task_id": "...", "description": "...", "domain": "crew"},
                "agent": "crew/backend-engineer",
                "model": "claude-sonnet-4-6",
                "steps": [...],
                "outcome": {"success": true, "total_steps": 5, ...},
                "metadata": {"employee": "backend-engineer", "channel": "claude-code", ...}
            }
        """
        task = raw.get("task", {})
        outcome = raw.get("outcome", {})
        metadata = raw.get("metadata", {})
        steps = raw.get("steps", [])

        # 转换 steps 为 CAS 兼容格式
        # crew 格式: tool_call.name / tool_call.parameters / tool_result.output
        # CAS content_hash 需要: tool / params / output
        cas_steps = []
        for s in steps:
            tool_call = s.get("tool_call", {})
            tool_result = s.get("tool_result", {})
            cas_step = {
                "tool": tool_call.get("name", s.get("tool_name", "")),
                "params": tool_call.get("parameters", s.get("tool_params", {})),
                "output": tool_result.get("output", s.get("tool_output", "")),
                "thought": s.get("thought", ""),
            }
            cas_steps.append(cas_step)

        # 从 agent 字段提取 framework
        agent = raw.get("agent", "")
        framework = agent.split("/")[0] if "/" in agent else "crew"

        # 提取 employee/source/domain 到 metadata
        employee = metadata.get("employee", "")
        if not employee and "/" in agent:
            employee = agent.split("/", 1)[1]

        source = metadata.get("channel", metadata.get("source", "crew"))
        domain = task.get("domain", metadata.get("domain", "crew"))

        enriched_metadata = {
            **metadata,
            "employee": employee,
            "source": source,
            "domain": domain,
        }

        return Trajectory(
            task_id=task.get("task_id", ""),
            agent_framework=framework,
            agent_model=raw.get("model", ""),
            steps=cas_steps,
            total_steps=outcome.get("total_steps", len(cas_steps)),
            success=outcome.get("success", False),
            reward=outcome.get("reward", 0.0),
            duration_seconds=outcome.get("duration_seconds", 0.0),
            metadata=enriched_metadata,
        )

    def _load_cursor(self) -> dict[str, int]:
        """加载增量游标."""
        if self._cursor_path.exists():
            try:
                with open(self._cursor_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cursor(self, cursor: dict[str, int]) -> None:
        """保存增量游标."""
        with open(self._cursor_path, "w", encoding="utf-8") as f:
            json.dump(cursor, f, indent=2)
