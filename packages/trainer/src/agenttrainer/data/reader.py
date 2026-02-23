"""数据读取 - 读取 hub exporter 导出的 JSONL 格式.

Phase 3 新增：read_from_cas() 直接从 CAS SQLite 读取训练数据，
跳过 JSONL 导出步骤。trainer 包不依赖 hub 包，使用 sqlite3 直连。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_sft(path: str | Path) -> list[dict[str, Any]]:
    """读取 SFT JSONL 文件.

    兼容 hub ``export_sft`` 导出的格式，每行字段:
    instruction, input, response, task_id, reward, metadata
    """
    records: list[dict[str, Any]] = []
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError:
                logger.warning("第 %d 行 JSON 解析失败，跳过", i)
    logger.info("读取 SFT 数据: %d 条 <- %s", len(records), path)
    return records


def read_dpo(path: str | Path) -> list[dict[str, Any]]:
    """读取 DPO JSONL 文件.

    兼容 hub ``export_dpo`` 导出的格式，每行字段:
    prompt, chosen, rejected, task_id, reward_margin, metadata
    """
    records: list[dict[str, Any]] = []
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError:
                logger.warning("第 %d 行 JSON 解析失败，跳过", i)
    logger.info("读取 DPO 数据: %d 条 <- %s", len(records), path)
    return records


def read_grpo_groups(path: str | Path) -> list[dict[str, Any]]:
    """读取 GRPO 分组 JSONL 文件.

    兼容 hub ``export_grpo`` 导出的格式，每行字段:
    task_id, prompt, trajectories (list of {response, reward})

    Agent 增强格式的 trajectories 还支持:
    trajectories[].steps (list of {thought, action, observation, reward})
    """
    records: list[dict[str, Any]] = []
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError:
                logger.warning("第 %d 行 JSON 解析失败，跳过", i)
    logger.info("读取 GRPO 数据: %d 组 <- %s", len(records), path)
    return records


# ------------------------------------------------------------------
# Phase 3: CAS 直读
# ------------------------------------------------------------------


def _steps_to_text(steps: list[dict[str, Any]]) -> str:
    """将步骤列表转为文本格式（简化版，参考 hub exporter._steps_to_text）.

    # NOTE: 此函数与 hub/exporter.py 的同名函数逻辑同步。
    # 修改任一处时请同步另一处，或考虑迁移至 knowlyr-core 包。

    兼容多种字段约定:
    - 标准: action / observation
    - Hub 内部: tool + params / output
    - Crew 轨迹: tool_call / tool_result
    """
    parts = []
    for i, step in enumerate(steps, 1):
        thought = step.get("thought", "")

        # --- action ---
        action = step.get("action", "")
        if not action:
            tool = step.get("tool", "")
            if tool:
                params = step.get("params")
                action = (
                    f"{tool}({json.dumps(params, ensure_ascii=False)})"
                    if params
                    else tool
                )
        if not action:
            tc = step.get("tool_call")
            if isinstance(tc, dict):
                name = tc.get("name", "")
                params = tc.get("parameters")
                action = (
                    f"{name}({json.dumps(params, ensure_ascii=False)})"
                    if params
                    else name
                )

        # --- observation ---
        observation = step.get("observation", "")
        if not observation:
            observation = step.get("output", "")
        if not observation:
            tr = step.get("tool_result")
            if isinstance(tr, dict):
                observation = tr.get("output", "")

        step_text = f"Step {i}:"
        if thought:
            step_text += f"\nThought: {thought}"
        if action:
            step_text += f"\nAction: {action}"
        if observation:
            step_text += f"\nObservation: {observation}"
        parts.append(step_text)

    return "\n\n".join(parts) if parts else ""


def read_from_cas(
    db_path: str,
    format: str = "sft",
    min_reward: float = 0.0,
    employee: str | None = None,
    domain: str | None = None,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """直接从 CAS SQLite 读取训练数据.

    跳过 JSONL 导出步骤，直接从 CAS 读取并转换为训练格式。
    使用 sqlite3 直连，**不依赖 hub 包**。

    Args:
        db_path: CAS SQLite 路径
        format: "sft" | "dpo" | "grpo"
        min_reward: 最低 reward 阈值
        employee: 按员工筛选（可选）
        domain: 按领域筛选（可选）
        limit: 最多读取条数

    Returns:
        训练格式的 dict 列表（与 read_sft/read_dpo 输出格式一致）
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        rows = _query_cas(conn, min_reward=min_reward, employee=employee, domain=domain, limit=limit)

        if format == "sft":
            return _cas_to_sft(rows)
        elif format == "dpo":
            return _cas_to_dpo(rows)
        elif format == "grpo":
            return _cas_to_grpo(rows)
        else:
            raise ValueError(f"不支持的格式: {format}")
    finally:
        conn.close()


def _query_cas(
    conn: sqlite3.Connection,
    *,
    min_reward: float = 0.0,
    employee: str | None = None,
    domain: str | None = None,
    limit: int = 10000,
) -> list[sqlite3.Row]:
    """从 CAS 查询轨迹行."""
    sql = "SELECT * FROM trajectories WHERE reward >= ?"
    params: list[Any] = [min_reward]

    if employee:
        sql += " AND employee = ?"
        params.append(employee)
    if domain:
        sql += " AND domain = ?"
        params.append(domain)

    sql += " ORDER BY reward DESC LIMIT ?"
    params.append(limit)

    return conn.execute(sql, params).fetchall()


def _parse_row(row: sqlite3.Row) -> dict[str, Any]:
    """将 CAS 行解析为 dict，展开 data JSON."""
    d = dict(row)
    d["success"] = bool(d.get("success", 0))
    if "data" in d:
        try:
            extra = json.loads(d.pop("data"))
            d.update(extra)
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _cas_to_sft(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """CAS 行 -> SFT 训练格式."""
    records = []
    for row in rows:
        d = _parse_row(row)
        if not d.get("success"):
            continue

        steps = d.get("steps", [])
        meta = d.get("metadata", {})
        response_text = _steps_to_text(steps)

        input_data = {}
        for key in ("repo", "base_commit", "test_command"):
            if meta.get(key):
                input_data[key] = meta[key]

        records.append({
            "instruction": meta.get(
                "task_description", f"Solve task: {d.get('task_id', '')}"
            ),
            "input": json.dumps(input_data, ensure_ascii=False) if input_data else "",
            "response": response_text,
            "task_id": d.get("task_id", ""),
            "reward": d.get("reward", 0.0),
            "metadata": {
                "agent_framework": d.get("agent_framework", ""),
                "agent_model": d.get("agent_model", ""),
                "total_steps": d.get("total_steps", 0),
                "employee": d.get("employee", ""),
                "domain": d.get("domain", ""),
            },
        })
    logger.info("CAS -> SFT: %d 条记录", len(records))
    return records


def _cas_to_dpo(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """CAS 行 -> DPO 训练格式.

    同 task_id 的轨迹两两配对（高 reward = chosen，低 reward = rejected）。
    """
    # 按 task_id 分组
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d = _parse_row(row)
        tid = d.get("task_id", "")
        if tid:
            task_groups[tid].append(d)

    records = []
    for task_id, group in task_groups.items():
        if len(group) < 2:
            continue

        # 按 reward 降序
        sorted_group = sorted(group, key=lambda x: x.get("reward", 0.0), reverse=True)

        # 两两配对：相邻的高/低
        for i in range(len(sorted_group) - 1):
            chosen = sorted_group[i]
            rejected = sorted_group[i + 1]

            chosen_meta = chosen.get("metadata", {})

            records.append({
                "prompt": chosen_meta.get(
                    "task_description",
                    f"Solve the following task:\n\nTask ID: {task_id}",
                ),
                "chosen": _steps_to_text(chosen.get("steps", [])),
                "rejected": _steps_to_text(rejected.get("steps", [])),
                "task_id": task_id,
                "reward_margin": round(
                    chosen.get("reward", 0.0) - rejected.get("reward", 0.0), 4
                ),
                "metadata": {
                    "chosen_model": chosen.get("agent_model", ""),
                    "rejected_model": rejected.get("agent_model", ""),
                    "chosen_reward": chosen.get("reward", 0.0),
                    "rejected_reward": rejected.get("reward", 0.0),
                },
            })
    logger.info("CAS -> DPO: %d 条记录", len(records))
    return records


def _cas_to_grpo(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """CAS 行 -> GRPO 分组格式."""
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d = _parse_row(row)
        tid = d.get("task_id", "")
        if tid:
            task_groups[tid].append(d)

    records = []
    for task_id, group in task_groups.items():
        if len(group) < 2:
            continue

        sorted_group = sorted(group, key=lambda x: x.get("reward", 0.0), reverse=True)
        meta = sorted_group[0].get("metadata", {})

        records.append({
            "task_id": task_id,
            "prompt": meta.get("task_description", f"Solve task: {task_id}"),
            "trajectories": [
                {
                    "response": _steps_to_text(t.get("steps", [])),
                    "reward": t.get("reward", 0.0),
                }
                for t in sorted_group
            ],
        })
    logger.info("CAS -> GRPO: %d 组", len(records))
    return records
