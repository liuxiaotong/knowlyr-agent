"""数据读取 - 读取 hub exporter 导出的 JSONL 格式."""

from __future__ import annotations

import json
import logging
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
