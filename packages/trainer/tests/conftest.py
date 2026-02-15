"""pytest fixtures - 测试数据和小模型."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sft_sample_file(tmp_path: Path) -> Path:
    """生成 SFT JSONL 测试数据（模拟 hub exporter 输出）."""
    path = tmp_path / "sft_train.jsonl"
    records = [
        {
            "instruction": "Fix the off-by-one bug in sort function",
            "input": json.dumps({"repo": "owner/repo", "base_commit": "abc123"}),
            "response": (
                "Step 1:\nThought: Let me read the file\nAction: read_file /sort.py\n"
                "Observation: def sort(arr): ...\n\n"
                "Step 2:\nThought: Fix the bug\nAction: edit_file /sort.py\n"
                "Observation: File edited"
            ),
            "task_id": "task-001",
            "reward": 0.85,
            "metadata": {
                "agent_framework": "openhands",
                "agent_model": "test-model",
                "total_steps": 2,
            },
        },
        {
            "instruction": "Add input validation to API endpoint",
            "input": json.dumps({"repo": "owner/repo2", "base_commit": "def456"}),
            "response": (
                "Step 1:\nThought: Read the endpoint code\nAction: read_file /api.py\n"
                "Observation: def handler(): ..."
            ),
            "task_id": "task-002",
            "reward": 0.72,
            "metadata": {
                "agent_framework": "swe-agent",
                "agent_model": "test-model",
                "total_steps": 1,
            },
        },
        {
            "instruction": "Write unit tests for the parser module",
            "input": json.dumps({"repo": "owner/repo3"}),
            "response": (
                "Step 1:\nThought: Examine the parser\nAction: read_file /parser.py\n"
                "Observation: class Parser: ...\n\n"
                "Step 2:\nAction: edit_file /test_parser.py\nObservation: File created\n\n"
                "Step 3:\nAction: bash pytest\nObservation: 3 passed"
            ),
            "task_id": "task-003",
            "reward": 0.91,
            "metadata": {
                "agent_framework": "openhands",
                "agent_model": "test-model",
                "total_steps": 3,
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def dpo_sample_file(tmp_path: Path) -> Path:
    """生成 DPO JSONL 测试数据."""
    path = tmp_path / "dpo_train.jsonl"
    records = [
        {
            "prompt": "Solve the following task:\n\nTask ID: task-001",
            "chosen": "Step 1:\nAction: read_file\n\nStep 2:\nAction: edit_file",
            "rejected": "Step 1:\nAction: bash rm -rf /\n\nStep 2:\nAction: bash echo done",
            "task_id": "task-001",
            "reward_margin": 0.55,
            "metadata": {
                "chosen_model": "good-model",
                "rejected_model": "bad-model",
                "chosen_reward": 0.85,
                "rejected_reward": 0.30,
            },
        },
        {
            "prompt": "Solve the following task:\n\nTask ID: task-002",
            "chosen": "Step 1:\nAction: read_file /api.py",
            "rejected": "Step 1:\nAction: read_file /wrong.py",
            "task_id": "task-002",
            "reward_margin": 0.25,
            "metadata": {
                "chosen_model": "model-a",
                "rejected_model": "model-b",
                "chosen_reward": 0.72,
                "rejected_reward": 0.47,
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def grpo_sample_file(tmp_path: Path) -> Path:
    """生成 GRPO 分组 JSONL 测试数据."""
    path = tmp_path / "grpo_train.jsonl"
    groups = [
        {
            "task_id": "task-001",
            "prompt": "Fix the off-by-one bug in sort function",
            "trajectories": [
                {"response": "Step 1:\nAction: read_file\n\nStep 2:\nAction: edit_file", "reward": 0.85},
                {"response": "Step 1:\nAction: bash cat file\n\nStep 2:\nAction: bash echo fix", "reward": 0.55},
                {"response": "Step 1:\nAction: read_file\n\nStep 2:\nAction: bash rm *", "reward": 0.20},
            ],
        },
        {
            "task_id": "task-002",
            "prompt": "Add input validation",
            "trajectories": [
                {"response": "Step 1:\nAction: read_file /api.py", "reward": 0.72},
                {"response": "Step 1:\nAction: bash grep pattern", "reward": 0.40},
            ],
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for grp in groups:
            f.write(json.dumps(grp, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def small_model_name() -> str:
    """返回 HuggingFace 上的微型测试模型."""
    return "sshleifer/tiny-gpt2"
