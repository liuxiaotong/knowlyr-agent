"""Task loading and management - 任务加载与管理."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    """单个任务定义.

    Attributes:
        task_id: 唯一任务标识
        description: 任务描述
        type: 任务类型 (bug_fix / feature / refactor / test)
        language: 编程语言 (python / javascript / go / etc.)
        difficulty: 难度等级 (easy / medium / hard)
        repo: GitHub 仓库地址 (e.g. "owner/repo")
        base_commit: 基准 commit SHA
        test_command: 验证命令 (e.g. "pytest tests/test_xxx.py")
        metadata: 额外元数据
    """

    task_id: str = ""
    description: str = ""
    type: str = "bug_fix"
    language: str = "python"
    difficulty: str = "medium"
    repo: str = ""
    base_commit: str = ""
    test_command: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskLoader:
    """任务加载器.

    支持从多种来源加载任务：
    - JSONL 文件 (本地任务定义)
    - SWE-bench 数据集 (通过 HuggingFace datasets 加载)

    Usage:
        loader = TaskLoader()

        # 从本地 JSONL 加载
        tasks = loader.load_from_jsonl("tasks.jsonl")

        # 从 SWE-bench 加载
        tasks = loader.load_from_swebench("princeton-nlp/SWE-bench_Verified", limit=100)

        # 过滤任务
        python_tasks = loader.filter_tasks(tasks, language="python", difficulty="medium")
    """

    def load_from_jsonl(self, path: str) -> List[Task]:
        """从 JSONL 文件加载任务.

        JSONL 格式 (每行一个 JSON):
        ```json
        {
            "task_id": "repo__issue-123",
            "description": "Fix the bug in parser module",
            "type": "bug_fix",
            "language": "python",
            "difficulty": "medium",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "test_command": "pytest tests/test_parser.py"
        }
        ```

        Args:
            path: JSONL 文件路径

        Returns:
            List[Task]: 加载的任务列表
        """
        tasks = []
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"任务文件不存在: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    task = Task(
                        task_id=data.get("task_id", f"task_{line_num}"),
                        description=data.get("description", ""),
                        type=data.get("type", "bug_fix"),
                        language=data.get("language", "python"),
                        difficulty=data.get("difficulty", "medium"),
                        repo=data.get("repo", ""),
                        base_commit=data.get("base_commit", ""),
                        test_command=data.get("test_command", ""),
                        metadata=data.get("metadata", {}),
                    )
                    tasks.append(task)
                except json.JSONDecodeError:
                    continue

        return tasks

    def load_from_swebench(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> List[Task]:
        """从 SWE-bench 数据集加载任务.

        需要安装 datasets 库: pip install datasets

        Args:
            dataset_name: HuggingFace 数据集名称
            split: 数据集分割 (test / train / dev)
            limit: 限制加载数量

        Returns:
            List[Task]: 加载的任务列表
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "加载 SWE-bench 需要 datasets 库。请运行: pip install datasets"
            )

        dataset = load_dataset(dataset_name, split=split)

        tasks = []
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break

            task = Task(
                task_id=item.get("instance_id", f"swebench_{i}"),
                description=item.get("problem_statement", ""),
                type="bug_fix",
                language="python",
                difficulty=self._estimate_difficulty(item),
                repo=item.get("repo", ""),
                base_commit=item.get("base_commit", ""),
                test_command=item.get("test_cmd", ""),
                metadata={
                    "source": dataset_name,
                    "patch": item.get("patch", ""),
                    "hints_text": item.get("hints_text", ""),
                    "created_at": item.get("created_at", ""),
                    "version": item.get("version", ""),
                },
            )
            tasks.append(task)

        return tasks

    def filter_tasks(
        self,
        tasks: List[Task],
        language: Optional[str] = None,
        difficulty: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[Task]:
        """按条件过滤任务.

        Args:
            tasks: 任务列表
            language: 按编程语言过滤
            difficulty: 按难度过滤
            task_type: 按任务类型过滤

        Returns:
            List[Task]: 过滤后的任务列表
        """
        filtered = tasks

        if language:
            filtered = [t for t in filtered if t.language.lower() == language.lower()]

        if difficulty:
            filtered = [t for t in filtered if t.difficulty.lower() == difficulty.lower()]

        if task_type:
            filtered = [t for t in filtered if t.type.lower() == task_type.lower()]

        return filtered

    def _estimate_difficulty(self, item: Dict[str, Any]) -> str:
        """根据 SWE-bench 数据估算任务难度.

        简单启发式: 按 patch 大小估算。
        """
        patch = item.get("patch", "")
        patch_lines = len(patch.split("\n"))

        if patch_lines < 10:
            return "easy"
        elif patch_lines < 50:
            return "medium"
        else:
            return "hard"
