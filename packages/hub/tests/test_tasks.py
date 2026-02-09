"""测试 TaskLoader 任务加载."""

import json

import pytest

from knowlyrcore import TaskInfo
from trajectoryhub.tasks import TaskLoader


def _write_tasks_jsonl(path, tasks=None):
    """写测试用任务文件."""
    if tasks is None:
        tasks = [
            {
                "task_id": "task-001",
                "description": "修复排序 bug",
                "type": "bug_fix",
                "language": "python",
                "difficulty": "easy",
                "repo": "org/repo",
                "base_commit": "abc123",
                "test_command": "pytest tests/test_sort.py",
            },
            {
                "task_id": "task-002",
                "description": "添加缓存功能",
                "type": "feature",
                "language": "python",
                "difficulty": "medium",
            },
            {
                "task_id": "task-003",
                "description": "Fix JS linting",
                "type": "bug_fix",
                "language": "javascript",
                "difficulty": "easy",
            },
        ]
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


class TestLoadFromJsonl:
    """测试 JSONL 加载."""

    def test_basic_load(self, tmp_path):
        """加载基本任务文件."""
        path = tmp_path / "tasks.jsonl"
        _write_tasks_jsonl(path)

        loader = TaskLoader()
        tasks = loader.load_from_jsonl(str(path))

        assert len(tasks) == 3
        assert all(isinstance(t, TaskInfo) for t in tasks)
        assert tasks[0].task_id == "task-001"
        assert tasks[0].description == "修复排序 bug"
        assert tasks[0].repo == "org/repo"

    def test_file_not_found(self):
        """文件不存在应抛出 FileNotFoundError."""
        loader = TaskLoader()
        with pytest.raises(FileNotFoundError, match="任务文件不存在"):
            loader.load_from_jsonl("/nonexistent/path.jsonl")

    def test_empty_lines_skipped(self, tmp_path):
        """空行应被跳过."""
        path = tmp_path / "tasks.jsonl"
        with open(path, "w") as f:
            f.write('{"task_id": "t1", "description": "test"}\n')
            f.write("\n")
            f.write('{"task_id": "t2", "description": "test2"}\n')

        loader = TaskLoader()
        tasks = loader.load_from_jsonl(str(path))
        assert len(tasks) == 2

    def test_invalid_json_skipped(self, tmp_path):
        """无效 JSON 行应被跳过."""
        path = tmp_path / "tasks.jsonl"
        with open(path, "w") as f:
            f.write('{"task_id": "t1", "description": "ok"}\n')
            f.write("not valid json\n")
            f.write('{"task_id": "t2", "description": "ok2"}\n')

        loader = TaskLoader()
        tasks = loader.load_from_jsonl(str(path))
        assert len(tasks) == 2

    def test_auto_task_id(self, tmp_path):
        """缺少 task_id 应自动生成."""
        path = tmp_path / "tasks.jsonl"
        with open(path, "w") as f:
            f.write('{"description": "no id task"}\n')

        loader = TaskLoader()
        tasks = loader.load_from_jsonl(str(path))
        assert tasks[0].task_id == "task_1"

    def test_default_fields(self, tmp_path):
        """缺少字段应用默认值."""
        path = tmp_path / "tasks.jsonl"
        with open(path, "w") as f:
            f.write('{"task_id": "t1"}\n')

        loader = TaskLoader()
        tasks = loader.load_from_jsonl(str(path))
        assert tasks[0].type == "bug_fix"
        assert tasks[0].language == "python"
        assert tasks[0].difficulty == "medium"


class TestFilterTasks:
    """测试任务过滤."""

    def _load_test_tasks(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        _write_tasks_jsonl(path)
        return TaskLoader().load_from_jsonl(str(path))

    def test_filter_by_language(self, tmp_path):
        """按语言过滤."""
        tasks = self._load_test_tasks(tmp_path)
        loader = TaskLoader()
        filtered = loader.filter_tasks(tasks, language="python")
        assert len(filtered) == 2
        assert all(t.language == "python" for t in filtered)

    def test_filter_by_difficulty(self, tmp_path):
        """按难度过滤."""
        tasks = self._load_test_tasks(tmp_path)
        loader = TaskLoader()
        filtered = loader.filter_tasks(tasks, difficulty="easy")
        assert len(filtered) == 2

    def test_filter_by_type(self, tmp_path):
        """按类型过滤."""
        tasks = self._load_test_tasks(tmp_path)
        loader = TaskLoader()
        filtered = loader.filter_tasks(tasks, task_type="feature")
        assert len(filtered) == 1
        assert filtered[0].task_id == "task-002"

    def test_filter_combined(self, tmp_path):
        """组合过滤."""
        tasks = self._load_test_tasks(tmp_path)
        loader = TaskLoader()
        filtered = loader.filter_tasks(tasks, language="python", difficulty="easy")
        assert len(filtered) == 1
        assert filtered[0].task_id == "task-001"

    def test_filter_no_match(self, tmp_path):
        """无匹配应返回空列表."""
        tasks = self._load_test_tasks(tmp_path)
        loader = TaskLoader()
        filtered = loader.filter_tasks(tasks, language="rust")
        assert len(filtered) == 0

    def test_filter_case_insensitive(self, tmp_path):
        """过滤应不区分大小写."""
        tasks = self._load_test_tasks(tmp_path)
        loader = TaskLoader()
        filtered = loader.filter_tasks(tasks, language="Python")
        assert len(filtered) == 2


class TestEstimateDifficulty:
    """测试难度估算."""

    def test_easy(self):
        """短 patch 应为 easy."""
        loader = TaskLoader()
        assert loader._estimate_difficulty({"patch": "a\nb\nc"}) == "easy"

    def test_medium(self):
        """中等 patch 应为 medium."""
        loader = TaskLoader()
        patch = "\n".join([f"line {i}" for i in range(30)])
        assert loader._estimate_difficulty({"patch": patch}) == "medium"

    def test_hard(self):
        """长 patch 应为 hard."""
        loader = TaskLoader()
        patch = "\n".join([f"line {i}" for i in range(60)])
        assert loader._estimate_difficulty({"patch": patch}) == "hard"

    def test_no_patch(self):
        """无 patch 应为 easy."""
        loader = TaskLoader()
        assert loader._estimate_difficulty({}) == "easy"
