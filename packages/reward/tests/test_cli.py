"""测试 Reward CLI 命令."""

import json

from click.testing import CliRunner

from agentreward.cli import main


def _write_trajectory(path, outcome=None):
    """写一个测试轨迹文件."""
    traj = {
        "task": "修复排序 bug",
        "steps": [
            {"tool": "read_file", "params": {"path": "/a.py"}, "output": "def sort(): pass"},
            {"tool": "edit_file", "params": {"path": "/a.py"}, "output": "File edited"},
            {"tool": "bash", "params": {"command": "pytest"}, "output": "1 passed"},
        ],
        "outcome": outcome or {"success": True, "tests_passed": 1, "tests_total": 1},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traj, f, ensure_ascii=False)


def _write_preferences_input(path):
    """写偏好对构建输入文件."""
    data = {
        "task-1": [
            {"id": "a", "reward": 0.9, "step_count": 3},
            {"id": "b", "reward": 0.3, "step_count": 8},
        ],
        "task-2": [
            {"id": "c", "reward": 0.8},
            {"id": "d", "reward": 0.2},
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


class TestScoreCommand:
    """测试 score 命令."""

    def test_score_stdout(self, tmp_path):
        """评分并输出到 stdout."""
        traj_path = tmp_path / "traj.json"
        _write_trajectory(traj_path)

        runner = CliRunner()
        result = runner.invoke(main, ["score", str(traj_path)])

        assert result.exit_code == 0
        assert "总分" in result.output
        assert "结果分" in result.output
        assert "过程分" in result.output

    def test_score_to_file(self, tmp_path):
        """评分并保存到文件."""
        traj_path = tmp_path / "traj.json"
        _write_trajectory(traj_path)
        out_path = tmp_path / "result.json"

        runner = CliRunner()
        result = runner.invoke(main, ["score", str(traj_path), "-o", str(out_path)])

        assert result.exit_code == 0
        assert "评分结果已保存" in result.output
        assert out_path.exists()

        with open(out_path) as f:
            data = json.load(f)
        assert "total_score" in data
        assert "step_rewards" in data

    def test_score_custom_weights(self, tmp_path):
        """自定义权重应生效."""
        traj_path = tmp_path / "traj.json"
        _write_trajectory(traj_path)

        runner = CliRunner()
        result = runner.invoke(main, [
            "score", str(traj_path),
            "--rule-weight", "0.8",
            "--model-weight", "0.2",
        ])

        assert result.exit_code == 0
        assert "总分" in result.output

    def test_score_failed_trajectory(self, tmp_path):
        """失败轨迹应正常评分."""
        traj_path = tmp_path / "traj.json"
        _write_trajectory(traj_path, outcome={"success": False, "tests_passed": 0, "tests_total": 1})

        runner = CliRunner()
        result = runner.invoke(main, ["score", str(traj_path)])

        assert result.exit_code == 0
        assert "结果分: 0.0000" in result.output


class TestCompareCommand:
    """测试 compare 命令."""

    def test_compare_two(self, tmp_path):
        """比较两条轨迹."""
        t1 = tmp_path / "t1.json"
        t2 = tmp_path / "t2.json"
        _write_trajectory(t1, {"success": True, "tests_passed": 1, "tests_total": 1})
        _write_trajectory(t2, {"success": False, "tests_passed": 0, "tests_total": 1})

        runner = CliRunner()
        result = runner.invoke(main, ["compare", str(t1), str(t2)])

        assert result.exit_code == 0
        assert "轨迹比较结果" in result.output
        assert "#1" in result.output
        assert "#2" in result.output

    def test_compare_needs_two(self, tmp_path):
        """少于 2 条应报错."""
        t1 = tmp_path / "t1.json"
        _write_trajectory(t1)

        runner = CliRunner()
        result = runner.invoke(main, ["compare", str(t1)])

        assert result.exit_code != 0
        assert "至少需要 2 条" in result.output

    def test_compare_with_output(self, tmp_path):
        """比较结果保存到文件."""
        t1 = tmp_path / "t1.json"
        t2 = tmp_path / "t2.json"
        _write_trajectory(t1)
        _write_trajectory(t2)
        out = tmp_path / "compare.json"

        runner = CliRunner()
        result = runner.invoke(main, ["compare", str(t1), str(t2), "-o", str(out)])

        assert result.exit_code == 0
        assert out.exists()


class TestPreferencesCommand:
    """测试 preferences 命令."""

    def test_preferences_stdout(self, tmp_path):
        """构建偏好对并输出到 stdout."""
        pref_path = tmp_path / "input.json"
        _write_preferences_input(pref_path)

        runner = CliRunner()
        result = runner.invoke(main, ["preferences", str(pref_path)])

        assert result.exit_code == 0
        assert "偏好对构建完成" in result.output
        assert "总对数: 2" in result.output

    def test_preferences_to_file(self, tmp_path):
        """保存偏好对到文件."""
        pref_path = tmp_path / "input.json"
        _write_preferences_input(pref_path)
        out = tmp_path / "pairs.json"

        runner = CliRunner()
        result = runner.invoke(main, [
            "preferences", str(pref_path), "-o", str(out),
        ])

        assert result.exit_code == 0
        assert "偏好对已保存" in result.output
        assert out.exists()

    def test_preferences_custom_margin(self, tmp_path):
        """自定义 min_margin."""
        pref_path = tmp_path / "input.json"
        _write_preferences_input(pref_path)

        runner = CliRunner()
        result = runner.invoke(main, [
            "preferences", str(pref_path), "--min-margin", "0.5",
        ])

        assert result.exit_code == 0


class TestRubricsCommand:
    """测试 rubrics 命令."""

    def test_rubrics_list(self):
        """列出 rubric 维度."""
        runner = CliRunner()
        result = runner.invoke(main, ["rubrics"])

        assert result.exit_code == 0
        assert "goal_progress" in result.output
        assert "tool_choice" in result.output
        assert "总权重" in result.output
