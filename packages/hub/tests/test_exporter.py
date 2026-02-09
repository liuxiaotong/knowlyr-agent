"""Tests for DatasetExporter format conversion."""

import json
import tempfile
from pathlib import Path

from trajectoryhub.exporter import DatasetExporter, ExportResult


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_default_result(self):
        """Test default export result."""
        result = ExportResult()
        assert result.success is False
        assert result.total_records == 0
        assert result.error is None

    def test_success_result(self):
        """Test successful export result."""
        result = ExportResult(
            success=True,
            output_path="/tmp/output.jsonl",
            total_records=100,
            format="sft",
        )
        assert result.success is True
        assert result.total_records == 100
        assert result.format == "sft"


class TestDatasetExporter:
    """Tests for DatasetExporter."""

    def _create_test_trajectories(self, tmp_dir: Path) -> Path:
        """创建测试用的轨迹文件."""
        traj_path = tmp_dir / "trajectories.jsonl"
        trajectories = [
            {
                "task_id": "task-001",
                "agent_framework": "openhands",
                "agent_model": "claude-sonnet-4-20250514",
                "steps": [
                    {"thought": "Read the file", "action": "file_read /test.py", "observation": "def test(): pass"},
                    {"thought": "Fix the bug", "action": "file_write /test.py", "observation": "File written"},
                ],
                "total_steps": 2,
                "success": True,
                "reward": 0.85,
                "step_rewards": [0.8, 0.9],
                "duration_seconds": 15.0,
                "metadata": {"task_description": "Fix the test function"},
            },
            {
                "task_id": "task-001",
                "agent_framework": "sweagent",
                "agent_model": "gpt-4o",
                "steps": [
                    {"thought": "Look at file", "action": "file_read /test.py", "observation": "def test(): pass"},
                ],
                "total_steps": 1,
                "success": False,
                "reward": 0.3,
                "step_rewards": [0.3],
                "duration_seconds": 8.0,
                "metadata": {"task_description": "Fix the test function"},
            },
        ]
        with open(traj_path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        return traj_path

    def _create_test_preferences(self, tmp_dir: Path) -> Path:
        """创建测试用的偏好对文件."""
        pref_path = tmp_dir / "preferences.jsonl"
        preferences = [
            {
                "task_id": "task-001",
                "chosen": {
                    "agent_framework": "openhands",
                    "agent_model": "claude-sonnet-4-20250514",
                    "steps": [
                        {"thought": "Read the file", "action": "file_read /test.py", "observation": "content"},
                    ],
                    "reward": 0.85,
                    "success": True,
                },
                "rejected": {
                    "agent_framework": "sweagent",
                    "agent_model": "gpt-4o",
                    "steps": [
                        {"thought": "Look at file", "action": "file_read /test.py", "observation": "content"},
                    ],
                    "reward": 0.3,
                    "success": False,
                },
                "reward_margin": 0.55,
            }
        ]
        with open(pref_path, "w", encoding="utf-8") as f:
            for pref in preferences:
                f.write(json.dumps(pref, ensure_ascii=False) + "\n")
        return pref_path

    def test_export_sft(self):
        """Test SFT format export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_test_trajectories(tmp_path)
            output_path = tmp_path / "sft_output.jsonl"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.export_sft(str(output_path))

            assert result.success is True
            assert result.format == "sft"
            assert result.total_records == 1  # 只有 1 条成功的

            # 验证输出格式
            with open(output_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            assert len(records) == 1
            record = records[0]
            assert "instruction" in record
            assert "response" in record
            assert "task_id" in record
            assert record["task_id"] == "task-001"
            assert record["reward"] == 0.85

    def test_export_dpo(self):
        """Test DPO format export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_test_trajectories(tmp_path)
            pref_path = self._create_test_preferences(tmp_path)
            output_path = tmp_path / "dpo_output.jsonl"

            exporter = DatasetExporter(
                trajectories_dir=str(traj_path),
                preferences_dir=str(pref_path),
            )
            result = exporter.export_dpo(str(output_path))

            assert result.success is True
            assert result.format == "dpo"
            assert result.total_records == 1

            # 验证输出格式
            with open(output_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            assert len(records) == 1
            record = records[0]
            assert "prompt" in record
            assert "chosen" in record
            assert "rejected" in record
            assert "reward_margin" in record
            assert record["reward_margin"] == 0.55

    def test_export_dpo_no_preferences(self):
        """Test DPO export fails gracefully without preferences file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_test_trajectories(tmp_path)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.export_dpo(str(tmp_path / "dpo_output.jsonl"))

            assert result.success is False
            assert result.error is not None

    def test_export_benchmark(self):
        """Test benchmark format export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_test_trajectories(tmp_path)
            output_path = tmp_path / "benchmark_output.jsonl"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.export_benchmark(str(output_path))

            assert result.success is True
            assert result.format == "benchmark"

    def test_generate_datacard(self):
        """Test dataset card generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_test_trajectories(tmp_path)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            card = exporter.generate_datacard()

            assert "Agent Trajectory Dataset" in card
            assert "Total trajectories" in card
            assert "license: mit" in card

    def test_export_huggingface_without_lib(self):
        """无 huggingface_hub 时应返回安装提示."""
        import trajectoryhub.exporter as exp_mod
        original = exp_mod._HAS_HF
        try:
            exp_mod._HAS_HF = False
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                traj_path = self._create_test_trajectories(tmp_path)

                exporter = DatasetExporter(trajectories_dir=str(traj_path))
                result = exporter.export_huggingface("test/repo")

                assert result.success is False
                assert "huggingface" in result.error.lower()
        finally:
            exp_mod._HAS_HF = original

    def test_export_huggingface_with_mock(self):
        """有 huggingface_hub 时应调用 API 上传."""
        from unittest.mock import MagicMock, patch
        import trajectoryhub.exporter as exp_mod

        original = exp_mod._HAS_HF
        try:
            exp_mod._HAS_HF = True
            mock_api = MagicMock()

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                traj_path = self._create_test_trajectories(tmp_path)

                exporter = DatasetExporter(trajectories_dir=str(traj_path))

                with patch.object(exp_mod, "HfApi", return_value=mock_api, create=True):
                    result = exporter.export_huggingface("user/test-dataset")

                assert result.success is True
                assert result.format == "huggingface"
                assert result.total_records > 0
                assert "huggingface.co" in result.output_path
                mock_api.create_repo.assert_called_once_with(
                    "user/test-dataset", repo_type="dataset", exist_ok=True,
                )
                mock_api.upload_folder.assert_called_once()
        finally:
            exp_mod._HAS_HF = original
