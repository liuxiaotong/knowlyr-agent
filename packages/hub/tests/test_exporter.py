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

    def test_export_grpo(self):
        """Test GRPO format export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # 需要同 task_id 的多条轨迹
            traj_path = tmp_path / "trajectories.jsonl"
            trajectories = [
                {
                    "task_id": "task-001",
                    "agent_model": "model-a",
                    "steps": [{"action": "read", "observation": "ok"}],
                    "success": True,
                    "reward": 0.9,
                    "metadata": {"task_description": "Fix bug"},
                },
                {
                    "task_id": "task-001",
                    "agent_model": "model-b",
                    "steps": [{"action": "write", "observation": "done"}],
                    "success": True,
                    "reward": 0.5,
                    "metadata": {"task_description": "Fix bug"},
                },
                {
                    "task_id": "task-002",
                    "agent_model": "model-a",
                    "steps": [{"action": "read", "observation": "ok"}],
                    "success": True,
                    "reward": 0.7,
                    "metadata": {"task_description": "Add feature"},
                },
            ]
            with open(traj_path, "w", encoding="utf-8") as f:
                for traj in trajectories:
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")

            output_path = tmp_path / "grpo_output.jsonl"
            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.export_grpo(str(output_path), group_size=4)

            assert result.success is True
            assert result.format == "grpo"
            assert result.total_records == 1  # 只有 task-001 有 >=2 条

            with open(output_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            assert len(records) == 1
            assert records[0]["task_id"] == "task-001"
            assert len(records[0]["trajectories"]) == 2
            # 按 reward 降序排列
            assert records[0]["trajectories"][0]["reward"] >= records[0]["trajectories"][1]["reward"]

    def test_export_grpo_empty(self):
        """所有 task 只有 1 条轨迹时 GRPO 应返回 0 条."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = tmp_path / "trajectories.jsonl"
            with open(traj_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"task_id": "t-1", "reward": 0.5, "steps": []}) + "\n")

            output_path = tmp_path / "grpo_output.jsonl"
            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.export_grpo(str(output_path))

            assert result.success is True
            assert result.total_records == 0

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


# ── validate_dataset 测试 ─────────────────────────────────────────


class TestValidateDataset:
    """validate_dataset() 质量验证测试."""

    def _create_trajectories(self, tmp_dir: Path, trajectories: list) -> Path:
        """写入轨迹 JSONL."""
        traj_path = tmp_dir / "trajectories.jsonl"
        with open(traj_path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        return traj_path

    def _create_preferences(self, tmp_dir: Path, preferences: list) -> Path:
        """写入偏好对 JSONL."""
        pref_path = tmp_dir / "preferences.jsonl"
        with open(pref_path, "w", encoding="utf-8") as f:
            for pref in preferences:
                f.write(json.dumps(pref, ensure_ascii=False) + "\n")
        return pref_path

    def test_valid_sft_dataset(self):
        """合格 SFT 数据集应通过验证."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, [
                {
                    "task_id": "t-001",
                    "steps": [
                        {"action": "read /a.py", "observation": "content"},
                        {"action": "edit /a.py", "observation": "done"},
                    ],
                    "success": True,
                    "reward": 0.85,
                },
                {
                    "task_id": "t-002",
                    "steps": [{"action": "bash ls", "observation": "files"}],
                    "success": False,
                    "reward": 0.3,
                },
            ])

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.validate_dataset(format="sft")

            assert result["total_records"] == 2
            assert result["is_valid"] is True
            assert result["issues"] == []
            assert result["reward_stats"]["min"] == 0.3
            assert result["reward_stats"]["max"] == 0.85

    def test_empty_dataset(self):
        """空数据集应返回 is_valid=False."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, [])

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.validate_dataset()

            assert result["total_records"] == 0
            assert result["is_valid"] is False
            assert len(result["issues"]) > 0

    def test_missing_fields_detected(self):
        """缺失字段应报告问题."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # 创建大量缺失 task_id 的记录 (>5% 触发告警)
            trajs = [
                {"steps": [{"action": "a", "observation": "b"}], "success": True, "reward": 0.5}
                for _ in range(10)
            ]
            traj_path = self._create_trajectories(tmp_path, trajs)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.validate_dataset()

            assert result["missing_rate"]["task_id"] == 1.0
            assert any("task_id" in issue for issue in result["issues"])
            assert result["is_valid"] is False

    def test_all_zero_rewards(self):
        """全零 reward 应报告问题."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trajs = [
                {
                    "task_id": f"t-{i}",
                    "steps": [{"action": "a", "observation": "b"}],
                    "success": True,
                    "reward": 0.0,
                }
                for i in range(5)
            ]
            traj_path = self._create_trajectories(tmp_path, trajs)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.validate_dataset()

            assert result["reward_stats"]["all_zero"] is True
            assert any("reward" in issue and "0.0" in issue for issue in result["issues"])

    def test_empty_steps_detected(self):
        """无步骤的记录应报告问题."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trajs = [
                {"task_id": "t-1", "steps": [], "success": True, "reward": 0.5},
                {"task_id": "t-2", "steps": [{"action": "a", "observation": "b"}],
                 "success": True, "reward": 0.7},
            ]
            traj_path = self._create_trajectories(tmp_path, trajs)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.validate_dataset()

            assert any("无步骤" in issue for issue in result["issues"])

    def test_validate_dpo_format(self):
        """DPO 格式验证应检查 chosen/rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, [])
            pref_path = self._create_preferences(tmp_path, [
                {
                    "task_id": "t-001",
                    "chosen": {"steps": [{"action": "a"}], "reward": 0.9},
                    "rejected": {"steps": [{"action": "b"}], "reward": 0.3},
                    "reward_margin": 0.6,
                },
            ])

            exporter = DatasetExporter(
                trajectories_dir=str(traj_path),
                preferences_dir=str(pref_path),
            )
            result = exporter.validate_dataset(format="dpo")

            assert result["total_records"] == 1
            assert result["is_valid"] is True

    def test_validate_dpo_negative_margin(self):
        """DPO 负 reward_margin 应报告问题."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, [])
            pref_path = self._create_preferences(tmp_path, [
                {
                    "task_id": "t-001",
                    "chosen": {"steps": [{"action": "a"}], "reward": 0.3},
                    "rejected": {"steps": [{"action": "b"}], "reward": 0.9},
                    "reward_margin": -0.6,
                },
            ])

            exporter = DatasetExporter(
                trajectories_dir=str(traj_path),
                preferences_dir=str(pref_path),
            )
            result = exporter.validate_dataset(format="dpo")

            assert any("reward_margin < 0" in issue for issue in result["issues"])

    def test_length_stats_computed(self):
        """应计算 response 长度统计."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trajs = [
                {
                    "task_id": f"t-{i}",
                    "steps": [{"action": "a" * 100, "observation": "b" * 100}],
                    "success": True,
                    "reward": 0.5,
                }
                for i in range(3)
            ]
            traj_path = self._create_trajectories(tmp_path, trajs)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            result = exporter.validate_dataset()

            assert "min" in result["length_stats"]
            assert "max" in result["length_stats"]
            assert "mean" in result["length_stats"]
            assert result["length_stats"]["min"] > 0
