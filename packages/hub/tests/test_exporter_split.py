"""Tests for DatasetExporter split export (Phase 3)."""

import json
import tempfile
from pathlib import Path

from trajectoryhub.exporter import DatasetExporter


class TestExportSFTSplit:
    """export_sft_split() 分割导出测试."""

    def _create_trajectories(self, tmp_dir: Path, count: int = 10) -> Path:
        """创建 N 条成功的测试轨迹."""
        traj_path = tmp_dir / "trajectories.jsonl"
        with open(traj_path, "w", encoding="utf-8") as f:
            for i in range(count):
                traj = {
                    "task_id": f"task-{i:03d}",
                    "agent_framework": "openhands",
                    "agent_model": "test-model",
                    "steps": [
                        {"thought": f"Step for task {i}", "action": f"action_{i}", "observation": "ok"},
                    ],
                    "total_steps": 1,
                    "success": True,
                    "reward": 0.5 + i * 0.05,
                    "metadata": {"task_description": f"Test task {i}"},
                }
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        return traj_path

    def test_default_split_ratios(self):
        """默认比例 0.8/0.1/0.1 应正确分割."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, count=10)
            output_dir = tmp_path / "split_output"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            results = exporter.export_sft_split(str(output_dir))

            assert "train" in results
            assert "val" in results
            assert "test" in results

            total = sum(r.total_records for r in results.values())
            assert total == 10

            assert results["train"].success is True
            assert results["val"].success is True
            assert results["test"].success is True

            # train: 8, val: 1, test: 1
            assert results["train"].total_records == 8
            assert results["val"].total_records == 1
            assert results["test"].total_records == 1

    def test_custom_split_ratios(self):
        """自定义比例应正确分割."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, count=20)
            output_dir = tmp_path / "split_output"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            results = exporter.export_sft_split(
                str(output_dir),
                split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            )

            total = sum(r.total_records for r in results.values())
            assert total == 20

            assert results["train"].total_records == 14  # int(20 * 0.7)
            assert results["val"].total_records == 3  # int(20 * 0.15)
            # test gets remainder: 20 - 14 - 3 = 3
            assert results["test"].total_records == 3

    def test_reproducibility(self):
        """相同 seed 应产生相同结果."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, count=20)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))

            output_dir_1 = tmp_path / "split_1"
            exporter.export_sft_split(str(output_dir_1), seed=123)

            output_dir_2 = tmp_path / "split_2"
            exporter.export_sft_split(str(output_dir_2), seed=123)

            # 读取并比较 train.jsonl
            with open(output_dir_1 / "train.jsonl") as f1, open(output_dir_2 / "train.jsonl") as f2:
                lines_1 = f1.readlines()
                lines_2 = f2.readlines()

            assert lines_1 == lines_2

    def test_different_seed_different_order(self):
        """不同 seed 应产生不同顺序."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, count=20)

            exporter = DatasetExporter(trajectories_dir=str(traj_path))

            output_dir_1 = tmp_path / "split_a"
            exporter.export_sft_split(str(output_dir_1), seed=1)

            output_dir_2 = tmp_path / "split_b"
            exporter.export_sft_split(str(output_dir_2), seed=999)

            with open(output_dir_1 / "train.jsonl") as f1, open(output_dir_2 / "train.jsonl") as f2:
                lines_1 = f1.readlines()
                lines_2 = f2.readlines()

            # 顺序应不同（20 条数据几乎不可能 shuffle 到相同顺序）
            assert lines_1 != lines_2

    def test_empty_data(self):
        """空数据应返回所有分割各 0 条."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = tmp_path / "trajectories.jsonl"
            traj_path.write_text("")
            output_dir = tmp_path / "split_output"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            results = exporter.export_sft_split(str(output_dir))

            assert all(r.success is True for r in results.values())
            assert all(r.total_records == 0 for r in results.values())

    def test_only_failed_trajectories(self):
        """只有失败轨迹时，SFT split 应返回 0 条."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = tmp_path / "trajectories.jsonl"
            with open(traj_path, "w") as f:
                f.write(json.dumps({
                    "task_id": "t1", "success": False, "reward": 0.1,
                    "steps": [{"action": "a"}],
                }) + "\n")

            output_dir = tmp_path / "split_output"
            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            results = exporter.export_sft_split(str(output_dir))

            assert all(r.total_records == 0 for r in results.values())

    def test_output_files_created(self):
        """应在输出目录创建对应的 JSONL 文件."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, count=5)
            output_dir = tmp_path / "split_output"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            exporter.export_sft_split(str(output_dir))

            assert (output_dir / "train.jsonl").exists()
            assert (output_dir / "val.jsonl").exists()
            assert (output_dir / "test.jsonl").exists()

    def test_sft_record_format(self):
        """分割后的记录应包含标准 SFT 字段."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = self._create_trajectories(tmp_path, count=5)
            output_dir = tmp_path / "split_output"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            exporter.export_sft_split(str(output_dir))

            with open(output_dir / "train.jsonl") as f:
                for line in f:
                    rec = json.loads(line)
                    assert "instruction" in rec
                    assert "input" in rec
                    assert "response" in rec
                    assert "task_id" in rec
                    assert "reward" in rec


class TestExportDPOSplit:
    """export_dpo_split() 分割导出测试."""

    def _create_preferences(self, tmp_dir: Path, count: int = 10) -> Path:
        """创建 N 条测试偏好对."""
        pref_path = tmp_dir / "preferences.jsonl"
        with open(pref_path, "w", encoding="utf-8") as f:
            for i in range(count):
                pref = {
                    "task_id": f"task-{i:03d}",
                    "chosen": {
                        "steps": [{"action": f"good_action_{i}", "observation": "ok"}],
                        "reward": 0.8 + i * 0.01,
                        "agent_model": "good-model",
                    },
                    "rejected": {
                        "steps": [{"action": f"bad_action_{i}", "observation": "fail"}],
                        "reward": 0.2,
                        "agent_model": "bad-model",
                    },
                    "reward_margin": 0.6 + i * 0.01,
                }
                f.write(json.dumps(pref, ensure_ascii=False) + "\n")
        return pref_path

    def test_dpo_split_ratios(self):
        """DPO 分割应按比例正确分割."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # 需要一个（空的）trajectories 文件
            traj_path = tmp_path / "trajectories.jsonl"
            traj_path.write_text("")
            pref_path = self._create_preferences(tmp_path, count=10)
            output_dir = tmp_path / "dpo_split"

            exporter = DatasetExporter(
                trajectories_dir=str(traj_path),
                preferences_dir=str(pref_path),
            )
            results = exporter.export_dpo_split(str(output_dir))

            total = sum(r.total_records for r in results.values())
            assert total == 10
            assert results["train"].total_records == 8
            assert all(r.success is True for r in results.values())

    def test_dpo_split_no_preferences(self):
        """无偏好对文件时应返回失败."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = tmp_path / "trajectories.jsonl"
            traj_path.write_text("")
            output_dir = tmp_path / "dpo_split"

            exporter = DatasetExporter(trajectories_dir=str(traj_path))
            results = exporter.export_dpo_split(str(output_dir))

            assert all(r.success is False for r in results.values())
            assert all("偏好对文件不存在" in r.error for r in results.values())

    def test_dpo_record_format(self):
        """分割后的记录应包含标准 DPO 字段."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_path = tmp_path / "trajectories.jsonl"
            traj_path.write_text("")
            pref_path = self._create_preferences(tmp_path, count=5)
            output_dir = tmp_path / "dpo_split"

            exporter = DatasetExporter(
                trajectories_dir=str(traj_path),
                preferences_dir=str(pref_path),
            )
            exporter.export_dpo_split(str(output_dir))

            with open(output_dir / "train.jsonl") as f:
                for line in f:
                    rec = json.loads(line)
                    assert "prompt" in rec
                    assert "chosen" in rec
                    assert "rejected" in rec
                    assert "reward_margin" in rec
