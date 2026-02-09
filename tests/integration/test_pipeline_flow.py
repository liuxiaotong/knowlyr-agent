"""集成测试 — Hub Pipeline 数据流 (preference pairs + export)."""

import json

from trajectoryhub.config import PipelineConfig
from trajectoryhub.exporter import DatasetExporter
from trajectoryhub.pipeline import Pipeline, Trajectory


class TestPreferencePairs:
    """验证 preference pair 构建逻辑."""

    def test_build_preference_pairs(self, tmp_path):
        """同任务多轨迹应按 reward 排序配对."""
        config = PipelineConfig(output_dir=str(tmp_path))
        pipeline = Pipeline(config)

        # 3 条同 task 轨迹，不同 reward
        trajectories = [
            Trajectory(task_id="t1", agent_framework="oh", agent_model="m1",
                       reward=0.9, success=True, steps=[{"action": "good"}]),
            Trajectory(task_id="t1", agent_framework="swe", agent_model="m2",
                       reward=0.5, success=True, steps=[{"action": "ok"}]),
            Trajectory(task_id="t1", agent_framework="oh", agent_model="m3",
                       reward=0.2, success=False, steps=[{"action": "bad"}]),
        ]

        prefs_path = tmp_path / "preferences.jsonl"
        pipeline._build_preference_pairs(trajectories, prefs_path)

        # 读取偏好对
        pairs = []
        with open(prefs_path) as f:
            for line in f:
                pairs.append(json.loads(line))

        # 3 个轨迹 → 2 个偏好对 (0.9>0.5, 0.5>0.2)
        assert len(pairs) == 2

        # 每对的 chosen.reward > rejected.reward
        for pair in pairs:
            assert pair["chosen"]["reward"] > pair["rejected"]["reward"]
            assert pair["reward_margin"] > 0

    def test_preference_pairs_skip_ties(self, tmp_path):
        """相同 reward 的轨迹不产生偏好对."""
        config = PipelineConfig(output_dir=str(tmp_path))
        pipeline = Pipeline(config)

        trajectories = [
            Trajectory(task_id="t1", reward=0.5, steps=[]),
            Trajectory(task_id="t1", reward=0.5, steps=[]),
        ]

        prefs_path = tmp_path / "preferences.jsonl"
        pipeline._build_preference_pairs(trajectories, prefs_path)

        with open(prefs_path) as f:
            pairs = [json.loads(line) for line in f if line.strip()]

        assert len(pairs) == 0

    def test_preference_pairs_multi_task(self, tmp_path):
        """不同 task 的轨迹独立配对."""
        config = PipelineConfig(output_dir=str(tmp_path))
        pipeline = Pipeline(config)

        trajectories = [
            Trajectory(task_id="t1", reward=0.9, steps=[]),
            Trajectory(task_id="t1", reward=0.3, steps=[]),
            Trajectory(task_id="t2", reward=0.8, steps=[]),
            Trajectory(task_id="t2", reward=0.1, steps=[]),
        ]

        prefs_path = tmp_path / "preferences.jsonl"
        pipeline._build_preference_pairs(trajectories, prefs_path)

        with open(prefs_path) as f:
            pairs = [json.loads(line) for line in f if line.strip()]

        assert len(pairs) == 2
        task_ids = {p["task_id"] for p in pairs}
        assert task_ids == {"t1", "t2"}


class TestTrajectoryRoundTrip:
    """验证轨迹序列化/反序列化完整性."""

    def test_save_load_trajectories(self, tmp_path):
        """保存后加载应恢复所有字段."""
        config = PipelineConfig(output_dir=str(tmp_path))
        pipeline = Pipeline(config)

        original = [
            Trajectory(
                task_id="rt-001",
                agent_framework="openhands",
                agent_model="claude-sonnet-4-20250514",
                steps=[
                    {"action": "read", "path": "/test.py"},
                    {"action": "edit", "path": "/test.py", "content": "fixed"},
                ],
                total_steps=2,
                success=True,
                reward=0.85,
                step_rewards=[0.7, 0.9],
                duration_seconds=42.5,
                metadata={"run_id": "r1", "nested": {"key": "value"}},
            ),
        ]

        path = tmp_path / "traj.jsonl"
        pipeline._save_trajectories(path, original)
        loaded = pipeline._load_trajectories(path)

        assert len(loaded) == 1
        t = loaded[0]
        assert t.task_id == "rt-001"
        assert t.agent_framework == "openhands"
        assert len(t.steps) == 2
        assert t.reward == 0.85
        assert t.step_rewards == [0.7, 0.9]
        assert t.duration_seconds == 42.5
        assert t.metadata["nested"]["key"] == "value"


class TestExportPipeline:
    """验证 export 完整流程."""

    def _write_trajectories(self, path, trajectories):
        """辅助: 将 hub Trajectory 写为 JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for t in trajectories:
                record = {
                    "task_id": t.task_id,
                    "agent_framework": t.agent_framework,
                    "agent_model": t.agent_model,
                    "steps": t.steps,
                    "total_steps": t.total_steps,
                    "success": t.success,
                    "reward": t.reward,
                    "metadata": t.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_preferences(self, path, pairs):
        """辅助: 将偏好对写为 JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def test_sft_export(self, tmp_path):
        """SFT 导出: 只包含成功轨迹, 按 reward 排序."""
        traj_path = tmp_path / "trajectories.jsonl"
        self._write_trajectories(traj_path, [
            Trajectory(
                task_id="t1", success=True, reward=0.5,
                steps=[{"action": "edit", "thought": "fix it"}],
                metadata={"task_description": "Fix bug"},
            ),
            Trajectory(
                task_id="t2", success=False, reward=0.1,
                steps=[{"action": "bad"}],
                metadata={"task_description": "Another task"},
            ),
            Trajectory(
                task_id="t3", success=True, reward=0.9,
                steps=[{"action": "perfect", "thought": "optimal fix"}],
                metadata={"task_description": "Easy fix"},
            ),
        ])

        exporter = DatasetExporter(str(traj_path))
        result = exporter.export_sft(str(tmp_path / "sft.jsonl"))

        assert result.success is True
        assert result.total_records == 2  # 只有 2 条成功的

        with open(tmp_path / "sft.jsonl") as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 2
        # 按 reward 降序
        assert records[0]["reward"] == 0.9
        assert records[1]["reward"] == 0.5
        # instruction 来自 metadata
        assert records[0]["instruction"] == "Easy fix"

    def test_dpo_export(self, tmp_path):
        """DPO 导出: 将偏好对转为 prompt/chosen/rejected."""
        traj_path = tmp_path / "trajectories.jsonl"
        prefs_path = tmp_path / "preferences.jsonl"

        # 写空轨迹文件（DPO 只需 preferences）
        traj_path.write_text("")
        self._write_preferences(prefs_path, [
            {
                "task_id": "t1",
                "chosen": {
                    "agent_model": "m1",
                    "steps": [{"action": "good", "thought": "smart fix"}],
                    "reward": 0.9,
                },
                "rejected": {
                    "agent_model": "m2",
                    "steps": [{"action": "bad", "thought": "wrong approach"}],
                    "reward": 0.2,
                },
                "reward_margin": 0.7,
            },
        ])

        exporter = DatasetExporter(str(traj_path), str(prefs_path))
        result = exporter.export_dpo(str(tmp_path / "dpo.jsonl"))

        assert result.success is True
        assert result.total_records == 1

        with open(tmp_path / "dpo.jsonl") as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 1
        assert "t1" in records[0]["prompt"]
        assert records[0]["reward_margin"] == 0.7
        assert records[0]["metadata"]["chosen_reward"] == 0.9

    def test_full_pipeline_to_export(self, tmp_path):
        """完整流程: trajectories → preferences → SFT + DPO 导出."""
        # 模拟 pipeline 输出
        config = PipelineConfig(output_dir=str(tmp_path))
        pipeline = Pipeline(config)

        trajectories = [
            Trajectory(
                task_id="full-001", agent_framework="oh", agent_model="claude",
                steps=[{"action": "read"}, {"action": "edit"}],
                total_steps=2, success=True, reward=0.85,
                metadata={"task_description": "Full pipeline test"},
            ),
            Trajectory(
                task_id="full-001", agent_framework="swe", agent_model="gpt",
                steps=[{"action": "read"}],
                total_steps=1, success=True, reward=0.4,
                metadata={"task_description": "Full pipeline test"},
            ),
        ]

        # 保存轨迹
        traj_path = tmp_path / "trajectories.jsonl"
        pipeline._save_trajectories(traj_path, trajectories)

        # 构建偏好对
        prefs_path = tmp_path / "preferences.jsonl"
        pipeline._build_preference_pairs(trajectories, prefs_path)

        # SFT 导出
        exporter = DatasetExporter(str(traj_path), str(prefs_path))
        sft_result = exporter.export_sft(str(tmp_path / "sft.jsonl"))
        assert sft_result.success is True
        assert sft_result.total_records == 2

        # DPO 导出
        dpo_result = exporter.export_dpo(str(tmp_path / "dpo.jsonl"))
        assert dpo_result.success is True
        assert dpo_result.total_records == 1

        # Benchmark 导出
        bench_result = exporter.export_benchmark(str(tmp_path / "benchmark.jsonl"))
        assert bench_result.success is True
        assert bench_result.total_records == 1  # 1 个 task_id

        # 验证 benchmark 内容
        with open(tmp_path / "benchmark.jsonl") as f:
            bench = json.loads(f.readline())
        assert bench["task_id"] == "full-001"
        assert len(bench["reference_trajectories"]) == 2
        assert bench["expected_reward_range"] == [0.4, 0.85]
