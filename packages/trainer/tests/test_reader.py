"""测试 data.reader - JSONL 读取."""

from agenttrainer.data.reader import read_sft, read_dpo, read_grpo_groups


class TestReadSFT:
    def test_read_correct_count(self, sft_sample_file):
        records = read_sft(sft_sample_file)
        assert len(records) == 3

    def test_read_fields(self, sft_sample_file):
        records = read_sft(sft_sample_file)
        rec = records[0]
        assert "instruction" in rec
        assert "input" in rec
        assert "response" in rec
        assert "task_id" in rec
        assert "reward" in rec
        assert "metadata" in rec

    def test_read_values(self, sft_sample_file):
        records = read_sft(sft_sample_file)
        assert records[0]["task_id"] == "task-001"
        assert records[0]["reward"] == 0.85

    def test_read_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        records = read_sft(path)
        assert records == []

    def test_skip_blank_lines(self, tmp_path):
        path = tmp_path / "blank.jsonl"
        path.write_text('{"task_id": "t1"}\n\n{"task_id": "t2"}\n')
        records = read_sft(path)
        assert len(records) == 2


class TestReadDPO:
    def test_read_correct_count(self, dpo_sample_file):
        records = read_dpo(dpo_sample_file)
        assert len(records) == 2

    def test_read_fields(self, dpo_sample_file):
        records = read_dpo(dpo_sample_file)
        rec = records[0]
        assert "prompt" in rec
        assert "chosen" in rec
        assert "rejected" in rec
        assert "reward_margin" in rec

    def test_read_values(self, dpo_sample_file):
        records = read_dpo(dpo_sample_file)
        assert records[0]["reward_margin"] == 0.55


class TestReadGRPOGroups:
    def test_read_correct_count(self, grpo_sample_file):
        groups = read_grpo_groups(grpo_sample_file)
        assert len(groups) == 2

    def test_group_structure(self, grpo_sample_file):
        groups = read_grpo_groups(grpo_sample_file)
        grp = groups[0]
        assert "task_id" in grp
        assert "prompt" in grp
        assert "trajectories" in grp
        assert len(grp["trajectories"]) == 3

    def test_trajectory_fields(self, grpo_sample_file):
        groups = read_grpo_groups(grpo_sample_file)
        traj = groups[0]["trajectories"][0]
        assert "response" in traj
        assert "reward" in traj
