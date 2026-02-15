"""测试 data.curriculum — 课程学习采样器."""

from agenttrainer.data.curriculum import CurriculumSampler, compute_difficulties


class TestCurriculumSampler:
    def test_initial_epoch_subset(self):
        """初始 epoch 应该只使用部分数据."""
        difficulties = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        sampler = CurriculumSampler(
            difficulties, num_epochs=5, start_ratio=0.3, warmup_epochs=3
        )
        sampler.set_epoch(0)
        indices = list(sampler)
        assert len(indices) == 3  # 30% of 10

    def test_final_epoch_full_data(self):
        """warmup 后应使用全部数据."""
        difficulties = [1.0, 2.0, 3.0, 4.0, 5.0]
        sampler = CurriculumSampler(
            difficulties, num_epochs=5, start_ratio=0.3, warmup_epochs=2
        )
        sampler.set_epoch(3)  # 超过 warmup_epochs
        indices = list(sampler)
        assert len(indices) == 5

    def test_progressive_increase(self):
        """epoch 增加时数据量应逐步增大."""
        difficulties = list(range(100))
        sampler = CurriculumSampler(
            difficulties, num_epochs=10, start_ratio=0.2, warmup_epochs=5
        )
        sizes = []
        for epoch in range(6):
            sampler.set_epoch(epoch)
            sizes.append(len(sampler))
        # 应该是非递减的
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_sorted_by_difficulty(self):
        """初始 epoch 应该使用最简单的样本."""
        difficulties = [5.0, 1.0, 3.0, 2.0, 4.0]
        sampler = CurriculumSampler(
            difficulties, num_epochs=5, start_ratio=0.4, warmup_epochs=3
        )
        sampler.set_epoch(0)
        indices = list(sampler)
        # 选出的索引应该都是低难度的样本
        # start_ratio=0.4, 5 个样本 → 使用 2 个
        assert len(indices) == 2
        # 应该包含 index 1 (难度 1.0) 和 index 3 (难度 2.0)
        selected_difficulties = sorted([difficulties[i] for i in indices])
        assert selected_difficulties == [1.0, 2.0]

    def test_len_matches_iter(self):
        """__len__ 应该和 __iter__ 返回的数量一致."""
        difficulties = [3.0, 1.0, 2.0, 5.0, 4.0]
        sampler = CurriculumSampler(
            difficulties, num_epochs=3, start_ratio=0.5, warmup_epochs=2
        )
        for epoch in range(3):
            sampler.set_epoch(epoch)
            assert len(sampler) == len(list(sampler))


class TestComputeDifficulties:
    def test_basic(self):
        records = [
            {"reward": 0.9, "metadata": {"total_steps": 2}},
            {"reward": 0.3, "metadata": {"total_steps": 5}},
            {"reward": 0.6, "metadata": {"total_steps": 3}},
        ]
        diffs = compute_difficulties(records)
        assert len(diffs) == 3
        # 高 reward + 少步骤 = 低难度
        assert diffs[0] < diffs[1]  # 0.9 reward, 2 steps vs 0.3 reward, 5 steps

    def test_structured_steps(self):
        """支持 steps 字段."""
        records = [
            {"reward": 0.5, "steps": [{"thought": "a"}, {"thought": "b"}]},
            {"reward": 0.5, "steps": [{"thought": "a"}]},
        ]
        diffs = compute_difficulties(records)
        assert diffs[0] > diffs[1]  # 更多步骤 = 更高难度

    def test_no_metadata(self):
        """缺少 metadata 时使用默认值."""
        records = [{"reward": 0.5}]
        diffs = compute_difficulties(records)
        assert len(diffs) == 1
