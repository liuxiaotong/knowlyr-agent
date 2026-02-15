"""测试 data.chunker — 长轨迹分块."""

import pytest
from transformers import AutoTokenizer

from agenttrainer.data.agent_format import AgentStep
from agenttrainer.data.chunker import chunk_trajectory


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


class TestChunkTrajectory:
    def test_short_trajectory_single_chunk(self, tokenizer):
        """短轨迹不需要分块，返回单个 chunk."""
        steps = [
            AgentStep(thought="Read", action="read_file /a.py", observation="code"),
        ]
        chunks = chunk_trajectory("Fix bug", "", steps, tokenizer, max_length=2048)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
        assert len(chunks[0].context_steps) == 0

    def test_long_trajectory_splits(self, tokenizer):
        """长轨迹应该被拆分为多个 chunk."""
        # 创建很长的轨迹
        steps = [
            AgentStep(
                thought=f"Thinking about step {i} " * 20,
                action=f"action_{i} " * 20,
                observation=f"result_{i} " * 20,
                reward=0.5 + i * 0.1,
            )
            for i in range(10)
        ]
        chunks = chunk_trajectory("Task", "", steps, tokenizer, max_length=256)
        assert len(chunks) > 1
        # 所有 chunk 的 total_chunks 应该一致
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)

    def test_chunk_indices_sequential(self, tokenizer):
        """chunk_index 应该是连续递增的."""
        steps = [
            AgentStep(
                thought=f"Think {i} " * 30,
                action=f"act {i} " * 30,
                observation=f"obs {i} " * 30,
            )
            for i in range(8)
        ]
        chunks = chunk_trajectory("Task", "", steps, tokenizer, max_length=256)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_overlap_context(self, tokenizer):
        """后续 chunk 应该包含前一个 chunk 的上下文步骤."""
        steps = [
            AgentStep(
                thought=f"Think {i} " * 30,
                action=f"act {i} " * 30,
                observation=f"obs {i} " * 30,
            )
            for i in range(8)
        ]
        chunks = chunk_trajectory(
            "Task", "", steps, tokenizer, max_length=256, overlap_steps=1
        )
        if len(chunks) > 1:
            # 第二个 chunk 应该有上下文步骤
            assert len(chunks[1].context_steps) > 0

    def test_empty_steps(self, tokenizer):
        """空步骤列表应返回空 chunks."""
        chunks = chunk_trajectory("Task", "", [], tokenizer, max_length=2048)
        assert len(chunks) == 0

    def test_all_steps_covered(self, tokenizer):
        """所有步骤应被覆盖（不丢失）."""
        steps = [
            AgentStep(
                thought=f"Think {i} " * 20,
                action=f"act {i}",
                observation=f"obs {i} " * 20,
                reward=float(i),
            )
            for i in range(5)
        ]
        chunks = chunk_trajectory("Task", "", steps, tokenizer, max_length=256)
        # 收集所有 chunk 的步骤（不含 context）
        all_chunk_steps = []
        for chunk in chunks:
            all_chunk_steps.extend(chunk.steps)
        assert len(all_chunk_steps) == len(steps)
