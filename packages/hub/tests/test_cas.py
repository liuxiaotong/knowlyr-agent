"""CAS 内容寻址存储测试."""

import pytest

from trajectoryhub.cas import CAStore, content_hash
from trajectoryhub.pipeline import Trajectory


class TestContentHash:
    """content_hash() 函数测试."""

    def test_deterministic(self):
        steps = [{"tool": "read", "params": {"path": "/a.py"}, "output": "code"}]
        h1 = content_hash(steps)
        h2 = content_hash(steps)
        assert h1 == h2

    def test_length(self):
        steps = [{"tool": "read", "params": {}, "output": ""}]
        h = content_hash(steps)
        assert len(h) == 16

    def test_hex_string(self):
        h = content_hash([{"tool": "x", "params": {}, "output": "y"}])
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_steps_different_hash(self):
        h1 = content_hash([{"tool": "read", "params": {"path": "/a.py"}, "output": "v1"}])
        h2 = content_hash([{"tool": "read", "params": {"path": "/b.py"}, "output": "v1"}])
        assert h1 != h2

    def test_ignores_thought(self):
        """thought 字段不参与哈希，只看 tool/params/output."""
        steps1 = [{"thought": "思考A", "tool": "read", "params": {}, "output": "x"}]
        steps2 = [{"thought": "思考B", "tool": "read", "params": {}, "output": "x"}]
        assert content_hash(steps1) == content_hash(steps2)

    def test_empty_steps(self):
        h = content_hash([])
        assert len(h) == 16

    def test_missing_fields_handled(self):
        h = content_hash([{"something": "else"}])
        assert len(h) == 16


class TestCAStore:
    """CAStore SQLite 存储测试."""

    @pytest.fixture()
    def store(self, tmp_path):
        s = CAStore(tmp_path / "test.sqlite")
        yield s
        s.close()

    def _make_traj(self, task_id="t1", tool="read", output="code", reward=0.8):
        return Trajectory(
            task_id=task_id,
            agent_framework="test",
            agent_model="gpt-4o",
            steps=[{"tool": tool, "params": {"path": "/a.py"}, "output": output}],
            total_steps=1,
            success=True,
            reward=reward,
            step_rewards=[reward],
            duration_seconds=1.0,
            metadata={"test": True},
        )

    def test_put_and_get(self, store):
        traj = self._make_traj()
        h = store.put(traj)
        assert len(h) == 16

        result = store.get(h)
        assert result is not None
        assert result["task_id"] == "t1"
        assert result["reward"] == 0.8
        assert result["success"] is True
        assert result["steps"] == traj.steps

    def test_put_dedup(self, store):
        """相同步骤只存一次."""
        traj1 = self._make_traj(reward=0.5)
        traj2 = self._make_traj(reward=0.9)
        h1 = store.put(traj1)
        h2 = store.put(traj2)
        assert h1 == h2
        assert store.count() == 1
        # reward 取较高值
        result = store.get(h1)
        assert result["reward"] == 0.9

    def test_different_trajectories_stored_separately(self, store):
        traj1 = self._make_traj(output="code_v1")
        traj2 = self._make_traj(output="code_v2")
        h1 = store.put(traj1)
        h2 = store.put(traj2)
        assert h1 != h2
        assert store.count() == 2

    def test_get_nonexistent(self, store):
        assert store.get("does_not_exist") is None

    def test_list_ordered_by_gdi(self, store):
        t1 = self._make_traj(output="a", reward=0.3)
        t2 = self._make_traj(output="b", reward=0.9)
        h1 = store.put(t1)
        h2 = store.put(t2)
        store.update_gdi(h1, 0.2)
        store.update_gdi(h2, 0.8)

        results = store.list(order_by="gdi_score", limit=10)
        assert len(results) == 2
        assert results[0]["gdi_score"] == 0.8
        assert results[1]["gdi_score"] == 0.2

    def test_list_filter_by_task(self, store):
        store.put(self._make_traj(task_id="t1", output="a"))
        store.put(self._make_traj(task_id="t2", output="b"))
        results = store.list(task_id="t1")
        assert len(results) == 1
        assert results[0]["task_id"] == "t1"

    def test_update_gdi(self, store):
        h = store.put(self._make_traj())
        store.update_gdi(h, 0.72)
        result = store.get(h)
        assert result["gdi_score"] == 0.72

    def test_increment_export(self, store):
        h = store.put(self._make_traj())
        assert store.get(h)["export_count"] == 0
        store.increment_export(h)
        store.increment_export(h)
        assert store.get(h)["export_count"] == 2

    def test_update_gdi_batch(self, store):
        h1 = store.put(self._make_traj(output="a"))
        h2 = store.put(self._make_traj(output="b"))
        store.update_gdi_batch({h1: 0.5, h2: 0.9})
        assert store.get(h1)["gdi_score"] == 0.5
        assert store.get(h2)["gdi_score"] == 0.9

    def test_stats(self, store):
        store.put(self._make_traj(task_id="t1", output="a", reward=0.6))
        store.put(self._make_traj(task_id="t2", output="b", reward=0.8))
        s = store.stats()
        assert s["total_trajectories"] == 2
        assert s["unique_tasks"] == 2
        assert s["avg_reward"] == 0.7

    def test_context_manager(self, tmp_path):
        with CAStore(tmp_path / "ctx.sqlite") as store:
            store.put(self._make_traj())
            assert store.count() == 1

    def test_count(self, store):
        assert store.count() == 0
        store.put(self._make_traj(output="a"))
        store.put(self._make_traj(output="b"))
        assert store.count() == 2
        assert store.count(task_id="t1") == 2
