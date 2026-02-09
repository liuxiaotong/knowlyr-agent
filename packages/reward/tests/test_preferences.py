"""测试偏好对构建模块."""

from agentreward.preferences import (
    PreferencePair,
    build_preferences,
    preferences_summary,
    preferences_to_dicts,
)


# ── PreferencePair 测试 ──────────────────────────────────────────


class TestPreferencePair:
    """测试 PreferencePair 数据类."""

    def test_basic(self):
        """基本构造."""
        pair = PreferencePair(
            task_id="task-1",
            chosen_trajectory_id="traj-a",
            rejected_trajectory_id="traj-b",
            chosen_reward=0.9,
            rejected_reward=0.3,
        )
        assert pair.task_id == "task-1"
        assert pair.chosen_reward == 0.9

    def test_margin(self):
        """margin 应为分数差."""
        pair = PreferencePair(
            task_id="t", chosen_trajectory_id="a", rejected_trajectory_id="b",
            chosen_reward=0.8, rejected_reward=0.3,
        )
        assert abs(pair.margin() - 0.5) < 0.001

    def test_to_dict(self):
        """to_dict 应包含所有字段和 margin."""
        pair = PreferencePair(
            task_id="t1", chosen_trajectory_id="a", rejected_trajectory_id="b",
            chosen_reward=0.9, rejected_reward=0.4, rationale="chosen is better",
        )
        d = pair.to_dict()
        assert d["task_id"] == "t1"
        assert d["margin"] == 0.5
        assert d["rationale"] == "chosen is better"

    def test_to_dict_rounds(self):
        """分数应四舍五入到 4 位."""
        pair = PreferencePair(
            task_id="t", chosen_trajectory_id="a", rejected_trajectory_id="b",
            chosen_reward=0.33333, rejected_reward=0.11111,
        )
        d = pair.to_dict()
        assert d["chosen_reward"] == 0.3333
        assert d["rejected_reward"] == 0.1111


# ── build_preferences 测试 ───────────────────────────────────────


class TestBuildPreferences:
    """测试偏好对构建."""

    def test_basic_pairing(self):
        """2 条轨迹应产生 1 对."""
        by_task = {
            "task-1": [
                {"id": "a", "reward": 0.9},
                {"id": "b", "reward": 0.3},
            ],
        }
        pairs = build_preferences(by_task)
        assert len(pairs) == 1
        assert pairs[0].chosen_trajectory_id == "a"
        assert pairs[0].rejected_trajectory_id == "b"

    def test_multiple_pairs(self):
        """4 条轨迹应产生 2 对 (外向内配对)."""
        by_task = {
            "task-1": [
                {"id": "a", "reward": 0.9},
                {"id": "b", "reward": 0.7},
                {"id": "c", "reward": 0.5},
                {"id": "d", "reward": 0.1},
            ],
        }
        pairs = build_preferences(by_task)
        assert len(pairs) == 2
        # 最好 vs 最差
        assert pairs[0].chosen_trajectory_id == "a"
        assert pairs[0].rejected_trajectory_id == "d"
        # 第二 vs 倒二
        assert pairs[1].chosen_trajectory_id == "b"
        assert pairs[1].rejected_trajectory_id == "c"

    def test_single_trajectory_no_pair(self):
        """只有 1 条轨迹不应产生偏好对."""
        by_task = {"task-1": [{"id": "a", "reward": 0.9}]}
        pairs = build_preferences(by_task)
        assert len(pairs) == 0

    def test_min_margin_filter(self):
        """低于 min_margin 的对应被过滤."""
        by_task = {
            "task-1": [
                {"id": "a", "reward": 0.51},
                {"id": "b", "reward": 0.50},
            ],
        }
        # 默认 min_margin=0.05, margin=0.01 应被过滤
        pairs = build_preferences(by_task, min_margin=0.05)
        assert len(pairs) == 0

    def test_min_margin_pass(self):
        """达到 min_margin 的对应保留."""
        by_task = {
            "task-1": [
                {"id": "a", "reward": 0.9},
                {"id": "b", "reward": 0.3},
            ],
        }
        pairs = build_preferences(by_task, min_margin=0.5)
        assert len(pairs) == 1

    def test_multi_task(self):
        """多个任务应独立配对."""
        by_task = {
            "task-1": [
                {"id": "a1", "reward": 0.9},
                {"id": "b1", "reward": 0.3},
            ],
            "task-2": [
                {"id": "a2", "reward": 0.8},
                {"id": "b2", "reward": 0.2},
            ],
        }
        pairs = build_preferences(by_task)
        assert len(pairs) == 2
        task_ids = {p.task_id for p in pairs}
        assert task_ids == {"task-1", "task-2"}

    def test_empty_input(self):
        """空输入应返回空列表."""
        assert build_preferences({}) == []

    def test_auto_id_generation(self):
        """缺少 id 字段应自动生成."""
        by_task = {
            "task-1": [
                {"reward": 0.9},
                {"reward": 0.3},
            ],
        }
        pairs = build_preferences(by_task)
        assert len(pairs) == 1
        assert pairs[0].chosen_trajectory_id == "traj_0"
        assert pairs[0].rejected_trajectory_id == "traj_1"

    def test_rationale_generated(self):
        """rationale 应包含分数信息."""
        by_task = {
            "task-1": [
                {"id": "a", "reward": 0.9},
                {"id": "b", "reward": 0.3},
            ],
        }
        pairs = build_preferences(by_task)
        assert "0.900" in pairs[0].rationale
        assert "0.300" in pairs[0].rationale

    def test_rationale_step_count(self):
        """rationale 应包含步数比较."""
        by_task = {
            "task-1": [
                {"id": "a", "reward": 0.9, "step_count": 3},
                {"id": "b", "reward": 0.3, "step_count": 10},
            ],
        }
        pairs = build_preferences(by_task)
        assert "fewer steps" in pairs[0].rationale

    def test_rationale_outcome(self):
        """rationale 应包含 outcome 比较."""
        by_task = {
            "task-1": [
                {"id": "a", "reward": 0.9, "outcome_score": 1.0},
                {"id": "b", "reward": 0.3, "outcome_score": 0.0},
            ],
        }
        pairs = build_preferences(by_task)
        assert "better task outcome" in pairs[0].rationale


# ── preferences_to_dicts 测试 ────────────────────────────────────


class TestPreferencesToDicts:
    """测试序列化."""

    def test_to_dicts(self):
        """应返回 dict 列表."""
        pairs = [
            PreferencePair(
                task_id="t1", chosen_trajectory_id="a",
                rejected_trajectory_id="b",
                chosen_reward=0.9, rejected_reward=0.3,
            ),
        ]
        dicts = preferences_to_dicts(pairs)
        assert len(dicts) == 1
        assert dicts[0]["task_id"] == "t1"

    def test_empty(self):
        """空列表应返回空列表."""
        assert preferences_to_dicts([]) == []


# ── preferences_summary 测试 ─────────────────────────────────────


class TestPreferencesSummary:
    """测试统计汇总."""

    def test_basic_summary(self):
        """基本统计."""
        pairs = [
            PreferencePair(
                task_id="t1", chosen_trajectory_id="a",
                rejected_trajectory_id="b",
                chosen_reward=0.9, rejected_reward=0.3,
            ),
            PreferencePair(
                task_id="t2", chosen_trajectory_id="c",
                rejected_trajectory_id="d",
                chosen_reward=0.8, rejected_reward=0.2,
            ),
        ]
        s = preferences_summary(pairs)
        assert s["total_pairs"] == 2
        assert s["unique_tasks"] == 2
        assert abs(s["avg_margin"] - 0.6) < 0.001
        assert s["min_margin"] == 0.6
        assert s["max_margin"] == 0.6

    def test_empty_summary(self):
        """空偏好对列表."""
        s = preferences_summary([])
        assert s["total_pairs"] == 0
        assert s["unique_tasks"] == 0
        assert s["avg_margin"] == 0.0

    def test_same_task_multiple_pairs(self):
        """同任务多对."""
        pairs = [
            PreferencePair(
                task_id="t1", chosen_trajectory_id="a",
                rejected_trajectory_id="d",
                chosen_reward=0.9, rejected_reward=0.1,
            ),
            PreferencePair(
                task_id="t1", chosen_trajectory_id="b",
                rejected_trajectory_id="c",
                chosen_reward=0.7, rejected_reward=0.3,
            ),
        ]
        s = preferences_summary(pairs)
        assert s["total_pairs"] == 2
        assert s["unique_tasks"] == 1  # 同一个 task
        assert s["min_margin"] == 0.4
        assert s["max_margin"] == 0.8
