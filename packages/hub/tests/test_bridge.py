"""AntgatherBridge 测试."""

import json
from unittest.mock import patch

from trajectoryhub.bridge import AntgatherBridge
from trajectoryhub.cas import CAStore


def _make_store_with_data(tmp_path, trajectories=None):
    """创建一个带数据的 CAStore."""
    store = CAStore(tmp_path / "data" / "cas.sqlite")
    if trajectories:
        for traj in trajectories:
            store._conn.execute(
                """INSERT INTO trajectories
                   (content_hash, task_id, agent_framework, agent_model,
                    total_steps, success, reward, created_at, data,
                    employee, source, domain)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    traj.get("content_hash", "abc123"),
                    traj.get("task_id", "t1"),
                    traj.get("agent_framework", "crew"),
                    traj.get("agent_model", "claude-sonnet"),
                    traj.get("total_steps", 1),
                    1 if traj.get("success", True) else 0,
                    traj.get("reward", 0.5),
                    traj.get("created_at", 1000000.0),
                    json.dumps(traj.get("data", {"steps": [], "metadata": {}})),
                    traj.get("employee", "backend-engineer"),
                    traj.get("source", "claude-code"),
                    traj.get("domain", "crew"),
                ),
            )
        store._conn.commit()
    return store


class TestAntgatherBridgeInit:
    """AntgatherBridge 初始化测试."""

    def test_init_with_params(self, tmp_path):
        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(
            store=store,
            base_url="https://test.example.com",
            token="test-token",
            dataset_id="DS001",
        )
        assert bridge.base_url == "https://test.example.com"
        assert bridge.token == "test-token"
        assert bridge.dataset_id == "DS001"
        store.close()

    def test_init_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTGATHER_BASE_URL", "https://env.example.com/")
        monkeypatch.setenv("ANTGATHER_TOKEN", "env-token")
        monkeypatch.setenv("ANTGATHER_DATASET_ID", "DS_ENV")

        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store)
        # rstrip("/") 应移除尾部斜杠
        assert bridge.base_url == "https://env.example.com"
        assert bridge.token == "env-token"
        assert bridge.dataset_id == "DS_ENV"
        store.close()

    def test_init_defaults(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTGATHER_BASE_URL", raising=False)
        monkeypatch.delenv("ANTGATHER_TOKEN", raising=False)
        monkeypatch.delenv("ANTGATHER_DATASET_ID", raising=False)

        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store)
        assert bridge.base_url == "https://antgather.knowlyr.com"
        assert bridge.token == ""
        assert bridge.dataset_id == ""
        store.close()


class TestPushTrajectories:
    """push_trajectories 测试."""

    def test_empty_cas(self, tmp_path):
        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="DS001")
        result = bridge.push_trajectories()
        assert result.pushed == 0
        assert result.skipped == 0
        assert result.errors == 0
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_push_with_data(self, mock_request, tmp_path):
        mock_request.return_value = {"ok": True, "id": "DS001", "updated": ["sample_count", "contributor_count"]}
        store = _make_store_with_data(
            tmp_path,
            [
                {
                    "content_hash": "hash1",
                    "task_id": "t1",
                    "employee": "backend-engineer",
                    "created_at": 1000000.0,
                    "data": {
                        "steps": [{"step_id": 1, "tool": "Bash", "output": "ok"}],
                        "metadata": {},
                    },
                },
                {
                    "content_hash": "hash2",
                    "task_id": "t2",
                    "employee": "frontend-engineer",
                    "created_at": 1000001.0,
                    "data": {
                        "steps": [{"step_id": 1, "tool": "Read", "output": "done"}],
                        "metadata": {},
                    },
                },
            ],
        )
        bridge = AntgatherBridge(store=store, dataset_id="DS001")
        result = bridge.push_trajectories()
        assert result.pushed == 2
        assert result.errors == 0
        assert len(result.sanitized_data) == 2
        # 验证调用了蚁聚统计 API
        mock_request.assert_called_once_with(
            "PATCH",
            "/api/projects/DS001/stats",
            {
                "sample_count": 2,
                "contributor_count": 2,
                "covered_categories": 2,
            },
        )
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_push_with_since(self, mock_request, tmp_path):
        mock_request.return_value = {"ok": True, "id": "DS001", "updated": []}
        store = _make_store_with_data(
            tmp_path,
            [
                {
                    "content_hash": "old",
                    "task_id": "t1",
                    "created_at": 900000.0,
                    "data": {"steps": [], "metadata": {}},
                },
                {
                    "content_hash": "new",
                    "task_id": "t2",
                    "created_at": 1100000.0,
                    "data": {"steps": [], "metadata": {}},
                },
            ],
        )
        bridge = AntgatherBridge(store=store, dataset_id="DS001")
        result = bridge.push_trajectories(since=1000000.0)
        assert result.pushed == 1  # 只有 created_at > 1000000.0 的
        # stats API 用 CAS 总量（2 条），不是本次增量（1 条）
        # 两条数据都是默认 employee="backend-engineer"，所以 contributor_count=1
        mock_request.assert_called_once_with(
            "PATCH",
            "/api/projects/DS001/stats",
            {
                "sample_count": 2,
                "contributor_count": 1,
                "covered_categories": 2,
            },
        )
        store.close()

    def test_push_bad_data(self, tmp_path):
        """data 字段为非法 JSON 时计入 errors."""
        store = CAStore(tmp_path / "cas.sqlite")
        store._conn.execute(
            """INSERT INTO trajectories
               (content_hash, task_id, agent_framework, agent_model,
                total_steps, success, reward, created_at, data,
                employee, source, domain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("badhash", "t1", "crew", "model", 1, 1, 0.5, 1000000.0,
             "NOT VALID JSON", "eng", "src", "dom"),
        )
        store._conn.commit()

        bridge = AntgatherBridge(store=store, dataset_id="DS001")
        result = bridge.push_trajectories()
        assert result.errors == 1
        assert result.pushed == 0
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_push_stats_api_failure_does_not_block(self, mock_request, tmp_path):
        """stats API 失败时 warn 但不阻塞推送结果."""
        mock_request.side_effect = Exception("503 service unavailable")
        store = _make_store_with_data(
            tmp_path,
            [
                {
                    "content_hash": "hash1",
                    "task_id": "t1",
                    "employee": "backend-engineer",
                    "created_at": 1000000.0,
                    "data": {"steps": [], "metadata": {}},
                },
            ],
        )
        bridge = AntgatherBridge(store=store, dataset_id="DS001")
        result = bridge.push_trajectories()
        # 推送本身成功
        assert result.pushed == 1
        assert result.errors == 0
        # stats API 被调用了（虽然失败）
        mock_request.assert_called_once()
        store.close()

    def test_push_no_dataset_id_skips_stats(self, tmp_path):
        """没有 dataset_id 时不调用 stats API."""
        store = _make_store_with_data(
            tmp_path,
            [
                {
                    "content_hash": "hash1",
                    "task_id": "t1",
                    "employee": "backend-engineer",
                    "created_at": 1000000.0,
                    "data": {"steps": [], "metadata": {}},
                },
            ],
        )
        bridge = AntgatherBridge(store=store, dataset_id="")
        with patch.object(bridge, "_request") as mock_req:
            result = bridge.push_trajectories()
            assert result.pushed == 1
            mock_req.assert_not_called()
        store.close()


class TestPullJudgments:
    """pull_judgments 测试."""

    def test_no_dataset_id(self, tmp_path):
        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="")
        result = bridge.pull_judgments()
        assert result.pulled == 0
        assert result.dpo_pairs == 0
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_pull_empty_judgments(self, mock_request, tmp_path):
        mock_request.return_value = {"data": []}
        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="DS001", token="tok")
        result = bridge.pull_judgments()
        assert result.pulled == 0
        assert result.dpo_pairs == 0
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_pull_with_consensus(self, mock_request, tmp_path):
        """有共识的判断应生成 DPO 对."""
        mock_request.side_effect = [
            # 第一次调用：列表
            {
                "data": [
                    {"id": 42, "title": "哪个回答更好？"},
                ]
            },
            # 第二次调用：详情
            {
                "id": 42,
                "title": "哪个回答更好？",
                "status": "closed",
                "total_responses": 3,
                "distribution": [
                    {"index": 0, "label": "轨迹A更好", "votes": 2, "percent": 66.7},
                    {"index": 1, "label": "轨迹B更好", "votes": 1, "percent": 33.3},
                ],
                "majority": {
                    "index": 0,
                    "label": "轨迹A更好",
                    "votes": 2,
                    "consensus": True,
                },
            },
        ]

        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="DS001", token="tok")
        result = bridge.pull_judgments()
        assert result.pulled == 1
        assert result.dpo_pairs == 1
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_pull_no_consensus(self, mock_request, tmp_path):
        """无共识的判断不生成 DPO 对."""
        mock_request.side_effect = [
            {"data": [{"id": 99}]},
            {
                "id": 99,
                "total_responses": 3,
                "majority": {"index": 0, "label": "A", "votes": 1, "consensus": False},
            },
        ]

        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="DS001", token="tok")
        result = bridge.pull_judgments()
        assert result.pulled == 1
        assert result.dpo_pairs == 0
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_pull_api_error(self, mock_request, tmp_path):
        """API 异常时 graceful 降级."""
        mock_request.side_effect = Exception("connection refused")

        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="DS001", token="tok")
        result = bridge.pull_judgments()
        assert result.pulled == 0
        assert result.dpo_pairs == 0
        store.close()


class TestCreateJudgment:
    """create_judgment 测试."""

    def test_no_dataset_id(self, tmp_path):
        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="")
        result = bridge.create_judgment(
            title="test", description="desc", options=["A", "B"]
        )
        assert result is None
        store.close()

    def test_title_truncation(self, tmp_path):
        """标题超过 200 字应被截断."""
        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(
            store=store, dataset_id="DS001", token="tok",
            base_url="https://test.example.com",
        )

        long_title = "X" * 300
        long_desc = "Y" * 3000
        many_options = [f"opt{i}" for i in range(15)]

        with patch.object(bridge, "_request") as mock_req:
            mock_req.return_value = {"ok": True, "id": 1, "frozen": 5}
            result = bridge.create_judgment(
                title=long_title,
                description=long_desc,
                options=many_options,
            )

        assert result is not None
        # 验证截断逻辑
        call_data = mock_req.call_args[0][2]
        assert len(call_data["title"]) == 200
        assert len(call_data["description"]) == 2000
        assert len(call_data["options"]) == 10
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_create_api_error(self, mock_request, tmp_path):
        """API 失败时返回 None."""
        mock_request.side_effect = Exception("500 server error")

        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="DS001", token="tok")
        result = bridge.create_judgment(
            title="test", description="desc", options=["A", "B"]
        )
        assert result is None
        store.close()


class TestAutoJudge:
    """auto_judge 自动判断测试."""

    def _make_store_with_uncertainty(self, tmp_path, task_id="t1", rewards=None):
        """创建含不确定性轨迹的 CAS."""
        if rewards is None:
            rewards = [0.4, 0.5, 0.6]  # 都在 (0.3, 0.7) 内
        store = CAStore(tmp_path / "data" / "cas.sqlite")
        for i, r in enumerate(rewards):
            store._conn.execute(
                """INSERT INTO trajectories
                   (content_hash, task_id, agent_framework, agent_model,
                    total_steps, success, reward, created_at, data,
                    employee, source, domain)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"hash_{task_id}_{i}",
                    task_id,
                    "crew",
                    "test-model",
                    2,
                    1,
                    r,
                    1000000.0 + i,
                    json.dumps({
                        "steps": [
                            {"thought": f"思考 {i}", "action": f"action_{i}", "observation": "ok"},
                        ],
                        "metadata": {
                            "task_description": f"测试任务 {task_id}",
                            "employee": "backend-engineer",
                        },
                    }),
                    "backend-engineer",
                    "claude-code",
                    "engineering",
                ),
            )
        store._conn.commit()
        return store

    def test_no_dataset_id(self, tmp_path):
        """未配置 dataset_id 时应返回全零."""
        store = self._make_store_with_uncertainty(tmp_path)
        bridge = AntgatherBridge(store=store, dataset_id="")
        result = bridge.auto_judge()
        assert result["created"] == 0
        assert result["errors"] == 0
        store.close()

    def test_no_candidates(self, tmp_path):
        """没有候选任务时返回 0."""
        store = CAStore(tmp_path / "cas.sqlite")
        bridge = AntgatherBridge(store=store, dataset_id="DS001")
        result = bridge.auto_judge()
        assert result["created"] == 0
        store.close()

    def test_single_task_not_enough_trajectories(self, tmp_path):
        """只有 1 条轨迹的任务不应参与."""
        store = self._make_store_with_uncertainty(tmp_path, rewards=[0.5])
        bridge = AntgatherBridge(store=store, dataset_id="DS001")
        result = bridge.auto_judge()
        assert result["created"] == 0
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_creates_judgments(self, mock_request, tmp_path):
        """有候选时应正确创建判断."""
        mock_request.return_value = {"ok": True, "id": 1, "frozen": 5}
        store = self._make_store_with_uncertainty(tmp_path, rewards=[0.4, 0.5, 0.6])
        bridge = AntgatherBridge(
            store=store, dataset_id="DS001", token="tok",
            base_url="https://test.example.com",
        )
        result = bridge.auto_judge()
        # 3 条轨迹 -> 2 个相邻对
        assert result["created"] == 2
        assert result["errors"] == 0
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_batch_size_limit(self, mock_request, tmp_path):
        """batch_size 应限制最大创建数."""
        mock_request.return_value = {"ok": True, "id": 1, "frozen": 5}
        store = self._make_store_with_uncertainty(
            tmp_path, rewards=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        )
        bridge = AntgatherBridge(
            store=store, dataset_id="DS001", token="tok",
            base_url="https://test.example.com",
        )
        result = bridge.auto_judge(batch_size=3)
        assert result["created"] == 3
        store.close()

    @patch.object(AntgatherBridge, "_request")
    def test_api_error_counted(self, mock_request, tmp_path):
        """API 失败应计入 errors."""
        mock_request.side_effect = Exception("boom")
        store = self._make_store_with_uncertainty(tmp_path, rewards=[0.4, 0.5])
        bridge = AntgatherBridge(
            store=store, dataset_id="DS001", token="tok",
            base_url="https://test.example.com",
        )
        result = bridge.auto_judge()
        assert result["errors"] == 1
        assert result["created"] == 0
        store.close()

    def test_rewards_outside_range(self, tmp_path):
        """reward 不在不确定区间内的轨迹不应参与."""
        store = self._make_store_with_uncertainty(tmp_path, rewards=[0.1, 0.9])
        bridge = AntgatherBridge(
            store=store, dataset_id="DS001", token="tok",
            base_url="https://test.example.com",
        )
        result = bridge.auto_judge(reward_uncertainty_range=(0.3, 0.7))
        assert result["created"] == 0
        store.close()
