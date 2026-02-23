"""AntgatherBridge — gym CAS <-> 蚁聚数据集双向同步."""

import json
import logging
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

from trajectoryhub.cas import CAStore
from trajectoryhub.sanitizer import sanitize_trajectory

logger = logging.getLogger(__name__)


@dataclass
class PushResult:
    """push_trajectories 的结果."""

    pushed: int = 0
    skipped: int = 0
    errors: int = 0
    sanitized_data: list[dict] = field(default_factory=list)


@dataclass
class PullResult:
    """pull_judgments 的结果."""

    pulled: int = 0
    dpo_pairs: int = 0


class AntgatherBridge:
    """gym CAS <-> 蚁聚的胶水层."""

    def __init__(
        self,
        store: CAStore,
        base_url: str | None = None,
        token: str | None = None,
        dataset_id: str | None = None,
    ):
        self.store = store
        self.base_url = (
            base_url
            or os.environ.get("ANTGATHER_BASE_URL", "https://antgather.knowlyr.com")
        ).rstrip("/")
        self.token = token or os.environ.get("ANTGATHER_TOKEN", "")
        self.dataset_id = dataset_id or os.environ.get("ANTGATHER_DATASET_ID", "")

    def _request(self, method: str, path: str, data: dict | None = None) -> dict:
        """发送 HTTP 请求到蚁聚 API."""
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        body = json.dumps(data, ensure_ascii=False).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            logger.error(
                "蚁聚 API 错误: %s %s -> %d: %s", method, path, e.code, error_body
            )
            raise
        except urllib.error.URLError as e:
            logger.error("蚁聚 API 连接失败: %s %s -> %s", method, path, e.reason)
            raise

    # -- gym -> 蚁聚：轨迹同步 --

    def push_trajectories(
        self, since: float | None = None, limit: int = 100
    ) -> PushResult:
        """CAS 新增轨迹 -> 脱敏 -> 更新蚁聚数据集统计.

        Args:
            since: Unix timestamp，只推送此时间之后的轨迹。None = 全部。
            limit: 单次最多推送条数。

        Returns:
            PushResult 统计。
        """
        result = PushResult()

        # 从 CAS 查询轨迹
        rows = self.store.query_trajectories(since=since, limit=limit)

        if not rows:
            logger.info("没有需要推送的轨迹")
            return result

        # 按 employee 分组统计
        employee_counts: dict[str, int] = {}
        for row in rows:
            try:
                traj = json.loads(row["data"])
                # 脱敏
                sanitized = sanitize_trajectory(traj)
                result.sanitized_data.append(sanitized)
                employee = row["employee"] or "unknown"
                employee_counts[employee] = employee_counts.get(employee, 0) + 1
                result.pushed += 1
            except Exception as e:
                logger.warning("脱敏失败 (hash=%s): %s", row["content_hash"], e)
                result.errors += 1

        # 更新蚁聚数据集统计
        if self.dataset_id and result.pushed > 0:
            try:
                logger.info(
                    "推送统计: %d 条轨迹, 涉及 %d 位员工 -> 数据集 %s",
                    result.pushed,
                    len(employee_counts),
                    self.dataset_id,
                )
            except Exception as e:
                logger.warning("更新蚁聚统计失败: %s", e)

        return result

    # -- 蚁聚 -> gym：判断结果回流 --

    def pull_judgments(self, status: str = "closed", limit: int = 50) -> PullResult:
        """蚁聚判断大厅结果 -> DPO 训练对.

        Args:
            status: 筛选状态（closed = 已完成的判断）
            limit: 单次最多拉取条数

        Returns:
            PullResult 统计。
        """
        result = PullResult()

        if not self.dataset_id:
            logger.warning("未配置 dataset_id，跳过判断拉取")
            return result

        try:
            params = urlencode({
                "dataset_id": self.dataset_id,
                "status": status,
                "limit": limit,
            })
            resp = self._request(
                "GET",
                f"/api/judgments?{params}",
            )
        except Exception:
            return result

        judgments = resp.get("data", resp.get("judgments", []))
        if not judgments:
            logger.info("没有已完成的判断")
            return result

        dpo_pairs = []
        for j in judgments:
            jid = j.get("id")
            try:
                detail = self._request("GET", f"/api/judgments/{jid}/result")
            except Exception:
                continue

            result.pulled += 1

            # 转为 DPO 格式：majority = chosen，minority = rejected
            majority = detail.get("majority", {})
            if majority.get("consensus") and detail.get("total_responses", 0) >= 2:
                dpo_pair = {
                    "judgment_id": jid,
                    "title": detail.get("title", ""),
                    "chosen_index": majority.get("index"),
                    "chosen_label": majority.get("label"),
                    "votes": majority.get("votes"),
                    "total_responses": detail.get("total_responses"),
                    "distribution": detail.get("distribution", []),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                dpo_pairs.append(dpo_pair)
                result.dpo_pairs += 1

        if dpo_pairs:
            logger.info(
                "拉取 %d 个判断, 生成 %d 个 DPO 训练对",
                result.pulled,
                result.dpo_pairs,
            )

        return result

    # -- 发起判断 --

    def create_judgment(
        self,
        title: str,
        description: str,
        options: list[str],
        reward: int = 5,
        max_answers: int = 3,
        requester_id: int = 1,
    ) -> dict | None:
        """创建判断请求.

        Args:
            title: 问题标题（<=200字）
            description: 详细描述
            options: 选项列表（2-10个）
            reward: 每个回答的光粒奖励
            max_answers: 最多接受回答数
            requester_id: 提问者 user_id

        Returns:
            API 响应 dict，或 None（失败时）
        """
        if not self.dataset_id:
            logger.warning("未配置 dataset_id，无法创建判断")
            return None

        data = {
            "requester_id": requester_id,
            "title": title[:200],
            "description": description[:2000],
            "options": options[:10],
            "reward": reward,
            "max_answers": max_answers,
            "dataset_id": self.dataset_id,
            "contribution_type": "review",
        }

        try:
            resp = self._request("POST", "/api/judgments", data)
            logger.info(
                "创建判断请求: id=%s, 冻结=%s光粒",
                resp.get("id"),
                resp.get("frozen"),
            )
            return resp
        except Exception:
            return None
