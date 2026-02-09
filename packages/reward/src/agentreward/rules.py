"""Rule-based reward functions for agent trajectory evaluation.

Each function takes trajectory data and returns scores in [0.0, 1.0].
These functions handle the "rule" evaluator layer of the 3-layer architecture.

通过 DomainProfile / ToolClassifier 实现领域无关的工具分类，
不再硬编码工具名。
"""

from __future__ import annotations

from typing import Any

from knowlyrcore.domain import (
    CODING_PROFILE,
    DomainProfile,
    OutcomeSpec,
    ToolCategory,
    ToolSpec,
)


class ToolClassifier:
    """从 DomainProfile 构建 tool→category 映射，替代硬编码工具名集合.

    当 profile 无工具定义时（GENERIC_PROFILE），所有分类查询返回 False，
    规则层退化为"完全相同调用才算冗余"的启发式模式。
    """

    def __init__(self, profile: DomainProfile | None = None):
        self._profile = profile or CODING_PROFILE
        self._map: dict[str, ToolSpec] = {}
        for spec in self._profile.tools:
            self._map[spec.name] = spec
            for alias in spec.aliases:
                self._map[alias] = spec

    def category(self, tool_name: str) -> ToolCategory | None:
        """获取工具的功能分类."""
        spec = self._map.get(tool_name)
        return spec.category if spec else None

    def is_read(self, tool_name: str) -> bool:
        """是否为读取/搜索类工具."""
        cat = self.category(tool_name)
        return cat in (ToolCategory.READ, ToolCategory.SEARCH)

    def is_write(self, tool_name: str) -> bool:
        """是否为写入/修改类工具."""
        return self.category(tool_name) == ToolCategory.WRITE

    def is_execute(self, tool_name: str) -> bool:
        """是否为执行类工具."""
        return self.category(tool_name) == ToolCategory.EXECUTE

    def stateful_key(self, tool_name: str) -> str:
        """获取标识操作目标的参数名 (如 file_path / url)."""
        spec = self._map.get(tool_name)
        return spec.stateful_key if spec else ""

    def target_param(self, tool_name: str, params: dict[str, Any]) -> str:
        """从参数中提取操作目标值."""
        key = self.stateful_key(tool_name)
        if key and key in params:
            return str(params[key])
        # 回退: 常见参数名
        for fallback in ("file_path", "path", "url", "element_id"):
            if fallback in params:
                return str(params[fallback])
        return ""


def _normalize_step(step: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    """从步骤字典中提取 tool/params/output，兼容两种格式.

    格式 1（简单）: {tool, params, output}
    格式 2（recorder）: {tool_call: {name, parameters}, tool_result: {output}}

    Returns:
        (tool_name, params_dict, output_str)
    """
    # 格式 1
    tool = step.get("tool", "")
    params = step.get("params", {})
    output = step.get("output", "")

    # 格式 2: recorder 转换输出
    if not tool and "tool_call" in step:
        tc = step["tool_call"]
        tool = tc.get("name", "")
        params = tc.get("parameters", {})
    if not output and "tool_result" in step:
        tr = step["tool_result"]
        output = tr.get("output", "")

    return tool, params, output


def check_redundancy(
    steps: list[dict[str, Any]],
    profile: DomainProfile | None = None,
) -> list[float]:
    """检测轨迹中的冗余操作.

    检查:
    - 对同一目标的重复读取（中间无写入）
    - 完全相同参数的重复调用
    - 写入后又恢复的模式

    Args:
        steps: 步骤列表
        profile: 领域配置（默认 CODING_PROFILE）

    Returns:
        每步评分列表，1.0 = 非冗余，0.0 = 完全冗余
    """
    if not steps:
        return []

    classifier = ToolClassifier(profile)
    scores = []
    seen_calls: list[tuple[str, str]] = []

    for i, step in enumerate(steps):
        tool, params, _output = _normalize_step(step)
        score = 1.0

        param_key = _normalize_params(params)
        call_key = (tool, param_key)

        is_read = classifier.is_read(tool)
        edited_since_last_read = False

        if is_read and i > 0:
            target = classifier.target_param(tool, params)
            if target:
                last_edit_idx = _find_last_write_index(
                    steps[:i], target, classifier
                )
                last_read_idx = _find_last_read_index(
                    steps[:i], target, classifier
                )
                if last_edit_idx is not None and (
                    last_read_idx is None or last_edit_idx > last_read_idx
                ):
                    edited_since_last_read = True

        if call_key in seen_calls and not edited_since_last_read:
            if is_read:
                score = 0.0
            else:
                score = 0.3

        if is_read and i > 0 and not edited_since_last_read:
            target = classifier.target_param(tool, params)
            if target:
                last_read_idx = _find_last_read_index(
                    steps[:i], target, classifier
                )
                if last_read_idx is not None:
                    score = 0.0

        scores.append(score)
        seen_calls.append(call_key)

    return scores


def check_efficiency(steps: list[dict[str, Any]], reference_steps: int) -> float:
    """基于步数比计算效率分.

    Args:
        steps: 步骤列表
        reference_steps: 参考步数

    Returns:
        [0.0, 1.0]，1.0 = 步数 <= 参考
    """
    if reference_steps <= 0:
        return 1.0
    actual_steps = len(steps)
    if actual_steps == 0:
        return 0.0
    ratio = reference_steps / actual_steps
    return min(1.0, max(0.0, ratio))


def check_outcome(
    outcome: dict[str, Any],
    outcome_spec: OutcomeSpec | None = None,
) -> float:
    """基于任务结果评分.

    通过 OutcomeSpec 定义如何提取成功/分数字段，支持任意领域。
    默认兼容 coding 领域的 tests_passed/tests_total。

    Args:
        outcome: 结果字典
        outcome_spec: 结果判定规格（默认使用 CODING_PROFILE 的 OutcomeSpec）

    Returns:
        [0.0, 1.0]
    """
    if not outcome:
        return 0.0

    spec = outcome_spec or CODING_PROFILE.outcome_spec

    # 手动部分得分
    if spec.partial_credit_field and spec.partial_credit_field in outcome:
        return max(0.0, min(1.0, float(outcome[spec.partial_credit_field])))

    # 分数/总量比例
    if spec.score_field and spec.total_field:
        score_val = outcome.get(spec.score_field, 0)
        total_val = outcome.get(spec.total_field, 0)
        if total_val > 0:
            return score_val / total_val

    # 布尔成功
    if spec.success_field and outcome.get(spec.success_field, False):
        return 1.0

    return 0.0


def check_regression(
    steps: list[dict[str, Any]],
    profile: DomainProfile | None = None,
) -> list[float]:
    """检测写入-撤销模式（回归）.

    Args:
        steps: 步骤列表
        profile: 领域配置

    Returns:
        每步评分，1.0 = 无回归，0.0 = 完全撤销
    """
    if not steps:
        return []

    classifier = ToolClassifier(profile)
    scores = [1.0] * len(steps)
    edits_history: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        tool, params, _output = _normalize_step(step)

        if classifier.is_write(tool):
            target = classifier.target_param(tool, params)
            old_string = params.get("old_string", "")
            new_string = params.get("new_string", "")

            if old_string and new_string:
                for prev_edit in edits_history:
                    if (
                        prev_edit["target"] == target
                        and prev_edit["old_string"] == new_string
                        and prev_edit["new_string"] == old_string
                    ):
                        scores[i] = 0.0
                        break

                edits_history.append({
                    "target": target,
                    "old_string": old_string,
                    "new_string": new_string,
                    "step_index": i,
                })

    return scores


def check_info_utilization(
    steps: list[dict[str, Any]],
    profile: DomainProfile | None = None,
) -> list[float]:
    """检查每步是否利用了之前获取的信息.

    Args:
        steps: 步骤列表
        profile: 领域配置

    Returns:
        每步评分，1.0 = 良好利用，低分 = 浪费
    """
    if not steps:
        return []

    classifier = ToolClassifier(profile)
    scores = []
    collected_info: set[str] = set()

    for i, step in enumerate(steps):
        if i == 0:
            scores.append(1.0)
        else:
            tool, params, _step_output = _normalize_step(step)
            score = 1.0

            # 读取/搜索类工具是在收集信息 — 总是 OK
            if classifier.is_read(tool):
                score = 1.0
            # 写入/执行类工具 — 检查是否引用了之前的信息
            elif classifier.is_write(tool) or classifier.is_execute(tool):
                params_str = str(params).lower()
                has_reference = False
                for info_piece in collected_info:
                    if info_piece.lower() in params_str:
                        has_reference = True
                        break
                if collected_info and not has_reference:
                    score = 0.7

            scores.append(score)

        _tool, _params, output = _normalize_step(step)
        if output:
            for token in _extract_info_tokens(output):
                collected_info.add(token)

    return scores


# --- Helper functions ---


def _normalize_params(params: dict[str, Any]) -> str:
    """参数归一化为字符串 key."""
    sorted_items = sorted(params.items())
    return str(sorted_items)


def _find_last_read_index(
    steps: list[dict[str, Any]],
    target: str,
    classifier: ToolClassifier,
) -> int | None:
    """查找最后一次读取指定目标的步骤索引."""
    for i in range(len(steps) - 1, -1, -1):
        tool, params, _output = _normalize_step(steps[i])
        if classifier.is_read(tool):
            step_target = classifier.target_param(tool, params)
            if step_target == target:
                return i
    return None


def _find_last_write_index(
    steps: list[dict[str, Any]],
    target: str,
    classifier: ToolClassifier,
) -> int | None:
    """查找最后一次写入指定目标的步骤索引."""
    for i in range(len(steps) - 1, -1, -1):
        tool, params, _output = _normalize_step(steps[i])
        if classifier.is_write(tool):
            step_target = classifier.target_param(tool, params)
            if step_target == target:
                return i
    return None


def _extract_info_tokens(output: str) -> list[str]:
    """从步骤输出中提取关键信息 token."""
    tokens = []
    for word in output.split():
        if "/" in word and len(word) > 3:
            cleaned = word.strip(".,;:()[]{}\"'`")
            if cleaned:
                tokens.append(cleaned)
        if "_" in word and len(word) > 3 and word.isidentifier():
            tokens.append(word)
    return tokens
