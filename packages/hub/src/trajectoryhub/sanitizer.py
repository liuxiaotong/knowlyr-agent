"""Sanitizer — 轨迹数据脱敏（Phase 1: 硬规则正则）.

将轨迹中的敏感信息替换为占位符，返回脱敏文本和审计日志。

两层策略：
  1. 硬规则（本模块）：正则匹配 token/密码/手机号/邮箱/身份证/内网地址
  2. 软规则（Phase 2）：LLM 语义脱敏（公司名/人名/金额等）

用法::

    from trajectoryhub.sanitizer import sanitize, sanitize_trajectory

    clean_text, audit = sanitize("token is sk-abc123def456...")
    clean_traj = sanitize_trajectory(trajectory_dict)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── 脱敏规则定义 ──────────────────────────────────────────────

# 每条规则: (名称, 编译后正则, 替换占位符)
# 顺序很重要：长 token 放前面（避免被子串规则部分匹配）

_RULES: list[tuple[str, re.Pattern, str]] = [
    # API tokens — 已知前缀
    (
        "api_token_prefix",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36,}|gho_[A-Za-z0-9]{36,}"
            r"|github_pat_[A-Za-z0-9_]{22,}|xoxb-[A-Za-z0-9\-]+"
            r"|xoxp-[A-Za-z0-9\-]+|AKIA[A-Z0-9]{16})\b"
        ),
        "[REDACTED_TOKEN]",
    ),
    # password= / token= / secret= / key= 赋值
    (
        "credential_assignment",
        re.compile(
            r"(?:password|passwd|token|secret|api_?key|access_?key|auth)"
            r"\s*[=:]\s*['\"]?([A-Za-z0-9_/+\-.]{8,})['\"]?",
            re.IGNORECASE,
        ),
        "[REDACTED_CREDENTIAL]",
    ),
    # 长随机串（至少 32 字符的 Base64/hex，像 token）
    (
        "long_random_string",
        re.compile(r"\b[A-Za-z0-9_+\-]{32,}\b"),
        "[REDACTED_TOKEN]",
    ),
    # 中国身份证号（18 位）
    (
        "id_card",
        re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"),
        "[REDACTED_ID_CARD]",
    ),
    # 中国手机号
    (
        "phone",
        re.compile(r"\b1[3-9]\d{9}\b"),
        "[REDACTED_PHONE]",
    ),
    # 邮箱
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    # 银行卡号（16-19 位纯数字）
    (
        "bank_card",
        re.compile(r"\b(?:6[0-9]{15,18}|4[0-9]{15,18}|5[1-5][0-9]{14,17}|3[47][0-9]{13,16})\b"),
        "[REDACTED_BANK_CARD]",
    ),
    # 内网 IP (10.x / 172.16-31.x / 192.168.x)
    (
        "internal_ip",
        re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3})\b"
        ),
        "[REDACTED_INTERNAL_IP]",
    ),
    # 内网 URL（包含内网域名关键词）
    (
        "internal_url",
        re.compile(
            r"https?://[A-Za-z0-9.\-]*(?:internal|intranet|local|corp|private"
            r"|staging|dev\.)[A-Za-z0-9.\-]*(?::\d+)?(?:/[^\s]*)?"
        ),
        "[REDACTED_INTERNAL_URL]",
    ),
]


@dataclass
class AuditEntry:
    """单条脱敏审计记录."""

    rule_name: str
    original: str
    replacement: str
    position: int  # 在原文中的起始位置


@dataclass
class SanitizeResult:
    """脱敏结果."""

    text: str
    audit_log: list[AuditEntry] = field(default_factory=list)

    @property
    def redacted_count(self) -> int:
        return len(self.audit_log)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "redacted_count": self.redacted_count,
            "audit_log": [
                {
                    "rule": e.rule_name,
                    "original": e.original,
                    "replacement": e.replacement,
                    "position": e.position,
                }
                for e in self.audit_log
            ],
        }


# ── 脱敏入口 ──────────────────────────────────────────────────


def sanitize(text: str) -> SanitizeResult:
    """对文本应用硬规则脱敏.

    Args:
        text: 原始文本.

    Returns:
        SanitizeResult 包含脱敏后文本和审计日志.
    """
    if not text:
        return SanitizeResult(text="")

    audit_log: list[AuditEntry] = []
    result = text

    for rule_name, pattern, replacement in _RULES:
        new_result = ""
        last_end = 0
        for match in pattern.finditer(result):
            if rule_name == "credential_assignment" and match.lastindex:
                # 只替换捕获组（密码值），保留 password= 等前缀
                group_start = match.start(1)
                group_end = match.end(1)
                audit_log.append(
                    AuditEntry(
                        rule_name=rule_name,
                        original=match.group(1),
                        replacement=replacement,
                        position=group_start,
                    )
                )
                new_result += result[last_end:group_start] + replacement
                last_end = group_end
            else:
                audit_log.append(
                    AuditEntry(
                        rule_name=rule_name,
                        original=match.group(0),
                        replacement=replacement,
                        position=match.start(),
                    )
                )
                new_result += result[last_end : match.start()] + replacement
                last_end = match.end()
        new_result += result[last_end:]
        result = new_result

    return SanitizeResult(text=result, audit_log=audit_log)


def sanitize_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """对整条轨迹的文本字段做脱敏.

    递归处理 steps 中的 thought/output/params 等文本字段。

    Args:
        trajectory: 原始轨迹 dict（不会被修改，返回深拷贝）.

    Returns:
        脱敏后的轨迹 dict，在 metadata 中附加 sanitize_audit。
    """
    traj = copy.deepcopy(trajectory)
    total_audit: list[dict[str, Any]] = []

    # 脱敏 steps
    for step in traj.get("steps", []):
        # thought
        if "thought" in step and isinstance(step["thought"], str):
            r = sanitize(step["thought"])
            step["thought"] = r.text
            total_audit.extend(
                {"step_id": step.get("step_id"), "field": "thought", **e}
                for e in _audit_entries_to_dicts(r.audit_log)
            )

        # tool_call.parameters — 递归脱敏字符串值
        tool_call = step.get("tool_call", {})
        if isinstance(tool_call, dict):
            params = tool_call.get("parameters", {})
            if isinstance(params, dict):
                _sanitize_dict_values(params, total_audit, step.get("step_id"), "params")

        # tool_result.output
        tool_result = step.get("tool_result", {})
        if isinstance(tool_result, dict) and "output" in tool_result:
            if isinstance(tool_result["output"], str):
                r = sanitize(tool_result["output"])
                tool_result["output"] = r.text
                total_audit.extend(
                    {"step_id": step.get("step_id"), "field": "tool_output", **e}
                    for e in _audit_entries_to_dicts(r.audit_log)
                )

        # 兼容扁平格式（tool/params/output 直接在 step 上）
        if "output" in step and isinstance(step["output"], str):
            r = sanitize(step["output"])
            step["output"] = r.text
            total_audit.extend(
                {"step_id": step.get("step_id"), "field": "output", **e}
                for e in _audit_entries_to_dicts(r.audit_log)
            )
        if "params" in step and isinstance(step["params"], dict):
            _sanitize_dict_values(step["params"], total_audit, step.get("step_id"), "params")

    # task.description
    task = traj.get("task", {})
    if isinstance(task, dict) and "description" in task:
        r = sanitize(str(task["description"]))
        task["description"] = r.text
        total_audit.extend(
            {"step_id": None, "field": "task.description", **e}
            for e in _audit_entries_to_dicts(r.audit_log)
        )

    # 写入审计日志
    if total_audit:
        meta = traj.setdefault("metadata", {})
        meta["sanitize_audit"] = {
            "total_redacted": len(total_audit),
            "entries": total_audit[:100],  # 最多记 100 条
        }

    return traj


# ── 辅助函数 ──────────────────────────────────────────────────


def _sanitize_dict_values(
    d: dict,
    audit_list: list[dict],
    step_id: Any,
    field_prefix: str,
) -> None:
    """递归脱敏 dict 中的字符串值（原地修改）."""
    for k, v in d.items():
        if isinstance(v, str):
            r = sanitize(v)
            if r.redacted_count > 0:
                d[k] = r.text
                audit_list.extend(
                    {"step_id": step_id, "field": f"{field_prefix}.{k}", **e}
                    for e in _audit_entries_to_dicts(r.audit_log)
                )
        elif isinstance(v, dict):
            _sanitize_dict_values(v, audit_list, step_id, f"{field_prefix}.{k}")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str):
                    r = sanitize(item)
                    if r.redacted_count > 0:
                        v[i] = r.text
                        audit_list.extend(
                            {"step_id": step_id, "field": f"{field_prefix}.{k}[{i}]", **e}
                            for e in _audit_entries_to_dicts(r.audit_log)
                        )
                elif isinstance(item, dict):
                    _sanitize_dict_values(
                        item, audit_list, step_id, f"{field_prefix}.{k}[{i}]"
                    )


def _audit_entries_to_dicts(entries: list[AuditEntry]) -> list[dict[str, Any]]:
    """将 AuditEntry 列表转为 dict 列表."""
    return [
        {
            "rule": e.rule_name,
            "original": e.original[:100],  # 截断原始值（安全）
            "replacement": e.replacement,
        }
        for e in entries
    ]


# -- Phase 2: LLM 软规则（语义脱敏）-------------------------------


def sanitize_soft(text: str, model: str = "claude-haiku") -> SanitizeResult:
    """LLM 语义脱敏：识别上下文敏感信息.

    用小模型识别并替换：
    - 公司名 -> [COMPANY]
    - 人名 -> [PERSON]
    - 金额 -> [AMOUNT]
    - 客户名 -> [CLIENT]

    Args:
        text: 硬规则处理后的文本
        model: 使用的模型（默认 haiku，成本低）

    Returns:
        SanitizeResult
    """
    if not text or len(text) < 10:
        return SanitizeResult(text=text)

    # 尝试调用 LLM
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.debug("未设置 ANTHROPIC_API_KEY，跳过 LLM 软规则")
        return SanitizeResult(text=text)

    # 截断过长文本（控制成本）
    truncated = text[:4000] if len(text) > 4000 else text

    # 构造提示词（用截断后的文本，与实际发给 API 的一致）
    prompt = (
        "你是数据脱敏助手。请将以下文本中的敏感信息替换为占位符，保留语义结构：\n"
        "- 公司/组织名 -> [COMPANY]\n"
        "- 人名（中英文） -> [PERSON]\n"
        "- 具体金额/薪资 -> [AMOUNT]\n"
        "- 客户/合作方名称 -> [CLIENT]\n"
        "- 产品名/项目代号 -> [PROJECT]\n\n"
        "只输出替换后的文本，不要解释。如果没有需要替换的内容，原样输出。\n\n"
        f"文本：\n{truncated}"
    )

    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": len(truncated) + 500,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = json.loads(resp.read().decode())

        content = resp_data.get("content", [])
        if content and isinstance(content, list):
            sanitized_text = content[0].get("text", truncated)

            # 如果原文被截断，拼接尾部原文
            if len(text) > 4000:
                sanitized_text = sanitized_text + text[4000:]

            # 简单审计：比较差异
            audit = []
            if sanitized_text != text:
                audit.append(
                    AuditEntry(
                        rule_name="llm_soft_rule",
                        original="(see diff)",
                        replacement="(LLM sanitized)",
                        position=0,
                    )
                )

            return SanitizeResult(text=sanitized_text, audit_log=audit)
    except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
        logger.warning("LLM 软规则异常: %s", e)

    return SanitizeResult(text=text)


def sanitize_full(text: str) -> SanitizeResult:
    """完整两层脱敏：硬规则 -> LLM 软规则.

    Args:
        text: 原始文本

    Returns:
        最终脱敏结果（合并两层审计日志）
    """
    # 第一层：硬规则
    hard_result = sanitize(text)

    # 第二层：LLM 软规则
    soft_result = sanitize_soft(hard_result.text)

    # 合并审计日志
    combined_audit = hard_result.audit_log + soft_result.audit_log

    return SanitizeResult(text=soft_result.text, audit_log=combined_audit)
