#!/usr/bin/env python3
"""
claude2crew.py — Claude Code 对话 JSONL → Crew 轨迹格式转换器

Claude Code 把每次对话完整保存在 .claude/projects/ 的 JSONL 文件中，
包含主对话和所有子 agent。这个脚本提取其中的工具调用，转换为 Crew 轨迹格式，
可以直接 POST 到服务器或写入本地文件。

用法:
    # 转换单个文件
    python scripts/claude2crew.py /path/to/session.jsonl -o output.jsonl

    # 转换整个 session（自动扫描子 agent）
    python scripts/claude2crew.py --session /path/to/session.jsonl -o output.jsonl

    # 转换并 POST 到服务器
    python scripts/claude2crew.py --session /path/to/session.jsonl --post

    # 批量扫描目录
    python scripts/claude2crew.py --scan-dir ~/.claude/projects/ --since 2026-02-22 -o output.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── 花名册：角色名 → slug ─────────────────────────────────────

CHARACTER_MAP = {
    "陆明哲": "product-manager",
    "林锐": "code-reviewer",
    "苏文": "doc-writer",
    "程薇": "test-engineer",
    "顾然": "refactor-guide",
    "秦合": "pr-creator",
    "谢安": "security-auditor",
    "贺铭": "debug-expert",
    "唐思远": "api-designer",
    "马骁": "devops-engineer",
    "钟瑞": "performance-optimizer",
    "方逸凡": "algorithm-researcher",
    "沈若兰": "sociology-researcher",
    "韩泽民": "economics-researcher",
    "叶心蕾": "hr-manager",
    "曹正宇": "finance-expert",
    "林晓桐": "data-quality-expert",
    "陈启明": "benchmark-specialist",
    "苏映彤": "nlp-researcher",
    "黄维达": "solutions-architect",
    "周念慈": "community-operator",
    "卫子昂": "frontend-engineer",
    "罗清河": "data-engineer",
    "姜墨言": "ceo-assistant",
    "傅语桥": "i18n-expert",
    "宋正言": "legal-counsel",
    "丁雪筠": "e2e-tester",
    "许鹏举": "bd-manager",
    "温若瑜": "ux-designer",
    "郑锐航": "mlops-engineer",
    "孙策安": "dba",
    "赵云帆": "backend-engineer",
    "柳若曦": "customer-success",
}

# 也支持用 slug 本身匹配
SLUG_SET = set(CHARACTER_MAP.values())

# 简称 → slug（agent-setting 中用的是简称如 "墨言" 而非 "姜墨言"）
SHORT_NAME_MAP = {}
for full_name, slug in CHARACTER_MAP.items():
    SHORT_NAME_MAP[full_name] = slug
    # 去掉姓（第一个字）作为简称
    if len(full_name) >= 3:
        SHORT_NAME_MAP[full_name[1:]] = slug
    # 也映射 2 字名
    if len(full_name) == 2:
        SHORT_NAME_MAP[full_name] = slug

CREW_API_URL = "https://crew.knowlyr.com/api/trajectory/report"
CREW_API_TOKEN = os.environ.get("CREW_API_TOKEN", "")

# ── 核心：解析 JSONL ──────────────────────────────────────────


def parse_jsonl(filepath: str) -> list[dict]:
    """读取 JSONL 文件，返回解析后的条目列表。"""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # 跳过损坏的行
    return entries


def detect_employee(entries: list[dict], filepath: str) -> str:
    """从 JSONL 条目中推断员工名。"""

    # 1. 检查 agent-setting 条目（主对话）
    for entry in entries[:5]:
        if entry.get("type") == "agent-setting":
            setting = entry.get("agentSetting", "")
            # 先查全名，再查简称
            if setting in CHARACTER_MAP:
                return CHARACTER_MAP[setting]
            if setting in SHORT_NAME_MAP:
                return SHORT_NAME_MAP[setting]

    # 2. 检查第一条 user 消息（子 agent 的 prompt）
    for entry in entries[:10]:
        if entry.get("type") == "user":
            msg = entry.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                # 匹配 "你是XXX" 模式
                m = re.search(r"你是([\u4e00-\u9fff]{2,4})", content)
                if m:
                    name = m.group(1)
                    if name in CHARACTER_MAP:
                        return CHARACTER_MAP[name]
                    if name in SHORT_NAME_MAP:
                        return SHORT_NAME_MAP[name]
                # 匹配 slug
                for slug in SLUG_SET:
                    if slug in content:
                        return slug
            break

    # 3. 从文件路径猜（如果有 employee 名字）
    fp = str(filepath)
    for name, slug in CHARACTER_MAP.items():
        if name in fp or slug in fp:
            return slug

    return "unknown-agent"


def detect_model(entries: list[dict]) -> str:
    """从 assistant 消息的 model 字段提取模型名。"""
    for entry in entries:
        msg = entry.get("message", {})
        if msg.get("role") == "assistant":
            model = msg.get("model", "")
            if model:
                return model
    return "claude-sonnet-4-6"


def extract_task_description(entries: list[dict]) -> str:
    """从第一条 user 消息提取任务描述。"""
    for entry in entries:
        if entry.get("type") == "user":
            msg = entry.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # 截取前 200 字符
                desc = content.strip()[:200]
                return desc
            elif isinstance(content, list):
                # 提取 text 块
                texts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        texts.append(c.get("text", ""))
                if texts:
                    return " ".join(texts)[:200]
    return "未知任务"


def extract_session_id(entries: list[dict], filepath: str) -> str:
    """提取 session ID。"""
    for entry in entries[:5]:
        sid = entry.get("sessionId", "")
        if sid:
            return sid
    # 从文件名提取
    stem = Path(filepath).stem
    # 去掉 agent- 前缀
    if stem.startswith("agent-"):
        stem = stem[6:]
    return stem[:36]  # UUID 长度


def truncate(text: str, max_len: int = 8000) -> str:
    """截断文本到指定长度。"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [截断, 原始 {len(text)} 字符]"


def extract_steps(entries: list[dict]) -> list[dict]:
    """
    从 JSONL 条目中提取工具调用步骤。

    逻辑：
    1. 遍历 assistant 消息，提取 tool_use 块和 text 块（thought）
    2. 在紧跟的 user 消息中按 tool_use_id 匹配 tool_result
    3. 组装为 crew 轨迹格式的 step
    """
    steps = []
    step_id = 0

    # 建一个工具结果索引：先扫描所有 user 消息中的 tool_result
    tool_results = {}
    for entry in entries:
        msg = entry.get("message", {})
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tuid = block.get("tool_use_id", "")
                if tuid:
                    # 提取 content（可能是 str 或 list）
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        parts = []
                        for rc in result_content:
                            if isinstance(rc, dict) and rc.get("type") == "text":
                                parts.append(rc.get("text", ""))
                        result_content = "\n".join(parts)
                    tool_results[tuid] = {
                        "output": result_content,
                        "is_error": block.get("is_error", False),
                    }

    # 遍历 assistant 消息，提取 tool_use
    for entry in entries:
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        timestamp = entry.get("timestamp", "")
        usage = msg.get("usage", {})
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        # 提取 thought（text 块）
        thought_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                thought_parts.append(block.get("text", ""))

        thought = "\n".join(thought_parts) if thought_parts else ""

        # 提取 tool_use 块
        tool_uses = [
            b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
        ]

        if not tool_uses:
            # 纯思考步骤：assistant 消息只有 text，没有 tool_use
            if thought:
                step_id += 1
                step = {
                    "step_id": step_id,
                    "thought": truncate(thought, 8000),
                    "tool_call": {
                        "name": "thinking",
                        "parameters": {},
                    },
                    "tool_result": {
                        "output": "",
                        "exit_code": 0,
                    },
                    "timestamp": timestamp,
                    "token_count": total_tokens,
                }
                steps.append(step)
            continue

        # 每个 tool_use 生成一个 step
        # 如果有多个 tool_use，thought 只分配给第一个
        for i, tu in enumerate(tool_uses):
            step_id += 1
            tool_name = tu.get("name", "")
            tool_params = tu.get("input", {})
            tool_id = tu.get("id", "")

            # 匹配 tool_result
            result = tool_results.get(tool_id, {})
            tool_output = result.get("output", "")
            is_error = result.get("is_error", False)

            step = {
                "step_id": step_id,
                "thought": truncate(thought if i == 0 else "", 8000),
                "tool_call": {
                    "name": tool_name,
                    "parameters": tool_params,
                },
                "tool_result": {
                    "output": truncate(str(tool_output), 8000),
                    "exit_code": 1 if is_error else 0,
                },
                "timestamp": timestamp,
                "token_count": total_tokens if i == 0 else 0,
            }
            steps.append(step)

    return steps


def convert_file(filepath: str) -> dict | None:
    """将一个 Claude Code JSONL 文件转换为 crew 轨迹格式。"""
    entries = parse_jsonl(filepath)
    if not entries:
        return None

    steps = extract_steps(entries)
    if not steps:
        return None

    employee = detect_employee(entries, filepath)
    model = detect_model(entries)
    task_desc = extract_task_description(entries)
    session_id = extract_session_id(entries, filepath)

    # 计算总 token
    total_tokens = sum(s.get("token_count", 0) for s in steps)

    trajectory = {
        "task": {
            "task_id": f"claude-{session_id[:8]}",
            "description": task_desc,
            "domain": "crew",
        },
        "agent": f"crew/{employee}",
        "model": model,
        "steps": steps,
        "outcome": {
            "success": True,
            "total_steps": len(steps),
            "total_tokens": total_tokens,
        },
        "metadata": {
            "employee": employee,
            "channel": "claude-code",
            "source_session": session_id,
            "source_file": str(filepath),
        },
    }

    return trajectory


def convert_session(session_file: str) -> list[dict]:
    """
    转换整个 session：主对话 + 所有子 agent。

    session_file: 主对话 JSONL 路径
    子 agent 在 {session_id}/subagents/ 目录下
    """
    results = []

    # 1. 主对话
    main_traj = convert_file(session_file)
    if main_traj:
        results.append(main_traj)
        print(
            f"  主对话: {main_traj['metadata']['employee']} | "
            f"{main_traj['outcome']['total_steps']} 步",
            file=sys.stderr,
        )

    # 2. 子 agent
    session_path = Path(session_file)
    session_id = session_path.stem
    subagent_dir = session_path.parent / session_id / "subagents"

    if subagent_dir.exists():
        agent_files = sorted(subagent_dir.glob("agent-*.jsonl"))
        # 排除 compact 文件（压缩后的上下文摘要）
        agent_files = [f for f in agent_files if "compact" not in f.name]

        for agent_file in agent_files:
            traj = convert_file(str(agent_file))
            if traj:
                results.append(traj)
                print(
                    f"  子 agent: {traj['metadata']['employee']} | "
                    f"{traj['outcome']['total_steps']} 步 | "
                    f"{agent_file.name}",
                    file=sys.stderr,
                )

    return results


def scan_directory(
    scan_dir: str, since: str | None = None
) -> list[dict]:
    """扫描目录下所有 session JSONL 文件。"""
    results = []
    scan_path = Path(scan_dir)

    since_dt = None
    if since:
        since_dt = datetime.strptime(since, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    # 查找所有顶层 JSONL 文件（session 文件）
    for jsonl_file in sorted(scan_path.glob("*.jsonl")):
        # 过滤时间
        if since_dt:
            mtime = datetime.fromtimestamp(
                jsonl_file.stat().st_mtime, tz=timezone.utc
            )
            if mtime < since_dt:
                continue

        print(f"\n处理 session: {jsonl_file.name}", file=sys.stderr)
        session_trajs = convert_session(str(jsonl_file))
        results.extend(session_trajs)

    return results


def post_to_server(trajectory: dict) -> bool:
    """POST 单条轨迹到 crew 服务器。"""
    payload = {
        "employee_name": trajectory["metadata"]["employee"],
        "task_description": trajectory["task"]["description"],
        "model": trajectory["model"],
        "channel": "claude-code",
        "steps": trajectory["steps"],
        "success": trajectory["outcome"]["success"],
    }

    payload_json = json.dumps(payload, ensure_ascii=False)

    # 如果太大，写临时文件
    if len(payload_json) > 50000:
        tmp = "/tmp/_claude2crew_post.json"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload_json)
        cmd = [
            "curl", "-s", "-X", "POST", CREW_API_URL,
            "-H", f"Authorization: Bearer {CREW_API_TOKEN}",
            "-H", "Content-Type: application/json",
            "-d", f"@{tmp}",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            os.remove(tmp)
            return result.returncode == 0
        except Exception:
            return False
    else:
        cmd = [
            "curl", "-s", "-X", "POST", CREW_API_URL,
            "-H", f"Authorization: Bearer {CREW_API_TOKEN}",
            "-H", "Content-Type: application/json",
            "-d", payload_json,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False


def write_output(trajectories: list[dict], output_file: str, append: bool = False):
    """写入输出 JSONL 文件。"""
    mode = "a" if append else "w"
    with open(output_file, mode, encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")


# ── CLI ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code JSONL → Crew 轨迹格式转换器"
    )

    # 输入源（互斥）
    parser.add_argument(
        "file", nargs="?", help="单个 JSONL 文件路径"
    )
    parser.add_argument(
        "--session", help="Session JSONL 路径（自动扫描子 agent）"
    )
    parser.add_argument(
        "--scan-dir", help="扫描目录下所有 session"
    )
    parser.add_argument(
        "--since", help="只处理此日期之后的文件 (YYYY-MM-DD)"
    )

    # 输出
    parser.add_argument(
        "-o", "--output", help="输出 JSONL 文件路径"
    )
    parser.add_argument(
        "--append", action="store_true", help="追加模式"
    )
    parser.add_argument(
        "--post", action="store_true", help="POST 到 crew 服务器"
    )

    # 其他
    parser.add_argument(
        "--dry-run", action="store_true", help="只统计，不输出"
    )

    args = parser.parse_args()

    # 至少需要一个输入源
    if not args.file and not args.session and not args.scan_dir:
        parser.print_help()
        sys.exit(1)

    # 转换
    trajectories = []

    if args.scan_dir:
        trajectories = scan_directory(args.scan_dir, args.since)
    elif args.session:
        trajectories = convert_session(args.session)
    elif args.file:
        traj = convert_file(args.file)
        if traj:
            trajectories.append(traj)

    if not trajectories:
        print("没有提取到任何轨迹", file=sys.stderr)
        sys.exit(0)

    # 统计
    total_steps = sum(t["outcome"]["total_steps"] for t in trajectories)
    employees = set(t["metadata"]["employee"] for t in trajectories)
    print(
        f"\n=== 提取完成 ===\n"
        f"  轨迹数: {len(trajectories)}\n"
        f"  总步数: {total_steps}\n"
        f"  涉及员工: {', '.join(sorted(employees))}\n",
        file=sys.stderr,
    )

    if args.dry_run:
        # 打印摘要
        for t in trajectories:
            print(
                f"  [{t['metadata']['employee']}] "
                f"{t['outcome']['total_steps']} 步 | "
                f"{t['task']['description'][:60]}"
            )
        return

    # 输出
    if args.output:
        write_output(trajectories, args.output, args.append)
        print(f"  写入: {args.output}", file=sys.stderr)

    if args.post:
        if not CREW_API_TOKEN:
            print("错误: 未设置 CREW_API_TOKEN 环境变量", file=sys.stderr)
            sys.exit(1)
        posted = 0
        failed = 0
        for t in trajectories:
            if post_to_server(t):
                posted += 1
            else:
                failed += 1
        print(
            f"  POST: 成功 {posted}, 失败 {failed}",
            file=sys.stderr,
        )

        # 双写本地 CAS（如果配置了路径）
        cas_path = os.environ.get("KNOWLYR_CAS_PATH", "")
        if cas_path and posted > 0:
            try:
                from trajectoryhub.cas import CAStore
                from trajectoryhub.ingest import CrewIngestor
                import tempfile

                # 将成功 POST 的轨迹写入临时 JSONL，再用 CrewIngestor 导入
                store = CAStore(cas_path)
                ingestor = CrewIngestor(store)

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
                ) as tmp:
                    for t in trajectories:
                        tmp.write(json.dumps(t, ensure_ascii=False) + "\n")
                    tmp_path = tmp.name

                result = ingestor.ingest(tmp_path)
                os.remove(tmp_path)
                # 清除临时文件的游标记录
                cursor_path = store.db_path.parent / ".ingest_cursor.json"
                if cursor_path.exists():
                    with open(cursor_path, "r", encoding="utf-8") as f:
                        cursor = json.load(f)
                    cursor.pop(tmp_path, None)
                    with open(cursor_path, "w", encoding="utf-8") as f:
                        json.dump(cursor, f, indent=2)

                print(
                    f"  CAS 双写: 新增 {result.ingested} 条 → {cas_path}",
                    file=sys.stderr,
                )
                store.close()
            except ImportError:
                print(
                    "  CAS 双写: 跳过 (trajectoryhub 未安装)",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"  CAS 双写: 失败 ({e})",
                    file=sys.stderr,
                )

    # 如果没指定输出，默认打印到 stdout
    if not args.output and not args.post:
        for t in trajectories:
            print(json.dumps(t, ensure_ascii=False))


if __name__ == "__main__":
    main()
