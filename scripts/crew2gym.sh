#!/usr/bin/env bash
# ============================================================
# crew2gym.sh — Crew 轨迹 → knowlyr-gym 训练数据 增量转换脚本
#
# 用法:
#   1. 先在 knowlyr-crew 目录跑 make pull 拉取最新轨迹
#   2. 运行本脚本:
#        bash scripts/crew2gym.sh
#      或指定轨迹文件:
#        CREW_TRAJECTORIES=/path/to/trajectories.jsonl bash scripts/crew2gym.sh
#
# 功能:
#   - 增量提取: 游标记录上次处理位置，只处理新增行
#   - 转换+评分: 调用 knowlyr-hub 的 Python 接口处理轨迹
#   - 导出 SFT / DPO 格式 (追加模式)
#   - 高分轨迹 (reward >= 0.7) 单独导出供 crew few-shot 消费
#   - 每日汇报写入 daily_report.json
#
# 依赖:
#   - python3 (已安装 knowlyr-hub, knowlyr-reward)
#   - jq (可选，用于 daily_report 美化)
# ============================================================
set -euo pipefail

# ── 配置 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 轨迹源文件（默认: knowlyr-crew 项目里 make pull 拉下来的）
CREW_TRAJECTORIES="${CREW_TRAJECTORIES:-$HOME/knowlyr-crew/.crew/trajectories/trajectories.jsonl}"

# 游标文件: 记录上次处理到第几行
CURSOR_FILE="${CURSOR_FILE:-$HOME/.knowlyr/crew2gym_cursor}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"

# 高分阈值
HIGH_QUALITY_THRESHOLD="${HIGH_QUALITY_THRESHOLD:-0.7}"

# ── 工具函数 ──────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

# ── 前置检查 ──────────────────────────────────────────────────
[[ -f "$CREW_TRAJECTORIES" ]] || die "轨迹文件不存在: $CREW_TRAJECTORIES"

command -v python3 >/dev/null 2>&1 || die "需要 python3"

# 确保输出目录和游标目录存在
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$CURSOR_FILE")"

# ── Step 1: 增量提取 ─────────────────────────────────────────
log "=== Step 1: 增量提取 ==="

# 读取游标（上次处理到第几行，从 1 开始计数）
if [[ -f "$CURSOR_FILE" ]]; then
    CURSOR=$(cat "$CURSOR_FILE")
    # 验证游标是正整数
    if ! [[ "$CURSOR" =~ ^[0-9]+$ ]] || [[ "$CURSOR" -lt 1 ]]; then
        CURSOR=1
    fi
else
    CURSOR=1
fi

# 计算总行数（包含空行）
TOTAL_LINES=$(wc -l < "$CREW_TRAJECTORIES" | tr -d ' ')

log "游标位置: 第 $CURSOR 行 / 共 $TOTAL_LINES 行"

if [[ "$CURSOR" -gt "$TOTAL_LINES" ]]; then
    log "No new trajectories. 游标=$CURSOR, 总行数=$TOTAL_LINES"
    exit 0
fi

# 提取新增行到临时文件（跳过空行）
INCREMENTAL_FILE=$(mktemp /tmp/crew2gym_incremental.XXXXXX.jsonl)
trap "rm -f $INCREMENTAL_FILE" EXIT

tail -n +"$CURSOR" "$CREW_TRAJECTORIES" | grep -v '^\s*$' > "$INCREMENTAL_FILE" || true

NEW_COUNT=$(wc -l < "$INCREMENTAL_FILE" | tr -d ' ')

if [[ "$NEW_COUNT" -eq 0 ]]; then
    log "No new trajectories (全是空行)"
    # 更新游标到末尾
    echo "$((TOTAL_LINES + 1))" > "$CURSOR_FILE"
    exit 0
fi

log "新增轨迹: $NEW_COUNT 条"

# ── Step 2: 转换 + 评分 ──────────────────────────────────────
log "=== Step 2: 转换 + 评分 ==="

# 使用 Python 完成: 读取增量 JSONL → reward 评分 → 输出标准 hub 格式
# 同时生成 SFT / DPO / 高分轨迹
python3 - "$INCREMENTAL_FILE" "$OUTPUT_DIR" "$HIGH_QUALITY_THRESHOLD" << 'PYTHON_SCRIPT'
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

incremental_file = sys.argv[1]
output_dir = Path(sys.argv[2])
hq_threshold = float(sys.argv[3])

# ---- 加载增量轨迹 ----
raw_trajectories = []
with open(incremental_file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw_trajectories.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [WARN] 跳过第 {line_no} 行: JSON 解析失败 - {e}", file=sys.stderr)

if not raw_trajectories:
    print("  没有有效轨迹")
    # 写一个空结果让 bash 脚本知道
    result = {"processed": 0, "avg_reward": 0.0, "high_quality": 0}
    with open(output_dir / "_batch_result.json", "w") as f:
        json.dump(result, f)
    sys.exit(0)

print(f"  读取 {len(raw_trajectories)} 条原始轨迹")

# ---- 尝试加载 reward engine ----
reward_engine = None
try:
    from agentreward.reward import RewardEngine
    reward_engine = RewardEngine()
    print("  Reward engine 已加载")
except ImportError:
    print("  [INFO] knowlyr-reward 未安装，使用 outcome.score 作为 reward")
except Exception as e:
    print(f"  [WARN] Reward engine 初始化失败: {e}，回退到 outcome.score")

# ---- 转换 + 评分 ----
processed = []

for traj in raw_trajectories:
    task = traj.get("task", {})
    outcome = traj.get("outcome", {})
    steps = traj.get("steps", [])
    metadata = traj.get("metadata", {})

    # 计算 reward
    reward = outcome.get("score", 0.0)

    if reward_engine is not None and steps:
        try:
            # 构建 reward engine 输入
            reward_steps = []
            for s in steps:
                tc = s.get("tool_call", {})
                tr = s.get("tool_result", {})
                reward_steps.append({
                    "tool": tc.get("name", ""),
                    "params": tc.get("parameters", {}),
                    "output": tr.get("output", ""),
                })
            reward_input = {
                "task": task.get("description", ""),
                "steps": reward_steps,
                "outcome": {
                    "success": outcome.get("success", False),
                    "tests_passed": outcome.get("tests_passed", 0),
                    "tests_total": outcome.get("tests_passed", 0) + outcome.get("tests_failed", 0),
                },
            }
            result = reward_engine.score(reward_input)
            reward = result.total_score
        except Exception as e:
            print(f"  [WARN] 评分失败 task={task.get('task_id','?')}: {e}", file=sys.stderr)

    # 标准 hub 格式
    hub_traj = {
        "task_id": task.get("task_id", ""),
        "agent_framework": traj.get("agent", ""),
        "agent_model": traj.get("model", ""),
        "steps": steps,
        "total_steps": len(steps),
        "success": outcome.get("success", False),
        "reward": reward,
        "step_rewards": [],
        "duration_seconds": 0.0,
        "metadata": {
            "task_description": task.get("description", ""),
            "task_type": task.get("type", ""),
            "domain": task.get("domain", "crew"),
            "employee": metadata.get("employee", ""),
            "total_tokens": outcome.get("total_tokens", 0),
            "source": "crew2gym",
        },
    }
    processed.append(hub_traj)

print(f"  转换完成: {len(processed)} 条")

# ---- Step 3: 导出 ----

# 3a. 追加到主轨迹文件
traj_path = output_dir / "trajectories.jsonl"
with open(traj_path, "a", encoding="utf-8") as f:
    for t in processed:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"  轨迹追加到 {traj_path}")

# 3b. 追加 SFT 格式
sft_path = output_dir / "sft_train.jsonl"
sft_count = 0
with open(sft_path, "a", encoding="utf-8") as f:
    for t in processed:
        if not t["success"]:
            continue
        # 构建 response: 将 steps 拼成文本
        response_parts = []
        for i, step in enumerate(t["steps"], 1):
            thought = step.get("thought", "")
            tc = step.get("tool_call", {})
            tr = step.get("tool_result", {})
            parts = [f"Step {i}:"]
            if thought:
                parts.append(f"Thought: {thought}")
            if tc.get("name"):
                parts.append(f"Tool: {tc['name']}({json.dumps(tc.get('parameters', {}), ensure_ascii=False)})")
            output = tr.get("output", "")
            if output:
                # 截断过长输出
                if len(output) > 500:
                    output = output[:500] + "..."
                parts.append(f"Output: {output}")
            response_parts.append("\n".join(parts))

        record = {
            "instruction": t["metadata"].get("task_description", ""),
            "input": json.dumps({
                "domain": t["metadata"].get("domain", ""),
                "employee": t["metadata"].get("employee", ""),
            }, ensure_ascii=False),
            "response": "\n\n".join(response_parts),
            "task_id": t["task_id"],
            "reward": t["reward"],
            "metadata": {
                "agent_framework": t["agent_framework"],
                "agent_model": t["agent_model"],
                "total_steps": t["total_steps"],
            },
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        sft_count += 1
print(f"  SFT 追加 {sft_count} 条 -> {sft_path}")

# 3c. 追加 DPO 格式（同 task 不同 reward 配对）
# 先按 task_description 分组当前批次
from collections import defaultdict
task_groups = defaultdict(list)
for t in processed:
    desc = t["metadata"].get("task_description", "")
    if desc:
        task_groups[desc].append(t)

dpo_path = output_dir / "dpo_train.jsonl"
dpo_count = 0
with open(dpo_path, "a", encoding="utf-8") as f:
    for desc, group in task_groups.items():
        if len(group) < 2:
            continue
        sorted_group = sorted(group, key=lambda x: x["reward"], reverse=True)
        for i in range(len(sorted_group) - 1):
            chosen = sorted_group[i]
            rejected = sorted_group[i + 1]
            if chosen["reward"] == rejected["reward"]:
                continue
            def steps_to_text(steps):
                parts = []
                for j, s in enumerate(steps, 1):
                    thought = s.get("thought", "")
                    tc = s.get("tool_call", {})
                    text = f"Step {j}:"
                    if thought:
                        text += f"\n{thought}"
                    if tc.get("name"):
                        text += f"\nTool: {tc['name']}"
                    parts.append(text)
                return "\n\n".join(parts)

            pair = {
                "prompt": desc,
                "chosen": steps_to_text(chosen["steps"]),
                "rejected": steps_to_text(rejected["steps"]),
                "task_id": chosen["task_id"],
                "reward_margin": chosen["reward"] - rejected["reward"],
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            dpo_count += 1
print(f"  DPO 追加 {dpo_count} 条 -> {dpo_path}")

# 3d. 高分轨迹提取
hq_path = output_dir / "high_quality_examples.jsonl"
hq_count = 0
with open(hq_path, "a", encoding="utf-8") as f:
    for t in processed:
        if t["reward"] >= hq_threshold:
            # 导出为 crew few-shot 友好格式
            example = {
                "task_id": t["task_id"],
                "task_description": t["metadata"].get("task_description", ""),
                "employee": t["metadata"].get("employee", ""),
                "model": t["agent_model"],
                "steps": t["steps"],
                "reward": t["reward"],
                "total_steps": t["total_steps"],
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            hq_count += 1
print(f"  高分轨迹 (>= {hq_threshold}) {hq_count} 条 -> {hq_path}")

# ---- 统计 ----
rewards = [t["reward"] for t in processed]
avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

batch_result = {
    "processed": len(processed),
    "sft_exported": sft_count,
    "dpo_exported": dpo_count,
    "high_quality": hq_count,
    "avg_reward": round(avg_reward, 4),
    "min_reward": round(min(rewards), 4) if rewards else 0.0,
    "max_reward": round(max(rewards), 4) if rewards else 0.0,
    "success_count": sum(1 for t in processed if t["success"]),
    "success_rate": round(sum(1 for t in processed if t["success"]) / len(processed), 4) if processed else 0.0,
}

# 写临时结果供 bash 读取
with open(output_dir / "_batch_result.json", "w") as f:
    json.dump(batch_result, f, indent=2)

print(f"\n  === 批次统计 ===")
print(f"  处理: {batch_result['processed']} 条")
print(f"  成功率: {batch_result['success_rate']:.1%}")
print(f"  平均 reward: {batch_result['avg_reward']:.4f}")
print(f"  reward 范围: [{batch_result['min_reward']:.4f}, {batch_result['max_reward']:.4f}]")
print(f"  SFT: {sft_count}, DPO: {dpo_count}, 高分: {hq_count}")

PYTHON_SCRIPT

# ── Step 3: 更新游标 ─────────────────────────────────────────
log "=== Step 3: 更新游标 ==="

NEW_CURSOR=$((TOTAL_LINES + 1))
echo "$NEW_CURSOR" > "$CURSOR_FILE"
log "游标更新: $CURSOR -> $NEW_CURSOR"

# ── Step 4: 汇报 ─────────────────────────────────────────────
log "=== Step 4: 写入日报 ==="

RESULT_FILE="$OUTPUT_DIR/_batch_result.json"
REPORT_FILE="$OUTPUT_DIR/daily_report.json"

if [[ -f "$RESULT_FILE" ]]; then
    # 读取批次结果，追加到 daily_report.json
    python3 - "$RESULT_FILE" "$REPORT_FILE" << 'REPORT_SCRIPT'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

result_file = Path(sys.argv[1])
report_file = Path(sys.argv[2])

with open(result_file) as f:
    batch = json.load(f)

# 构建日报条目
entry = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    **batch,
}

# 追加到日报（JSON Lines 格式，每行一个日期的记录）
with open(report_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"  日报已追加: {report_file}")

REPORT_SCRIPT

    # 清理临时结果
    rm -f "$RESULT_FILE"
fi

# ── 完成 ──────────────────────────────────────────────────────
log "=== 完成 ==="
log "输出目录: $OUTPUT_DIR"
log "文件列表:"
ls -lh "$OUTPUT_DIR"/*.jsonl "$OUTPUT_DIR"/daily_report.json 2>/dev/null | while read line; do
    log "  $line"
done
