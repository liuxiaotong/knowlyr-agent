"""AgentReward MCP Server - Model Context Protocol 服务."""

import json
from typing import Any, Dict, List

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from agentreward.config import RewardConfig
from agentreward.reward import RewardEngine
from agentreward.rubrics import get_default_rubric_set
from agentreward.preferences import build_preferences, preferences_summary
from agentreward.calibration import calibrate


def create_server() -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-reward[mcp]")

    server = Server("agentreward")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="score_trajectory",
                description="对单条 Agent 轨迹计算过程级 Reward",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trajectory": {
                            "type": "object",
                            "description": "轨迹数据，包含 task, steps, outcome 等字段",
                        },
                        "rule_weight": {
                            "type": "number",
                            "description": "规则层权重 (默认: 0.6)",
                            "default": 0.6,
                        },
                        "model_weight": {
                            "type": "number",
                            "description": "模型层权重 (默认: 0.4)",
                            "default": 0.4,
                        },
                    },
                    "required": ["trajectory"],
                },
            ),
            Tool(
                name="build_preferences",
                description="从多条轨迹构建偏好对 (用于 RLHF/DPO 训练)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trajectories_by_task": {
                            "type": "object",
                            "description": "按任务分组的轨迹数据 {task_id: [trajectory, ...]}",
                        },
                        "min_margin": {
                            "type": "number",
                            "description": "最小分数差阈值 (默认: 0.05)",
                            "default": 0.05,
                        },
                    },
                    "required": ["trajectories_by_task"],
                },
            ),
            Tool(
                name="calibrate_reward",
                description="将自动 Reward 与人工标注进行校准",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reward_scores": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "自动 Reward 分数列表",
                        },
                        "human_scores": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "对应的人工标注分数列表",
                        },
                    },
                    "required": ["reward_scores", "human_scores"],
                },
            ),
            Tool(
                name="list_rubrics",
                description="列出可用的评估 Rubric 维度",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """调用工具."""

        if name == "score_trajectory":
            trajectory = arguments["trajectory"]
            rule_weight = arguments.get("rule_weight", 0.6)
            model_weight = arguments.get("model_weight", 0.4)

            config = RewardConfig(
                rule_weight=rule_weight,
                model_weight=model_weight,
            )
            engine = RewardEngine(config)
            result = engine.score(trajectory)

            result_dict = result.to_dict()
            result_text = json.dumps(result_dict, ensure_ascii=False, indent=2)

            return [
                TextContent(
                    type="text",
                    text=f"## 轨迹评分结果\n\n"
                    f"- 总分: {result.total_score:.4f}\n"
                    f"- 结果分: {result.outcome_score:.4f}\n"
                    f"- 过程分: {result.process_score:.4f}\n"
                    f"- 步骤数: {len(result.step_rewards)}\n\n"
                    f"### 详细结果\n```json\n{result_text}\n```",
                )
            ]

        elif name == "build_preferences":
            trajectories_by_task = arguments["trajectories_by_task"]
            min_margin = arguments.get("min_margin", 0.05)

            pairs = build_preferences(trajectories_by_task, min_margin=min_margin)
            summary = preferences_summary(pairs)
            pairs_data = [p.to_dict() for p in pairs]

            return [
                TextContent(
                    type="text",
                    text=f"## 偏好对构建结果\n\n"
                    f"- 总对数: {summary['total_pairs']}\n"
                    f"- 涉及任务: {summary['unique_tasks']}\n"
                    f"- 平均 margin: {summary['avg_margin']:.4f}\n\n"
                    f"### 偏好对详情\n```json\n"
                    f"{json.dumps(pairs_data, ensure_ascii=False, indent=2)}\n```",
                )
            ]

        elif name == "calibrate_reward":
            reward_scores = arguments["reward_scores"]
            human_scores = arguments["human_scores"]

            try:
                result = calibrate(reward_scores, human_scores)
                result_dict = result.to_dict()

                return [
                    TextContent(
                        type="text",
                        text=f"## 校准结果\n\n"
                        f"- Pearson r: {result.pearson_r:.4f}\n"
                        f"- Spearman rho: {result.spearman_rho:.4f}\n"
                        f"- 一致率: {result.agreement_rate:.4f}\n\n"
                        f"### 详细指标\n```json\n"
                        f"{json.dumps(result_dict, ensure_ascii=False, indent=2)}\n```",
                    )
                ]
            except ValueError as e:
                return [TextContent(type="text", text=f"校准失败: {e}")]

        elif name == "list_rubrics":
            rubric_set = get_default_rubric_set()
            lines = ["## 评估 Rubric 维度\n"]
            for r in rubric_set.rubrics:
                lines.append(
                    f"- **{r.id}** ({r.name}): {r.description}\n"
                    f"  - 权重: {r.weight}, 评估方式: {r.evaluator}"
                )
            lines.append(f"\n总权重: {rubric_set.total_weight():.2f}")

            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

    return server


async def serve():
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-reward[mcp]")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """主入口."""
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
