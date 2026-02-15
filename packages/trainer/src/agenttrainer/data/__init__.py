"""数据加载和预处理.

包含:
- reader: 读取 hub exporter 导出的 JSONL
- formatter: 标准 chat template 格式化
- agent_format: Agent 轨迹多轮格式化 + 观察遮蔽 + 步骤加权
- collator: padding DataCollator
- chunker: 长轨迹分块
- curriculum: 课程学习采样器
"""
