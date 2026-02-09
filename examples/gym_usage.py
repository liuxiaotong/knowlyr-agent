"""Gymnasium 风格 API 使用示例.

演示如何用 AgentEnv 协议 + Wrapper 组合 + collect() 收集轨迹。
"""

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep
from knowlyrcore.registry import register, make
from knowlyrcore.wrappers import MaxStepsWrapper, RewardWrapper, RecorderWrapper
from trajectoryhub.collect import collect


# ── 1. 自定义环境 ─────────────────────────────────────────────────


class MathEnv(AgentEnv):
    """数学计算环境 — 演示用."""

    domain = "math"

    def __init__(self):
        self._target = 0
        self._current = 0

    def reset(self, *, task=None, seed=None) -> TimeStep:
        """重置环境."""
        self._target = 42 if task is None else task
        self._current = 0
        return TimeStep(
            observation=f"目标: {self._target}, 当前: {self._current}",
            info={"target": self._target},
        )

    def step(self, action: dict) -> TimeStep:
        """执行一步."""
        op = action.get("tool", "add")
        value = action.get("params", {}).get("value", 1)

        if op == "add":
            self._current += value
        elif op == "multiply":
            self._current *= value
        elif op == "submit":
            return TimeStep(
                observation=f"提交: {self._current}",
                terminated=True,
                info={"answer": self._current, "correct": self._current == self._target},
            )

        return TimeStep(
            observation=f"当前: {self._current}",
            info={"current": self._current},
        )

    @property
    def available_tools(self):
        return ["add", "multiply", "submit"]


# ── 2. 注册到 Registry ───────────────────────────────────────────

register("example/math", MathEnv, domain="math", description="数学计算演示环境")


# ── 3. 基础用法: reset/step/close ────────────────────────────────

def basic_usage():
    """基础 Gymnasium 风格交互."""
    env = make("example/math")

    ts = env.reset(task=10)
    print(f"初始观察: {ts.observation}")

    ts = env.step({"tool": "add", "params": {"value": 5}})
    print(f"加 5: {ts.observation}")

    ts = env.step({"tool": "multiply", "params": {"value": 2}})
    print(f"乘 2: {ts.observation}")

    ts = env.step({"tool": "submit"})
    print(f"提交: {ts.observation}, 正确={ts.info['correct']}")

    env.close()


# ── 4. Wrapper 组合 ──────────────────────────────────────────────

def wrapper_usage():
    """Wrapper 可组合模式."""
    # Reward 函数: 答案越接近目标, reward 越高
    def proximity_reward(steps, action):
        if action.get("tool") == "submit":
            return 1.0
        return 0.1

    env = MathEnv()
    env = MaxStepsWrapper(env, max_steps=10)
    env = RewardWrapper(env, reward_fn=proximity_reward)
    env = RecorderWrapper(env, agent_name="demo-agent", model_name="gpt-4o")

    ts = env.reset(task=6)
    ts = env.step({"tool": "add", "params": {"value": 3}})
    ts = env.step({"tool": "multiply", "params": {"value": 2}})
    ts = env.step({"tool": "submit"})

    traj = env.get_trajectory()
    print(f"轨迹: {len(traj['steps'])} 步, outcome={traj['outcome']}")
    for step in traj["steps"]:
        print(f"  step {step['step_id']}: {step['tool']} → reward={step['reward']}")

    env.close()


# ── 5. collect() 批量收集 ────────────────────────────────────────

def collect_usage():
    """用 collect() 批量收集轨迹."""
    call_count = 0

    def simple_agent(observation: str) -> dict:
        """简单 agent: 加 7 后 submit."""
        nonlocal call_count
        call_count += 1
        if call_count % 3 == 0:
            call_count = 0
            return {"tool": "submit"}
        return {"tool": "add", "params": {"value": 7}}

    trajs = collect(
        MathEnv(),
        agent=simple_agent,
        n_episodes=3,
        max_steps=10,
        agent_name="simple-math-agent",
        model_name="demo",
    )

    for i, traj in enumerate(trajs):
        print(f"Episode {i+1}: {len(traj['steps'])} 步, "
              f"success={traj['outcome'].get('success')}")


# ── 6. Context Manager ──────────────────────────────────────────

def context_manager_usage():
    """with 语句自动清理资源."""
    with MathEnv() as env:
        ts = env.reset(task=100)
        ts = env.step({"tool": "add", "params": {"value": 50}})
        ts = env.step({"tool": "multiply", "params": {"value": 2}})
        ts = env.step({"tool": "submit"})
        print(f"答案: {ts.info['answer']}, 正确={ts.info['correct']}")


# ── 运行 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 基础用法 ===")
    basic_usage()

    print("\n=== Wrapper 组合 ===")
    wrapper_usage()

    print("\n=== collect() 批量收集 ===")
    collect_usage()

    print("\n=== Context Manager ===")
    context_manager_usage()
