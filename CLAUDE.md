# knowlyr-agent

Monorepo，包含 4 个独立 Python 包。修改代码时注意保持各包的独立性。

## 结构

```
packages/
├── sandbox/    knowlyr-sandbox   Code Agent 执行沙箱
├── recorder/   knowlyr-recorder  Agent 轨迹录制
├── reward/     knowlyr-reward    过程级 Reward 计算
└── hub/        knowlyr-hub       Pipeline 编排
```

## 规则

- 每个子包有独立的 `pyproject.toml`，修改依赖时改对应子包的，不要改错
- 子包之间**不允许互相 import**（sandbox/recorder/reward 保持独立）
- hub 通过可选依赖引用其余三个包，import 时用 try/except 保护
- 数据模型（Pydantic/dataclass）目前各包各自定义，暂不统一
- 代码风格：ruff，line-length=100，target-version=py310
- 中文注释和 docstring

## 测试

```bash
make test              # 全部
make test-sandbox      # 单个包
```

每个子包的测试在 `packages/<name>/tests/` 下，用 pytest 运行。

## 发布

各包独立发布到 PyPI，版本号在各自 `pyproject.toml` 中管理。
