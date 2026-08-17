# AGENTS.md — 项目维护约定

## 文档同步（重要）

- 每次开发/修改代码后，**必须检查 `README.md`（英文）与 `README.zh-CN.md`（简体中文）是否需要联动更新**：
  - API 签名、构造函数参数、环境变量、配置项变化
  - Quick Start / Installation / Backend Configuration 示例变化
  - 新增或移除公开导出（`__all__`）
- 两份 README 内容必须保持同步（仅语言不同），顶部保留语言切换行 `[English](README.md) | [简体中文](README.zh-CN.md)`。

## 工程命令

- 单元测试：`python -m pytest tests/unit_tests`
  - 覆盖率门槛 90%（`--cov-fail-under=90`，已配置于 `pyproject.toml`）
  - 集成测试在 `tests/integration_tests/`，需真实昇腾硬件，默认跳过
- 依赖管理：Poetry（`pyproject.toml`）。内网环境若 Poetry 无法直连 pypi.org，
  可用 `$env:POETRY_REPOSITORIES_PYPI_URL="https://mirrors.tools.huawei.com/pypi/simple"` 或改用 pip 镜像安装后直接 `python -m pytest`。

## 开发完成质量门禁（强制，任何代码变更必做）

- **所有代码变更完成后、提交之前，必须执行一次：**

  ```bash
  python scripts/quality_gate.py
  ```

  该命令同时校验：
  1. **pylint**：对 `git ls-files '*.py'` 全部文件评分，必须达到 `10.00/10`
     （pylint 配置见 `pyproject.toml` 的 `[tool.pylint]`，禁止为凑分新增 disable）
  2. **pytest**：`tests/unit_tests` 全部通过，且覆盖率 ≥ 90%
     （集成测试需真实昇腾硬件，默认跳过，不算失败）
- 任何一步失败：**必须修复后重新运行**，直至全部通过才允许提交。
- 新增 Python 文件后必须重新运行门禁（pylint 按 `git ls-files` 扫描，未跟踪文件需先 `git add`）。
- 若新增依赖，须同步更新 `pyproject.toml` 的 dev 依赖组（pylint / pytest 等）。

## Git 规范

- 提交严格遵循 Conventional Commits（`feat:` / `fix:` / `docs:` / `test:` / `chore:` / `refactor:`）。
- 远端仓库：https://github.com/ascend-agentos-bridge/langchain-ascend-affinity（分支 `main`）。
- 本机经公司代理 + 自签证书环境，push 时使用单次命令 `git -c http.sslVerify=false push`（勿修改全局 SSL 配置）。
- 绝不提交任何 Token / 密钥；Token 若已写入 `.git/config`，交付后提示用户清除。
## 架构裁决（2026-08 修订）

- **否决（保持）**：v0.1 的实现形态——`callbacks/`（AscendAffinityCallbackHandler）、
  `backends/`（offload/prefetch/evict 适配器）、独立 `/agent-hints` 端点。
  任何新代码不得复活回调接线或后端适配器形态。
- **演进（2026-08 P1 决议）**：openjiuwen agent-core 于 2026-07 合入的
  `agent_hint` 生命周期协议（`session_id`/`parent_session_id` +
  `context_management` 的 `evict/offload/prefetch`）为本项目演进方向，
  按「阶段 A 协议对齐（身份字段 + 管理方法，opt-in）→ 阶段 B 模型内自动
  调度 → 真机验证」分阶段引入。协议构造须与 agent-core
  `AscendAffinityModelClient` 保持字段级一致；引擎不支持时未知字段被忽略，
  必须安全降级（非致命、可观测）。
- **评审时**：以本节约束为准；与 openjiuwen agent-core 的协议兼容性
  （字段、URL、序列化）逐项核对，发现漂移即提出修复。
