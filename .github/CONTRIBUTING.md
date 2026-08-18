# 贡献指南

感谢你关注 langchain-ascend-affinity！本文档说明如何参与本项目开发。

## 项目简介

langchain-ascend-affinity 将 openJiuwen agent-core 的昇腾算力亲和机制移植到
LangChain 生态。换一个模型对象 `AscendAffinityChatModel`，用 langchain /
langgraph / deepagents 构建的智能体即可获得对推理引擎的 KV-Cache 亲和调度——
无需回调、无需挂载 handler。

## 开发环境搭建

1. **Python 版本**：3.9+
2. **安装依赖**（Poetry 推荐）：

   ```bash
   pip install poetry
   poetry install
   ```

3. **内网环境**：若 Poetry 无法直连 pypi.org，设置镜像：

   ```powershell
   $env:POETRY_REPOSITORIES_PYPI_URL="https://mirrors.tools.huawei.com/pypi/simple"
   ```

   或直接用 pip 安装依赖后 `python -m pytest`。

## 开发工作流

```
Fork → 创建功能分支 → 开发 → 运行质量门禁 → 提交 PR 到 main
```

### 质量门禁（强制）

**所有代码变更完成后、提交之前，必须执行：**

```bash
python scripts/quality_gate.py
```

该命令同时校验：

1. **pylint**：对 `git ls-files '*.py'` 全部文件评分，必须达到 `10.00/10`
   （pylint 配置见 `pyproject.toml` 的 `[tool.pylint]`，**禁止为凑分新增 disable**）
2. **pytest**：`tests/unit_tests` 全部通过，且覆盖率 ≥ 90%
   （集成测试在 `tests/integration_tests/`，需真实昇腾硬件，默认跳过）

任何一步失败：**必须修复后重新运行**，直至全部通过才允许提交。

**注意**：新增 Python 文件后必须 `git add` 该文件再运行门禁（pylint 按
`git ls-files` 扫描，未跟踪文件不会被校验）。

## 提交规范

严格遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：

- `feat:` — 新功能
- `fix:` — 缺陷修复
- `docs:` — 文档变更
- `test:` — 测试变更
- `chore:` — 构建/工具/CI 变更
- `refactor:` — 重构（不改变公开 API）

## 文档同步规则（重要）

每次开发/修改代码后，**必须检查** `README.md`（英文）与 `README.zh-CN.md`
（简体中文）是否需要联动更新：

- API 签名、构造函数参数、环境变量、配置项变化
- Quick Start / Installation / Backend Configuration 示例变化
- 新增或移除公开导出（`__all__`）

两份 README 内容必须保持同步（仅语言不同），顶部保留语言切换行。

### 兼容性矩阵集中维护

引擎能力 × 支持版本匹配列表、收益失效分析、LLM 网关透传矩阵**集中维护**在
`COMPATIBILITY.md` / `COMPATIBILITY.zh-CN.md`（两份同步，仅语言不同）。其他 md
（根 README 双语、`benchmark/PRINCIPLES.md`、`benchmark/README*.md`）只做一句话
要点简述并链接，**不得复制细节表格或版本号清单**，防止多份拷贝漂移。

**COMPATIBILITY 不是免检权威**：它同样可能过时或出错。任何相关变更
（引擎/网关上游 release、PR/RFC 状态变化、本库协议字段调整）都必须
**交叉比对多方来源**（上游官方文档、release notes、PR/issue、真机探测结果），
而非单信 COMPATIBILITY 已有结论；发现其与上游事实不符时，先改 COMPATIBILITY，
再同步全项目。

**任何 md 文件修改后**，必须通看项目内全部 md 文件（根 README 双语、
COMPATIBILITY 双语、`benchmark/README*.md`、`benchmark/PRINCIPLES.md`、
`AGENTS.md`、`REQUIREMENTS.md`、`openspec/` 下的 md），核对：

- 交叉引用链接是否有效、要点简述是否仍与 COMPATIBILITY 一致
- 事实陈述（版本号、PR/RFC 状态、字段语义）是否与上游来源交叉比对后的结论一致
- 双语文件之间内容是否同步

发现漂移立即修复后才能提交。

## 架构裁决（重要——新贡献者必读）

以下裁决约束所有新代码，评审时会逐项核对：

### 否决（保持）

v0.1 的实现形态——`callbacks/`（AscendAffinityCallbackHandler）、
`backends/`（offload/prefetch/evict 适配器）、独立 `/agent-hints` 端点。
**任何新代码不得复活回调接线或后端适配器形态。**

### 演进（2026-08 P1 决议）

openjiuwen agent-core 于 2026-07 合入的 `agent_hint` 生命周期协议
（`session_id`/`parent_session_id` + `context_management` 的
`evict/offload/prefetch`）为本项目演进方向，按以下阶段引入：

- **阶段 A**（2026-08）：协议对齐（身份字段 + 管理方法，opt-in）
- **阶段 B**：模型内自动调度
- **真机验证**

协议构造须与 agent-core `AscendAffinityModelClient` 保持字段级一致；引擎不支持时
未知字段被忽略，必须安全降级（非致命、可观测）。

## 测试要求

- 新增或修改代码必须补充/更新 `tests/unit_tests/` 下的单元测试
- 覆盖率门槛 90%（已配置于 `pyproject.toml` 的 `--cov-fail-under=90`）
- 集成测试在 `tests/integration_tests/`，需真实昇腾硬件，默认跳过

## 其他规范

- 远端仓库：https://github.com/ascend-agentos-bridge/langchain-ascend-affinity
  （分支 `main`）
- **绝不提交任何 Token / 密钥**；若 Token 已写入 `.git/config`，交付后提示用户清除
- 推送时若有自签证书问题，使用单次命令 `git -c http.sslVerify=false push`
  （勿修改全局 SSL 配置）
