# Changelog

本文档遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式
与 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

相关文档：

- 本仓库：[langchain-ascend-affinity](https://github.com/ascend-agentos-bridge/langchain-ascend-affinity)
- 功能说明：[README.md](README.md)（[简体中文](README.zh-CN.md)）
- 引擎与网关兼容矩阵：[COMPATIBILITY.md](COMPATIBILITY.md)（[简体中文](COMPATIBILITY.zh-CN.md)）

## [Unreleased]

### Added

- **社区协作基础设施**（`.github/`）：`CONTRIBUTING.md`（质量门禁、提交
  规范、双语文档同步规则、架构裁决）、Issue 模板（bug / feature）、PR
  模板、`SECURITY.md`、`CODE_OF_CONDUCT.md`、`dependabot.yml`。

### Fixed

- **流式路径丢失 run metadata 导致 cache_salt 失效（真机 benchmark 暴露）**：
  langchain-core 在 `streaming=True` 时绕过 `_generate`/`_agenerate` 直接调用
  `_stream` 且不传 run manager，使 `config.metadata` 中的 `session_id` 无法到达
  亲和管线——`affinity_stats.salt_bound_requests` 恒为 0。现通过
  `_should_stream` 拦截将生成路径统一收敛到 `_generate`/`_agenerate`
  （内部流式聚合、token 回调不变），异步路径经 `run_manager.get_sync()` 线程
  安全转发；仅显式 `stream=True`（`stream()`/`astream()` API）保留框架流式分支。
- **benchmark 采集端丢弃 token 用量**：`usage_metadata` 在 OpenAI 兼容集成中
  是 `TypedDict`（运行时 dict），采集端却用 `getattr` 读取，导致 prompt /
  completion / cached tokens 全部为空（Prefill、Decode、TPOT、KV 命中率 ➖）。
  新增 `usage_field` 同时兼容 dict 与命名空间对象，`lc`/`oj` 两侧采集统一修复。

### Changed

- **`chat_ascend.py` 模块化拆分**：拆分为 transport / serialization /
  affinity-pipeline / agent-hint 四个 mixin 模块
  （`langchain_ascend/llms/`），公开 API 与协议行为完全不变（纯重构）。
- **亲和决策 DEBUG 日志**：salt 绑定、前缀分叉释放决策（会话、释放下标）
  在 DEBUG 级别输出；失败始终以 WARNING 记录。
- **benchmark 引擎侧指标自动识别**：前缀缓存指标名跨 vLLM 版本自动发现
  （V0 hit/query 计数器、V1 `prefix_cache_hit_rate` gauge、hits+miss 配对、
  改名后的 usage 指标），同一套代码兼容 vLLM / vLLM-Ascend / 网关透传部署。
- **benchmark release 自动禁用**：探测发现引擎无 `/release_kv_cache` 时，
  亲和 agent 自动关闭 release 请求（`cache_salt` 绑定保留），不再产生 404
  噪声与 `releases_failed` 虚高。
- **benchmark 可观测性**：新增 `stream_usage` 引擎探测（`include_usage` 是否
  生效，✗ 时报告提示 token 类指标缺失原因）；`affinity_stats` 改为跨轮累计；
  框架构建失败（如 openJiuwen 缺失）以 `build_error` 行显式标注而非静默零数据。
- **benchmark 全链路运行日志**：新增 `--log-level`（默认 INFO）与
  `--log-file`（默认 `benchmark/run.log`）；每次 LLM 调用一行（TTFT/E2E/token 用量/`salt=yes|no`），
  任务、阶段（含该轮亲和计数 `salt=绑定/总数`）、引擎窗口（命中率/KV/NPU）、
  预热与构建失败均有日志；`DEBUG` 级输出亲和管线每次请求的 salt/释放决策与
  请求体。报告附录引用日志文件路径。
- **默认采集开箱即用**：`--metrics-url` 默认 `http://172.24.107.130:7000/metrics`，
  `--log-file` 默认 `benchmark/run.log`，不再需要手动配置即可采集引擎侧指标
  和运行日志；传 `--metrics-url ""` 可关闭引擎采集。

## [0.2.0] - 2026-08

重大重构版本：将 v0.1 的回调式亲和处理器替换为 openJiuwen agent-core 完整
算力亲和机制的 LangChain 移植——单入口模型 + 前缀差异调度 + 部分 KV 释放，
零回调、零接线、零后端适配器。

### Added

- **`AscendAffinityChatModel`**：单入口 LangChain 聊天模型（继承
  `BaseChatModel`），亲和默认开启（`enable_affinity=True`），替代 v0.1 的
  `AscendChatLLM`。每次 LLM 调用自动完成 salt 绑定、前缀差异调度和部分
  KV 释放。
- **`PrefixCacheTracker` / `ReleasePlan` 公开导出**：前缀差异调度算法（移植
  自 agent-core `KVCacheManager`）及其产出结构，作为公开 API 暴露。
- **前缀差异调度**：对比当前 `(messages, tools)` 窗口与会话上一个窗口，区分
  纯追加（缓存保持热）与历史改写（首个分歧下标），仅在分叉时触发部分释放。
- **部分 KV 释放**：通过 `POST {engine-root}/release_kv_cache` 向引擎发送
  失效窗口与 `messages_released_index` / `tools_released_index`，引擎只丢弃
  脏块、保留有效前缀。释放失败仅告警，不中断生成。
- **agent_hint 生命周期协议（阶段 A，opt-in）**：`enable_agent_hint` 可选
  启用，注入 `session_id` / `parent_session_id` 身份字段，暴露管理方法
  `evict_kvc` / `offload_kvc` / `prefetch_kvc`（与 agent-core 同名同语义）。
  支持 inference-then-manage（`manage_request=false`）和空闲自动驱逐
  （`idle_evict_after_seconds`）。
- **流式支持**：SSE 事件聚合，`on_llm_new_token` 回调 + `usage` 统计信息
  透传。
- **双语 README 三框架快速开始**：`langchain`（`create_agent`）、
  `langgraph`（`StateGraph`）、`deepagents`（`create_deep_agent`）三种
  框架的 copy-paste 示例，中英文同步。
- **4-agent 真实引擎 benchmark 化验单框架**：`lc/oj × baseline/affinity`
  单变量对照，核心四指标（TTFT↓ / Prefill/call↓ / KV 命中率↑ / E2E↓）同步
  改善才判定"真亲和"，否则报告"疑似假亲和"警报。
- **质量门禁脚本**：`scripts/quality_gate.py`，校验 pylint 10.00/10 + 单元
  测试覆盖率 ≥ 90%。

### Removed

- **`callbacks/` 子包**：移除 `AscendAffinityCallbackHandler`——v0.1 的回调式
  亲和处理器，在 LangGraph 嵌套运行树下存在错序/误驱逐问题。
- **`backends/` 子包**：移除 `offload` / `prefetch` / `evict` 后端适配器
  （`BaseAscendBackend`、`get_backend`、`SUPPORTED_BACKENDS`）。
- **`AscendChatLLM`**：被 `AscendAffinityChatModel` 取代。
- **独立 `/agent-hints` 端点**：移除独立 hint 通道，亲和字段全部内联到
  `/v1/chat/completions` 请求体。
- **`example/` 目录**：移除 mock 引擎 + 验证框架（模拟 TTFT 数字无实际
  验证价值；协议行为由单元测试契约套件覆盖）。

### Changed

- **亲和默认开启**：从 v0.1 的 opt-in 改为 `enable_affinity=True` 默认启用；
  关闭后模型退化为普通 OpenAI 兼容客户端。
- **会话解析顺序**：每次调用按 `per-call / bind kwargs → config.metadata →
  构造参数 session_id → 无会话` 优先级解析；无会话时不发送任何亲和字段，
  保持普通 OpenAI 客户端，避免匿名请求共享缓存桶造成跨会话污染。
- **传输层零运行时依赖**：用 stdlib `urllib` 实现 HTTP 通信，仅依赖
  `langchain-core`（≥1.0.0）。

### Fixed

- **v0.1 回调式调度在 LangGraph 嵌套运行树下的错序/误驱逐问题**：通过架构级
  重构解决（移除回调→模型内调度），非补丁修复。

## [0.1.0] - 2026-07

初始版本。

### Added

- **`AscendAffinityCallbackHandler`**：回调式亲和处理器，通过 LangChain
  回调系统挂载亲和调度。
- **`backends/` 适配器**：`offload` / `prefetch` / `evict` 三种后端适配器，
  用于 KV 缓存管理操作。
- **独立 `/agent-hints` 端点**：通过独立 HTTP 端点向引擎发送 hint 指令。
- **`AscendChatLLM`**：初始聊天模型实现。
- **mock 硬件示例**：`example/` 目录下的模拟引擎与验证框架。

> **注意**：以上形态已被 v0.2 架构裁决否决（见 [AGENTS.md](AGENTS.md)
> 「架构裁决」节），回调式调度在 LangGraph 嵌套运行树下存在错序/误驱逐的
> 根本性缺陷。本节仅供历史参考，新代码不得复活回调接线或后端适配器形态。

## 版本对比链接

> **注**：`v0.1.0` tag 未存在于本仓库——v0.1 回调式实现代码在仓库重建时
> 已移除，`v0.1.0` 的 compare 链接仅作历史参考。`v0.2.0` 为当前可追溯的
> 首个发布 tag。

[0.2.0]: https://github.com/ascend-agentos-bridge/langchain-ascend-affinity/compare/v0.2.0
[0.1.0]: https://github.com/ascend-agentos-bridge/langchain-ascend-affinity/releases/tag/v0.1.0
