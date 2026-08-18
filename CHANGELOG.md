# Changelog

本文档遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式
与 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

相关文档：

- 本仓库：[langchain-ascend-affinity](https://github.com/ascend-agentos-bridge/langchain-ascend-affinity)
- 功能说明：[README.md](README.md)（[简体中文](README.zh-CN.md)）
- 引擎与网关兼容矩阵：[COMPATIBILITY.md](COMPATIBILITY.md)（[简体中文](COMPATIBILITY.zh-CN.md)）

## [Unreleased]

（暂无计划内容。本区块用于记录合并到 main 但尚未发布版本的变更。）

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

[0.2.0]: https://github.com/ascend-agentos-bridge/langchain-ascend-affinity/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ascend-agentos-bridge/langchain-ascend-affinity/releases/tag/v0.1.0
