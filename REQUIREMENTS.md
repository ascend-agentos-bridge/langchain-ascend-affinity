# 项目需求规格（Requirements）

> 本文档描述 langchain-ascend-affinity 的当前架构与约束（v0.2）。
> 详细设计见
> [openspec/projects/affinity-core/proposals/openjiuwen-affinity-port/](openspec/projects/affinity-core/proposals/openjiuwen-affinity-port/)。

## SPEC-01: 项目定位

为 LangChain 生态提供无侵入的昇腾算力亲和能力：让使用 **langchain /
langgraph / deepagents** 构建的智能体，在与昇腾推理引擎（MindIE /
vLLM-Ascend，OpenAI 兼容 API）交互时获得 KV-Cache 亲和调度，降低多智能体
场景的 TTFT 与单位成本。

## SPEC-02: 核心架构（v0.2）

单一入口 `AscendAffinityChatModel(BaseChatModel)`，每次 LLM 调用完成三件事：

1. **会话盐绑定** —— 请求携带 `cache_sharing: true` + `cache_salt: <session_id>`
   （对齐 vLLM-Ascend 前缀缓存 salt 语义）；**未绑定 session 时不发送任何亲和
   字段**，保持普通 OpenAI 客户端，避免匿名请求共享缓存桶造成跨会话污染。
2. **前缀差异调度** —— `PrefixCacheTracker` 对比上一窗口与当前窗口
   `(messages, tools)`，区分纯追加（缓存保持热）与历史改写（首个分叉索引）。
3. **部分 KV 释放** —— 检测到分叉时向 `POST {engine-root}/release_kv_cache`
   发送 previous window + `messages_released_index` / `tools_released_index`，
   只丢弃脏块、保留有效前缀；失败仅告警，不中断生成。

v0.1 的 `callbacks/`、`backends/`（Agent Hint offload/prefetch/evict）机制已
移除：无回调、无接线、无后端适配器，替换一个模型对象即可生效。

## SPEC-03: 测试与质量门禁

- 单元测试：`python -m pytest tests/unit_tests`，覆盖率 ≥ 90%。
- 质量门禁：`python scripts/quality_gate.py`（pylint 10.00/10 + pytest）。
- 集成测试：需真实昇腾硬件/引擎，默认跳过。
- CI：push / PR 自动运行质量门禁（Python 3.11 / 3.12，benchmark 依赖
  deepagents 要求 ≥3.11）。

## SPEC-04: 基准测试

真实引擎上的 4-agent 单变量对照（lc/oj × baseline/affinity），核心四指标
（TTFT↓、Prefill/call↓、KV 命中率↑、E2E↓）**同步改善**才判定"真亲和"，否则
报告给出"疑似假亲和"警报。详见
[benchmark/PRINCIPLES.md](benchmark/PRINCIPLES.md)。

## SPEC-05: 提交与推送规范

- Conventional Commits（`feat:` / `fix:` / `docs:` / `chore:` / `refactor:`）。
- 每次代码变更后运行质量门禁，全部通过才允许提交。
- 推送使用 `git -c http.sslVerify=false push`（不修改全局 SSL 配置）。
- **绝不提交任何 Token / 密钥**；认证信息只从环境变量或交互输入读取。
