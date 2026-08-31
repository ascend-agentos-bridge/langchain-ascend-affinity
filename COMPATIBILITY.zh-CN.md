# 引擎与网关兼容性矩阵

[English](COMPATIBILITY.md) | [简体中文](COMPATIBILITY.zh-CN.md)

> **为什么有这份文档**：openjiuwen 社区的亲和协议演进快于推理引擎侧的跟进。
> 客户端单方面实现的字段，引擎不认识就是无效载荷。本矩阵回答三个问题：
> 哪个协议版本配哪个引擎版本能拿到什么收益（第 2 节）、不匹配时哪些收益
> 失效（第 3 节）、亲和参数能否穿过常见 LLM 网关（第 4 节）。
>
> **最近核对**：2026-08-31。维护规则见第 5 节。
>
> **勘误（2026-08-31）**：本项目 2026-08-20/24 两次真机运行中的工具任务
> HTTP 501，根因是**客户端缺陷**——流式聚合的历史消息 role 被序列化为
> 类名 `AIMessageChunk`（引擎拒绝未知 role），并非引擎拒绝 salt；已修复
> （2026-08-31），两次运行所涉引擎（vLLM-family，`dsv4-0731`）经探测
> **接受** salt + 工具调用组合（`salt_tool_calls` 探测 200）。下文涉及
> MindIE 类服务器"拒绝 salt+工具调用 → HTTP 501"的表述均源自 agent-core
> 联调经验，**本仓库未独立复核**，降级机制作为防御保留。

---

## 1. 背景：三方进度错位

截至核对日，公开生态里**不存在任何官方引擎 release 完整实现 openjiuwen
亲和协议**。真实状态是一个"部分匹配"梯度：

- openjiuwen agent-core 存在**两代协议**（见 2.1）：旧 release 协议
  （`cache_salt`/`cache_sharing` + `/release_kv_cache`）随 v0.1.16 发布；
  新 `agent_hint` 生命周期协议只合入 develop 分支，**未发任何 release tag**。
- 引擎侧唯一公开实现是 **vllm-ascend PR #6722**（jiuwen affinity kv cache
  插件，基于 vLLM v0.15.0 验证，含 `/release_kv_cache` 端点）——**至今
  Open 未合入**。`agent_hint` 的联调对象是华为内部自研 vLLM，公开仓库
  无对应实现。
- **MindIE 全版本（≤ 3.1.0）零支持**：无 `cache_salt` 请求字段、无
  `/release_kv_cache`、无 `agent_hint` 等价能力；其 Prefix Cache 插件为
  内容哈希跨会话复用，且有特性叠加约束（见 3.2）。
- 唯一在官方 release 中真实存在的亲和能力是 vLLM 核心的 **`cache_salt`**
  （v0.9.0 起，PR #17045）——它提供会话隔离命名空间，但不提供驻留保证
  与主动释放。

### 1.1 本库发送的接口契约

符号约定：`base_url` 为 OpenAI 兼容基址（如 `http://host:8000/v1`）；
`engine-root` 为去掉 `/v1` 后缀的同源地址，释放端点位于该路径下。
`base_url` 支持裸 origin、带 `/v1` 基址或完整 `/chat/completions` 端点
三种形态；认证可选（匿名引擎设 `api_key=""`）。

**必需基线（任何引擎都能跑）**

| 接口 | 要求 |
|---|---|
| `POST {base_url}/chat/completions` | OpenAI 兼容 `messages`；工具调用智能体需支持 `tools`；测 TTFT 需支持 `stream`（SSE） |
| 认证 | `Authorization: Bearer <api_key>` |

**亲和契约（决定有没有收益）**

| 本库发送什么 | 引擎应做什么 | 缺失/被忽略时的行为 |
|---|---|---|
| 请求体携带 `cache_sharing: true` | 允许该会话加入前缀缓存共享 | 无收益，无害 |
| 请求体携带 `cache_salt: <session_id>` | 按 vLLM 前缀缓存 salt 语义：salt 注入首块哈希，同 salt 会话获得隔离的 KV 命名空间，异 salt 请求无法复用（显存压力下的驱逐策略仍由引擎决定） | 退回共享缓存桶——无隔离、无收益 |
| 检测到前缀被改写时 `POST {engine-root}/release_kv_cache`，携带 `model`、`cache_salt`、`cache_sharing`、`messages`、`messages_released_index`（以及可选的 `tools`、`tools_released_index`） | 按 agent-core 兼容的部分释放语义：从释放下标起丢弃脏块，保留有效前缀 | 记入 `releases_failed` 计数并告警；改写频繁的智能体失去释放收益 |

**未绑定 session**——模型不发送任何亲和字段，保持普通 OpenAI 客户端。
这是有意为之：没有 `cache_salt` 却发送 `cache_sharing`，会让所有匿名请求
挤进同一个共享缓存桶，存在跨会话 KV 污染风险。

降级始终安全：符合规范的网关会忽略未知字段，释放失败仅产生非致命告警，
模型作为普通 OpenAI 客户端照常工作。引擎**主动拒绝**亲和字段时同样如此
（据 agent-core 联调经验，本仓库未独立复核）：MindIE 类服务器对携带
`cache_sharing` / `cache_salt` 且 messages 含工具调用（assistant
`tool_calls` / `tool` 角色）的 `/v1/chat/completions` 请求返回 **HTTP 501
Not Implemented**——纯消息 + salt 正常，工具消息无 salt 也正常。客户端
自动降级：被拒请求去掉 salt 字段重试一次，随后**该 session** 禁用 salt
绑定（`salt_degraded_requests` 计数 + WARNING 日志，其他 session 不受
影响），生成以普通 OpenAI 客户端继续；`salt_enabled=False` 可在能力探测
（`salt_tool_calls`）已判定不支持时预先禁用，benchmark 会先探测该组合并
在化验单上如实标注。

> 本仓库已实测的行为（2026-08-24/31）：vLLM-family 引擎（`dsv4-0731`，
> v0.25.1）与 LM Studio 均接受 salt + 工具调用组合（探测 200）；501 降级
> 路径经 mock 引擎与单元测试验证。

## 2. 版本匹配列表

### 2.1 openjiuwen agent-core 协议时间线

| 版本 / commit | 日期 | 协议形态 | 客户端发送内容 |
|---|---|---|---|
| v0.1.16（最新 release，PyPI 同步） | 2026-07-14 | **release 协议**（旧） | `cache_sharing: true` + `cache_salt: <session_id>`；改写历史时 `POST /release_kv_cache`（`model`/`cache_salt`/`cache_sharing`/`messages`/`messages_released_index`/`tools`/`tools_released_index`） |
| develop `63380f17e8` | 2026-07-22 | **agent_hint 生命周期**（新，+8858/−1054，删除旧 KVCacheManager） | `agent_hint: {session_id, parent_session_id, context_management: {manage_request, edits: [{type, target, start, end}]}}`；`evict_kvc`/`offload_kvc`/`prefetch_kvc` 方法 |
| develop `75adc2b44e` | 2026-08-17 | vLLM 联调修复 | URL 归一化三分支；SSE 多形态兼容；推理后管理（`manage_request=false`） |

> 本库 `AscendAffinityChatModel` 同时实现两条协议：release 协议默认开启
> （与 agent-core `InferenceAffinityModelClient.release()` 字节兼容，前缀
> 差异调度自动完成），agent_hint 协议 opt-in（`enable_agent_hint`），
> 字段级对齐 develop 分支。

### 2.2 引擎能力 → 支持版本（只列支持的，每项一个代表版本）

| 能力 | 支持的引擎 / 版本 |
|---|---|
| `cache_salt` | vLLM ≥ v0.9.0（PR #17045；vLLM-Ascend ≥ v0.9.1 全系继承，需开启 prefix caching） |
| `cache_sharing` + `/release_kv_cache` | 仅 vLLM-Ascend v0.15 + PR #6722 插件（Open 未合入） |
| `agent_hint` | 仅华为内部自研 vLLM（不公开） |
| 全量 KV 释放 / agent 提示（参考） | SGLang `/flush_cache`；NVIDIA Dynamo `nvext.agent_hints` + Session Control（实验，非本库协议） |

### 2.3 可行组合（"匹配列表"）

| 组合 | 协议覆盖 | 说明 |
|---|---|---|
| **完整 release 协议（4/4 契约）**：openjiuwen release 协议 × vLLM-Ascend v0.15 + PR #6722 补丁 | chat + salt + metrics + release | 唯一能拿到"部分释放"收益的组合；需自行携带 patch，无官方 release |
| **salt 档（3/4 契约）**：openjiuwen 任意版本 × vLLM-Ascend ≥ v0.9.1（stock） | chat + salt + metrics | **当前推荐的真实引擎验证平台**：salt 隔离与跨轮命中真实生效；release 404 属预期 |
| **全局缓存档（1/4 契约）**：openjiuwen 任意版本 × MindIE 任意版本（stock） | 仅 chat | 纯消息请求亲和字段被安全忽略，但 **salt + 工具调用请求据报被主动拒绝（HTTP 501，agent-core 联调经验，本仓库未独立复核）**——客户端自动降级；只剩引擎全局内容哈希前缀缓存（叠加约束见 3.2） |
| **生命周期档**：openjiuwen develop（agent_hint）× 华为内部自研 vLLM | agent_hint 全量 | 公开生态不可复现，等待引擎侧跟进或 PR #6722 类补丁扩展 |

## 3. 不匹配时的收益失效分析

### 3.1 逐机制失效矩阵

| # | 亲和机制 | 依赖的引擎能力 | vLLM-Ascend ≥ 0.9.1（stock） | + PR #6722 | MindIE ≤ 3.1.0（stock） | 失效后果 |
|---|---|---|---|---|---|---|
| 1 | 会话盐绑定（`cache_salt`） | 引擎消费 salt，注入首块哈希 | ✅ 生效 | ✅ | ❌ 忽略；**salt + 工具调用消息组合据报被主动拒绝（HTTP 501，agent-core 联调经验，未独立复核）** → 客户端自动降级为普通客户端 | MindIE：无会话隔离命名空间，跨会话 KV 混布于全局缓存，隔离收益 = 0；工具调用型 agent 若无客户端的 salt 拒绝降级将整轮 501 |
| 2 | `cache_sharing` 标记 | 引擎消费（非标字段） | ⚠️ 忽略（无害） | ✅ | ⚠️ 忽略（无害） | 无独立收益，仅伴随 salt 生效 |
| 3 | 前缀差异检测（prefix diff） | 无（纯客户端） | ✅ 始终工作 | ✅ | ✅ 始终工作 | 无失效；但 release 不可用时检测结果无处上报 |
| 4 | 部分释放（`/release_kv_cache`） | 引擎端点 | ❌ 404 | ✅ | ❌ 404 | **历史改写场景收益全失效**：脏 KV 块滞留显存只能等 LRU 逐出；`releases_failed` 持续增长属预期，非故障 |
| 5 | `agent_hint` 身份字段 | 引擎消费 | ⚠️ 忽略 | ⚠️ 忽略 | ⚠️ 忽略 | 身份透传无效，连带管理动作失去寻址依据 |
| 6 | `evict/offload/prefetch_kvc` | 引擎管理实现 | ❌ | ❌ | ❌ | **生命周期管理全失效**：空闲会话 KV 驻留与否全凭引擎 LRU |
| 7 | 推理后管理（`manage_request=false`） | 引擎生成后原子执行 | ❌ | ❌ | ❌ | 同上 |
| 8 | 空闲自动 evict | 同 #6 | ❌ | ❌ | ❌ | 同上 |

### 3.2 按引擎的收益保留度

| 引擎环境 | 保留的收益 | 失效的收益 | 判定方式 |
|---|---|---|---|
| vLLM-Ascend ≥ 0.9.1（stock） | 工具调用间隙同 salt 跨轮命中 ↑、prefill ↓、TTFT ↓（**真实可测**）；跨会话隔离 | 部分释放、agent_hint 全部生命周期管理；显存压力下 salt 桶仍会被 LRU 逐出（salt 是隔离命名空间，不是驻留保证） | benchmark 中 affinity 配对 KV 命中率/prefill 应见差异；`releases_failed` 增长属预期 |
| vLLM-Ascend + PR #6722 | 上述全部 + 改写历史的精确释放（脏块即弃，腾槽位给活跃会话） | agent_hint 生命周期管理（插件不实现） | release 计数应转为成功 |
| MindIE（stock） | 仅引擎全局 prefix cache 的公共前缀命中（系统提示词、工具定义等），无会话隔离 | salt 隔离、部分释放、生命周期管理全部失效。**注意叠加约束**：MindIE prefix cache 不支持与 function call(multiturn)、context parallel + sequence parallel 叠加——工具调用型 agent 可能连公共前缀命中都拿不到，收益趋零。**主动拒绝（据报，未独立复核）**：`cache_salt` + 工具调用消息返回 HTTP 501——客户端自动降级（去 salt 重试后该 session 禁用 salt），benchmark 经 `salt_tool_calls` 探测预先禁用 salt | salt 绑定与 release 指标为 0/失败属预期；若开启 Prefix Cache 插件可对照 `--metrics-url` 验证残留收益；affinity agent 的 `salt_degraded_requests` > 0 即标记该自动降级 |
| 华为内部自研 vLLM | agent_hint 全量（身份 + evict/offload/prefetch + 推理后管理） | —（公开生态无法验证） | 不适用 |

### 3.3 结论

- "一厢情愿"风险是真实的：**openjiuwen 客户端发出的 8 类亲和载荷，在
  stock 引擎上只有 1 类（salt）在 1 个引擎家族（vLLM 系 ≥ 0.9.x）上生效**。
- 因此本库坚持：全部字段安全降级（引擎忽略即退化为普通 OpenAI 请求）+
  benchmark 显式区分**真亲和 / 部分收益（纯前缀缓存）/ 假亲和**，而非假设
  收益存在。

## 4. LLM 网关透传矩阵

场景：客户端在 `/v1/chat/completions` 请求体携带非标字段
（`cache_salt` / `cache_sharing` / `agent_hint`），网关须原样转发到上游
引擎，不得剥离。**网关剥离字段与引擎不认字段同罪：亲和静默失效。**

| 网关 | 默认行为 | 亲和字段能否存活 | 配置方法 |
|---|---|---|---|
| Nginx / OpenResty 简单反代 | 字节级透传，不解析 JSON | ✅ 必然存活 | 无需配置；勿用 `proxy_set_body` 或 Lua 改写 body |
| AWS ALB / NLB | L7 转发不改请求体 | ✅ 存活 | 无 |
| vLLM api-server 直连 | `extra="allow"` 接受未知字段；`cache_salt` 为原生字段 | ✅ 原生支持 | 无 |
| Higress ai-proxy（openai/vllm provider） | 仅对 `model` 字段做单点替换，其余原样保留 | ✅ 存活 | provider 选 `openai` 或 `vllm`，勿开协议转换 |
| Kong AI Gateway | 同协议仅改写 `model`/`stream`；**跨协议转换会全量重建请求体** | ✅ 同协议存活 / ❌ 跨协议丢失 | 保持 openai → openai 直通 |
| LiteLLM proxy | 参数白名单校验 | ⚠️ 需配置 | `litellm_settings: drop_params: false` + `allowed_openai_params: ["cache_salt", "cache_sharing", "agent_hint"]` |
| New API（QuantumNous） | 默认 struct 重建请求体 | ⚠️ 需配置 | 渠道设置开启「透传请求体」（PR #1441，`PassThroughBodyEnabled`） |
| APISIX ai-proxy | 默认按协议解析→重建 | ⚠️ 需配置 | 使用 `passthrough` 协议（PR #13320） |
| One API（songquanpeng） | 无模型映射时原样透传；**配置模型重定向即 struct 重建，未知字段静默剥离**（issue #2295，修复 PR #2384 未合并） | ⚠️ 条件性（默认部署常用模型映射 → 丢字段） | 规避模型重定向，或自行应用 PR #2384 |

**部署建议**（按优先级）：Nginx 纯反代 / 直连引擎 → Higress（openai
provider）/ Kong（同协议）→ LiteLLM / New API（按上表配置）→ 规避 One API
模型重定向。**任何网关变更后，先跑一次 benchmark 的 release-endpoint 探测
与 `affinity_stats` 核对**，确认 salt 绑定数非零、release 不被网关拦截。

**探测注意事项——SPA catch-all 前端**（实测 `models.ascend.huawei.com`，
nginx/1.21.5）：部分网关对**任意未知路径**（`/release_kv_cache`、
`/version`、`/health`、`/metrics`）一律返回 **200 + 前端 index.html**，
朴素的"200 = 端点存在"判断会误报。benchmark 探测把 HTML 响应一律视为
"端点不存在"（`release_endpoint=false`），引擎身份块回退为"未知"；
亲和字段是否被接受仍可通过真实 chat 路由验证（`salt_tool_calls` /
`stream_usage` 探测）。

**探测注意事项——JSON catch-all 服务**（实测 LM Studio，2026-08-31）：
部分服务器对未知路径返回 **200 + `{"error": "..."}` JSON 错误体**而非
404，同样会误报端点存在——本轮已把"200 但响应体为 JSON error"也判定为
"端点不存在"（此前在 LM Studio 上曾把不存在的 `/release_kv_cache` 误报
为存在，产生 3 次假成功的 release 计数）。

## 5. 维护规则

本文档是兼容性事实的**集中维护点，但不是免检权威**——它同样可能过时
或出错。每次维护 pass（或上游出现新 release/PR 状态变化）必须：

1. **交叉比对多方来源**（下表入口：上游官方文档、release notes、
   PR/issue 讨论、必要时真机探测），而非沿用本文档既有结论；
2. 以交叉比对后的结论**先更新本文档**第 2/3/4 节矩阵并刷新核对日期；
3. **通看全项目 md**（根 README 双语、`benchmark/README*.md`、
   `benchmark/PRINCIPLES.md` 等），同步其中的要点简述，核对双语一致
   （规则详见 `AGENTS.md` 的"文档一致性"节）。

| 核对项 | 入口 |
|---|---|
| openjiuwen agent-core 协议演进 | <https://github.com/openJiuwen-ai/agent-core/releases> · develop 分支 `ascend_affinity_model_client.py` |
| vllm-ascend 亲和插件合入状态 | <https://github.com/vllm-project/vllm-ascend/pull/6722> |
| vLLM 主动释放 RFC | <https://github.com/vllm-project/vllm/issues/37168>（含 #37003 RetentionDirective、agentic-api #18） |
| vLLM `cache_salt` 语义 | <https://docs.vllm.ai/en/latest/design/prefix_caching/> |
| vLLM-Ascend 版本对应 | <https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html> |
| MindIE 接口清单与 Prefix Cache 约束 | <https://www.hiascend.com/document/detail/zh/mindie/latest/index/index.html> |
| One API 透传修复 | <https://github.com/songquanpeng/one-api/pull/2384> |

**更新触发条件**：openjiuwen 发布含 agent_hint 的 release tag、PR #6722
合入或关闭、RFC #37168 落地具体 vLLM 版本、MindIE 公开接口新增
`cache_salt`/主动释放能力、任一网关默认透传行为变化。
