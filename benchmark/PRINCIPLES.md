# 算力亲和原理与性能测试方法说明

[English](README.md) | [简体中文](README.zh-CN.md)

本文讲解三件事：**算力亲和到底是什么**（原理）、**本基准测试如何证明它有效**
（方法）、以及**读者最常见的疑问**（Q&A）。报告怎么读请参见
[README.zh-CN.md](README.zh-CN.md) 的"阅读化验单报告"一节。

---

## 一、算力亲和原理

### 1.1 要解决的问题：信息不对称

Agent 与推理引擎之间存在天然的信息不对称：

- **Agent 知道**：这个会话还会继续（工具返回后马上有下一轮请求）、
  哪些历史消息已被客户端改写、会话的标识是什么。
- **引擎不知道**：它只看到一个个独立到达的 HTTP 请求。对它而言，
  Agent 调工具的那几秒和"用户走了"没有任何区别。

于是引擎只能用**全局 LRU** 管理 KV Cache：新请求进来、缓存槽不够时，
把最久未用的会话缓存驱逐掉——哪怕那个会话 3 秒后就会回来。被驱逐的
会话再次到达时，只能全量重算 prefill，表现为 TTFT 飙升（冷启动）。

> 算力亲和的本质：**把 Agent 的任务、会话、上下文与生命周期信息向下
> 传递给推理引擎，使 KV Cache 与 NPU 算力能够按照 Agent 的未来行为
> 主动调度**，而不是等引擎被动地用 LRU 猜。

### 1.2 三个核心机制

本仓库的 `AscendAffinityChatModel`（对应 openJiuwen 的 InferenceAffinity）
在每次 LLM 调用上做三件事：

| 机制 | 做什么 | 判定依据 |
|---|---|---|
| **会话盐绑定**（cache_salt） | 每个请求注入 `cache_salt: <session_id>`，与 vLLM 原生 prefix-cache salt 语义对齐：salt 注入首块哈希，后续块哈希依赖父块，形成会话专属的隔离命名空间——同 salt 才能复用，跨会话互不串缓存 | 会话 ID（每任务唯一） |
| **前缀差异检测**（prefix diff） | 客户端逐消息比对当前请求与上一窗口的消息序列，检测"纯追加"还是"发生改写" | 消息序列 + 工具列表指纹 |
| **部分 KV 释放**（partial release） | 检测到分叉时，向引擎 `POST /release_kv_cache`，携带 `messages_released_index` 精确指出从第几条消息起失效，引擎只丢弃脏块、保留仍然命中的前缀 | 前缀差异检测结果 |

除上述默认的 **release 协议**外（阶段 A，2026-08），`AscendAffinityChatModel`
还支持 openjiuwen agent-core 的 **agent_hint 生命周期协议**（opt-in，
`enable_agent_hint=True`）：

| 能力 | 做什么 | 触发方式 |
|---|---|---|
| **身份字段** | 每次请求携带 `agent_hint: {session_id, parent_session_id}`（血缘支持团队/成员） | 自动（启用后随每次调用） |
| **生命周期管理** | `evict_kvc` / `offload_kvc` / `prefetch_kvc` 显式管理请求（`context_management.manage_request=true`） | 用户/框架在生命周期点显式调用 |
| **推理后管理** | 同一请求内 `manage_request=false`：推理完成后由引擎原子执行编辑（如 evict 附件尾巴） | 调用时传 `agent_hint_manage={...}` |
| **空闲自动 evict** | 生成后空闲超过 `idle_evict_after_seconds`（默认 0=关）自动 `evict_kvc`；新请求取消重排 | 配置启用 |

两条协议均**安全降级**：引擎忽略未知字段即退化为普通 OpenAI 请求；
管理失败仅计数/告警，绝不中断生成。

### 1.3 生效调用链

```
Agent（deepagents / openJiuwen ReActAgent）
  │  会话上下文 + 生命周期
  ▼
AscendAffinityChatModel / InferenceAffinity provider
  │  release：cache_salt 注入 + 前缀 diff + release 指令
  │  agent_hint：session_id/parent_session_id + evict/offload/prefetch
  ▼
推理引擎（MindIE / vLLM-Ascend，OpenAI 兼容 API）
  │  按 salt 调度 KV 槽位；按 release 索引丢脏块；按 agent_hint 管理
  ▼
昇腾 NPU（KV Cache 显存 + 计算核）
```

关键收益场景是**工具调用间隙**：调工具期间会话暂时静默；带 salt 的
下一轮请求直接命中同命名空间缓存，跳过大部分 prefill，TTFT 显著
下降。其次是**历史改写**：客户端改写了上文（如撤销某轮操作），
baseline 只能让引擎留着一整段脏缓存，亲和路径则精确释放失效部分，
腾出槽位给活跃会话。

> 精确地说：salt 提供的是**隔离与命中**，不是**驻留保证**——显存
> 压力下的驱逐策略仍由引擎决定（vLLM 为全局 LRU）。把会话 KV 主动
> "钉住/驱逐/卸载/预取"是 agent_hint 生命周期协议要补的一环。

> 注意：这一切的前提是**引擎侧实现**了 salt 调度 / release 端点 /
> agent_hint 管理。若引擎不认识这些字段，亲和请求会退化为普通请求——
> 报告里 `affinity_stats.releases_failed` / `management_failed` 与
> "假亲和警报"会把这种情况暴露出来。字段级完整契约（请求字段、端点、
> 降级行为）见根 README 的"引擎接口要求"与"协议兼容"小节，测试侧的
> 探测项与限流约束见 benchmark README 的"引擎接口要求"小节。

### 1.4 主流引擎现状对照（MindIE 3.0.0 公开文档 / vLLM-Ascend v0.23）

本库依赖的接口在两类引擎侧的落地情况（核对时间 2026-08）：

| 依赖 | MindIE 3.0.0（公开接口） | vLLM-Ascend（≥ v0.9.1） |
|---|---|---|
| `POST /v1/chat/completions` / `GET /v1/models` / `GET /metrics` | ✅ 支持。另有 `GET /metrics-json`：直接输出 TTFT/TBT 动态均值、执行/等待请求数、剩余 NPU block 数 | ✅ 支持（vLLM 原生 `/metrics` 含 prefix-cache hit 计数） |
| `cache_salt` 逐请求会话隔离 | ❌ 无此请求字段。Prefix Cache 为**内容哈希**跨会话复用，经服务端 `config.json` 的 `plugin_params: {"plugin_type":"prefix_cache"}` 开启 | ✅ **vLLM 核心原生字段**（chat/completions、completions、responses 三个端点均接受）。salt 注入首块哈希 → 同 salt 复用、异 salt 隔离。需服务端 `--enable-prefix-caching` |
| `POST /release_kv_cache` 部分释放 | ❌ 公开 RESTful 清单无此端点 | ❌ 无（vLLM RFC #37168 提案中：主动失效 + 会话引用计数 + Aging/Fresh 双区调度，未合入任何版本） |
| `agent_hint` 生命周期字段 | ❌ 公开文档无此字段（openJiuwen×昇腾联合特性，需定制版引擎） | ❌ 无（不在 vLLM 生态路线） |

**一个关键的语义辨析**：vLLM 的 `cache_salt` 本质是**隔离命名空间**
（官方动机是多租户防串缓存/防时序攻击），不是"钉住不被驱逐"——显存
不足时全局 LRU 照样逐出 salt 桶里的块。因此在 vLLM-Ascend 上：
salt 生效 → 同会话跨轮命中 ↑、prefill ↓（**这部分收益真实可测**）；
但工具调用间隙的"驻留保护"仍靠 LRU 运气。主动驱逐/钉住/预取正是
RFC #37168 与 agent_hint 要补的缺口。

MindIE Prefix Cache 的叠加约束（直接影响 agent 场景）：不支持
prefix cache 与 context parallel + sequence parallel +
function call(multiturn) 叠加；与 Multi-LoRA、SplitFuse+数据并行
互斥；复用按 block 粒度（blocksize 的整数倍）；模型限 Qwen2/2.5/3、
DeepSeek-R1/V3 系列。

**实际含义（按真机验证平台排序）**：

- **vLLM-Ascend（契约 3/4：chat + salt + metrics）**——salt 真实
  生效，benchmark 中 affinity 配对的命中率/prefill 应可见差异；
  release 计数失败属预期。
- **存量 MindIE（契约 1/4：仅内容哈希缓存）**——亲和字段被安全忽略，
  多轮 agent 的公共前缀仍可命中拿到部分 TTFT 收益，但无会话隔离、
  无主动释放，`affinity_stats` 显示 salt 未生效。
- **完整算力亲和（4/4）**——需要带 agent-hint 补丁的定制引擎。

这正是本基准测试要区分的三种结果：**真亲和 / 部分收益（纯前缀
缓存）/ 假亲和**。

---

## 二、性能测试原理

### 2.1 单变量对照设计

四个 Agent，两两配对比较，**每对内唯一变量是亲和开关**：

| Agent | 框架 | 模型/Provider | 亲和 |
|---|---|---|---|
| `lc-baseline` | deepagents | 原生 `ChatOpenAI` | 关 |
| `lc-affinity` | deepagents | `AscendAffinityChatModel` | 开 |
| `oj-baseline` | openJiuwen ReActAgent | provider `OpenAI` | 关 |
| `oj-affinity` | openJiuwen ReActAgent | provider `InferenceAffinity` | 开 |

同对内共享：同一引擎、同一模型、同一系统提示词、同一工具集、
同一任务输入、同一温度、同为流式。跨框架（lc vs oj）**不做直接
比较**——框架开销不同，只比同框架配对内 affinity 相对 baseline
的变化。

### 2.2 任务集与输入基线化

- 8 个金融顾问任务（调仓 / 风险测评 / 产品对比 / 市场问答），其中一半
  包含**客户端历史改写**——专门触发前缀差异检测与部分释放路径；
  另一半是纯追加——考察盐钉住的缓存能否被下一轮命中。
- 可选长时任务（`--include-longrun`）：25 个客户的批量持仓核查，
  约 100–150 次持续工具调用，检验长时运行下缓存驻留的累积收益。
- **任务集指纹**（sha256 前 16 位）随报告输出：证明每轮输入字节级
  一致，结果可复现、可对比。

### 2.3 轮次、预热与统计口径

- 默认 `--rounds 3`：每轮四个 Agent 各自跑完整任务集，**轮间轮转
  Agent 执行顺序**，抵消引擎残留缓存带来的顺序偏差。
- 每个 Agent 每轮一次**不计时的 warmup**：把引擎冷启动、首次建连、
  JIT 等一次性成本隔离在计时窗口之外。
- 头条数字取**跨轮中位数**（token 总量取均值），抑制单轮抖动。

### 2.4 指标体系（客户端 + 引擎侧 + NPU 侧）

| 层 | 指标 | 采集方式 |
|---|---|---|
| 客户端（必有） | TTFT mean/p50/p95 | `on_llm_start` → 首个非空 `on_llm_new_token` |
| | TPOT | `(E2E − TTFT) / (output_tokens − 1)` |
| | E2E（每次 LLM 调用） | 回调壁钟时间 |
| | Prefill / Decode tokens | `usage_metadata` 求和 |
| | KV 命中率（客户端） | `cached_tokens / prompt_tokens` |
| | Decode tokens/s | decode tokens / decode 时间 |
| 亲和行为 | salt 绑定数、释放尝试/失败 | `affinity_stats` |
| 引擎侧（可选） | 前缀缓存命中率、KV 显存占用 | `--metrics-url` Prometheus 快照差分 |
| NPU 侧（可选） | 利用率 / HBM 带宽 | `--npu-cmd` 采样器 |

引擎侧指标按 **Agent 执行窗口快照差分**（窗口前/后各采一次 Prometheus
读数）归因——只把当前 Agent 窗口内的变化算给它，不把别的 Agent 的
命中算进来。

### 2.5 化验单判定：什么样的结果才算"真亲和"

每个指标行有参考区间与判定（✅ PASS / ⚠️ WARN / ❌ FAIL / ➖ N/A），
默认阈值：

| 指标 | 期待方向 | ✅ 阈值 | ❌ 阈值 |
|---|---|---|---|
| TTFT mean | ↓ | 改善 ≥10% | 恶化 >5% |
| E2E mean | ↓ | 改善 ≥5% | 恶化 >5% |
| Prefill tokens/call | ↓ | 改善 ≥10% | 恶化 >5% |
| KV 命中率 | ↑ | 提升 ≥10pp | 下降 >2pp |
| Decode tokens/call | ≈ 持平 | \|Δ\|≤15% | \|Δ\|>30% |
| TPOT | ≈ 持平 | \|Δ\|≤10% | \|Δ\|>25% |
| Decode tokens/s | ≈ 持平 | \|Δ\|≤10% | \|Δ\|>20% |

**核心四指标**（TTFT↓、Prefill/call↓、KV 命中率↑、E2E↓）**同步改善**
才是算力亲和有效的证据：缓存真的被钉住了（命中率↑）、真的少算了
（prefill↓）、用户真的感知到了（TTFT/E2E↓）。

反过来，如果只看到 NPU 利用率/显存动了，核心四却持平——说明改变的
只是资源摆放，不是调度效率。报告会直接给出 **"疑似假亲和"警报**。

---

## 三、Q&A

**Q1：四个 Agent 共用一个 API key，会影响测试结果吗？引擎怎么知道
是 4 个不同的 Agent？**

不影响。API key 只是认证凭据，引擎的调度与 KV Cache 匹配完全不看它。
Agent 身份靠请求体里的 `cache_salt` 标识，命名规则
`bench-{agent}-{task}-r{round}`，每任务每轮唯一。两个 baseline 不带
salt，走引擎默认 LRU 调度。唯一需要留意的是引擎侧若配置了按 key 的
RPM/TPM 限流，四 Agent 会共享配额——但同框架配对双方面对同样的限流
条件，单变量性不受破坏。

**Q2：vLLM 的自动 prefix cache 是全局的，baseline 会不会"搭便车"
命中亲和 Agent 留下的缓存？**

salt 带隔离语义：带 salt 的请求与无 salt 请求的缓存空间隔离，亲和
Agent 的 KV 不会被 baseline 复用。此外分窗口执行 + 轮间轮转 +
窗口差分归因进一步消除顺序偏差。

**Q3：为什么 oj 两个 Agent 的 TTFT 显示 ➖？**

openJiuwen agent-core 没有暴露 token 级回调，客户端测不到首 token
时刻；oj 配对比较使用 E2E / token 用量 / 命中率等两侧都可测的指标。
TTFT 的完整对比请看 lc 配对。

**Q4：引擎不支持 `/release_kv_cache` 或 salt 字段会怎样？**

亲和请求退化为普通请求，性能与 baseline 无差，报告会通过
`releases_failed` 计数与核心四指标 FAIL/持平体现出来，必要时触发
"疑似假亲和"警报。这正是本基准测试的价值：**区分宣传与实效**。

**Q5：到底测几轮合适？**

默认 3 轮 + 跨轮中位数已能抑制大部分抖动。若单轮内指标波动大于
预期收益幅度（例如 TTFT 波动 20% 而预期收益 15%），请加轮数
（`--rounds 5`）而不是下结论。

**Q6：温度不是 0，输出不确定，对比还公平吗？**

采样随机性影响的是**内容**，不是**缓存调度**：KV 命中取决于输入
前缀（系统提示词 + 历史 + 工具结果），输入每轮字节级一致。token
用量统计上取多轮中位数即可收敛。若你的引擎支持，也可在
`agents.py` 中把 temperature 调成 0 复测。

**Q7：为什么不用模拟引擎，本机就能跑？**

模拟引擎只能证明"协议被调用了"，不能证明"性能真的变好"——TTFT
收益取决于引擎真实的 KV 调度实现。本基准测试坚持连真实引擎
（MindIE / vLLM-Ascend），不可达时直接退出并给出指引。

**Q8：跑一次的正确姿势是什么？**

```bash
python benchmark/run_benchmark.py --setup \
  --engine-url http://<engine-host>:<port>/v1 \
  --model <model-name> \
  --api-key <api-key> \
  --rounds 3 --include-longrun
```

可选 `--metrics-url`（引擎 Prometheus）与 `--npu-cmd`（NPU 采样命令）
补齐引擎侧与 NPU 侧指标。报告输出在 `benchmark/reports/`。

**Q9：在存量 MindIE 上跑，能看到多少收益？**

公开版 MindIE 3.0.0 未暴露 `cache_salt` / `/release_kv_cache` /
`agent_hint`（见 1.4 对照表），亲和请求退化为普通请求 + 引擎全局
内容哈希前缀缓存：多轮 agent 的公共前缀（系统提示词、工具定义、
早期历史）仍可命中，能拿到部分 TTFT 收益；但无会话隔离、无主动
释放。报告中 salt 绑定与 release 指标为 0/失败属预期行为，不代表
本库故障——引擎侧若开启了 Prefix Cache 插件，可对照
`--metrics-url` 的命中率变化验证这部分收益。

**Q10：在 vLLM-Ascend 上跑呢？salt 不是原生字段吗？**

是——`cache_salt` 是 vLLM 核心原生请求字段（三个 OpenAI 兼容端点
都接受），vLLM-Ascend 复用 vLLM v1 调度器，salt 直接生效（需
`--enable-prefix-caching`）。因此 affinity 配对的 KV 命中率与
prefill tokens 指标应能看到真实差异，这是当前**最有说服力的真机
验证平台**。但注意两点：其一，vLLM 的 salt 语义是**隔离命名空间**
（同 salt 才能复用），不是驻留保证——显存压力下 LRU 照样逐出，
"钉住"要等 RFC #37168 或 agent_hint 落地；其二，`/release_kv_cache`
不存在，`releases_failed` 计数持续增长属预期，报告中 release 相关
行会呈现 ➖/FAIL，不代表 salt 失效。
