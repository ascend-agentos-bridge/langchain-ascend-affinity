# 基准测试：4 Agent 化验单 × 昇腾算力亲和（真实引擎）

[English](README.md) | [简体中文](README.zh-CN.md)

面向**真实**昇腾推理引擎（MindIE / vLLM-Ascend）的四智能体基准测试。
**不提供模拟引擎**：引擎不可达时脚本直接退出并给出指引。

| Agent | 框架 | 模型 / Provider | KV 释放 |
|---|---|---|---|
| `lc-baseline` | deepagents | 原生 `ChatOpenAI` | 关 |
| `lc-affinity` | deepagents | `AscendAffinityChatModel` | 开（salt 绑定 + 前缀差异 + 部分释放） |
| `oj-baseline` | openJiuwen ReActAgent | provider `OpenAI` | 关 |
| `oj-affinity` | openJiuwen ReActAgent | provider `InferenceAffinity` | 开 |

每个亲和 Agent **只与同框架的 baseline 对比**（每对单一变量）。
`--agents all|lc|oj|<逗号列表>` 可选择子集。

算力亲和的原理、本基准测试如何证明其有效、以及常见疑问
（API key 共用、缓存交叉污染、轮次选择等），见
[PRINCIPLES.md](PRINCIPLES.md)。

## 快速开始

```bash
# 一键：安装基准依赖 + 引擎探测 + 跑全部四个智能体 + 出报告
python benchmark/run_benchmark.py --setup \
  --engine-url http://<引擎地址>:<端口>/v1 \
  --model <模型名> \
  --api-key <API密钥>
```

或使用环境变量：

```bash
export ASCEND_ENGINE_URL=http://<引擎地址>:<端口>/v1
export ASCEND_MODEL=<模型名>
export ASCEND_API_KEY=<API密钥>
python benchmark/run_benchmark.py
```

### 在封闭 / 无外网集群上运行

如果目标引擎机**没有互联网**（这是昇腾物理机集群的典型部署），**不要**
直接用 `--setup`（它会走 pip 联网安装依赖）。正确做法是：在一台能
联网的构建机上用 `scripts/build_benchmark.py` 打出自包含的包，
传到运行机后用包内附带的 `install_offline.sh` / `install_offline.ps1`
完成离线安装。

完整操作指引（含跨平台 wheel 处理、pydantic-core / numpy / orjson /
tiktoken / tokenizers 等 C 扩展包的兜底方案）见：
[`../OFFLINE_PACKAGING.zh-CN.md`](../OFFLINE_PACKAGING.zh-CN.md)。

参数：

- `--rounds N`（默认 3）：每轮完整任务集；agent 顺序每轮轮转；每 agent
  每轮前 1 次不计分预热；以跨轮中位数为主判定值。每轮输入字节级一致
  （任务集指纹写入报告）。
- `--include-longrun`：追加 25 客户组合核查长时任务（约 100-150 次
  LLM 调用的持续工具循环）。
- `--metrics-url`：可选 vLLM `/metrics` 端点 —— 采集每 agent 窗口的
  引擎侧前缀缓存命中率与 KV Cache 占用。
- `--npu-cmd`：可选采样命令，输出 `key=value`（如在引擎机上经
  `npu-smi` + `awk` 采集 NPU 利用率 / HBM 占用）。
- `--api-key`（回退 `ASCEND_API_KEY`，本地无鉴权引擎默认 `EMPTY`）、
  `--max-parallel`（默认 2）、`--turn-timeout`（默认 240 秒）、
  `--report-dir`。
- `--metrics-url`（默认 `http://172.24.107.130:7000/metrics`）：vLLM
  `/metrics` 端点——默认采集引擎侧缓存指标；传 `--metrics-url ""` 可关闭。
- `--log-level DEBUG|INFO|WARNING|ERROR`（默认 `INFO`）：控制台日志详细度。
  `DEBUG` 额外输出亲和管线每次请求的 salt 绑定/释放决策与请求体。
- `--log-file`（默认 `benchmark/run.log`）：同时将完整运行日志（UTF-8）追加
  写入文件；报告附录会引用该文件。

## 运行日志（全链路可观测）

默认 `INFO` 级别下，控制台按"每次 LLM 调用 / 每个任务 / 每个 agent 阶段 /
每个引擎窗口"各打一行——所有"静默失败"（salt 未绑定、usage 缺失、框架
被吞）都能当场看见：

```
[run] engine=http://.../v1 model=dsv4-0731 agents=['lc-baseline', ...] rounds=3
[probe] {"reachable": true, "model_listed": true, "release_endpoint": false, "streaming": true, "stream_usage": false, "salt_tool_calls": false}
=== round 1/3 order=[...] ===
[llm] r0 lc-affinity rebalance-C1001 ttft=1,112ms e2e=8,470ms prompt=1,203 comp=412 cached=0 salt=yes
[task] r0 lc-affinity rebalance-C1001 ok hits=2/3 turns=4 e2e=37,472ms
[phase] r0 lc-affinity: tasks=8 llm_calls=44 ttft_mean=1,615ms e2e_mean=8,188ms salt=45/45 releases=0/0
[engine] r0 lc-affinity hit_rate_delta=62.5% cache_usage_peak=0.872 npu=[{'util': 57.5}]
```

- `[llm]` —— 每次 LLM 调用一行：TTFT / E2E / prompt / completion / cached
  tokens，以及 `salt=yes|no`（本次调用是否携带会话 ID——`yes` 是
  `cache_salt` 绑定的前提）。引擎（或网关）不返回 usage 时
  `prompt/comp/cached` 显示 `None`。
- `[task]` —— 任务结果：关键词命中与总 E2E。
- `[phase]` —— 每 agent 每轮：调用量、均值与该轮亲和计数
  （`salt=绑定数/总数`、`releases=尝试/失败`、`degraded=salt 拒绝降级数`）。
  **`salt=44/44`（绑定数=总数）即证明每次请求都绑定了 salt。**
  `degraded=N` 表示引擎对 N 个 salt 绑定请求返回 HTTP 501，随后对应
  session 的 salt 绑定被禁用（其他 session 不受影响）。
- `[engine]` —— 引擎侧前缀命中率 / KV 占用 / NPU 采样（仅配置了
  `--metrics-url` / `--npu-cmd` 时出现）。
- `[warmup]` / `[build]` —— 预热结果与 agent 构建失败（如 openJiuwen
  缺失），否则这些失败会被静默吞掉。

`--log-level DEBUG` 时，亲和模型自身还会输出每次 salt 绑定与释放决策
（会话、释放下标）及请求体——当 `[phase]` 行出现 `salt=0/N` 时用它排查。

## 引擎接口要求

开跑前脚本会探测引擎并打印结果（`model_listed / release_endpoint /
streaming / stream_usage`）。各探测项对引擎的要求：

符号约定：`{base_url}` 为 OpenAI 兼容基址（`--engine-url`，缺 `/v1` 时自动
补上）；`{engine-root}` 为去掉 `/v1` 的同源地址（见
[COMPATIBILITY 第 1.1 节契约](../COMPATIBILITY.zh-CN.md#11-本库发送的接口契约)）。

| 探测项 | 端点 | 不满足的后果 |
|---|---|---|
| 可达性 | `GET {base_url}/models`（任意 HTTP 200） | **直接退出**并给出指引 |
| 模型列表 | `GET {base_url}/models`，返回 `data[].id` | 不阻断——打印 `model_listed=False` 后继续运行 |
| 流式 | `POST /chat/completions` 带 `stream: true`（SSE `data:` 帧） | 不阻断，但 lc 配对的 TTFT 失效 |
| 流式 usage | `POST /chat/completions` 带 `stream_options.include_usage: true`（末帧出现顶层 `"usage"`） | 不阻断；✗ 时 token 类指标（Prefill/Decode/KV 命中/TPOT）全部 ➖，请检查网关是否透传 `stream_options` |
| 部分释放 | `POST {engine-root}/release_kv_cache`（404/405、HTML catch-all、或 200 + JSON error 响应体均视为无此端点） | 不阻断；**自动禁用 release 请求**（cache_salt 绑定保留），报告会注明 |
| salt+工具调用 | `POST /chat/completions` 同时携带 `cache_sharing`/`cache_salt` 与工具调用消息（据报：MindIE 类引擎返回 HTTP 501，本仓库未独立复核） | 不阻断；✗ 时 **自动禁用 salt 绑定**（`salt_enabled=False`），工具任务以普通 OpenAI 客户端照常执行，报告会注明 |
| 引擎身份 | `GET /version`、`GET /health`、`GET /`、HTTP `Server` 头 | 不阻断；报告开头展示尽力而为的引擎类型/版本与探测依据（HTML/SPA catch-all 响应一律视为"端点不存在"） |

探测之外，要求数据可信还需：

- **usage 透传** —— 响应携带 `usage.prompt_tokens` /
  `completion_tokens` / `prompt_tokens_details.cached_tokens`；缺
  `cached_tokens` 时客户端 KV 命中率显示 ➖。采集端同时兼容
  `usage_metadata` 的 dict 形态（OpenAI 兼容）与命名空间对象形态。
- **亲和字段** —— 引擎须按
  [COMPATIBILITY 第 1.1 节契约](../COMPATIBILITY.zh-CN.md#11-本库发送的接口契约)
  处理 `cache_salt` / `cache_sharing` 与释放端点；否则亲和退化为普通
  客户端，化验单会如实呈现。`cache_salt` 必须**每次调用绑定**：runner
  通过 run metadata 传 `session_id`，亲和模型在 `_generate`/`_agenerate`
  内解析（流式路径也经这两个方法，保证 metadata 不丢失）。MindIE 类
  引擎**据报**会拒绝 salt + 工具调用消息（HTTP 501，agent-core 联调经验，
  本仓库未独立复核）：`salt_tool_calls` 探测会预先发现，客户端也会在
  运行时自动降级（去 salt 重试后对应 session 禁用 salt，其他 session
  不受影响）。
- **不要按 key 限流** —— 四个 agent 设计上共用一个 API key；若网关对
  该 key 配置 RPM/TPM 配额，排队噪声会污染所有时延指标。请在测试
  窗口内放开配额。
- 引擎侧 / NPU 侧指标可选，分别依赖 `--metrics-url`（vLLM 风格
  Prometheus `/metrics`）与 `--npu-cmd`（引擎机上的 `key=value`
  采样命令）。前缀缓存指标名自动识别（V0 计数器、V1
  `prefix_cache_hit_rate` gauge、改名后的 usage 指标），同一套代码
  兼容 vLLM / vLLM-Ascend / 网关透传部署。
- 框架构建失败（如 openJiuwen 未安装）会在报告中以 `build_error`
  行明确标注，而不是静默产出全零数据。`oj-*` 两个 agent 依赖私有包
  `openjiuwen`（内网源，不在 PyPI）——运行机需先
  `pip install openjiuwen`，否则会跳过并标注构建失败。

## 测量指标

| 指标 | 采集方式 | 说明 |
|---|---|---|
| TTFT mean/p50/p95 | `on_llm_start` → 首个 `on_llm_new_token`（两个 lc agent 均开启流式） | oj 侧显示 ➖（agent-core 无 token 级回调） |
| TPOT | `(E2E − TTFT) / (输出 tokens − 1)` | 依赖 usage 透传 |
| E2E（每次 LLM 调用） | 回调墙钟时间 | 四个 agent 均可测 |
| Prefill / decode tokens | `usage_metadata` 汇总（`prompt_tokens` / `completion_tokens`） | 四个 agent 均可测 |
| KV 命中率（客户端） | `cached_tokens / prompt_tokens` | 需引擎上报 `prompt_tokens_details.cached_tokens` |
| Decode tokens/s | decode tokens / decode 时长 | |
| KV 命中率 / KV 内存（引擎侧） | `--metrics-url` Prometheus 快照差分 | 可选，无则 ➖ |
| NPU 利用率 / 带宽 | `--npu-cmd` 采样 | 可选，无则 ➖ |
| 亲和行为 | `affinity_stats`（salt 绑定、释放尝试/失败，跨轮累计） | lc-affinity |
| 正确性 | 按任务的预期关键词命中，跨 agent 对比 | |

任务集：8 段金融投顾对话（调仓 / 风险测评 / 产品对比 / 市场问答），
其中一半含客户端历史改写（前缀差异调度必须检测并释放的形态），
另有可选的长时核查任务。

## 如何读化验单报告

报告输出到 `benchmark/reports/benchmark_report_<ts>.md`（附 `.json`
逐调用原始记录）。类似医疗化验报告，每个指标行带**参考区间**和判定：
✅ PASS / ⚠️ WARN / ❌ FAIL / ➖ N/A。

- **核心四项**：TTFT↓、Prefill tokens/call↓、KV 命中率↑、E2E↓。
  四项**同步改善** = 算力亲和真实生效（前缀缓存命中减少重算）。
- decode 侧指标（TPOT、tokens/s、decode tokens/call）应≈持平 ——
  亲和影响 prefill/缓存，不改变 decode 速度。
- **假亲和警报**：若仅 NPU 侧指标变化而核心四项持平，报告自动给出
  "疑似假亲和"判定。
- openJiuwen 侧以 E2E / Prefill / KV 命中率三项判定（无 token 级
  回调，TTFT 显示 ➖）。
- 主判定使用跨轮中位数，报告含分轮明细；建议 `rounds ≥ 3`。
