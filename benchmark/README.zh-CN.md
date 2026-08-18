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

## 引擎接口要求

开跑前脚本会探测引擎并打印结果（`model_listed / release_endpoint /
streaming`）。各探测项对引擎的要求：

符号约定：`{base_url}` 为 OpenAI 兼容基址（`--engine-url`，缺 `/v1` 时自动
补上）；`{engine-root}` 为去掉 `/v1` 的同源地址（见
[COMPATIBILITY 第 1.1 节契约](../COMPATIBILITY.zh-CN.md#11-本库发送的接口契约)）。

| 探测项 | 端点 | 不满足的后果 |
|---|---|---|
| 可达性 | `GET {base_url}/models`（任意 HTTP 200） | **直接退出**并给出指引 |
| 模型列表 | `GET {base_url}/models`，返回 `data[].id` | 不阻断——打印 `model_listed=False` 后继续运行 |
| 流式 | `POST /chat/completions` 带 `stream: true`（SSE `data:` 帧） | 不阻断，但 lc 配对的 TTFT 失效 |
| 部分释放 | `POST {engine-root}/release_kv_cache`（404/405 视为无此端点） | 不阻断，亲和释放收益丧失，`affinity_stats` 可见 |

探测之外，要求数据可信还需：

- **usage 透传** —— 响应携带 `usage.prompt_tokens` /
  `completion_tokens` / `prompt_tokens_details.cached_tokens`；缺
  `cached_tokens` 时客户端 KV 命中率显示 ➖。
- **亲和字段** —— 引擎须按
  [COMPATIBILITY 第 1.1 节契约](../COMPATIBILITY.zh-CN.md#11-本库发送的接口契约)
  处理 `cache_salt` / `cache_sharing` 与释放端点；否则亲和退化为普通
  客户端，化验单会如实呈现。
- **不要按 key 限流** —— 四个 agent 设计上共用一个 API key；若网关对
  该 key 配置 RPM/TPM 配额，排队噪声会污染所有时延指标。请在测试
  窗口内放开配额。
- 引擎侧 / NPU 侧指标可选，分别依赖 `--metrics-url`（vLLM 风格
  Prometheus `/metrics`）与 `--npu-cmd`（引擎机上的 `key=value`
  采样命令）。

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
| 亲和行为 | `affinity_stats`（salt 绑定、释放尝试/失败） | lc-affinity |
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
