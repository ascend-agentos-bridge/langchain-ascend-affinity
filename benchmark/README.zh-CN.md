# 基准测试：deepagents × 昇腾算力亲和（真实引擎）

[English](README.md) | [简体中文](README.zh-CN.md)

面向**真实**昇腾推理引擎（MindIE / vLLM-Ascend）的单变量基准测试。
**不提供模拟引擎**：引擎不可达时脚本直接退出并给出指引。

- **baseline**：`deepagents` 投顾智能体 + 原生 `ChatOpenAI`
- **affinity**：同一智能体（相同工具、相同指令、相同任务集）+
  `AscendAffinityChatModel` —— salt 绑定 + 前缀差异检测 + 部分 KV 释放

唯一变量是聊天模型对象。

## 快速开始

```bash
# 一键：安装基准依赖 + 引擎探测 + 跑双智能体 + 出报告
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

可选项：`--api-key`（兜底 `ASCEND_API_KEY` 环境变量，本地无鉴权引擎默认
`EMPTY`）、`--max-parallel`（任务并发数，默认 2）、`--turn-timeout`（单轮
超时秒数，默认 240）、`--report-dir`（报告目录）。

## 度量什么

| 指标 | 方式 |
|---|---|
| TTFT（mean / p50 / p95） | 每次真实 LLM 调用的首 token 时延（`on_llm_start` → 首个 `on_llm_new_token`） |
| 单轮 E2E | 每轮对话的墙钟时间 |
| 亲和行为 | `affinity_stats`：salt 绑定请求数、释放尝试/失败次数 |
| 正确性 | 按任务的预期关键词命中，两智能体对照 |

任务集为 8 段金融投顾对话（调仓 / 风险测评 / 产品对比 / 市场问答），
其中一半包含客户端历史改写（用户修改早前消息）——正是前缀差异调度
必须检测并释放的模式。

## 如何读报告

报告输出在 `benchmark/reports/benchmark_report_<时间戳>.md`（另有含全部
原始调用记录的 `.json`）。章节：环境与引擎能力探测、任务集、结果对比
表、按任务正确性、自动解读与公平性声明。单次运行样本量小，建议多轮
运行取中位数。
