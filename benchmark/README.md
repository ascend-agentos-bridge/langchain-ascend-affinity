# Benchmark: 4-agent lab sheet × Ascend affinity (real engine)

[English](README.md) | [简体中文](README.zh-CN.md)

A four-agent benchmark of compute affinity against a **real** Ascend
inference engine (MindIE / vLLM-Ascend). There is **no simulated engine**:
if the engine is unreachable, the runner exits with guidance.

| Agent | Framework | Model / provider | KV release |
|---|---|---|---|
| `lc-baseline` | deepagents | native `ChatOpenAI` | off |
| `lc-affinity` | deepagents | `AscendAffinityChatModel` | on (salt + prefix diff + partial release) |
| `oj-baseline` | openJiuwen ReActAgent | provider `OpenAI` | off |
| `oj-affinity` | openJiuwen ReActAgent | provider `InferenceAffinity` | on |

Each affinity agent is judged **only against its same-framework baseline**
(single variable per pair). `--agents all|lc|oj|<comma list>` selects a
subset.

## Quick start

```bash
# one-click: install benchmark deps + probe + run all four agents + report
python benchmark/run_benchmark.py --setup \
  --engine-url http://<engine-host>:<port>/v1 \
  --model <model-name> \
  --api-key <api-key>
```

Or use environment variables:

```bash
export ASCEND_ENGINE_URL=http://<engine-host>:<port>/v1
export ASCEND_MODEL=<model-name>
export ASCEND_API_KEY=<api-key>
python benchmark/run_benchmark.py
```

Options:

- `--rounds N` (default 3): full task-set rounds per agent, agent order
  rotates each round, one untimed warm-up per agent per round, medians as
  headline numbers. Inputs are byte-identical across rounds (task-set
  fingerprint printed in the report).
- `--include-longrun`: add a 25-customer portfolio-sweep task
  (~100-150 LLM calls of sustained tool calling).
- `--metrics-url`: optional vLLM `/metrics` endpoint — engine-side
  prefix-cache hit rate + KV-cache usage per agent window.
- `--npu-cmd`: optional sampler command printing `key=value` pairs (e.g.
  NPU utilization / HBM usage via `npu-smi` + `awk` on the engine host).
- `--api-key` (falls back to `ASCEND_API_KEY`, default `EMPTY` for local
  no-auth engines), `--max-parallel` (default 2), `--turn-timeout`
  (default 240 s), `--report-dir`.

## What it measures

| Metric | How | Notes |
|---|---|---|
| TTFT mean/p50/p95 | `on_llm_start` → first `on_llm_new_token` (streaming enabled on both lc agents) | oj agents render ➖ (agent-core has no token-level callbacks) |
| TPOT | `(E2E − TTFT) / (output_tokens − 1)` | from usage passthrough |
| E2E (per LLM call) | callback wall time | all four agents |
| Prefill / decode tokens | `usage_metadata` sums (`prompt_tokens` / `completion_tokens`) | all four agents |
| KV hit rate (client) | `cached_tokens / prompt_tokens` | needs engine to report `prompt_tokens_details.cached_tokens` |
| Decode tokens/s | decode tokens / decode time | |
| KV hit rate / KV memory (engine) | `--metrics-url` Prometheus snapshot deltas | optional, ➖ when absent |
| NPU utilization / bandwidth | `--npu-cmd` sampler | optional, ➖ when absent |
| Affinity behaviour | `affinity_stats` (salt-bound, releases attempted/failed) | lc-affinity |
| Correctness | per-task expected-keyword hits, compared across agents | |

Task set: 8 financial-advisor dialogues (rebalance / risk assessment /
fund comparison / market Q&A); half include a client-side history rewrite
(the pattern the prefix-diff scheduler must detect and release), plus the
optional long-horizon sweep.

## Reading the lab-sheet report

Reports land in `benchmark/reports/benchmark_report_<ts>.md` (+ `.json`
with raw per-call records). Like a medical lab report, every metric row
carries a **reference range** and a verdict: ✅ PASS / ⚠️ WARN / ❌ FAIL /
➖ N/A.

- **Core four**: TTFT ↓, prefill tokens/call ↓, KV hit rate ↑, E2E ↓.
  Improving **together** = real compute affinity (prefix-cache hits
  cutting recompute).
- Decode-side metrics (TPOT, tokens/s, decode tokens/call) should stay
  ≈flat — affinity affects prefill/cache, not decode speed.
- **False-affinity alert**: if only NPU-side metrics move while the core
  four stay flat, the report raises a "suspected false affinity" verdict.
- openJiuwen agents are judged on E2E / prefill / KV hit rate (no
  token-level callbacks for TTFT).
- Headline numbers are cross-round medians; per-round detail is included.
  Keep `rounds ≥ 3` for stable verdicts.
