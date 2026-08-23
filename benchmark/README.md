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

For how compute affinity works, how this benchmark proves it, and common
questions (shared API key, cross-contamination, round count...), see
[PRINCIPLES.md](PRINCIPLES.md) (Chinese).

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
- `--metrics-url` (default `http://172.24.107.130:7000/metrics`): vLLM
  `/metrics` endpoint — engine-side cache metrics are collected by default;
  pass `--metrics-url ""` to disable.
- `--log-level DEBUG|INFO|WARNING|ERROR` (default `INFO`): console log
  verbosity. `DEBUG` additionally shows the affinity pipeline's per-request
  salt/release decisions and request payloads.
- `--log-file` (default `benchmark/run.log`): also append the full run log
  (UTF-8) to a file; the report's appendix links it.

## Run log (full-chain observability)

With the default `INFO` level the console shows one line per LLM call, per
task, per agent phase and per engine window — everything needed to spot
*silent* failures (unbound salt, missing usage, dropped framework):

```
[run] engine=http://.../v1 model=dsv4-0731 agents=['lc-baseline', ...] rounds=3
[probe] {"reachable": true, "model_listed": true, "release_endpoint": false, "streaming": true, "stream_usage": false, "salt_tool_calls": false}
=== round 1/3 order=[...] ===
[llm] r0 lc-affinity rebalance-C1001 ttft=1,112ms e2e=8,470ms prompt=1,203 comp=412 cached=0 salt=yes
[task] r0 lc-affinity rebalance-C1001 ok hits=2/3 turns=4 e2e=37,472ms
[phase] r0 lc-affinity: tasks=8 llm_calls=44 ttft_mean=1,615ms e2e_mean=8,188ms salt=45/45 releases=0/0
[engine] r0 lc-affinity hit_rate_delta=62.5% cache_usage_peak=0.872 npu=[{'util': 57.5}]
```

- `[llm]` — one line per LLM call: TTFT / E2E / prompt / completion /
  cached tokens, and `salt=yes|no` (whether the call carried a session id —
  `yes` is the precondition for `cache_salt` binding). `prompt/comp/cached`
  show `None` when the engine (or gateway) does not return usage.
- `[task]` — task outcome with keyword hits and total E2E.
- `[phase]` — per agent per round: call volume, means, and the affinity
  counters for that round (`salt=bound/total`,
  `releases=attempted/failed`, `degraded=salt-rejection fallbacks`).
  **`salt=44/44` (bound == total) proves every request was salt-bound.**
  `degraded=N` means the engine rejected N salt-bound request(s) with
  HTTP 501 and salt binding was then disabled for the instance.
- `[engine]` — engine-side prefix hit rate / KV usage / NPU samples for the
  window (only when `--metrics-url` / `--npu-cmd` are configured).
- `[warmup]` / `[build]` — warm-up outcome and agent build failures (e.g.
  openJiuwen missing), which would otherwise be silent.

At `--log-level DEBUG`, the affinity model itself logs each salt binding and
release decision (session id, released indices) plus request payloads —
use this when a `[phase]` line shows `salt=0/N`.

## Engine interface requirements

At start the runner probes the engine and prints the verdict
(`model_listed / release_endpoint / streaming / stream_usage`). What each
probe needs:

Notation: `{base_url}` is the OpenAI-compatible base URL (`--engine-url`, `/v1`
appended if missing); `{engine-root}` is the same origin without `/v1` (see the
[interface contract](../COMPATIBILITY.md#11-interface-contract-sent-by-this-library)
in COMPATIBILITY.md).

| Probe | Endpoint | Blocks the run? |
|---|---|---|
| reachability | `GET {base_url}/models` (any HTTP 200) | **yes** — exits with guidance |
| model list | `GET {base_url}/models` returning `data[].id` | no — `model_listed=False` is printed; the run continues |
| streaming | `POST /chat/completions` with `stream: true` (SSE `data:` frames) | no — but lc-pair TTFT degrades |
| stream usage | `POST /chat/completions` with `stream_options.include_usage: true` (a final SSE event carrying top-level `"usage"`) | no — when ✗, token-derived metrics (Prefill/Decode/KV hit/TPOT) render ➖; check whether a gateway strips `stream_options` |
| partial release | `POST {engine-root}/release_kv_cache` (404/405 = absent) | no — release requests are **auto-disabled** (salt binding still applies); the lab sheet notes it |
| salt + tool calls | `POST /chat/completions` with `cache_sharing`/`cache_salt` **and** tool-call messages (MindIE-class engines answer HTTP 501) | no — when ✗, salt binding is **auto-disabled** for the affinity agent (`salt_enabled=False`), tool tasks run as a plain OpenAI client, and the lab sheet notes it |
| engine identity | `GET /version`, `GET /health`, `GET /`, HTTP `Server` header | no — informational; the report header shows best-effort engine type/version with the evidence (HTML/SPA catch-all responses are treated as "endpoint absent") |

Beyond the probes, for trustworthy numbers:

- **usage passthrough** — responses should carry
  `usage.prompt_tokens` / `completion_tokens` /
  `prompt_tokens_details.cached_tokens`; without `cached_tokens` the
  client-side KV hit rate renders ➖. The harness reads `usage_metadata`
  as both a dict (OpenAI-compatible) and a namespace (provider objects).
- **affinity fields** — the engine must honour `cache_salt` /
  `cache_sharing` and the release endpoint per the
  [COMPATIBILITY.md contract](../COMPATIBILITY.md#11-interface-contract-sent-by-this-library);
  otherwise affinity degrades to a plain client and the lab sheet will
  show it. The `cache_salt` must be bound *per call*: the runner passes
  `session_id` via run metadata, which the affinity model resolves inside
  `_generate`/`_agenerate` (streaming is routed through those methods so
  the metadata is never dropped). MindIE-class engines **reject**
  salt + tool-call messages with HTTP 501; the `salt_tool_calls` probe
  detects this up front and the client additionally auto-degrades at
  runtime (retry without salt, then disable salt for the instance).
- **no per-key rate limiting** — all four agents share one API key by
  design; RPM/TPM quotas on that key inject queuing noise into every
  latency figure. Lift the quota for the benchmark window.
- optional engine/NPU metrics need `--metrics-url` (vLLM-style Prometheus
  `/metrics`) and `--npu-cmd` (a `key=value` sampler on the engine host).
  Prefix-cache metric names are auto-discovered (V0 counters, V1
  `prefix_cache_hit_rate` gauge, renamed usage metrics), so the same
  harness works across vLLM / vLLM-Ascend / gateway-passthrough setups.
- a framework that fails to build (e.g. openJiuwen not installed) is
  reported as a `build_error` row instead of silently producing zero data.
  The `oj-*` agents need the proprietary `openjiuwen` package (internal
  index, not on PyPI) — `pip install openjiuwen` on the run host, or they
  are skipped and reported.

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
| Affinity behaviour | `affinity_stats` (salt-bound, releases attempted/failed, accumulated across rounds) | lc-affinity |
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
