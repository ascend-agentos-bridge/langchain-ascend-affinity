# Benchmark: deepagents × Ascend affinity (real engine)

[English](README.md) | [简体中文](README.zh-CN.md)

A single-variable benchmark of the compute-affinity plugin against a **real**
Ascend inference engine (MindIE / vLLM-Ascend). There is **no simulated
engine**: if the engine is unreachable, the runner exits with guidance.

- **baseline**: `deepagents` advisor on native `ChatOpenAI`
- **affinity**: the *same* advisor (same tools, same instructions, same task
  set) on `AscendAffinityChatModel` — salt binding + prefix diff + partial
  KV release

The only variable is the chat model object.

## Quick start

```bash
# one-click: install benchmark deps + probe + run both agents + report
python benchmark/run_benchmark.py --setup \
  --engine-url http://<engine-host>:<port>/v1 \
  --model <model-name>
```

Or use environment variables:

```bash
export ASCEND_ENGINE_URL=http://<engine-host>:<port>/v1
export ASCEND_MODEL=<model-name>
python benchmark/run_benchmark.py
```

Options: `--max-parallel` (concurrent tasks, default 2),
`--turn-timeout` (seconds, default 240), `--report-dir`.

## What it measures

| Metric | How |
|---|---|
| TTFT (mean / p50 / p95) | real per-LLM-call first-token time (`on_llm_start` → first `on_llm_new_token`) |
| Per-turn E2E | wall time of each dialogue turn |
| Affinity behaviour | `affinity_stats`: salt-bound requests, releases attempted/failed |
| Correctness | per-task expected-keyword hits, compared across agents |

The task set is 8 financial-advisor dialogues (rebalance / risk assessment /
fund comparison / market Q&A); half of them include a client-side history
rewrite (the user revises an earlier message), which is the pattern the
prefix-diff scheduler must detect and release.

## Reading the report

Reports land in `benchmark/reports/benchmark_report_<ts>.md` (+ `.json` with
raw per-call records). Sections: environment & engine capability probe,
task set, comparison table, per-task correctness, auto-interpretation with
fairness caveats. Single run, small sample — treat deltas as indicative and
re-run for medians.
