# Spec: deepagents Affinity Benchmark (real-engine)

## Change Description

Add a `benchmark/` directory containing two `deepagents` agents (baseline
native `ChatOpenAI` vs affinity `AscendAffinityChatModel`), a financial
multi-turn task set with history rewrites, a one-click runner that installs
dependencies, probes a real Ascend engine, executes both agents and emits a
Markdown + JSON comparison report. Extend `AscendAffinityChatModel` with
OpenAI SSE streaming and affinity counters to enable real first-token
measurement.

## Purpose

Measure the compute-affinity mechanism's real effect (TTFT, release
behaviour) on a real Ascend inference engine, with a single-variable
comparison (same framework, same prompt, same tasks; only the model object
differs).

## Requirements

### REQ-1: Library streaming + counters

**Requirement:** `AscendAffinityChatModel` MUST implement `_stream` (OpenAI
SSE `data:` line parsing, `[DONE]` termination, content and tool-call delta
chunks, `run_manager.on_llm_new_token` notifications) applying the identical
affinity pipeline (salt binding + prefix diff + partial release) before the
request. It MUST expose `affinity_stats` as a read-only dict with keys
`affinity_requests`, `salt_bound_requests`, `releases_attempted`,
`releases_failed`.

**Acceptance Criteria:**
- [ ] `.stream()` yields incremental content chunks assembled into the full
      message; tool-call deltas produce `tool_call_chunks`.
- [ ] Streaming requests carry the same affinity fields as non-streaming
      ones for a bound session.
- [ ] Counters increment correctly across bound/unbound/release-failure
      paths and cannot be mutated through the property.

### REQ-2: Financial task set

**Requirement:** `benchmark/tasks.py` MUST define ≥ 8 tasks across 4
categories (portfolio rebalance, risk assessment, product comparison, market
Q&A). Each task is a multi-turn dialogue (≥ 3 turns); at least half include
one client-side history rewrite (an earlier user message is edited and the
conversation re-invoked). Each task declares `expected_keywords` for
correctness scoring. Tools (`get_customer_holdings`, `get_fund_profile`,
`compute_portfolio_risk`) are deterministic, in-memory, no network.

**Acceptance Criteria:**
- [ ] Task set loads without engine access; ids are stable.
- [ ] Rewrite turns replace prior message content instead of appending.

### REQ-3: Two single-variable agents

**Requirement:** `benchmark/agents.py` MUST build both agents with
`deepagents.create_deep_agent` over the same tools, instructions and task
set; the only difference is the model: baseline = `ChatOpenAI`, affinity =
`AscendAffinityChatModel` with affinity enabled. The per-task cache salt is
delivered via invoke metadata (`session_id`), resolved by the affinity model.

**Acceptance Criteria:**
- [ ] Both factories accept identical engine/model parameters.
- [ ] Affinity model resolves a distinct salt per (agent, task) pair.

### REQ-4: One-click runner

**Requirement:** `python benchmark/run_benchmark.py` MUST, in one invocation
(optionally `--setup` to install `benchmark/requirements.txt` first):
probe the engine (models list, `/release_kv_cache` capability, streaming
capability), execute both agents over the task set (sequential turns,
configurable task concurrency), capture per-LLM-call TTFT via
`on_llm_start` → first `on_llm_new_token`, collect affinity counters, score
correctness, and write `benchmark/reports/benchmark_report_<ts>.md` + `.json`.
Engine endpoint comes from `--engine-url` / `--model` or
`ASCEND_ENGINE_URL` / `ASCEND_MODEL`. Unreachable engine → fail fast with
guidance; no simulated fallback.

**Acceptance Criteria:**
- [ ] `--setup` installs deps idempotently.
- [ ] Probe results appear in the report (salt/release/streaming flags).
- [ ] Report contains: environment matrix, task-set summary, comparison
      table (LLM calls, TTFT mean/p50/p95, per-turn E2E, salt-bound
      requests, releases), correctness, auto-interpretation with fairness
      caveats, appendix with raw per-call records.
- [ ] Exit code 0 only when both agents completed all tasks.

### REQ-5: Documentation

**Requirement:** Bilingual `benchmark/README.md` / `README.zh-CN.md`
(configuration, run, report reading) and a synced Benchmark section in both
root READMEs.

**Acceptance Criteria:**
- [ ] Both languages present, root READMEs updated in sync.

### REQ-6: Quality gates

**Requirement:** `python scripts/quality_gate.py` passes: pylint 10.00/10
(including `benchmark/*.py`), unit tests green, library coverage ≥ 90%.

**Acceptance Criteria:**
- [ ] Quality gate exits 0.
