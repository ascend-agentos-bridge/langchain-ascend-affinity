# Proposal: deepagents Affinity Benchmark (real-engine)

## Problem

After the redesign around `AscendAffinityChatModel`, the repo has no
runnable way to demonstrate or measure the plugin against a **real** Ascend
engine. The previous hardware-free mock example was removed (simulated TTFT
is self-fulfilling), so today there is:

1. No agent-level usage sample on `deepagents` (one of the three frameworks
   the README targets).
2. No benchmark that quantifies TTFT / release behaviour on a real engine
   with concurrent sessions and history rewrites.
3. No one-click entry point that sets up dependencies, probes engine
   capabilities, runs the comparison and emits a report.

## Proposed Solution

Add a `benchmark/` directory targeting a **real** OpenAI-compatible Ascend
engine (MindIE / vLLM-Ascend; endpoint supplied via `--engine-url` /
`ASCEND_ENGINE_URL`, never simulated):

- **Two agents, one variable** — both built with `deepagents.create_deep_agent`
  over the same financial tool set, system prompt and task set:
  - baseline: native `langchain_openai.ChatOpenAI`
  - affinity: `AscendAffinityChatModel` (salt binding + prefix diff + partial
    release), session salt injected per task via run metadata
- **Financial task set** — 4 categories (portfolio rebalance, risk
  assessment, product comparison, market Q&A), multi-turn dialogues that
  interleave pure appends with one client-side history rewrite (user edits an
  earlier message) to trigger prefix divergence.
- **Real TTFT capture** — requires library support for streaming
  (`_stream` SSE in `AscendAffinityChatModel`, new in this proposal): a
  callback recorder measures `on_llm_start → first on_llm_new_token` per LLM
  call for both agents.
- **One-click runner** — `python benchmark/run_benchmark.py --setup ...`:
  installs deps, probes engine (model list, `/release_kv_cache`, streaming),
  runs both agents over the task set (configurable concurrency), scores
  answer correctness against per-task keywords, renders a Markdown + JSON
  report with auto-derived interpretation and fairness caveats.
- Bilingual `benchmark/README.md` / `README.zh-CN.md`; root READMEs updated
  (in sync) with a Benchmark section.

## Library change (enabling)

`AscendAffinityChatModel` gains:

- `_stream()` — OpenAI SSE parsing (`stream: true`), yielding
  `ChatGenerationChunk` incl. tool-call chunks, with the identical affinity
  pipeline (salt + diff + release) applied before the request. Async
  streaming delegates to the sync implementation via the BaseChatModel
  default.
- `affinity_stats` — read-only counter dict
  (`affinity_requests`, `salt_bound_requests`, `releases_attempted`,
  `releases_failed`) so callers can observe affinity behaviour without
  sniffing traffic.

## Impact

- **New**: `benchmark/` (tasks, agents, runner, bilingual READMEs,
  requirements.txt), `openspec/` artifacts for this proposal.
- **Modified**: `langchain_ascend/llms/chat_ascend.py` (streaming + stats),
  unit tests (streaming/stats suites), root READMEs (Benchmark section),
  `.gitignore` (`benchmark/reports/`).
- No new runtime dependencies for the library itself; benchmark deps live in
  `benchmark/requirements.txt` (`deepagents`, `langchain-openai`).
- Without a reachable engine the runner fails fast with guidance; it never
  falls back to a simulated engine.
