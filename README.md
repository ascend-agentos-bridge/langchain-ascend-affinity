# langchain-ascend-affinity

[English](README.md) | [简体中文](README.zh-CN.md)

openJiuwen agent-core's Ascend compute-affinity mechanism, ported to the
LangChain ecosystem. Swap one model object and agents built with
**langchain**, **langgraph** or **deepagents** become cache-affine to the
inference engine — no callbacks, no handler wiring.

## How it works

`AscendAffinityChatModel` does exactly what agent-core's
`InferenceAffinityModelClient` + `KVCacheManager` do, per LLM call:

1. **Salt binding** — every `/v1/chat/completions` request carries
   `cache_sharing: true` + `cache_salt: <session_id>` (native vLLM /
   vLLM-Ascend prefix-cache salt), so each session gets an isolated KV-cache
   bucket instead of thrashing a shared one.
2. **Prefix-diff scheduling** — the outgoing `(messages, tools)` window is
   diffed against the previous window for that session. Pure appends (the
   normal agent loop) keep the prefix cache hot; rewritten history
   (`trim_messages`, summarization, deepagents context editing) yields the
   first divergent index.
3. **Partial release** — on divergence the model posts the previous window to
   `POST {engine}/release_kv_cache` with `messages_released_index` /
   `tools_released_index` (byte-compatible with agent-core), so the engine
   drops only the stale KV blocks and keeps the valid prefix resident.
   Release failures are logged, never fatal.

## Installation

```bash
pip install langchain-ascend-affinity
# requires langchain-core >=1.0; no other runtime deps
```

## Quick Start (LangChain 1.x)

The shared piece — **one model instance for all conversations**; the session
travels with each `invoke` call (run metadata → `cache_salt`), so one agent
can serve many users without instance-per-session churn:

```python
from langchain_ascend import AscendAffinityChatModel

llm = AscendAffinityChatModel(
    base_url="http://127.0.0.1:8000/v1",  # MindIE / vLLM-Ascend endpoint
    model="Qwen3-32B",
)

config = {"metadata": {"session_id": "user-123"}}  # per-conversation salt
```

### langchain

```python
from langchain.agents import create_agent

agent = create_agent(
    llm.bind_tools([lookup_quote, calculator]),
    system_prompt="You are a financial advisor.",
)
result = agent.invoke(
    {"messages": [("user", "Plan a 3-year monthly fund investment for me")]},
    config={"metadata": {"session_id": "user-123"}},
)
```

### langgraph

```python
from langgraph.graph import END, START, MessagesState, StateGraph

def advise(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("advise", advise)
graph.add_edge(START, "advise")
graph.add_edge("advise", END)
app = graph.compile()
app.invoke(
    {"messages": [("user", "Check SH000001 and advise")]},
    config={"metadata": {"session_id": "user-123"}},
)
```

### deepagents

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=llm,
    tools=[lookup_quote, calculator],
    system_prompt="You are a financial advisor.",
)
result = agent.invoke(
    {"messages": [("user", "Research index funds and draft a plan")]},
    config={"metadata": {"session_id": "user-123"}},
)
```

deepagents' mid-run context editing (summarization rewrites history) is where
the prefix-diff scheduler pays off: every rewrite triggers exactly one
partial release, keeping the valid prefix cache-resident.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible engine endpoint |
| `model` | `ascend-chat` | model name advertised to the engine |
| `session_id` | `None` | fallback salt when no per-call session is bound (single-session apps) |
| `enable_affinity` | `True` | `False` = plain OpenAI-compatible client |
| `release_endpoint` | `/release_kv_cache` | partial-release path; `""` disables |
| `enable_agent_hint` | `False` | opt-in agent_hint lifecycle protocol (see below) |
| `timeout` / `api_key` / `temperature` / `top_p` / `max_tokens` | — | standard request options |

Session resolution per call: per-call / `bind(session_id=...)` kwargs → run
metadata (`config={"metadata": {"session_id": ...}}`, recommended for
multi-session services — propagates through agents/graphs) → constructor
`session_id` (fallback).

## Engine interface requirements

`AscendAffinityChatModel` works with any OpenAI-compatible engine, but the
affinity gain is conditional on the following interface contract.

Notation: `base_url` is the OpenAI-compatible base (e.g. `http://host:8000/v1`);
`engine-root` is the same origin without the `/v1` suffix, where the release
endpoint lives.

**Required baseline (works everywhere)**

| Interface | Requirement |
|---|---|
| `POST {base_url}/chat/completions` | OpenAI-compatible `messages`; `tools` for tool-calling agents; `stream` (SSE) for TTFT measurement |
| Auth | `Authorization: Bearer <api_key>` |

**Affinity contract (decides whether there is any gain)**

| What this library sends | Engine expectation | If missing / ignored |
|---|---|---|
| `cache_sharing: true` in the request body | opt the session into prefix-cache sharing | no gain, harmless |
| `cache_salt: <session_id>` in the request body | vLLM-style prefix-cache salt: the salt is injected into the first block hash, giving same-salt sessions an isolated KV namespace that different-salt requests cannot reuse (eviction under memory pressure still follows engine policy) | falls back to the shared cache bucket — no isolation, no gain |
| `POST {engine-root}/release_kv_cache` with `model`, `cache_salt`, `cache_sharing`, `messages`, `messages_released_index` (+ `tools`, `tools_released_index`) — only when the client detects a rewritten prefix | agent-core-compatible partial release: drop blocks from the released index, keep the valid prefix | `releases_failed` counter + warning; rewrite-heavy agent loops lose the release gain |

**No session bound** — the model sends no affinity fields at all and stays
a plain OpenAI client. This is deliberate: ``cache_sharing`` without a
``cache_salt`` would lump every anonymous request into one shared cache
bucket and risk cross-session KV pollution.

Degradation is always safe: standards-compliant gateways ignore unknown
fields, release failures stay non-fatal warnings, and the model keeps working
as a plain OpenAI client. The benchmark makes the engine's actual behaviour
visible (release-endpoint probe, `affinity_stats`, suspected-false-affinity
alert) — see [benchmark/PRINCIPLES.md](benchmark/PRINCIPLES.md).

**MindIE status** (checked against MindIE 3.0.0 public docs, 2026-08):
the public RESTful surface exposes no per-request `cache_salt`, no
`/release_kv_cache` endpoint and no `agent_hint` field; its Prefix Cache
is content-hash based cross-session reuse, enabled server-side via
`plugin_params: {"plugin_type":"prefix_cache"}` in `config.json`, and
cannot stack with function call (multiturn) + context/sequence parallel.
On stock MindIE this library therefore degrades to "plain OpenAI client +
engine-global prefix cache" (multi-turn agents still gain from common-prefix
hits, but there is no session isolation or active release).

**vLLM-Ascend status**: `cache_salt` is a native vLLM core request field
(requires `--enable-prefix-caching`) and takes effect directly on
vLLM-Ascend — same-salt reuse, different-salt isolation — making it the
most convincing real-engine validation platform today. Note, however, that
its semantics are an isolation namespace rather than a residency guarantee,
and `/release_kv_cache` and `agent_hint` still do not exist. The full
affinity gain depends on vLLM RFC #37168 landing or a custom engine
carrying the agent-hint patch. See the per-item mapping in section 1.4 of
[benchmark/PRINCIPLES.md](benchmark/PRINCIPLES.md).


## openjiuwen agent-core protocol compatibility

This library tracks the affinity protocols of
[openjiuwen agent-core](https://github.com/openJiuwen-ai/agent-core) and is
checked against its affinity commits on every maintenance pass.

| Protocol | What the client sends | Status in this library |
|---|---|---|
| **release** (default) | `cache_sharing: true` + `cache_salt: <session_id>` on every bound request; on rewritten history `POST {engine-root}/release_kv_cache` with `model` / `cache_salt` / `cache_sharing` / `messages` / `messages_released_index` (+ `tools` / `tools_released_index`) | byte-compatible with agent-core `InferenceAffinityModelClient.release()`; prefix-diff scheduling is automatic |
| **agent_hint lifecycle** (stage A, opt-in) | `agent_hint: {session_id, parent_session_id}` identity on chat requests; `evict_kvc` / `offload_kvc` / `prefetch_kvc` management methods send `context_management: {manage_request: true, edits: [{type, target, start, end}]}` | field-for-field with agent-core `AscendAffinityModelClient` (2026-07 `63380f17e8`); management actions default **off** (`enable_agent_hint=True` to enable) |

Both degrade safely: engines that ignore unknown fields treat the request as
a plain OpenAI call, and management failures are logged/counted, never fatal.
The newer lifecycle protocol is introduced in stages — identity + explicit
management first, model-internal auto-scheduling only after real-engine
evidence (see `benchmark/PRINCIPLES.md` for the evidence discipline).

## Verification

Real affinity benefit is measured on a real Ascend engine (MindIE /
vLLM-Ascend prefix-cache stats). Without hardware, the unit tests verify the
full protocol contract (salt injection, prefix diff, release scheduling and
transport):

```bash
python -m pytest tests/unit_tests
```

## Benchmark

A single-variable benchmark against a real engine lives in
[benchmark/](benchmark/): two identical `deepagents` advisors over the same
financial task set — baseline on native `ChatOpenAI`, experiment on
`AscendAffinityChatModel`. It measures real TTFT (first token per LLM call)
and affinity behaviour (salt binding, partial releases):

```bash
python benchmark/run_benchmark.py --setup \
  --engine-url http://<engine-host>:<port>/v1 --model <model-name> \
  --api-key <api-key>
```

See [benchmark/README.md](benchmark/README.md) for configuration and how to
read the report.

## Development

```bash
python -m pytest tests/unit_tests   # coverage gate: 90%
python scripts/quality_gate.py      # pylint 10.00/10 + tests
```

Spec-driven design under [openspec/](openspec/projects/affinity-core/proposals/openjiuwen-affinity-port/).
