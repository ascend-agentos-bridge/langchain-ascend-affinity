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
# requires langchain-core >=0.3 (1.x recommended); no other runtime deps
```

## Quick Start (LangChain 1.x)

The shared piece — one model instance per conversation, session bound as the
cache salt:

```python
from langchain_ascend import AscendAffinityChatModel

llm = AscendAffinityChatModel(
    base_url="http://127.0.0.1:8000/v1",  # MindIE / vLLM-Ascend endpoint
    model="Qwen3-32B",
    session_id="user-123",                # cache_salt for this conversation
)
```

### langchain

```python
from langchain.agents import create_agent

agent = create_agent(
    llm.bind_tools([lookup_quote, calculator]),
    system_prompt="You are a financial advisor.",
)
result = agent.invoke(
    {"messages": [("user", "Plan a 3-year monthly fund investment for me")]}
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
app.invoke({"messages": [("user", "Check SH000001 and advise")]})
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
    {"messages": [("user", "Research index funds and draft a plan")]}
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
| `session_id` | `None` | cache salt when no per-call session is bound |
| `enable_affinity` | `True` | `False` = plain OpenAI-compatible client |
| `release_endpoint` | `/release_kv_cache` | partial-release path; `""` disables |
| `timeout` / `api_key` / `temperature` / `top_p` / `max_tokens` | — | standard request options |

Session resolution per call: per-call / `bind(session_id=...)` kwargs → run
metadata → constructor `session_id`.

**Engine requirements**: prefix-cache salt support (vLLM ≥ 0.9 style) for salt
binding, and an agent-core compatible `/release_kv_cache` endpoint for partial
release. Without them the model still works — affinity fields are simply
ignored by the engine and release failures stay non-fatal warnings.

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
