# langchain-ascend-affinity

[English](README.md) | [简体中文](README.zh-CN.md)

openJiuwen agent-core's Ascend compute-affinity mechanism, ported to the
LangChain ecosystem. Swap one model object and agents built with
**langchain**, **langgraph** or **deepagents** become cache-affine to the
inference engine — no callbacks, no handler wiring.

## How it works

Per LLM call, `AscendAffinityChatModel` does three things:

1. **Salt binding** — every request carries `cache_sharing: true` +
   `cache_salt: <session_id>`, giving each session an isolated KV-cache bucket.
2. **Prefix-diff scheduling** — the outgoing `(messages, tools)` window is
   diffed against the previous window; pure appends (the normal agent loop)
   keep the prefix cache hot, rewrites yield the first divergent index.
3. **Partial release** — on divergence the model posts the stale window to
   `POST {engine}/release_kv_cache`; the engine drops only the stale KV
   blocks. Release failures are logged, never fatal.

## Installation

```bash
pip install langchain-ascend-affinity
# requires langchain-core >=1.0; no other runtime deps
```

## Quick Start (LangChain 1.x)

**One model instance for all conversations** — the session travels with each
`invoke` call (run metadata → `cache_salt`), so one agent serves many users:

```python
from langchain_ascend import AscendAffinityChatModel

llm = AscendAffinityChatModel(
    base_url="http://127.0.0.1:8000/v1",  # MindIE / vLLM-Ascend endpoint
    model="Qwen3-32B",
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
    {"messages": [("user", "Plan a 3-year monthly fund investment for me")]},
    config={"metadata": {"session_id": "user-123"}},  # per-conversation salt
)
```

<details><summary>langgraph</summary>

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

</details>

<details><summary>deepagents</summary>

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
partial release.

</details>

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` | engine endpoint; accepts an origin, a `/v1` base, or a full `/chat/completions` URL |
| `model` | `ascend-chat` | model name advertised to the engine |
| `session_id` | `None` | fallback salt for single-session apps |
| `enable_affinity` | `True` | `False` = plain OpenAI-compatible client |
| `salt_enabled` | `True` | `False` = keep the pipeline counters but never inject `cache_sharing` / `cache_salt` (engines that reject salt-bound tool-call requests, see below) |
| `release_endpoint` | `/release_kv_cache` | partial-release path on the engine root; `""` disables |
| `enable_agent_hint` | `False` | opt into the agent_hint lifecycle protocol (identity fields + `evict` / `offload` / `prefetch` management) |
| `idle_evict_after_seconds` | `0` | auto-evict the session's KV cache after this many seconds idle (`0` = off; requires `enable_agent_hint`) |
| `streaming` | `False` | `invoke()` / `ainvoke()` stream via SSE internally and aggregate, emitting `on_llm_new_token` |
| `timeout` / `api_key` / `temperature` / `top_p` / `max_tokens` | — | standard request options (`api_key=""` for anonymous engines) |

Session resolution per call: per-call / `bind(session_id=...)` kwargs → run
metadata (`config={"metadata": {"session_id": ...}}`, recommended for
multi-session services — propagates through agents/graphs) → constructor
`session_id` (fallback). Without a session the model sends no affinity
fields and stays a plain OpenAI client.

Advanced: `enable_agent_hint` / `idle_evict_after_seconds` opt into the
agent_hint lifecycle protocol (identity + evict / offload / prefetch) —
see [COMPATIBILITY.md](COMPATIBILITY.md).

## Engine support

Works with any OpenAI-compatible engine; the affinity gain depends on the
engine consuming the affinity fields (`cache_salt`, `/release_kv_cache`).
Degradation is always safe: engines that ignore unknown fields treat
requests as plain OpenAI calls, and release failures stay non-fatal
warnings. Engines that actively *reject* the affinity fields (reported
from agent-core joint debugging: HTTP 501 on `cache_salt` + tool-call
messages, MindIE-class servers; not independently verified by this repo)
are handled automatically — the request is retried once without the salt
fields and salt binding is then disabled for the affected session
(`salt_degraded_requests` counter + warning, other sessions unaffected),
so tool-calling agents keep working as a plain OpenAI client.

Which engine versions support what (MindIE, vLLM-Ascend), the full
interface contract, protocol compatibility with openjiuwen agent-core, and
LLM-gateway passthrough behaviour are centrally maintained in
[COMPATIBILITY.md](COMPATIBILITY.md). The benchmark makes the engine's
actual behaviour visible (release-endpoint probe, `affinity_stats`) — see
[benchmark/PRINCIPLES.md](benchmark/PRINCIPLES.md).

## Observability

Every model instance exposes a read-only counter dict via `affinity_stats` —
the same numbers the benchmark reports:

| Key | Meaning |
|---|---|
| `affinity_requests` | requests that entered the affinity pipeline |
| `salt_bound_requests` | requests actually salt-bound to a session |
| `salt_degraded_requests` | requests retried without salt after the engine rejected a salt-bound request (HTTP 501); salt binding is then disabled for that session (other sessions unaffected) |
| `releases_attempted` | partial KV-release requests sent |
| `releases_failed` | release requests that failed (never fatal) |
| `management_requests` | agent_hint `evict` / `offload` / `prefetch` requests sent |
| `management_failed` | management requests that failed (never fatal) |

```python
stats = model.affinity_stats
# {"affinity_requests": 3, "salt_bound_requests": 3, ...}
```

At `DEBUG` log level the model records each salt binding and
prefix-divergence release decision (session id, released indices); failures
are always logged as warnings.

## Async & agent_hint usage

`ainvoke` runs the identical affinity pipeline; the agent_hint management
methods are protocol peers of `invoke` (agent-core parity):

```python
import asyncio

from langchain_core.messages import HumanMessage

from langchain_ascend import AscendAffinityChatModel


async def main() -> None:
    model = AscendAffinityChatModel(
        base_url="http://127.0.0.1:8000/v1",
        enable_agent_hint=True,  # opt into the lifecycle protocol
    )
    reply = await model.ainvoke(
        [HumanMessage(content="hello")],
        config={"metadata": {"session_id": "s1"}},
    )
    print(reply.content)

    # lifecycle management, same semantics as agent-core's methods
    model.evict_kvc(session_id="s1")
    model.offload_kvc(session_id="s1")
    model.prefetch_kvc(session_id="s1")


asyncio.run(main())
```

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
