# Design: openJiuwen Affinity Port

## 1. Source mechanism (openJiuwen agent-core, fixed reference)

| agent-core element | Behavior | Ported equivalent |
|---|---|---|
| `InferenceAffinityModelClient.chat()` | request body carries `cache_sharing: true` and `cache_salt: session_id` (salt only when a session is bound) | `AscendAffinityChatModel._build_payload()` |
| `Context.get_context_window()` + `KVCacheManager._check_release_needed()` | diff (messages, tools) vs previous window; first divergent index; pure append or shrink → no release | `PrefixCacheTracker.check_release_needed()` |
| `InferenceAffinityModelClient.release()` | `POST {engine}/release_kv_cache` with previous window + `messages_released_index` (+ optional tools indices); failure is logged, never fatal | `AscendAffinityChatModel._release_stale()` |
| `KVCacheManager.update()` | record the window that was just sent | `PrefixCacheTracker.update()` |

Payload contracts (byte-compatible with agent-core):

```jsonc
// chat/completions (additions)
{ "cache_sharing": true, "cache_salt": "<session-id>" }

// release_kv_cache
{
  "model": "<model>",
  "cache_salt": "<session-id>",
  "cache_sharing": true,
  "messages": [ /* PREVIOUS window */ ],
  "messages_released_index": 3,
  "tools": [ /* previous tools, only when tools diverged */ ],
  "tools_released_index": 1
}
```

## 2. Key decisions

### D1 — Model-embedded affinity, no callbacks
agent-core hooks context assembly (`get_context_window`), which in LangChain
maps to the chat model call itself: `_generate` / `_agenerate` see exactly the
final message list. Callbacks see run-tree events, misorder under LangGraph
nesting, and require user wiring — dropped entirely.

### D2 — Affinity ON by default
`AscendAffinityChatModel` is the *affinity* model; opting out is explicit
(`enable_affinity=False`). This inverts v0.1's opt-in `enable_cache_sharing`,
making the ported algorithm the product instead of an add-on.

### D3 — Session resolution order (per call)
1. `bind(session_id=...)` / per-call kwargs (`generate(..., session_id=...)`)
2. `invoke(..., config={"metadata": {"session_id": ...}})` — LangChain 1.x
   metadata propagates to the model run
3. constructor `session_id`
4. `None` → no salt, no diff (plain passthrough)

### D4 — Diff semantics identical to agent-core
- messages compared as normalized dicts, pairwise, first mismatch wins
- tools diffed the same way, independently
- pure append / tail-shrink → no release (engine reclaims unused blocks)
- release replay carries the **previous** window so the engine can truncate
  its block table from the divergence point

### D5 — Failure containment
Release POST failures log a warning and never block generation; a failed
release also does **not** corrupt the tracker (the window is still updated).

### D6 — Transport
stdlib `urllib` (no httpx/openai dependency), engine root derived from
`base_url` minus the `/v1` suffix; endpoints overridable via constructor.

## 3. Module layout (v0.2)

```
langchain_ascend/
  __init__.py        # AscendAffinityChatModel, PrefixCacheTracker, ReleasePlan
  prefix_tracker.py  # scheduling algorithm (unchanged from interim port)
  llms/
    __init__.py
    chat_ascend.py   # AscendAffinityChatModel
```


### D6 — agent_hint lifecycle protocol (stage A, opt-in, 2026-08)
Following openjiuwen agent-core's AscendAffinity commit (`63380f17e8`) and
its vLLM joint-debugging fixes (`75adc2b44e`), the model also speaks the
request-inline `agent_hint` lifecycle protocol, opt-in via
`enable_agent_hint`:

- normal calls carry `agent_hint: {session_id, parent_session_id}` identity;
- `evict_kvc` / `offload_kvc` / `prefetch_kvc` send pure management requests
  (`context_management.manage_request=true`, `edits` with `type/target/start/end`);
- per-call `agent_hint_manage={...}` carries `manage_request=false` edits on
  an inference request (inference-then-manage, atomic);
- `idle_evict_after_seconds` (default 0) arms a best-effort timer that evicts
  the session after idle; any new call cancels/re-arms.

All management paths default off, are never fatal, and degrade safely when
the engine ignores unknown fields. The release protocol (D2/D4) remains the
default and is byte-compatible with agent-core `InferenceAffinityModelClient.release()`.
