# Proposal: Complete openJiuwen Affinity Port (v0.2 redesign)

## Problem

The current plugin implements a **home-grown** compute-affinity protocol
(`AscendAffinityCallbackHandler` → `POST /agent-hints` with
`offload/prefetch/evict`) that does not exist in any real Ascend inference
engine, and an inline `agent_hint` field that no engine understands. The
benchmark built around it even needed a `RootAwareAffinityHandler` patch to
stop `on_chain_end` from evicting mid-task — evidence that callback-based
scheduling misfires inside LangGraph run trees.

Meanwhile the **source project** — openJiuwen `agent-core` — ships a complete,
working affinity mechanism that this repository was always meant to port:

1. **Request-inline salt affinity** — every `/v1/chat/completions` call carries
   `cache_sharing: true` + `cache_salt: <session_id>` (aligned with native
   vLLM / vLLM-Ascend prefix-cache salt).
2. **Prefix-diff scheduling algorithm** (`KVCacheManager`) — diff the window
   actually sent to the engine against the previous one; pure appends keep the
   prefix cache hot; rewritten history (trim / summarize / compress) yields the
   first divergent message/tool index.
3. **Partial KV release** — `POST /release_kv_cache` with the *previous* window
   and `messages_released_index` / `tools_released_index`, so the engine drops
   only the stale suffix while keeping the valid prefix blocks.

## Proposed Solution

Redesign the package around a **single entry point** that is the direct
LangChain equivalent of agent-core's `InferenceAffinityModelClient` +
`KVCacheManager`:

- `AscendAffinityChatModel(BaseChatModel)` — an OpenAI-compatible chat model
  that, per call: injects `cache_sharing`/`cache_salt`, runs the prefix diff,
  and auto-releases stale KV blocks via `/release_kv_cache`. Affinity is ON by
  default; **no callbacks, no handler wiring, no backend adapters**.
- `PrefixCacheTracker` — the ported `KVCacheManager` scheduling algorithm
  (kept from the interim port, now the only scheduling core).
- Remove the obsolete machinery: `callbacks/` (hint handler), `backends/`
  (offload/prefetch/evict adapters), the inline `agent_hint` field, and the
  hardware-free mock example (simulated TTFT numbers prove nothing; protocol
  behavior is covered by the unit-test contract suites).
- Rewrite the bilingual root README Quick Start on **LangChain 1.x**, with
  three copy-paste recipes: `langchain` (`create_agent`), `langgraph`
  (`StateGraph` + `create_react_agent`), and `deepagents`
  (`create_deep_agent`).

## Impact

- **Removed**: `langchain_ascend/callbacks/`, `langchain_ascend/backends/`,
  `AscendChatLLM` (superseded by `AscendAffinityChatModel`), and the whole
  `example/` directory (mock engine + benchmark + verification harness).
- **Added**: `AscendAffinityChatModel`, rewritten contract test suites.
- **Breaking** for v0.1 users: callback handler and backend exports disappear;
  the model class replaces `AscendChatLLM` (same constructor shape plus
  affinity defaults).
- No new runtime dependencies (stdlib HTTP + langchain-core).
