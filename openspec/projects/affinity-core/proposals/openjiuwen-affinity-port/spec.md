# Spec: openJiuwen Affinity Port

## Change Description

Replace the home-grown Agent-Hint mechanism (callback handler + backend
adapters + `/agent-hints` channel) with a complete port of openJiuwen
agent-core's compute-affinity mechanism: request-inline `cache_salt` binding,
client-side prefix-diff scheduling, and partial KV release. Introduce
`AscendAffinityChatModel` as the single entry point, rewrite the example as a
lean verification harness, and rewrite the bilingual README Quick Start on
LangChain 1.x with `langchain` / `langgraph` / `deepagents` recipes.

## Purpose

Make agents built with **langchain, langgraph, or deepagents** compute-affine
to an Ascend inference engine exactly the way agent-core agents are — by
swapping one model object, with zero callback wiring — and keep every payload
byte-compatible with the agent-core engine contract.

## Requirements

### REQ-1: Package surface

**Requirement:** The package MUST export exactly:
`AscendAffinityChatModel`, `PrefixCacheTracker`, `ReleasePlan`, `__version__`.
The `callbacks/` and `backends/` subpackages, `AscendChatLLM`,
`AscendAffinityCallbackHandler`, `BaseAscendBackend`, `get_backend`,
`SUPPORTED_BACKENDS` MUST be removed.

**Acceptance Criteria:**
- [ ] `from langchain_ascend import AscendAffinityChatModel, PrefixCacheTracker, ReleasePlan` works.
- [ ] `langchain_ascend.callbacks` / `langchain_ascend.backends` no longer exist.

### REQ-2: AscendAffinityChatModel request contract

**Requirement:** Every generation request MUST carry `cache_sharing: true`
and, when a session is resolvable, `cache_salt: <session_id>`. Session
resolution order per call: per-call/`bind` kwargs → invoke `config.metadata`
→ constructor `session_id` → none. Affinity injection MUST be disable-able
via `enable_affinity=False`.

**Acceptance Criteria:**
- [ ] Bound session → payload contains both fields with correct values.
- [ ] No session → `cache_sharing` still present, `cache_salt` absent.
- [ ] `enable_affinity=False` → neither field present, no diff, no release.

### REQ-3: Prefix-diff scheduling (algorithm fidelity)

**Requirement:** Before every generation with a bound session, the model MUST
diff the outgoing (messages, tools) against the previous window for that
session: first divergent index per axis; pure append or shrink → no release;
divergence → release plan carrying the previous window.

**Acceptance Criteria:**
- [ ] ReAct-style pure appends across ≥3 turns issue zero releases.
- [ ] Rewriting one history message issues exactly one release with the
      previous messages and the correct `messages_released_index`.
- [ ] Tool-schema divergence alone issues a release with
      `tools`/`tools_released_index` set.
- [ ] Diff state is per-session (two interleaved sessions never cross-release).

### REQ-4: Partial release transport

**Requirement:** A release MUST be `POST {engine-root}/release_kv_cache` with
the agent-core payload (`model`, `cache_salt`, `cache_sharing`,
`messages`, `messages_released_index`, optional `tools` +
`tools_released_index`). Engine root is `base_url` without the `/v1` suffix.
Release failure MUST log a warning and MUST NOT abort generation; the tracker
window is updated regardless of release outcome.

**Acceptance Criteria:**
- [ ] Payload byte-shape matches agent-core `InferenceAffinityModelClient.release()`.
- [ ] Engine down/unreachable → generation still succeeds, warning logged.
- [ ] Async path (`ainvoke`) applies the identical affinity pipeline.

### REQ-5: Example verification harness

**Requirement:** `example/` MUST contain: `mock_engine.py` (OpenAI-compatible
chat + salt-bound KV-block simulation + `/release_kv_cache` + `/metrics`),
`verify_affinity.py` (two-phase deterministic loop with mid-session history
rewrite), bilingual READMEs, requirements.txt. It MUST run without NPU and
demonstrate: salt binding, release firing on rewrite, warm-TTFT preservation
vs the plain baseline.

**Acceptance Criteria:**
- [ ] `python example/verify_affinity.py` exits 0 and prints the comparison.
- [ ] Affinity phase shows ≥1 release; baseline shows 0.
- [ ] Both phases produce identical answers (determinism check).

### REQ-6: README Quick Start on LangChain 1.x (three frameworks)

**Requirement:** Both READMEs MUST ship a Quick Start with three runnable
recipes using `AscendAffinityChatModel` as the model:
`langchain` (`langchain.agents.create_agent`), `langgraph`
(`StateGraph` node or `create_react_agent`), `deepagents`
(`deepagents.create_deep_agent`). READMEs MUST stay in sync and describe the
affinity mechanism (salt binding + prefix diff + partial release), not the
removed hint protocol.

**Acceptance Criteria:**
- [ ] Three recipes present in both languages, top language-switch line kept.
- [ ] No references to `AscendAffinityCallbackHandler` / `/agent-hints` remain.

### REQ-7: Quality gates

**Requirement:** `python scripts/quality_gate.py` MUST pass: pylint 10.00/10
over all tracked files, unit tests green, coverage ≥ 90%.

**Acceptance Criteria:**
- [ ] Quality gate exits 0 after the redesign.
