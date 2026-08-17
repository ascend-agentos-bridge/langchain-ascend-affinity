# Engine & Gateway Compatibility Matrix

[English](COMPATIBILITY.md) | [简体中文](COMPATIBILITY.zh-CN.md)

> **Why this document exists**: the openjiuwen community ships affinity
> protocol changes faster than inference engines adopt them. A field the
> client sends unilaterally is dead payload unless the engine understands
> it. This matrix answers three questions: which protocol version paired
> with which engine version yields which gains (section 2), which gains
> are invalidated on mismatch (section 3), and whether affinity fields
> survive common LLM gateways (section 4).
>
> **Last verified**: 2026-08-17. Maintenance rules in section 5.

---

## 1. Background: a three-way pace mismatch

As of the verification date, **no official engine release fully implements
the openjiuwen affinity protocol**. The real state is a partial-match
gradient:

- openjiuwen agent-core carries **two protocol generations** (see 2.1):
  the older release protocol (`cache_salt`/`cache_sharing` +
  `/release_kv_cache`) shipped in v0.1.16; the newer `agent_hint`
  lifecycle protocol is only on the develop branch, **never released**.
- The only public engine-side implementation is **vllm-ascend PR #6722**
  (the jiuwen affinity kv-cache plugin, validated against vLLM v0.15.0,
  including the `/release_kv_cache` endpoint) — **still Open, not
  merged**. `agent_hint` was joint-debugged against a Huawei-internal
  vLLM build; no public engine implements it.
- **MindIE (all versions ≤ 3.1.0): zero support** — no `cache_salt`
  request field, no `/release_kv_cache`, no `agent_hint` equivalent; its
  Prefix Cache plugin is content-hash cross-session reuse with feature
  stacking constraints (see 3.2).
- The only affinity capability that genuinely exists in official releases
  is vLLM-core **`cache_salt`** (since v0.9.0, PR #17045) — an isolation
  namespace, not a residency guarantee or active release.

## 2. Version matching list

### 2.1 openjiuwen agent-core protocol timeline

| Version / commit | Date | Protocol shape | What the client sends |
|---|---|---|---|
| v0.1.16 (latest release, PyPI in sync) | 2026-07-14 | **release protocol** (old) | `cache_sharing: true` + `cache_salt: <session_id>`; on rewritten history `POST /release_kv_cache` (`model`/`cache_salt`/`cache_sharing`/`messages`/`messages_released_index`/`tools`/`tools_released_index`) |
| develop `63380f17e8` | 2026-07-22 | **agent_hint lifecycle** (new, +8858/−1054, removes old KVCacheManager) | `agent_hint: {session_id, parent_session_id, context_management: {manage_request, edits: [{type, target, start, end}]}}`; `evict_kvc`/`offload_kvc`/`prefetch_kvc` methods |
| develop `75adc2b44e` | 2026-08-17 | vLLM joint-debug fixes | three-branch URL normalization; multi-shape SSE compatibility; inference-then-manage (`manage_request=false`) |

> This library implements both protocols: release protocol on by default,
> agent_hint protocol opt-in (`enable_agent_hint`), field-aligned with the
> develop branch.

### 2.2 Engine capability × version matrix

| Engine / version | `cache_salt` | `cache_sharing` | `/release_kv_cache` | `agent_hint` |
|---|---|---|---|---|
| vLLM v0.8.x and earlier | ❌ | ❌ | ❌ | ❌ |
| vLLM ≥ v0.9.0 (→ current) | ✅ (PR #17045; `/v1/responses` added later) | ⚠️ accepted via extra=allow, ignored | ❌ (RFC #37168 in proposal) | ❌ |
| vLLM-Ascend v0.7.3 | ❌ (vLLM v0.7.3 lacks the field) | ❌ | ❌ | ❌ |
| vLLM-Ascend ≥ v0.9.1 (→ v0.23+/main, all releases) | ✅ (inherited from the vLLM frontend; requires prefix caching enabled — officially supported since v0.9.1) | ⚠️ ignored | ❌ | ❌ |
| vLLM-Ascend v0.15 + **PR #6722 plugin** (Open, unmerged) | ✅ | ✅ | ✅ | ❌ (the plugin implements the release protocol, not agent_hint) |
| Huawei-internal vLLM (not public) | ? | ? | ? | ✅ (the only agent_hint joint-debug target) |
| MindIE 2.x / 3.0.0 / 3.1.0 | ❌ | ❌ | ❌ | ❌ |
| SGLang (incl. Ascend backend) | ❌ (no per-request salt) | ❌ | ⚠️ full flush `/flush_cache` only | ⚠️ Dynamo-layer `nvext.agent_hints` + Session Control only (experimental) |

### 2.3 Working combinations (the "matching list")

| Combination | Contract coverage | Notes |
|---|---|---|
| **Full release protocol (4/4)**: openjiuwen release protocol × vLLM-Ascend v0.15 + PR #6722 patch | chat + salt + metrics + release | the only combination yielding partial-release gains; requires carrying the patch yourself, no official release |
| **Salt tier (3/4)**: openjiuwen any version × vLLM-Ascend ≥ v0.9.1 (stock) | chat + salt + metrics | **the recommended real-engine validation platform today**: salt isolation and cross-turn hits work; release 404 is expected |
| **Global-cache tier (1/4)**: openjiuwen any version × MindIE any version (stock) | chat only | affinity fields safely ignored; only the engine-global content-hash prefix cache remains (stacking constraints in 3.2) |
| **Lifecycle tier**: openjiuwen develop (agent_hint) × Huawei-internal vLLM | full agent_hint | not reproducible in the public ecosystem; waiting for engine adoption or a PR #6722-style extension |

## 3. Benefit-invalidation analysis on mismatch

### 3.1 Per-mechanism invalidation matrix

| # | Affinity mechanism | Required engine support | vLLM-Ascend ≥ 0.9.1 (stock) | + PR #6722 | MindIE ≤ 3.1.0 (stock) | Consequence when missing |
|---|---|---|---|---|---|---|
| 1 | Session salt binding (`cache_salt`) | engine consumes salt into the first block hash | ✅ works | ✅ | ❌ ignored | MindIE: no session isolation namespace; cross-session KV shares one global cache — isolation gain = 0 |
| 2 | `cache_sharing` flag | engine consumes (non-standard field) | ⚠️ ignored (harmless) | ✅ | ⚠️ ignored (harmless) | no standalone gain; only meaningful alongside salt |
| 3 | Prefix-diff detection | none (pure client-side) | ✅ always works | ✅ | ✅ always works | never invalid; but with no release endpoint the diff result has nowhere to go |
| 4 | Partial release (`/release_kv_cache`) | engine endpoint | ❌ 404 | ✅ | ❌ 404 | **rewritten-history gains fully lost**: stale KV blocks linger in memory until LRU eviction; growing `releases_failed` is expected, not a fault |
| 5 | `agent_hint` identity fields | engine consumes | ⚠️ ignored | ⚠️ ignored | ⚠️ ignored | identity passthrough ineffective; management actions lose their addressing basis |
| 6 | `evict/offload/prefetch_kvc` | engine management implementation | ❌ | ❌ | ❌ | **lifecycle management fully lost**: idle-session KV residency is up to engine LRU |
| 7 | Inference-then-manage (`manage_request=false`) | engine applies edits atomically after generation | ❌ | ❌ | ❌ | same as above |
| 8 | Idle auto-evict | same as #6 | ❌ | ❌ | ❌ | same as above |

### 3.2 Retained benefit by engine

| Engine environment | Retained gains | Invalidated gains | How to tell |
|---|---|---|---|
| vLLM-Ascend ≥ 0.9.1 (stock) | same-salt cross-turn hits during tool-call gaps, prefill ↓, TTFT ↓ (**real and measurable**); cross-session isolation | partial release, all agent_hint lifecycle management; under memory pressure salt buckets are still LRU-evictable (salt is an isolation namespace, not a residency guarantee) | the affinity pair should show KV hit-rate/prefill differences; growing `releases_failed` is expected |
| vLLM-Ascend + PR #6722 | all of the above + precise release on rewritten history (stale blocks dropped, slots freed for active sessions) | agent_hint lifecycle management (not implemented by the plugin) | release counters should turn successful |
| MindIE (stock) | only engine-global prefix-cache hits on common prefixes (system prompt, tool definitions), no session isolation | salt isolation, partial release, lifecycle management — all lost. **Stacking constraint**: MindIE prefix cache does not stack with function call (multiturn) or context parallel + sequence parallel — tool-calling agents may lose even the common-prefix gain, benefit ≈ 0 | zero salt binding and failed releases are expected; with the Prefix Cache plugin enabled, cross-check `--metrics-url` for residual gains |
| Huawei-internal vLLM | full agent_hint (identity + evict/offload/prefetch + inference-then-manage) | — (not verifiable publicly) | n/a |

### 3.3 Conclusion

- The "wishful thinking" risk is real: **of the 8 kinds of affinity payload
  the openjiuwen client sends, only 1 (salt) takes effect on 1 engine
  family (vLLM-based ≥ 0.9.x) in stock form**.
- Hence this library's stance: every field degrades safely (engines that
  ignore unknown fields treat requests as plain OpenAI calls), and the
  benchmark explicitly separates **true affinity / partial gain (plain
  prefix caching) / false affinity** instead of assuming the gain exists.

## 4. LLM gateway passthrough matrix

Scenario: the client carries non-standard fields (`cache_salt` /
`cache_sharing` / `agent_hint`) in the `/v1/chat/completions` body; the
gateway must forward them verbatim to the upstream engine. **A gateway
stripping fields is as fatal as an engine ignoring them: affinity fails
silently.**

| Gateway | Default behavior | Do affinity fields survive | Configuration |
|---|---|---|---|
| Nginx / OpenResty plain reverse proxy | byte-level passthrough, no JSON parsing | ✅ always survive | none; avoid `proxy_set_body` or Lua body rewrites |
| AWS ALB / NLB | L7 forwarding, body untouched | ✅ survive | none |
| vLLM api-server direct | `extra="allow"` accepts unknown fields; `cache_salt` is native | ✅ native support | none |
| Higress ai-proxy (openai/vllm provider) | single-point `model` patch only, rest preserved | ✅ survive | choose the `openai` or `vllm` provider; avoid protocol conversion |
| Kong AI Gateway | same-protocol rewrites only `model`/`stream`; **cross-protocol conversion rebuilds the whole body** | ✅ same-protocol / ❌ cross-protocol lost | keep openai → openai direct |
| LiteLLM proxy | parameter allow-listing | ⚠️ needs config | `litellm_settings: drop_params: false` + `allowed_openai_params: ["cache_salt", "cache_sharing", "agent_hint"]` |
| New API (QuantumNous) | struct-rebuilds the body by default | ⚠️ needs config | enable per-channel "pass-through request body" (PR #1441, `PassThroughBodyEnabled`) |
| APISIX ai-proxy | parses → rebuilds by protocol | ⚠️ needs config | use the `passthrough` protocol (PR #13320) |
| One API (songquanpeng) | verbatim without model mapping; **struct rebuild (unknown fields silently stripped) once model redirect is configured** (issue #2295; fix PR #2384 unmerged) | ⚠️ conditional (model mapping — a common default — drops fields) | avoid model redirect, or apply PR #2384 yourself |

**Deployment preference**: Nginx plain proxy / engine direct → Higress
(openai provider) / Kong (same protocol) → LiteLLM / New API (configured
per above) → avoid One API model redirect. **After any gateway change,
re-run the benchmark's release-endpoint probe and check `affinity_stats`**
to confirm non-zero salt bindings and that releases are not intercepted.

## 5. Maintenance rules

On every maintenance pass (or when an upstream release/PR changes state),
re-verify the following entries, update the matrices in sections 2/3/4,
and refresh the verification date above:

| Check | Entry |
|---|---|
| openjiuwen agent-core protocol evolution | <https://github.com/openJiuwen-ai/agent-core/releases> · develop branch `ascend_affinity_model_client.py` |
| vllm-ascend affinity plugin merge state | <https://github.com/vllm-project/vllm-ascend/pull/6722> |
| vLLM active-release RFC | <https://github.com/vllm-project/vllm/issues/37168> (incl. #37003 RetentionDirective, agentic-api #18) |
| vLLM `cache_salt` semantics | <https://docs.vllm.ai/en/latest/design/prefix_caching/> |
| vLLM-Ascend version mapping | <https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html> |
| MindIE API surface & Prefix Cache constraints | <https://www.hiascend.com/document/detail/zh/mindie/latest/index/index.html> |
| One API passthrough fix | <https://github.com/songquanpeng/one-api/pull/2384> |

**Update triggers**: openjiuwen ships a release tag containing agent_hint;
PR #6722 merges or closes; RFC #37168 lands in a concrete vLLM version;
MindIE's public API adds `cache_salt`/active-release capability; any
gateway's default passthrough behavior changes.
