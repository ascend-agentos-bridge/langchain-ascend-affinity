"""One-click 4-agent affinity benchmark against a REAL Ascend engine.

Usage (from the repository root):

    python benchmark/run_benchmark.py --setup \
        --engine-url http://<engine-host>:<port>/v1 --model <model-name> \
        --api-key <api-key>

Engine access can also be provided via ASCEND_ENGINE_URL / ASCEND_MODEL /
ASCEND_API_KEY environment variables. There is NO simulated fallback: if the
engine is unreachable the runner exits with guidance.

Four agents over the same financial task set (single variable per pair):

- ``lc-baseline``  deepagents + native ChatOpenAI
- ``lc-affinity``  deepagents + AscendAffinityChatModel (salt/prefix/release)
- ``oj-baseline``  openJiuwen ReActAgent, provider OpenAI, KV release off
- ``oj-affinity``  openJiuwen ReActAgent, provider InferenceAffinity, on

Rounds are baselined: byte-identical inputs (task fingerprint recorded),
rotated agent order per round, one untimed warm-up per agent per round,
cross-round medians as headline numbers. The Markdown report renders a
lab sheet: per-metric reference ranges plus PASS/WARN/FAIL verdicts, an
overall verdict and a suspected-false-affinity alert.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position  # imports need the repo-root path hack
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from benchmark.metrics import (
    AgentMetrics,
    CallMetrics,
    aggregate,
    cache_hit_rate_delta,
    cache_usage_peak,
    fetch_prometheus,
    median_metrics,
    sample_sidecar,
    usage_field,
)
from benchmark.reporting import _REPORT_DIR_DEFAULT, write_reports

ALL_AGENTS = ("lc-baseline", "lc-affinity", "oj-baseline", "oj-affinity")
VERDICT_METRICS = (
    "ttft_mean_ms",
    "prefill_per_call",
    "kv_hit_rate",
    "e2e_mean_ms",
    "decode_per_call",
    "tpot_mean_ms",
    "decode_tps",
)


@dataclass
class EngineConfig:
    """Resolved engine access parameters."""

    base_url: str
    engine_root: str
    model: str
    api_key: str


@dataclass
class LlmCallRecord:  # pylint: disable=too-many-instance-attributes  # data carrier
    """One LLM call observed via callbacks (timing + token usage)."""

    run_id: str
    agent: str
    task_id: str
    round_idx: int
    ttft_ms: Optional[float]
    e2e_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


@dataclass
class TaskResult:  # pylint: disable=too-many-instance-attributes  # data carrier
    """Per-task outcome of one agent in one round."""

    agent: str
    task_id: str
    category: str
    round_idx: int
    turn_e2e_ms: List[float] = field(default_factory=list)
    keyword_hits: int = 0
    keywords_total: int = 0
    error: Optional[str] = None
    final_reply: str = ""


@dataclass
class AgentSpec:
    """One built agent plus its metric collectors."""

    name: str
    kind: str  # "lc" or "oj"
    agent: Any
    recorder: Optional[Any] = None  # TTFTRecorder for lc agents
    collector: Optional[Any] = None  # OJCallCollector for oj agents
    affinity_source: Optional[Any] = None  # object exposing affinity_stats
    build_error: Optional[str] = None  # set when the framework could not build


class TTFTRecorder(BaseCallbackHandler):
    """Per-LLM-call TTFT/E2E via first-token callbacks + usage passthrough."""

    def __init__(self, agent_name: str, round_idx: int) -> None:
        self.agent_name = agent_name
        self.round_idx = round_idx
        self.calls: List[LlmCallRecord] = []
        self._active: Dict[UUID, Dict[str, Any]] = {}

    def _label(self, metadata: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        meta = metadata or {}
        return str(meta.get("bench_agent", self.agent_name)), str(
            meta.get("bench_task", "?")
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        agent, task = self._label(metadata)
        self._active[run_id] = {
            "start": time.perf_counter(),
            "agent": agent,
            "task": task,
            "first": None,
        }

    def on_llm_new_token(
        self, token: str, *, run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        if not token:  # skip framework finalization / empty deltas
            return
        entry = self._active.get(run_id) if run_id is not None else None
        if entry is not None and entry["first"] is None:
            entry["first"] = time.perf_counter()

    def on_llm_end(
        self, response: Any, *, run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        entry = self._active.pop(run_id, None) if run_id is not None else None
        if entry is None:
            return
        self.calls.append(
            _record_from_response(entry, run_id, response, self.round_idx)
        )


def _record_from_response(
    entry: Dict[str, Any], run_id: UUID, response: Any, round_idx: int
) -> LlmCallRecord:
    """Build one call record incl. usage from the LLMResult generations."""
    end = time.perf_counter()
    first = entry["first"]
    prompt_tokens = completion_tokens = cached_tokens = None
    try:
        message = response.generations[0][0].message
        usage = message.usage_metadata
        prompt_tokens = usage_field(usage, "input_tokens")
        completion_tokens = usage_field(usage, "output_tokens")
        details = (
            usage.get("input_token_details")
            if isinstance(usage, dict)
            else getattr(usage, "input_token_details", None)
        )
        cached_tokens = usage_field(details, "cache_read")
    except (AttributeError, IndexError, TypeError):
        pass
    return LlmCallRecord(
        run_id=str(run_id),
        agent=entry["agent"],
        task_id=entry["task"],
        round_idx=round_idx,
        ttft_ms=(first - entry["start"]) * 1000.0 if first else None,
        e2e_ms=(end - entry["start"]) * 1000.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )


def records_to_metrics(records: List[LlmCallRecord]) -> List[CallMetrics]:
    """Convert recorder records to the shared metric schema (skip warmups)."""
    return [
        CallMetrics(
            agent=record.agent,
            task_id=record.task_id,
            round_idx=record.round_idx,
            ttft_ms=record.ttft_ms,
            e2e_ms=record.e2e_ms,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            cached_tokens=record.cached_tokens,
        )
        for record in records
        if record.task_id != "warmup"
    ]


# -- engine probing -------------------------------------------------------------


def _http_json(
    url: str, *, headers: Dict[str, str], payload: Optional[Dict[str, Any]] = None
) -> Tuple[int, Any]:
    """Perform a GET/POST and return (status, parsed-json-or-raw-bytes)."""
    request = urllib.request.Request(url, headers=headers)
    if payload is not None:
        request.data = json.dumps(payload).encode("utf-8")
        request.add_header("Content-Type", "application/json")
        request.method = "POST"
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def probe_engine(engine: EngineConfig) -> Dict[str, Any]:
    """Probe reachability, model list, release endpoint and streaming."""
    auth = {"Authorization": f"Bearer {engine.api_key}"}
    probe: Dict[str, Any] = {"base_url": engine.base_url, "model": engine.model}
    try:
        status, data = _http_json(f"{engine.base_url.rstrip('/')}/models", headers=auth)
        if status != 200:
            raise OSError(f"models endpoint returned {status}")
        entries = data.get("data") if isinstance(data, dict) else None
        probe["reachable"] = True
        probe["model_listed"] = engine.model in [
            entry.get("id") for entry in (entries or [])
        ]
    except OSError as exc:
        probe["reachable"] = False
        probe["error"] = str(exc)
        return probe
    release_status, _ = _http_json(
        f"{engine.engine_root.rstrip('/')}/release_kv_cache",
        headers=auth,
        payload={
            "model": engine.model,
            "cache_salt": "bench-probe",
            "cache_sharing": True,
            "messages": [{"role": "user", "content": "ping"}],
            "messages_released_index": 0,
        },
    )
    probe["release_endpoint"] = release_status not in (404, 405)
    try:
        stream_status, stream_body = _http_json(
            f"{engine.base_url.rstrip('/')}/chat/completions",
            headers=auth,
            payload={
                "model": engine.model,
                "messages": [{"role": "user", "content": "回复：好"}],
                "max_tokens": 8,
                "stream": True,
            },
        )
        probe["streaming"] = stream_status == 200 and "data:" in str(stream_body)
    except OSError:
        probe["streaming"] = False
    try:
        usage_status, usage_body = _http_json(
            f"{engine.base_url.rstrip('/')}/chat/completions",
            headers=auth,
            payload={
                "model": engine.model,
                "messages": [{"role": "user", "content": "回复：好"}],
                "max_tokens": 8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        # vLLM-family engines emit a final SSE event carrying a top-level
        # "usage" block only when stream_options.include_usage is honored.
        probe["stream_usage"] = (
            usage_status == 200 and '"usage"' in str(usage_body)
        )
    except OSError:
        probe["stream_usage"] = False
    return probe


# -- task execution --------------------------------------------------------------


def _score_keywords(reply: str, task: Any, error: Optional[str]) -> Tuple[int, int]:
    hits = sum(1 for keyword in task.expected_keywords if keyword in reply)
    return (hits if error is None else 0), len(task.expected_keywords)


async def run_lc_task(  # pylint: disable=too-many-locals  # orchestration loop
    spec: AgentSpec, task: Any, round_idx: int, turn_timeout: float
) -> TaskResult:
    """One deepagents advisor over one task's full dialogue."""
    recorder = spec.recorder
    config = {
        "callbacks": [recorder],
        "metadata": {
            "session_id": f"bench-{spec.name}-{task.task_id}-r{round_idx}",
            "bench_agent": spec.name,
            "bench_task": task.task_id,
        },
        "recursion_limit": 160,
    }
    conversation: List[BaseMessage] = [HumanMessage(content=task.turns[0])]
    user_positions: Dict[int, int] = {0: 0}
    turn_e2e: List[float] = []
    error: Optional[str] = None
    for turn_idx, turn_text in enumerate(task.turns):
        if turn_idx > 0:
            if task.edit_replaces_turn == turn_idx - 1:
                position = user_positions[task.edit_replaces_turn]
                conversation[position] = HumanMessage(content=task.edit_replacement)
                del conversation[position + 1 :]
            conversation.append(HumanMessage(content=turn_text))
            user_positions[turn_idx] = len(conversation) - 1
        started = time.perf_counter()
        try:
            state = await asyncio.wait_for(
                spec.agent.ainvoke({"messages": conversation}, config=config),
                timeout=turn_timeout,
            )
        except Exception as exc:  # engine/tool failures: record, keep going
            error = f"turn {turn_idx}: {exc}"
            break
        turn_e2e.append(round((time.perf_counter() - started) * 1000.0, 1))
        reply = next(
            (
                str(message.content)
                for message in reversed(state["messages"])
                if isinstance(message, AIMessage) and message.content
            ),
            "",
        )
        conversation = conversation + [AIMessage(content=reply)]
    final_reply = next(
        (
            str(message.content)
            for message in reversed(conversation)
            if isinstance(message, AIMessage)
        ),
        "",
    )
    hits, total = _score_keywords(final_reply, task, error)
    return TaskResult(
        agent=spec.name,
        task_id=task.task_id,
        category=task.category,
        round_idx=round_idx,
        turn_e2e_ms=turn_e2e,
        keyword_hits=hits,
        keywords_total=total,
        error=error,
        final_reply=final_reply[:2000],
    )


async def run_oj_task(
    spec: AgentSpec, task: Any, round_idx: int, turn_timeout: float
) -> TaskResult:
    """One openJiuwen advisor over one task (session memory in-engine)."""
    spec.collector.bind_task(task.task_id, round_idx)
    conversation_id = f"{spec.name}-{task.task_id}-r{round_idx}"
    turn_e2e: List[float] = []
    error: Optional[str] = None
    reply = ""
    for turn_idx, turn_text in enumerate(task.turns):
        if turn_idx > 0 and task.edit_replaces_turn == turn_idx - 1:
            conversation_id += f"-fix{turn_idx}"
            turn_text = f"（更正此前需求，以下述为准重新推理）{turn_text}"
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                spec.agent.invoke(
                    {"query": turn_text, "conversation_id": conversation_id}
                ),
                timeout=turn_timeout,
            )
        except Exception as exc:
            error = f"turn {turn_idx}: {exc}"
            break
        turn_e2e.append(round((time.perf_counter() - started) * 1000.0, 1))
        reply = (
            result.get("output", "")
            if isinstance(result, dict)
            else getattr(result, "content", str(result))
        )
    hits, total = _score_keywords(reply, task, error)
    return TaskResult(
        agent=spec.name,
        task_id=task.task_id,
        category=task.category,
        round_idx=round_idx,
        turn_e2e_ms=turn_e2e,
        keyword_hits=hits,
        keywords_total=total,
        error=error,
        final_reply=str(reply)[:2000],
    )


async def run_task(spec: AgentSpec, task: Any, round_idx: int, timeout: float) -> TaskResult:
    """Dispatch one task to the right framework runner."""
    runner = run_lc_task if spec.kind == "lc" else run_oj_task
    return await runner(spec, task, round_idx, timeout)


async def run_warmup(spec: AgentSpec, round_idx: int) -> None:
    """One untimed warm-up dialogue so no agent starts on a cold engine."""
    try:
        if spec.kind == "lc":
            config = {
                "callbacks": [spec.recorder],
                "metadata": {
                    "session_id": f"warmup-{spec.name}-r{round_idx}",
                    "bench_agent": spec.name,
                    "bench_task": "warmup",
                },
                "recursion_limit": 20,
            }
            await asyncio.wait_for(
                spec.agent.ainvoke(
                    {"messages": [HumanMessage(content="回复：你好，仅预热。")]},
                    config=config,
                ),
                timeout=120,
            )
        else:
            spec.collector.bind_task("warmup", round_idx)
            await asyncio.wait_for(
                spec.agent.invoke(
                    {
                        "query": "回复：你好，仅预热。",
                        "conversation_id": f"warmup-{spec.name}-r{round_idx}",
                    }
                ),
                timeout=120,
            )
    except Exception as exc:  # warm-up failures never abort the benchmark
        print(f"[warmup] {spec.name}: {exc}", flush=True)


# -- agent building --------------------------------------------------------------


async def _build_oj_spec(
    name: str, engine: EngineConfig, round_idx: int
) -> AgentSpec:
    """Build one openJiuwen spec; a framework failure is captured on the
    spec (``build_error``) instead of aborting the whole benchmark."""
    from benchmark.oj_adapter import OJCallCollector, build_openjiuwen_agent

    collector = OJCallCollector(name)
    try:
        agent = await build_openjiuwen_agent(
            affinity=name == "oj-affinity",
            model=engine.model,
            base_url=engine.base_url,
            api_key=engine.api_key,
            collector=collector,
        )
        return AgentSpec(
            name=name, kind="oj", agent=agent, collector=collector,
            affinity_source=None,
        )
    except Exception as exc:  # openJiuwen missing/API drift: record, skip
        error = f"{type(exc).__name__}: {exc}"
        print(f"[build] {name} 构建失败，跳过：{error}", flush=True)
        return AgentSpec(
            name=name, kind="oj", agent=None, collector=collector,
            affinity_source=None, build_error=error,
        )


async def build_agents(
    names: List[str], engine: EngineConfig, round_idx: int, release_enabled: bool
) -> List[AgentSpec]:
    """Build the requested agents fresh per round (counters reset).

    ``release_enabled`` comes from the engine probe: when the engine has no
    ``/release_kv_cache`` endpoint, the affinity model skips release requests
    (salt binding still applies) so no 404 noise inflates ``releases_failed``.
    """
    specs: List[AgentSpec] = []
    for name in names:
        if name.startswith("lc-"):
            from benchmark.agents import build_agent, build_affinity_model, build_baseline_model

            model = (
                build_baseline_model(
                    model=engine.model,
                    base_url=engine.base_url,
                    api_key=engine.api_key,
                )
                if name == "lc-baseline"
                else build_affinity_model(
                    model=engine.model,
                    base_url=engine.base_url,
                    api_key=engine.api_key,
                    release_enabled=release_enabled,
                )
            )
            specs.append(
                AgentSpec(
                    name=name,
                    kind="lc",
                    agent=build_agent(model),
                    recorder=TTFTRecorder(name, round_idx),
                    affinity_source=model if name == "lc-affinity" else None,
                )
            )
        else:
            specs.append(await _build_oj_spec(name, engine, round_idx))
    return specs


# -- orchestration ----------------------------------------------------------------


@dataclass
class PhaseWindow:
    """Optional engine-side snapshot window for one agent phase."""

    prom_before: Optional[Dict[str, float]] = None
    prom_after: Optional[Dict[str, float]] = None
    hit_rate_delta: Optional[float] = None
    cache_usage_peak: Optional[float] = None
    npu_samples: List[Dict[str, float]] = field(default_factory=list)


async def run_agent_phase(
    spec: AgentSpec,
    tasks: List[Any],
    round_idx: int,
    args: argparse.Namespace,
    metrics_url: str,
) -> Tuple[List[TaskResult], PhaseWindow]:
    """Warm-up + all tasks for one agent, wrapped in engine-side snapshots."""
    window = PhaseWindow(prom_before=fetch_prometheus(metrics_url))
    window.npu_samples.append(sample_sidecar(args.npu_cmd) or {})
    if spec.agent is None:  # framework failed to build: skip the phase
        return [], window
    await run_warmup(spec, round_idx)
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def one(task: Any) -> TaskResult:
        async with semaphore:
            print(
                f"[r{round_idx}][{spec.name}] {task.task_id} ...", flush=True
            )
            return await run_task(spec, task, round_idx, args.turn_timeout)

    results = list(await asyncio.gather(*(one(task) for task in tasks)))
    window.prom_after = fetch_prometheus(metrics_url)
    window.npu_samples.append(sample_sidecar(args.npu_cmd) or {})
    window.hit_rate_delta = cache_hit_rate_delta(
        window.prom_before, window.prom_after
    )
    window.cache_usage_peak = cache_usage_peak(window.prom_before, window.prom_after)
    return results, window


def rotate(names: List[str], offset: int) -> List[str]:
    """Rotate agent order per round to cancel cache warm-up bias."""
    if not names:
        return []
    offset %= len(names)
    return names[offset:] + names[:offset]


async def run_benchmark(  # pylint: disable=too-many-locals  # orchestrates all rounds
    engine: EngineConfig,
    tasks: List[Any],
    agent_names: List[str],
    args: argparse.Namespace,
    release_enabled: bool = True,
) -> Dict[str, Any]:
    """All rounds (rotated order) + optional longrun phase."""
    per_agent_records: Dict[str, List[Any]] = {
        name: [] for name in agent_names
    }
    per_agent_rounds: Dict[str, List[AgentMetrics]] = {
        name: [] for name in agent_names
    }
    windows: Dict[str, List[PhaseWindow]] = {name: [] for name in agent_names}
    results: List[TaskResult] = []
    affinity_stats: Dict[str, Dict[str, int]] = {}
    build_errors: Dict[str, str] = {}
    for round_idx in range(args.rounds):
        order = rotate(agent_names, round_idx)
        print(f"=== round {round_idx + 1}/{args.rounds} order={order} ===", flush=True)
        specs = await build_agents(order, engine, round_idx, release_enabled)
        for spec in specs:
            round_results, window = await run_agent_phase(
                spec, tasks, round_idx, args, args.metrics_url or ""
            )
            results.extend(round_results)
            if spec.kind == "lc":
                records = [
                    record
                    for record in spec.recorder.calls
                    if record.task_id != "warmup"
                ]
                call_metrics = records_to_metrics(records)
            else:
                spec.collector.drop_warmup()
                call_metrics = list(spec.collector.records)
                records = list(call_metrics)
            per_agent_records[spec.name].extend(records)
            per_agent_rounds[spec.name].append(aggregate(call_metrics))
            windows[spec.name].append(window)
            if spec.build_error:
                build_errors[spec.name] = spec.build_error
            if spec.affinity_source is not None:
                # counters reset per round on a fresh model: accumulate so the
                # report shows the run-total, not just the last round
                stats = dict(getattr(spec.affinity_source, "affinity_stats", {}))
                previous = affinity_stats.get(spec.name, {})
                affinity_stats[spec.name] = {
                    key: previous.get(key, 0) + value for key, value in stats.items()
                }
    summaries: Dict[str, Dict[str, Any]] = {}
    for name in agent_names:
        rounds_metrics = per_agent_rounds[name]
        summaries[name] = {
            "median": asdict(median_metrics(rounds_metrics)),
            "per_round": [asdict(m) for m in rounds_metrics],
            "affinity_stats": affinity_stats.get(name, {}),
            "build_error": build_errors.get(name),
            "engine_windows": [
                {
                    "hit_rate_delta": w.hit_rate_delta,
                    "cache_usage_peak": w.cache_usage_peak,
                    "npu_samples": w.npu_samples,
                }
                for w in windows[name]
            ],
        }
    return {
        "summaries": summaries,
        "results": results,
        "llm_calls": [
            asdict(record) for name in agent_names for record in per_agent_records[name]
        ],
    }


# -- reporting ---------------------------------------------------------------------


# -- entry point --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """CLI arguments; engine params fall back to environment variables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-url", default=None,
        help="OpenAI-compatible base URL; falls back to ASCEND_ENGINE_URL",
    )
    parser.add_argument(
        "--model", default=None, help="model name; falls back to ASCEND_MODEL",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key; falls back to ASCEND_API_KEY (default EMPTY)",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="pip-install benchmark/requirements.txt before running",
    )
    parser.add_argument(
        "--agents", default="all",
        help="all | lc | oj | comma list of agent names",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="full task-set rounds per agent (median as headline)",
    )
    parser.add_argument(
        "--include-longrun", action="store_true",
        help="add the 25-customer sweep task (~100-150 calls)",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=2,
        help="concurrent tasks per phase",
    )
    parser.add_argument(
        "--turn-timeout", type=float, default=240.0,
        help="per-turn timeout (s)",
    )
    parser.add_argument(
        "--metrics-url", default=None,
        help="optional vLLM /metrics URL for engine-side cache metrics",
    )
    parser.add_argument(
        "--npu-cmd", default=None,
        help="optional sampler command printing key=value pairs (NPU util/HBM)",
    )
    parser.add_argument(
        "--report-dir", default=str(_REPORT_DIR_DEFAULT),
        help="report output dir",
    )
    return parser.parse_args()


def resolve_agents(value: str) -> List[str]:
    """Resolve the --agents selection to concrete agent names."""
    if value == "all":
        return list(ALL_AGENTS)
    if value == "lc":
        return ["lc-baseline", "lc-affinity"]
    if value == "oj":
        return ["oj-baseline", "oj-affinity"]
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in ALL_AGENTS]
    if unknown:
        print(f"未知 agent：{unknown}，可选 {ALL_AGENTS}")
        sys.exit(2)
    return names


def resolve_engine(args: argparse.Namespace) -> EngineConfig:
    """Resolve engine URL/model/key from args + environment."""
    url = args.engine_url or os.environ.get("ASCEND_ENGINE_URL", "")
    model = args.model or os.environ.get("ASCEND_MODEL", "")
    api_key = args.api_key or os.environ.get("ASCEND_API_KEY", "EMPTY")
    if not url or not model:
        print(
            "缺少引擎参数：请通过 --engine-url/--model 或环境变量 "
            "ASCEND_ENGINE_URL/ASCEND_MODEL 提供真实昇腾引擎"
            "（MindIE / vLLM-Ascend）。本基准测试不提供模拟引擎。"
        )
        sys.exit(2)
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return EngineConfig(
        base_url=base, engine_root=base[: -len("/v1")],
        model=model, api_key=api_key,
    )


def run_setup() -> None:
    """Install benchmark-only dependencies (idempotent)."""
    requirements = str(_REPO_ROOT / "benchmark" / "requirements.txt")
    print(f"[setup] pip install -r {requirements}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements], check=True
    )


def main() -> None:
    """One-click entry: setup -> probe -> rounds of 4 agents -> lab report."""
    args = parse_args()
    if args.setup:
        run_setup()
    engine = resolve_engine(args)
    agent_names = resolve_agents(args.agents)
    from benchmark.tasks import load_longrun_tasks, load_tasks, task_fingerprint

    print(f"[probe] {engine.base_url} model={engine.model}")
    probe = probe_engine(engine)
    if not probe.get("reachable"):
        print(
            f"[probe] 引擎不可达：{probe.get('error')}\n"
            "请确认地址正确、服务已启动，且本机可访问该引擎。"
            "本基准测试不提供模拟引擎。"
        )
        sys.exit(2)
    print(
        f"[probe] model_listed={probe.get('model_listed')} "
        f"release_endpoint={probe.get('release_endpoint')} "
        f"streaming={probe.get('streaming')} "
        f"stream_usage={probe.get('stream_usage')}"
    )
    tasks = load_tasks()
    if args.include_longrun:
        tasks += load_longrun_tasks()
    fingerprint = task_fingerprint(args.include_longrun)
    data = asyncio.run(
        run_benchmark(
            engine,
            tasks,
            agent_names,
            args,
            release_enabled=bool(probe.get("release_endpoint")),
        )
    )
    md_path = write_reports(
        report_dir=args.report_dir,
        engine=engine,
        probe=probe,
        tasks=tasks,
        agent_names=agent_names,
        data=data,
        args=args,
        fingerprint=fingerprint,
    )
    print(f"\n报告：{md_path}")
    failed = any(result.error for result in data["results"])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
