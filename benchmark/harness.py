"""Benchmark harness: 4-agent execution, metric collection, lab-sheet report.

Split out of ``run_benchmark.py`` (the CLI) so the module stays under the
pylint size budget. Everything here assumes it is imported as part of the
``benchmark`` package from the repository root.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from benchmark.metrics import (
    AgentMetrics,
    PASS_MARK,
    CallMetrics,
    aggregate,
    cache_hit_rate_delta,
    cache_usage_peak,
    fetch_prometheus,
    judge,
    median_metrics,
    overall_verdict,
    sample_sidecar,
    verdict_text,
)

ALL_AGENTS = ("lc-baseline", "lc-affinity", "oj-baseline", "oj-affinity")
BASELINE_OF = {"lc-affinity": "lc-baseline", "oj-affinity": "oj-baseline"}
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
class LlmCallRecord:
    """One LLM call observed via callbacks (timing + token usage)."""

    agent: str
    task_id: str
    round_idx: int
    ttft_ms: Optional[float]
    e2e_ms: float
    usage: Optional[Tuple[int, int, int]] = None  # (prompt, completion, cached)

    def to_metrics(self) -> CallMetrics:
        """Convert to the shared metric schema."""
        return CallMetrics(
            agent=self.agent,
            task_id=self.task_id,
            round_idx=self.round_idx,
            ttft_ms=self.ttft_ms,
            e2e_ms=self.e2e_ms,
            usage=self.usage,
        )


@dataclass
class TaskResult:
    """Per-task outcome of one agent in one round."""

    agent: str
    task_id: str
    round_idx: int
    turn_e2e_ms: List[float] = field(default_factory=list)
    keyword_score: str = "0/0"
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


class TTFTRecorder(BaseCallbackHandler):
    """Per-LLM-call TTFT/E2E via first-token callbacks + usage passthrough."""

    def __init__(self, agent_name: str, round_idx: int) -> None:
        self.agent_name = agent_name
        self.round_idx = round_idx
        self.calls: List[LlmCallRecord] = []
        self._active: Dict[UUID, Dict[str, Any]] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        self._active[run_id] = {
            "start": time.perf_counter(),
            "agent": str(meta.get("bench_agent", self.agent_name)),
            "task": str(meta.get("bench_task", "?")),
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
            LlmCallRecord(
                agent=entry["agent"],
                task_id=entry["task"],
                round_idx=self.round_idx,
                ttft_ms=_ttft_ms(entry),
                e2e_ms=(time.perf_counter() - entry["start"]) * 1000.0,
                usage=_usage_tuple(response),
            )
        )


def _ttft_ms(entry: Dict[str, Any]) -> Optional[float]:
    first = entry["first"]
    return (first - entry["start"]) * 1000.0 if first else None


def _usage_tuple(response: Any) -> Optional[Tuple[int, int, int]]:
    """Extract (prompt, completion, cached) from the LLMResult generations."""
    try:
        message = response.generations[0][0].message
        usage = message.usage_metadata
        details = getattr(usage, "input_token_details", None)
        return (
            usage.input_tokens,
            usage.output_tokens,
            getattr(details, "cache_read", 0) or 0,
        )
    except (AttributeError, IndexError, TypeError):
        return None


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
    return probe


# -- task execution --------------------------------------------------------------


def _score_reply(reply: str, task: Any, error: Optional[str]) -> str:
    hits = sum(
        1 for keyword in task.expected_keywords if keyword in reply
    ) if error is None else 0
    return f"{hits}/{len(task.expected_keywords)}"


def _next_conversation(
    conversation: List[BaseMessage],
    user_positions: Dict[int, int],
    task: Any,
    turn_idx: int,
    turn_text: str,
) -> None:
    """Append the next user turn in place; rewrite history on corrections."""
    if turn_idx == 0:
        return
    if task.edit_replaces_turn == turn_idx - 1:
        position = user_positions[task.edit_replaces_turn]
        conversation[position] = HumanMessage(content=task.edit_replacement)
        del conversation[position + 1 :]
    conversation.append(HumanMessage(content=turn_text))
    user_positions[turn_idx] = len(conversation) - 1


def _last_ai_reply(messages: List[BaseMessage]) -> str:
    return next(
        (
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, AIMessage) and message.content
        ),
        "",
    )


async def run_lc_task(
    spec: AgentSpec, task: Any, round_idx: int, turn_timeout: float
) -> TaskResult:
    """One deepagents advisor over one task's full dialogue."""
    config = {
        "callbacks": [spec.recorder],
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
    reply = ""
    for turn_idx, turn_text in enumerate(task.turns):
        _next_conversation(conversation, user_positions, task, turn_idx, turn_text)
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
        reply = _last_ai_reply(state["messages"])
        conversation = conversation + [AIMessage(content=reply)]
    return TaskResult(
        agent=spec.name,
        task_id=task.task_id,
        round_idx=round_idx,
        turn_e2e_ms=turn_e2e,
        keyword_score=_score_reply(reply, task, error),
        error=error,
        final_reply=reply[:2000],
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
    return TaskResult(
        agent=spec.name,
        task_id=task.task_id,
        round_idx=round_idx,
        turn_e2e_ms=turn_e2e,
        keyword_score=_score_reply(reply, task, error),
        error=error,
        final_reply=str(reply)[:2000],
    )


async def run_task(
    spec: AgentSpec, task: Any, round_idx: int, timeout: float
) -> TaskResult:
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


def _build_lc_spec(
    name: str, engine: EngineConfig, round_idx: int
) -> AgentSpec:
    """Build one deepagents spec (baseline or affinity model)."""
    from benchmark.agents import build_agent, build_affinity_model, build_baseline_model

    builder = (
        build_baseline_model if name == "lc-baseline" else build_affinity_model
    )
    model = builder(
        model=engine.model, base_url=engine.base_url, api_key=engine.api_key
    )
    return AgentSpec(
        name=name,
        kind="lc",
        agent=build_agent(model),
        recorder=TTFTRecorder(name, round_idx),
        affinity_source=model if name == "lc-affinity" else None,
    )


async def _build_oj_spec(
    name: str, engine: EngineConfig, round_idx: int
) -> AgentSpec:
    """Build one openJiuwen spec (provider InferenceAffinity or OpenAI)."""
    from benchmark.oj_adapter import OJCallCollector, build_openjiuwen_agent

    collector = OJCallCollector(name)
    agent = await build_openjiuwen_agent(
        affinity=name == "oj-affinity",
        model=engine.model,
        base_url=engine.base_url,
        api_key=engine.api_key,
        collector=collector,
    )
    return AgentSpec(
        name=name, kind="oj", agent=agent, collector=collector, affinity_source=None
    )


async def build_agents(
    names: List[str], engine: EngineConfig, round_idx: int
) -> List[AgentSpec]:
    """Build the requested agents fresh per round (counters reset)."""
    specs: List[AgentSpec] = []
    for name in names:
        if name.startswith("lc-"):
            specs.append(_build_lc_spec(name, engine, round_idx))
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

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready projection."""
        return {
            "hit_rate_delta": self.hit_rate_delta,
            "cache_usage_peak": self.cache_usage_peak,
            "npu_samples": self.npu_samples,
        }


@dataclass
class RoundBundle:
    """Mutable collectors shared across one round of all agents."""

    per_agent_records: Dict[str, List[Any]]
    per_agent_rounds: Dict[str, List[AgentMetrics]]
    windows: Dict[str, List[PhaseWindow]]
    results: List[TaskResult]
    affinity_stats: Dict[str, Dict[str, int]]

    @classmethod
    def create(cls, agent_names: List[str]) -> "RoundBundle":
        """Empty bundle for the given agents."""
        return cls(
            per_agent_records={name: [] for name in agent_names},
            per_agent_rounds={name: [] for name in agent_names},
            windows={name: [] for name in agent_names},
            results=[],
            affinity_stats={},
        )


def _collect_call_metrics(spec: AgentSpec) -> Tuple[List[Any], List[CallMetrics]]:
    """Fetch per-call records (warm-ups dropped) + shared-schema metrics."""
    if spec.kind == "lc":
        records = [
            record for record in spec.recorder.calls if record.task_id != "warmup"
        ]
        return records, [record.to_metrics() for record in records]
    spec.collector.drop_warmup()
    records = list(spec.collector.records)
    return records, list(records)


def _absorb_round(spec: AgentSpec, bundle: RoundBundle) -> None:
    """Merge one finished agent phase into the shared bundle."""
    records, call_metrics = _collect_call_metrics(spec)
    bundle.per_agent_records[spec.name].extend(records)
    bundle.per_agent_rounds[spec.name].append(aggregate(call_metrics))
    if spec.affinity_source is not None:
        stats = getattr(spec.affinity_source, "affinity_stats", {})
        bundle.affinity_stats[spec.name] = dict(stats)


async def run_agent_phase(
    spec: AgentSpec,
    tasks: List[Any],
    round_idx: int,
    args: Any,
) -> Tuple[List[TaskResult], PhaseWindow]:
    """Warm-up + all tasks for one agent, wrapped in engine-side snapshots."""
    window = PhaseWindow(prom_before=fetch_prometheus(args.metrics_url or ""))
    window.npu_samples.append(sample_sidecar(args.npu_cmd) or {})
    await run_warmup(spec, round_idx)
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def one(task: Any) -> TaskResult:
        async with semaphore:
            print(f"[r{round_idx}][{spec.name}] {task.task_id} ...", flush=True)
            return await run_task(spec, task, round_idx, args.turn_timeout)

    results = list(await asyncio.gather(*(one(task) for task in tasks)))
    window.prom_after = fetch_prometheus(args.metrics_url or "")
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


async def run_benchmark(
    engine: EngineConfig,
    tasks: List[Any],
    agent_names: List[str],
    args: Any,
) -> Dict[str, Any]:
    """All rounds (rotated order); returns summaries/results/raw calls."""
    bundle = RoundBundle.create(agent_names)
    for round_idx in range(args.rounds):
        order = rotate(agent_names, round_idx)
        print(f"=== round {round_idx + 1}/{args.rounds} order={order} ===", flush=True)
        for spec in await build_agents(order, engine, round_idx):
            round_results, window = await run_agent_phase(
                spec, tasks, round_idx, args
            )
            bundle.results.extend(round_results)
            bundle.windows[spec.name].append(window)
            _absorb_round(spec, bundle)
    summaries = {
        name: {
            "median": asdict(median_metrics(bundle.per_agent_rounds[name])),
            "per_round": [asdict(m) for m in bundle.per_agent_rounds[name]],
            "affinity_stats": bundle.affinity_stats.get(name, {}),
            "engine_windows": [w.as_dict() for w in bundle.windows[name]],
        }
        for name in agent_names
    }
    return {
        "summaries": summaries,
        "results": [asdict(r) for r in bundle.results],
        "llm_calls": [
            asdict(record)
            for name in agent_names
            for record in bundle.per_agent_records[name]
        ],
    }


# -- reporting ---------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def _flat_metrics(agent: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested timing/tokens groups for metric lookups."""
    timing = agent.get("timing") or {}
    tokens = agent.get("tokens") or {}
    return {**timing, **tokens}


def _version_of(package: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "?"


def _render_lab_sheet(
    pair_name: str, affinity: Dict[str, Any], baseline: Dict[str, Any]
) -> List[str]:
    """One lab-sheet table: affinity agent vs its same-framework baseline."""
    aff = _flat_metrics(affinity["median"])
    base = _flat_metrics(baseline["median"])
    verdicts = [
        judge(metric, aff.get(metric), base.get(metric)) for metric in VERDICT_METRICS
    ]
    npu_moved = any(
        any(sample for sample in window.get("npu_samples", []))
        for window in affinity.get("engine_windows", [])
    )
    rows = [
        f"### {pair_name}（跨轮中位数，判定相对其 baseline）",
        "",
        "| 指标 | 参考区间 | baseline → affinity (Δ) | 判定 |",
        "|---|---|---|---|",
    ]
    for verdict in verdicts:
        rows.append(
            f"| {verdict.label} | {verdict.reference} "
            f"| {verdict_text(verdict)} | {PASS_MARK[verdict.status]} |"
        )
    rows += ["", f"**结论：{overall_verdict(verdicts, npu_moved)}**", ""]
    return rows


def _render_rounds_table(
    agent_names: List[str], summaries: Dict[str, Dict[str, Any]]
) -> List[str]:
    rows = [
        "| Agent | 轮次 | LLM调用 | TTFT mean(ms) | E2E mean(ms) |"
        " KV命中率 | Prefill/call |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in agent_names:
        for round_idx, metrics in enumerate(summaries[name]["per_round"]):
            flat = _flat_metrics(metrics)
            rows.append(
                f"| {name} | {round_idx + 1} | {metrics['llm_calls']} "
                f"| {_fmt(flat.get('ttft_mean_ms'))} "
                f"| {_fmt(flat.get('e2e_mean_ms'))} "
                f"| {_fmt(flat.get('kv_hit_rate'))}% "
                f"| {_fmt(flat.get('prefill_per_call'))} |"
            )
    return rows


def _render_correctness(
    tasks: List[Any],
    results: List[Dict[str, Any]],
    agent_names: List[str],
) -> List[str]:
    by_key = {(r["agent"], r["task_id"]): r for r in results}
    rows = [
        "## 5. 正确性（关键词得分按任务）",
        "",
        f"| 任务 | {' | '.join(agent_names)} |",
        "|" + "---|" * (len(agent_names) + 1),
    ]
    for task in tasks:
        cells = []
        for name in agent_names:
            record = by_key.get((name, task.task_id))
            score = record["keyword_score"] if record else "n/a"
            if record and record.get("error"):
                score += f" ⚠️{str(record['error'])[:40]}"
            cells.append(score)
        rows.append(f"| {task.task_id} ({task.category}) | " + " | ".join(cells) + " |")
    return rows


def _render_affinity_evidence(
    summaries: Dict[str, Dict[str, Any]], agent_names: List[str]
) -> List[str]:
    rows = ["## 6. 亲和行为证据", ""]
    for name in agent_names:
        if "affinity" not in name:
            continue
        stats = summaries[name].get("affinity_stats", {})
        rows.append(f"- **{name}** 亲和计数：{stats or '（openJiuwen 侧见引擎日志）'}")
        windows = summaries[name]["engine_windows"]
        hit_rates = [w["hit_rate_delta"] for w in windows if w["hit_rate_delta"] is not None]
        if hit_rates:
            rows.append(f"  - 引擎侧前缀命中率（各轮窗口）：{hit_rates}")
        peaks = [w["cache_usage_peak"] for w in windows if w["cache_usage_peak"] is not None]
        if peaks:
            rows.append(f"  - KV Cache 占用峰值：{peaks}")
        npu = [w["npu_samples"] for w in windows if any(w["npu_samples"])]
        if npu:
            rows.append(f"  - NPU 采样：{npu}")
    rows.append("- 未配置 --metrics-url / --npu-cmd 时引擎侧指标为 ➖，不影响客户端判定。")
    return rows


def render_report_md(
    *,
    engine: EngineConfig,
    probe: Dict[str, Any],
    tasks: List[Any],
    agent_names: List[str],
    summaries: Dict[str, Dict[str, Any]],
    results: List[Dict[str, Any]],
    args: Any,
    fingerprint: str,
    json_name: str,
) -> str:
    """Render the full lab-sheet Markdown report."""
    env_lines = [
        "# 昇腾算力亲和基准测试报告（4 Agent 化验单）",
        "",
        "## 1. 测试环境",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 引擎：`{engine.base_url}`（模型 `{engine.model}`）",
        f"- 框架：deepagents {_version_of('deepagents')}, "
        f"langchain {_version_of('langchain')}, "
        f"openjiuwen {_version_of('openjiuwen')}, "
        f"langchain-ascend-affinity {_version_of('langchain-ascend-affinity')}",
        f"- 采样基线：rounds={args.rounds}，agent 顺序每轮轮转，"
        f"每 agent 每轮前 1 次不计分预热；temperature=0.3 固定；"
        f"并发={args.max_parallel}；单轮超时={args.turn_timeout}s",
        f"- 任务集指纹：`{fingerprint}`（输入字节级固定的证明）",
        "",
        "### 引擎能力探测",
        "",
        f"- 可达：{'✓' if probe.get('reachable') else '✗'}"
        f"（模型在列表中：{'✓' if probe.get('model_listed') else '✗'}）",
        f"- `/release_kv_cache`：{'✓' if probe.get('release_endpoint') else '✗'}",
        f"- 流式输出：{'✓' if probe.get('streaming') else '✗'}",
        "",
        "## 2. 任务集（金融场景）",
        "",
        f"- 共 {len(tasks)} 个任务；"
        f"含历史改写 {sum(1 for t in tasks if t.edit_replaces_turn >= 0)} 个",
        "",
        "## 3. 化验单（核心对比）",
        "",
    ]
    for affinity_name, baseline_name in BASELINE_OF.items():
        if affinity_name in summaries and baseline_name in summaries:
            env_lines += _render_lab_sheet(
                f"{affinity_name} vs {baseline_name}",
                summaries[affinity_name],
                summaries[baseline_name],
            )
    tail_lines = [
        "## 4. 分轮明细（每 Agent 每轮）",
        "",
        *_render_rounds_table(agent_names, summaries),
        "",
        *_render_correctness(tasks, results, agent_names),
        "",
        *_render_affinity_evidence(summaries, agent_names),
        "",
        "## 7. 如何读本报告（判定规则）",
        "",
        "- 每个指标自带参考区间：✅ 达标（绿）、⚠️ 边界、❌ 异常、➖ 数据不可得。",
        "- **核心四项**：TTFT↓、Prefill tokens/call↓、KV 命中率↑、E2E↓。",
        "  四项同步改善 = 算力亲和真实生效（前缀缓存命中减少重算）。",
        "- decode 相关（TPOT、tokens/s、decode tokens/call）应≈持平：",
        "  亲和只影响 prefill/缓存，decode 阶段速度理论上不变。",
        "- **假亲和警报**：若仅 NPU 利用率/带宽等资源指标变化，而核心四项持平，",
        "  报告自动给出 ❌ 疑似假亲和 —— 说明只是换了设备而非调度生效。",
        "- openJiuwen 侧 TTFT/TPOT 为 ➖：agent-core 回调无 token 级事件，",
        "  其判定依赖 E2E/Prefill/KV 命中三项。",
        "- 单轮样本噪声大，主判定用跨轮中位数；建议 rounds≥3。",
        "",
        "## 8. 附录",
        "",
        f"- 每次调用的原始记录（时延+token 用量）：`{json_name}`",
        "",
    ]
    return "\n".join(env_lines + tail_lines)


def write_reports(
    *,
    report_dir: Path,
    engine: EngineConfig,
    probe: Dict[str, Any],
    tasks: List[Any],
    agent_names: List[str],
    data: Dict[str, Any],
    args: Any,
    fingerprint: str,
) -> Path:
    """Write the JSON + Markdown reports; returns the Markdown path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"benchmark_report_{stamp}.json"
    md_path = report_dir / f"benchmark_report_{stamp}.md"
    json_path.write_text(
        json.dumps(
            {
                "meta": {
                    "started": datetime.now().isoformat(),
                    "engine": asdict(engine),
                    "rounds": args.rounds,
                    "agents": agent_names,
                    "task_fingerprint": fingerprint,
                },
                "probe": probe,
                **data,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(
        render_report_md(
            engine=engine,
            probe=probe,
            tasks=tasks,
            agent_names=agent_names,
            summaries=data["summaries"],
            results=data["results"],
            args=args,
            fingerprint=fingerprint,
            json_name=json_path.name,
        ),
        encoding="utf-8",
    )
    return md_path
