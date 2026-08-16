"""One-click affinity benchmark against a REAL Ascend engine.

Usage (from the repository root):

    python benchmark/run_benchmark.py --setup \
        --engine-url http://<engine-host>:<port>/v1 --model <model-name>

Engine access can also be provided via ASCEND_ENGINE_URL / ASCEND_MODEL /
ASCEND_API_KEY environment variables. There is NO simulated fallback: if the
engine is unreachable the runner exits with guidance.

The runner probes the engine (model list, /release_kv_cache, streaming),
runs two single-variable deepagents advisors (baseline ChatOpenAI vs
AscendAffinityChatModel) over the same financial task set, measures real
TTFT per LLM call (on_llm_start -> first on_llm_new_token), and writes a
Markdown + JSON comparison report under benchmark/reports/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPORT_DIR_DEFAULT = Path(__file__).resolve().parent / "reports"


@dataclass
class EngineConfig:
    """Resolved engine access parameters."""

    base_url: str
    engine_root: str
    model: str
    api_key: str


@dataclass
class LlmCallRecord:
    """One LLM call observed via callbacks."""

    run_id: str
    agent: str
    task_id: str
    ttft_ms: Optional[float]
    e2e_ms: float
    streamed: bool


@dataclass
class TaskResult:
    """Per-task outcome of one agent."""

    agent: str
    task_id: str
    category: str
    turn_e2e_ms: List[float]
    keyword_hits: int
    keywords_total: int
    error: Optional[str] = None


class TTFTRecorder(BaseCallbackHandler):
    """Measures per-LLM-call TTFT and E2E via first-token callbacks."""

    def __init__(self) -> None:
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
            "agent": str(meta.get("bench_agent", "?")),
            "task": str(meta.get("bench_task", "?")),
            "first": None,
        }

    def on_llm_new_token(
        self, token: str, *, run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        entry = self._active.get(run_id) if run_id is not None else None
        if entry is not None and entry["first"] is None:
            entry["first"] = time.perf_counter()

    def on_llm_end(
        self, response: Any, *, run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        entry = self._active.pop(run_id, None) if run_id is not None else None
        if entry is None:
            return
        end = time.perf_counter()
        first = entry["first"]
        self.calls.append(
            LlmCallRecord(
                run_id=str(run_id),
                agent=entry["agent"],
                task_id=entry["task"],
                ttft_ms=(first - entry["start"]) * 1000.0 if first else None,
                e2e_ms=(end - entry["start"]) * 1000.0,
                streamed=first is not None,
            )
        )


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


def _probe_models(engine: EngineConfig, auth: Dict[str, str]) -> List[Any]:
    """Probe /models; returns [] and sets flags via the probe dict upstream."""
    status, data = _http_json(f"{engine.base_url.rstrip('/')}/models", headers=auth)
    if status != 200:
        return []
    entries = data.get("data") if isinstance(data, dict) else None
    return list(entries or [])


def probe_engine(engine: EngineConfig) -> Dict[str, Any]:
    """Probe reachability, model list, release endpoint and streaming."""
    auth = {"Authorization": f"Bearer {engine.api_key}"}
    probe: Dict[str, Any] = {"base_url": engine.base_url, "model": engine.model}
    try:
        model_entries = _probe_models(engine, auth)
    except OSError as exc:
        probe["reachable"] = False
        probe["error"] = str(exc)
        return probe
    probe["reachable"] = True
    probe["model_listed"] = engine.model in [
        entry.get("id") for entry in model_entries
    ]
    release_payload = {
        "model": engine.model,
        "cache_salt": "bench-probe",
        "cache_sharing": True,
        "messages": [{"role": "user", "content": "ping"}],
        "messages_released_index": 0,
    }
    release_status, _ = _http_json(
        f"{engine.engine_root.rstrip('/')}/release_kv_cache",
        headers=auth,
        payload=release_payload,
    )
    probe["release_endpoint"] = release_status not in (404, 405)
    stream_payload = {
        "model": engine.model,
        "messages": [{"role": "user", "content": "回复：好"}],
        "max_tokens": 8,
        "stream": True,
    }
    try:
        stream_status, stream_body = _http_json(
            f"{engine.base_url.rstrip('/')}/chat/completions",
            headers=auth,
            payload=stream_payload,
        )
        probe["streaming"] = stream_status == 200 and "data:" in str(stream_body)
    except OSError:
        probe["streaming"] = False
    return probe


# -- execution -----------------------------------------------------------------


def _apply_rewrite(
    conversation: List[BaseMessage],
    user_positions: Dict[int, int],
    task: Any,
) -> None:
    """Rewrite the user message of turn ``edit_replaces_turn`` in place."""
    position = user_positions[task.edit_replaces_turn]
    conversation[position] = HumanMessage(content=task.edit_replacement)
    del conversation[position + 1 :]


async def _invoke_turn(
    agent: Any,
    conversation: List[BaseMessage],
    config: Dict[str, Any],
    turn_timeout: float,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """One agent invocation; returns (reply_text, e2e_ms, error)."""
    started = time.perf_counter()
    try:
        state = await asyncio.wait_for(
            agent.ainvoke({"messages": conversation}, config=config),
            timeout=turn_timeout,
        )
    except Exception as exc:  # engine/tool failures: record, keep other tasks
        return None, None, str(exc)
    e2e_ms = (time.perf_counter() - started) * 1000.0
    replies = [
        message
        for message in state["messages"]
        if isinstance(message, AIMessage) and message.content
    ]
    reply = str(replies[-1].content) if replies else None
    return reply, e2e_ms, None


def _task_config(agent_name: str, task: Any, recorder: TTFTRecorder) -> Dict[str, Any]:
    """Invoke config: TTFT callbacks + per-task salt and run labels."""
    return {
        "callbacks": [recorder],
        "metadata": {
            "session_id": f"bench-{agent_name}-{task.task_id}",
            "bench_agent": agent_name,
            "bench_task": task.task_id,
        },
    }


def _finish_task_result(
    agent_name: str,
    task: Any,
    turn_e2e: List[float],
    error: Optional[str],
    reply: str,
) -> TaskResult:
    """Build the TaskResult incl. expected-keyword scoring."""
    hits = sum(1 for keyword in task.expected_keywords if keyword in reply)
    return TaskResult(
        agent=agent_name,
        task_id=task.task_id,
        category=task.category,
        turn_e2e_ms=turn_e2e,
        keyword_hits=hits if error is None else 0,
        keywords_total=len(task.expected_keywords),
        error=error,
    )


async def run_task(
    agent: Any,
    task: Any,
    agent_name: str,
    recorder: TTFTRecorder,
    turn_timeout: float,
) -> TaskResult:
    """Run one agent over one task's full dialogue."""
    config = _task_config(agent_name, task, recorder)
    conversation: List[BaseMessage] = [HumanMessage(content=task.turns[0])]
    user_positions: Dict[int, int] = {0: 0}
    turn_e2e: List[float] = []
    error: Optional[str] = None
    for turn_idx, turn_text in enumerate(task.turns):
        if turn_idx > 0:
            if task.edit_replaces_turn == turn_idx - 1:
                _apply_rewrite(conversation, user_positions, task)
            conversation.append(HumanMessage(content=turn_text))
            user_positions[turn_idx] = len(conversation) - 1
        answer, e2e_ms, invoke_error = await _invoke_turn(
            agent, conversation, config, turn_timeout
        )
        if invoke_error is not None:
            error = f"turn {turn_idx}: {invoke_error}"
            break
        turn_e2e.append(round(e2e_ms or 0.0, 1))
        if answer:
            conversation = conversation + [AIMessage(content=answer)]
    return _finish_task_result(
        agent_name,
        task,
        turn_e2e,
        error,
        next(
            (
                str(message.content)
                for message in reversed(conversation)
                if isinstance(message, AIMessage)
            ),
            "",
        ),
    )


async def run_phase(
    agent_name: str,
    agent: Any,
    tasks: List[Any],
    recorder: TTFTRecorder,
    args: argparse.Namespace,
) -> List[TaskResult]:
    """Run one agent over all tasks with bounded concurrency."""
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def one(task: Any) -> TaskResult:
        async with semaphore:
            print(f"[{agent_name}] running {task.task_id} ...", flush=True)
            return await run_task(agent, task, agent_name, recorder, args.turn_timeout)

    return list(await asyncio.gather(*(one(task) for task in tasks)))


# -- reporting ------------------------------------------------------------------


def _percentile(values: List[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round(ratio * (len(ordered) - 1)))
    return ordered[int(index)]


def summarize(
    results: List[TaskResult], calls: List[LlmCallRecord], stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Aggregate one agent's task results and LLM-call measurements."""
    ttfts = [call.ttft_ms for call in calls if call.ttft_ms is not None]
    e2e_turns = [value for result in results for value in result.turn_e2e_ms]
    return {
        "tasks_completed": sum(1 for r in results if r.error is None),
        "tasks_failed": sum(1 for r in results if r.error is not None),
        "llm_calls": len(calls),
        "streamed_calls": sum(1 for call in calls if call.streamed),
        "ttft_mean_ms": round(statistics.fmean(ttfts), 1) if ttfts else None,
        "ttft_p50_ms": round(_percentile(ttfts, 0.50), 1) if ttfts else None,
        "ttft_p95_ms": round(_percentile(ttfts, 0.95), 1) if ttfts else None,
        "turn_e2e_mean_ms": (
            round(statistics.fmean(e2e_turns), 1) if e2e_turns else None
        ),
        "keyword_score": f"{sum(r.keyword_hits for r in results)}/"
        f"{sum(r.keywords_total for r in results)}",
        "affinity_stats": stats,
    }


def _fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _delta(new: Optional[float], old: Optional[float]) -> str:
    if new is None or old is None or old == 0:
        return "n/a"
    return f"{(new - old) / old * 100.0:+.1f}%"


def _version_of(package: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "?"


def _render_comparison(
    baseline: Dict[str, Any], affinity: Dict[str, Any]
) -> List[str]:
    rows = [
        "| 指标 | baseline | affinity | Δ |",
        "|---|---|---|---|",
        f"| 完成任务 | {baseline['tasks_completed']} | "
        f"{affinity['tasks_completed']} | |",
        f"| LLM 调用 / 流式 | {baseline['llm_calls']} / {baseline['streamed_calls']}"
        f" | {affinity['llm_calls']} / {affinity['streamed_calls']} | |",
    ]
    for label, key in (
        ("TTFT mean (ms)", "ttft_mean_ms"),
        ("TTFT p50 (ms)", "ttft_p50_ms"),
        ("TTFT p95 (ms)", "ttft_p95_ms"),
        ("单轮 E2E mean (ms)", "turn_e2e_mean_ms"),
    ):
        rows.append(
            f"| {label} | {_fmt_ms(baseline[key])} | {_fmt_ms(affinity[key])}"
            f" | {_delta(affinity[key], baseline[key])} |"
        )
    rows.append(
        f"| 关键词得分 | {baseline['keyword_score']} | "
        f"{affinity['keyword_score']} | |"
    )
    return rows


def _render_env_section(engine: EngineConfig, probe: Dict[str, Any]) -> List[str]:
    return [
        "## 1. 测试环境",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 引擎地址：`{engine.base_url}`（模型 `{engine.model}`）",
        f"- 框架版本：deepagents {_version_of('deepagents')}, "
        f"langchain {_version_of('langchain')}, "
        f"langchain-ascend-affinity {_version_of('langchain-ascend-affinity')}",
        "",
        "### 引擎能力探测",
        "",
        f"- 可达：{'✓' if probe.get('reachable') else '✗'}"
        f"（模型在列表中：{'✓' if probe.get('model_listed') else '✗'}）",
        f"- `/release_kv_cache`：{'✓' if probe.get('release_endpoint') else '✗'}",
        f"- 流式输出：{'✓' if probe.get('streaming') else '✗'}",
        "",
    ]


def _render_taskset_section(tasks: List[Any]) -> List[str]:
    categories: Dict[str, int] = {}
    for task in tasks:
        categories[task.category] = categories.get(task.category, 0) + 1
    rewrite_count = sum(1 for task in tasks if task.edit_replaces_turn >= 0)
    return [
        "## 2. 任务集（金融场景）",
        "",
        f"- 共 {len(tasks)} 个任务，分布：{categories}；"
        f"含历史改写任务 {rewrite_count} 个",
        "",
    ]


def _render_correctness_rows(
    tasks: List[Any], results: List[TaskResult]
) -> List[str]:
    by_key = {(r.agent, r.task_id): r for r in results}
    rows = [
        "## 4. 正确性（关键词得分按任务）",
        "",
        "| 任务 | baseline | affinity |",
        "|---|---|---|",
    ]
    for task in tasks:
        base = by_key.get(("baseline", task.task_id))
        aff = by_key.get(("affinity", task.task_id))
        base_score = f"{base.keyword_hits}/{base.keywords_total}" if base else "n/a"
        aff_score = f"{aff.keyword_hits}/{aff.keywords_total}" if aff else "n/a"
        base_err = f"（异常：{base.error}）" if base and base.error else ""
        aff_err = f"（异常：{aff.error}）" if aff and aff.error else ""
        rows.append(
            f"| {task.task_id} | {base_score}{base_err} | {aff_score}{aff_err} |"
        )
    return rows


def _render_interpretation(
    probe: Dict[str, Any], affinity: Dict[str, Any]
) -> List[str]:
    stats = affinity["affinity_stats"]
    lines = []
    if affinity.get("ttft_mean_ms") is None:
        lines.append(
            "- 未能采集流式 TTFT（引擎或框架未产生 token 级回调），"
            "请检查引擎 `stream` 支持；单轮 E2E 仍具参考性。"
        )
    if stats.get("releases_attempted", 0) > 0:
        lines.append(
            f"- 亲和模型共发起 {stats['releases_attempted']} 次部分释放"
            f"（失败 {stats.get('releases_failed', 0)} 次），"
            "说明历史改写被前缀差异检测捕获。"
        )
    else:
        lines.append(
            "- 本次运行未触发部分释放：请确认任务集中含改写轮且 salt 生效。"
        )
    if not probe.get("release_endpoint", False):
        lines.append(
            "- 注意：引擎未提供 `/release_kv_cache` 端点，释放请求无法生效"
            "（salt 绑定本身不受影响）。"
        )
    lines.append(
        "- 公平性声明：两智能体使用同一框架（deepagents）、同一工具与提示词、"
        "同一任务集，唯一变量是模型对象；但本报告为单次运行，样本量小，"
        "结论仅供参考，建议多次运行取中位数。"
    )
    return lines


def render_report_md(
    *,
    engine: EngineConfig,
    probe: Dict[str, Any],
    tasks: List[Any],
    summaries: Dict[str, Dict[str, Any]],
    results: List[TaskResult],
    json_name: str,
) -> str:
    """Render the full Markdown report."""
    sections = [
        "# 昇腾算力亲和基准测试报告",
        "",
        *_render_env_section(engine, probe),
        *_render_taskset_section(tasks),
        "## 3. 结果对比",
        "",
        *_render_comparison(summaries["baseline"], summaries["affinity"]),
        "",
        *_render_correctness_rows(tasks, results),
        "",
        "## 5. 结论与告警",
        "",
        *_render_interpretation(probe, summaries["affinity"]),
        "",
        "## 6. 附录",
        "",
        f"- 每请求原始记录（含每次 LLM 调用的 TTFT/E2E）：`{json_name}`",
        "",
    ]
    return "\n".join(sections)


# -- entry point ----------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """CLI arguments; engine params fall back to environment variables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-url",
        default=None,
        help="OpenAI-compatible engine base URL (e.g. http://host:1025/v1); "
        "falls back to ASCEND_ENGINE_URL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model name served by the engine; falls back to ASCEND_MODEL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key; falls back to ASCEND_API_KEY (default EMPTY)",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="pip-install benchmark/requirements.txt before running",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=2, help="concurrent tasks per phase"
    )
    parser.add_argument(
        "--turn-timeout", type=float, default=240.0, help="per-turn timeout (s)"
    )
    parser.add_argument(
        "--report-dir", default=str(_REPORT_DIR_DEFAULT), help="report output dir"
    )
    return parser.parse_args()


def resolve_engine(args: argparse.Namespace) -> EngineConfig:
    """Resolve engine URL/model/key from args + environment."""
    url = args.engine_url or os.environ.get("ASCEND_ENGINE_URL", "")
    model = args.model or os.environ.get("ASCEND_MODEL", "")
    api_key = args.api_key or os.environ.get("ASCEND_API_KEY", "EMPTY")
    if not url or not model:
        print(
            "缺少引擎参数：请通过 --engine-url/--model 或环境变量 "
            "ASCEND_ENGINE_URL/ASCEND_MODEL 提供真实昇腾引擎（MindIE / vLLM-Ascend）。"
            "本基准测试不提供模拟引擎。"
        )
        sys.exit(2)
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    root = base[: -len("/v1")]
    return EngineConfig(base_url=base, engine_root=root, model=model, api_key=api_key)


def run_setup() -> None:
    """Install benchmark-only dependencies (idempotent)."""
    requirements = str(_REPO_ROOT / "benchmark" / "requirements.txt")
    print(f"[setup] pip install -r {requirements}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements], check=True
    )


async def _execute_phases(
    engine: EngineConfig,
    tasks: List[Any],
    args: argparse.Namespace,
) -> Tuple[
    List[TaskResult], Dict[str, Dict[str, Any]], List[LlmCallRecord], List[LlmCallRecord]
]:
    """Run both phases; returns (results, summaries, baseline/affinity calls)."""
    from benchmark.agents import build_agent, build_affinity_model, build_baseline_model

    baseline_model = build_baseline_model(
        model=engine.model, base_url=engine.base_url, api_key=engine.api_key
    )
    affinity_model = build_affinity_model(
        model=engine.model, base_url=engine.base_url, api_key=engine.api_key
    )
    baseline_recorder = TTFTRecorder()
    affinity_recorder = TTFTRecorder()
    baseline_results = await run_phase(
        "baseline", build_agent(baseline_model), tasks, baseline_recorder, args
    )
    affinity_results = await run_phase(
        "affinity", build_agent(affinity_model), tasks, affinity_recorder, args
    )
    summaries = {
        "baseline": summarize(
            baseline_results, baseline_recorder.calls, {"enabled": False}
        ),
        "affinity": summarize(
            affinity_results,
            affinity_recorder.calls,
            getattr(affinity_model, "affinity_stats", {}),
        ),
    }
    results = baseline_results + affinity_results
    return results, summaries, baseline_recorder.calls, affinity_recorder.calls


def _write_reports(
    *,
    report_dir: str,
    engine: EngineConfig,
    probe: Dict[str, Any],
    tasks: List[Any],
    summaries: Dict[str, Dict[str, Any]],
    results: List[TaskResult],
    baseline_calls: List[LlmCallRecord],
    affinity_calls: List[LlmCallRecord],
) -> Path:
    """Write the JSON + Markdown reports; returns the Markdown path."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_report_{stamp}.json"
    md_path = out_dir / f"benchmark_report_{stamp}.md"
    json_path.write_text(
        json.dumps(
            {
                "meta": {
                    "started": datetime.now().isoformat(),
                    "engine": asdict(engine),
                },
                "probe": probe,
                "summaries": summaries,
                "results": [asdict(result) for result in results],
                "llm_calls": [asdict(call) for call in baseline_calls]
                + [asdict(call) for call in affinity_calls],
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
            summaries=summaries,
            results=results,
            json_name=json_path.name,
        ),
        encoding="utf-8",
    )
    return md_path


def _print_summary(summaries: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== 摘要 ===")
    for name in ("baseline", "affinity"):
        summary = summaries[name]
        done = summary["tasks_completed"]
        total = done + summary["tasks_failed"]
        print(
            f"{name:9s} 任务 {done}/{total}"
            f" LLM调用 {summary['llm_calls']}"
            f" TTFT mean {_fmt_ms(summary['ttft_mean_ms'])}ms"
            f" p95 {_fmt_ms(summary['ttft_p95_ms'])}ms"
        )


def main() -> None:
    """One-click entry: setup -> probe -> run both agents -> report."""
    sys.path.insert(0, str(_REPO_ROOT))
    args = parse_args()
    if args.setup:
        run_setup()
    engine = resolve_engine(args)
    from benchmark.tasks import load_tasks

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
        f"streaming={probe.get('streaming')}"
    )
    tasks = load_tasks()
    results, summaries, base_calls, aff_calls = asyncio.run(
        _execute_phases(engine, tasks, args)
    )
    md_path = _write_reports(
        report_dir=args.report_dir,
        engine=engine,
        probe=probe,
        tasks=tasks,
        summaries=summaries,
        results=results,
        baseline_calls=base_calls,
        affinity_calls=aff_calls,
    )
    _print_summary(summaries)
    print(f"\n报告：{md_path}")
    failed = any(result.error for result in results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
