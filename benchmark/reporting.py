"""Benchmark report rendering and persistence (lab-sheet Markdown + JSON).

Split out of ``run_benchmark.py`` so the CLI stays under pylint's
too-many-lines budget; import-only module with no side effects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from benchmark.metrics import (
    NA,
    PASS_MARK,
    PairValidity,
    judge,
    overall_verdict,
    pair_validity,
    verdict_text,
)

_REPORT_DIR_DEFAULT = Path(__file__).resolve().parent / "reports"

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


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def _version_of(package: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "?"


def _identity_summary(probe: Dict[str, Any]) -> str:
    """Best-effort engine type/version fingerprint for the report header."""
    identity = probe.get("identity") or {}
    engine_type = identity.get("engine_type") or "未知"
    version = identity.get("version") or "未提供"
    signals: List[str] = []
    if identity.get("version_endpoint"):
        signals.append("/version=200")
    if identity.get("health"):
        signals.append("/health=200")
    if identity.get("server_header"):
        signals.append(f"Server={identity['server_header']}")
    basis = "；".join(signals) if signals else "无可用信号"
    return f"类型={engine_type}，版本={version}（探测依据：{basis}）"


def _task_success_sets(
    results: List[Any], *agent_names: str
) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """Task ids completed without ANY round error, per agent (pair guard).

    A task counts only when every round of that agent finished it without
    error — a flaky task on one side already means the two sides ran
    different workloads. An agent with no records at all (build failure)
    yields ``None``, which skips the set check in :func:`pair_validity`.
    """
    sets: List[Optional[Set[str]]] = []
    for name in agent_names:
        records = [record for record in results if record.agent == name]
        if not records:
            sets.append(None)
            continue
        failed = {record.task_id for record in records if record.error}
        sets.append({record.task_id for record in records} - failed)
    return sets[0], sets[1]


def _pair_validity(
    baseline: Dict[str, Any],
    affinity: Dict[str, Any],
    baseline_ok: Optional[Set[str]],
    affinity_ok: Optional[Set[str]],
) -> PairValidity:
    """Comparability check for one (affinity, baseline) lab-sheet pair."""
    return pair_validity(
        baseline_calls=(baseline.get("median") or {}).get("llm_calls"),
        affinity_calls=(affinity.get("median") or {}).get("llm_calls"),
        baseline_ok=baseline_ok,
        affinity_ok=affinity_ok,
    )


def _render_lab_sheet(
    pair_name: str,
    affinity: Dict[str, Any],
    baseline: Dict[str, Any],
    validity: Optional[PairValidity] = None,
) -> List[str]:
    """One lab-sheet table: affinity agent vs its same-framework baseline.

    An invalid pair (one side skipped/failed a different workload) keeps its
    raw numbers visible but has every verdict forced to ➖ plus an explicit
    alert, so survivorship artifacts can never surface as ✅/❌ conclusions.
    """
    aff, base = affinity["median"], baseline["median"]
    verdicts = [
        judge(metric, aff.get(metric), base.get(metric)) for metric in VERDICT_METRICS
    ]
    invalid = validity is not None and not validity.comparable
    if invalid:
        verdicts = [replace(verdict, status=NA) for verdict in verdicts]
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
            f"| {verdict.label} | {verdict.reference} | "
            f"{verdict_text(verdict)} | {PASS_MARK[verdict.status]} |"
        )
    if invalid:
        conclusion = (
            f"⛔ 对照无效：{validity.reason}；上表判定一律按 ➖ 处理，"
            "不作为亲和收益或损失的证据。"
        )
    else:
        conclusion = overall_verdict(verdicts, npu_moved)
    rows += ["", f"**结论：{conclusion}**", ""]
    return rows


def _render_rounds_table(
    agent_names: List[str], summaries: Dict[str, Dict[str, Any]]
) -> List[str]:
    rows = [
        "| Agent | 轮次 | LLM调用 | TTFT mean(ms) | E2E mean(ms) | KV命中率 | Prefill/call |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in agent_names:
        build_error = summaries[name].get("build_error")
        if build_error:
            rows.append(
                f"| {name} | — | 0 | n/a | n/a | n/a | n/a |  "
                f"⚠️ 构建失败，未执行：{build_error[:80]} |"
            )
            continue
        for round_idx, metrics in enumerate(summaries[name]["per_round"]):
            rows.append(
                f"| {name} | {round_idx + 1} | {metrics['llm_calls']} "
                f"| {_fmt(metrics['ttft_mean_ms'])} | {_fmt(metrics['e2e_mean_ms'])} "
                f"| {_fmt(metrics['kv_hit_rate'])}% "
                f"| {_fmt(metrics['prefill_per_call'])} |"
            )
    return rows


def _render_correctness(
    tasks: List[Any],
    results: List[Any],
    agent_names: List[str],
    summaries: Dict[str, Dict[str, Any]],
) -> List[str]:
    by_key = {(r.agent, r.task_id): r for r in results}
    header = "| 任务 | " + " | ".join(agent_names) + " |"
    rows = [
        "## 5. 正确性（关键词得分按任务）",
        "",
        header,
        "|" + "---|" * (len(agent_names) + 1),
    ]
    for task in tasks:
        cells = []
        for name in agent_names:
            if summaries[name].get("build_error"):
                cells.append("⚠️ 未执行")
                continue
            record = by_key.get((name, task.task_id))
            score = (
                f"{record.keyword_hits}/{record.keywords_total}" if record else "n/a"
            )
            if record and record.error:
                score += f" ⚠️{record.error[:40]}"
            cells.append(score)
        rows.append(f"| {task.task_id} | " + " | ".join(cells) + " |")
    return rows


def _render_affinity_evidence(
    summaries: Dict[str, Dict[str, Any]],
    agent_names: List[str],
    probe: Dict[str, Any],
) -> List[str]:
    rows = ["## 6. 亲和行为证据", ""]
    for name in agent_names:
        if "affinity" not in name:
            continue
        stats = summaries[name].get("affinity_stats", {})
        rows.append(
            f"- **{name}** 亲和计数（全程累计）：{stats or '（openJiuwen 侧见引擎日志）'}"
        )
        if (
            name == "lc-affinity"
            and stats.get("salt_bound_requests", 0) == 0
            and not probe.get("salt_tool_calls")
        ):
            rows.append(
                "  - salt 绑定为 0：引擎探测 `salt_tool_calls=✗`，salt 绑定被"
                "自动禁用（引擎拒绝 cache_salt + 工具调用请求，HTTP 501），"
                "affinity 以普通客户端运行，属安全降级。"
            )
        degraded = stats.get("salt_degraded_requests", 0)
        if degraded:
            rows.append(
                f"  - ⚠️ 运行中发生 {degraded} 次 salt 拒绝降级（按 session 生效，"
                "被拒会话退化为普通客户端；详见 run.log 中引擎 HTTP 错误响应体）。"
            )
        hit_rates = [
            window.get("hit_rate_delta")
            for window in summaries[name]["engine_windows"]
            if window.get("hit_rate_delta") is not None
        ]
        if hit_rates:
            rows.append(
                f"  - 引擎侧前缀命中率（各轮窗口）：{hit_rates}"
            )
        peaks = [
            window.get("cache_usage_peak")
            for window in summaries[name]["engine_windows"]
            if window.get("cache_usage_peak") is not None
        ]
        if peaks:
            rows.append(f"  - KV Cache 占用峰值：{peaks}")
        npu = [
            window.get("npu_samples")
            for window in summaries[name]["engine_windows"]
            if any(window.get("npu_samples", []))
        ]
        if npu:
            rows.append(f"  - NPU 采样：{npu}")
    rows.append(
        "- 未配置 --metrics-url / --npu-cmd 时引擎侧指标为 ➖，不影响客户端判定。"
    )
    return rows


def render_report_md(
    *,
    engine: Any,
    probe: Dict[str, Any],
    tasks: List[Any],
    agent_names: List[str],
    summaries: Dict[str, Dict[str, Any]],
    results: List[Any],
    args: argparse.Namespace,
    fingerprint: str,
    json_name: str,
) -> str:
    """Render the full lab-sheet Markdown report."""
    sections = [
        "# 昇腾算力亲和基准测试报告（4 Agent 化验单）",
        "",
        "## 1. 测试环境",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 引擎：`{engine.base_url}`（模型 `{engine.model}`）",
        f"- 引擎身份（探测，尽力而为）：{_identity_summary(probe)}",
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
        f"- `/release_kv_cache`：{'✓' if probe.get('release_endpoint') else '✗'}"
        f"（缺失时自动禁用 release 请求，仅保留 cache_salt 绑定）",
        (
            f"- salt+工具调用（cache_salt + tool messages）："
            f"{'✓' if probe.get('salt_tool_calls') else '✗'}"
            "（✗ 时引擎拒绝带 salt 的工具调用请求 → 自动禁用 salt 绑定，"
            "affinity 退化为普通客户端，工具任务照常执行）"
        ),
        f"- 流式输出：{'✓' if probe.get('streaming') else '✗'}",
        (
            f"- 流式 usage（include_usage）：{'✓' if probe.get('stream_usage') else '✗'}"
            "（✗ 时客户端无法获得 token 用量 → Prefill/Decode/KV 命中率/TPOT 均为 ➖；"
            "请检查网关是否透传 stream_options，或配置 --metrics-url 走引擎侧指标）"
        ),
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
            sections += _render_lab_sheet(
                f"{affinity_name} vs {baseline_name}",
                summaries[affinity_name],
                summaries[baseline_name],
                validity=_pair_validity(
                    summaries[baseline_name],
                    summaries[affinity_name],
                    *_task_success_sets(results, baseline_name, affinity_name),
                ),
            )
    sections += [
        "## 4. 分轮明细（每 Agent 每轮）",
        "",
        *_render_rounds_table(agent_names, summaries),
        "",
        *_render_correctness(tasks, results, agent_names, summaries),
        "",
        *_render_affinity_evidence(summaries, agent_names, probe),
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
        "- **对照无效警报（⛔）**：配对双方任务完成集不一致或 LLM 调用数差异",
        "  过大时，两侧样本不再是同一工作负载，全部指标强制 ➖ 并给出原因，",
        "  防止幸存者偏差（如一侧任务早夭导致的假 E2E 收益）冒充结论。",
        "- openJiuwen 侧 TTFT/TPOT 为 ➖：agent-core 回调无 token 级事件，",
        "  其判定依赖 E2E/Prefill/KV 命中三项。",
        "- KV 命中率为 ➖ 表示引擎/网关未返回 cached_tokens（非命中为 0）：",
        "  此时核心四项仅三项可判，需网关透传 prompt_tokens_details 或",
        "  配置 --metrics-url 走引擎侧指标。",
        "- 单轮样本噪声大，主判定用跨轮中位数；建议 rounds≥3。",
        "",
        "## 8. 附录",
        "",
        f"- 每次调用的原始记录（时延+token 用量）：`{json_name}`",
    ]
    log_file = getattr(args, "log_file", None)
    if log_file:
        sections.append(
            f"- 运行日志（全链路，含每次 LLM 调用与亲和计数）：`{log_file}`"
        )
    sections.append("")
    return "\n".join(sections)


def write_reports(
    *,
    report_dir: str,
    engine: Any,
    probe: Dict[str, Any],
    tasks: List[Any],
    agent_names: List[str],
    data: Dict[str, Any],
    args: argparse.Namespace,
    fingerprint: str,
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
                    "rounds": args.rounds,
                    "agents": agent_names,
                    "task_fingerprint": fingerprint,
                },
                "probe": probe,
                **data,
                # TaskResult dataclasses are not JSON-serializable; convert at
                # the serialization boundary (internal code keeps attribute
                # access on the objects).
                "results": [asdict(result) for result in data["results"]],
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
