"""Benchmark report rendering and persistence (lab-sheet Markdown + JSON).

Split out of ``run_benchmark.py`` so the CLI stays under pylint's
too-many-lines budget; import-only module with no side effects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from benchmark.metrics import PASS_MARK, judge, overall_verdict, verdict_text

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


def _render_lab_sheet(
    pair_name: str, affinity: Dict[str, Any], baseline: Dict[str, Any]
) -> List[str]:
    """One lab-sheet table: affinity agent vs its same-framework baseline."""
    aff, base = affinity["median"], baseline["median"]
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
            f"| {verdict.label} | {verdict.reference} | "
            f"{verdict_text(verdict)} | {PASS_MARK[verdict.status]} |"
        )
    rows += ["", f"**结论：{overall_verdict(verdicts, npu_moved)}**", ""]
    return rows


def _render_rounds_table(
    agent_names: List[str], summaries: Dict[str, Dict[str, Any]]
) -> List[str]:
    rows = [
        "| Agent | 轮次 | LLM调用 | TTFT mean(ms) | E2E mean(ms) | KV命中率 | Prefill/call |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in agent_names:
        for round_idx, metrics in enumerate(summaries[name]["per_round"]):
            rows.append(
                f"| {name} | {round_idx + 1} | {metrics['llm_calls']} "
                f"| {_fmt(metrics['ttft_mean_ms'])} | {_fmt(metrics['e2e_mean_ms'])} "
                f"| {_fmt(metrics['kv_hit_rate'])}% "
                f"| {_fmt(metrics['prefill_per_call'])} |"
            )
    return rows


def _render_correctness(
    tasks: List[Any], results: List[Any], agent_names: List[str]
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
    summaries: Dict[str, Dict[str, Any]], agent_names: List[str]
) -> List[str]:
    rows = ["## 6. 亲和行为证据", ""]
    for name in agent_names:
        if "affinity" not in name:
            continue
        stats = summaries[name].get("affinity_stats", {})
        rows.append(
            f"- **{name}** 亲和计数：{stats or '（openJiuwen 侧见引擎日志）'}"
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
            sections += _render_lab_sheet(
                f"{affinity_name} vs {baseline_name}",
                summaries[affinity_name],
                summaries[baseline_name],
            )
    sections += [
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
