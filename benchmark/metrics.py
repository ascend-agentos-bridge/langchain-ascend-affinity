"""Client-side metric aggregation + lab-sheet verdict engine.

Layered metric model (per the benchmark spec):

1. Client metrics (always collected): TTFT / TPOT / E2E / decode tokens-per-sec
   / prefill & decode token counts / client-side KV hit rate
   (``cached_tokens / prompt_tokens`` from usage passthrough).
2. Affinity behaviour: salt-bound requests, releases attempted/failed.
3. Engine & NPU side (optional): ``--metrics-url`` Prometheus snapshot deltas
   (vLLM prefix-cache counters), ``--npu-cmd`` key=value sampler. Absent
   sources render as N/A, never fail the run.

The verdict engine turns each (affinity, baseline) metric pair into a
lab-sheet row: reference range + PASS / WARN / FAIL / N/A, mirroring a
medical lab report. Core-four signals (TTFT down, prefill down, KV hit up,
E2E down) improving together prove real compute affinity; NPU-only movement
with a flat core four raises the "suspected false affinity" alert.
"""

from __future__ import annotations

import logging
import re
import statistics
import subprocess  # nosec B404 - user-supplied sampler command, opt-in only
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
NA = "N/A"

PASS_MARK = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "N/A": "➖"}


@dataclass
class CallMetrics:  # pylint: disable=too-many-instance-attributes  # data carrier
    """Metrics of one LLM call observed from the client side."""

    agent: str
    task_id: str
    round_idx: int
    ttft_ms: Optional[float]
    e2e_ms: Optional[float]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


@dataclass
class AgentMetrics:  # pylint: disable=too-many-instance-attributes  # data carrier
    """Aggregated client-side metrics for one agent (one round or median)."""

    llm_calls: int = 0
    streamed_calls: int = 0
    ttft_mean_ms: Optional[float] = None
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    tpot_mean_ms: Optional[float] = None
    e2e_mean_ms: Optional[float] = None
    decode_tps: Optional[float] = None
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_per_call: Optional[float] = None
    decode_per_call: Optional[float] = None
    kv_hit_rate: Optional[float] = None  # percent: cached / prompt


def _safe_mean(values: List[float]) -> Optional[float]:
    return round(statistics.fmean(values), 2) if values else None


def usage_field(usage: Any, key: str) -> Optional[int]:
    """Read one field from LangChain ``usage_metadata``.

    ``usage_metadata`` is a ``UsageMetadata`` TypedDict (a plain ``dict`` at
    runtime) for every OpenAI-compatible integration, but some providers hand
    out namespace objects. Accept both so token metrics never silently drop.
    """
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _percentile(values: List[float], ratio: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(ratio * (len(ordered) - 1)))
    return round(ordered[int(index)], 2)


def aggregate(calls: List[CallMetrics]) -> AgentMetrics:
    """Aggregate per-call records into one agent-level metric set."""
    ttfts = [c.ttft_ms for c in calls if c.ttft_ms is not None]
    e2es = [c.e2e_ms for c in calls if c.e2e_ms is not None]
    tpots = [
        (c.e2e_ms - c.ttft_ms) / (c.completion_tokens - 1)
        for c in calls
        if c.ttft_ms is not None
        and c.e2e_ms is not None
        and c.completion_tokens not in (None, 0, 1)
    ]
    decode_seconds = [
        (c.e2e_ms - c.ttft_ms) / 1000.0
        for c in calls
        if c.ttft_ms is not None and c.e2e_ms is not None and c.e2e_ms > c.ttft_ms
    ]
    decode_tokens = sum(c.completion_tokens or 0 for c in calls)
    prompt_tokens = sum(c.prompt_tokens or 0 for c in calls)
    cached_tokens = sum(c.cached_tokens or 0 for c in calls)
    decode_time_total = sum(decode_seconds)
    return AgentMetrics(
        llm_calls=len(calls),
        streamed_calls=len(ttfts),
        ttft_mean_ms=_safe_mean(ttfts),
        ttft_p50_ms=_percentile(ttfts, 0.50),
        ttft_p95_ms=_percentile(ttfts, 0.95),
        tpot_mean_ms=_safe_mean(tpots),
        e2e_mean_ms=_safe_mean(e2es),
        decode_tps=round(decode_tokens / decode_time_total, 1)
        if decode_time_total > 0
        else None,
        prefill_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        prefill_per_call=round(prompt_tokens / len(calls), 1) if calls else None,
        decode_per_call=round(decode_tokens / len(calls), 1) if calls else None,
        kv_hit_rate=round(cached_tokens / prompt_tokens * 100.0, 1)
        if prompt_tokens
        else None,
    )


def median_metrics(per_round: List[AgentMetrics]) -> AgentMetrics:
    """Cross-round median of per-call metrics; token totals take the mean."""
    if not per_round:
        return AgentMetrics()
    if len(per_round) == 1:
        return per_round[0]
    fields = [
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_mean_ms",
        "e2e_mean_ms",
        "decode_tps",
        "prefill_per_call",
        "decode_per_call",
        "kv_hit_rate",
    ]
    data = asdict(per_round[0])
    data["llm_calls"] = round(sum(m.llm_calls for m in per_round) / len(per_round))
    data["streamed_calls"] = round(
        sum(m.streamed_calls for m in per_round) / len(per_round)
    )
    data["prefill_tokens"] = round(
        sum(m.prefill_tokens for m in per_round) / len(per_round)
    )
    data["decode_tokens"] = round(
        sum(m.decode_tokens for m in per_round) / len(per_round)
    )
    for name in fields:
        values = [getattr(m, name) for m in per_round if getattr(m, name) is not None]
        data[name] = round(statistics.median(values), 2) if values else None
    return AgentMetrics(**data)


# -- verdict engine --------------------------------------------------------------


@dataclass
class Rule:
    """Lab-sheet reference range for one metric.

    ``direction``: -1 improvement is a decrease, +1 an increase, 0 flat.
    ``pass_pct``: improvement threshold for PASS (positive number).
    ``warn_pct``: tolerated deterioration threshold for WARN (positive
    number); beyond it the verdict is FAIL.
    """

    label: str
    direction: int
    pass_pct: float
    warn_pct: float
    unit: str = "pct"  # "pct" relative change or "pp" percentage points


RULES: Dict[str, Rule] = {
    "ttft_mean_ms": Rule("TTFT mean", -1, 10.0, 5.0),
    "e2e_mean_ms": Rule("E2E mean", -1, 5.0, 5.0),
    "prefill_per_call": Rule("Prefill tokens/call", -1, 10.0, 5.0),
    "decode_per_call": Rule("Decode tokens/call", 0, 15.0, 30.0),
    "tpot_mean_ms": Rule("TPOT mean", 0, 10.0, 25.0),
    "decode_tps": Rule("Decode tokens/s", 0, 10.0, 20.0),
}

KV_HIT_RULE = Rule("KV 命中率 (cached/prompt)", 1, 10.0, 2.0, unit="pp")
CORE_FOUR = ("ttft_mean_ms", "prefill_per_call", "kv_hit_rate", "e2e_mean_ms")


@dataclass
class Verdict:  # pylint: disable=too-many-instance-attributes  # data carrier
    """One lab-sheet row: metric, reference range, delta and status."""

    metric: str
    label: str
    reference: str
    baseline: Optional[float]
    affinity: Optional[float]
    delta: Optional[float]
    unit: str
    status: str


def _fmt_value(value: Optional[float], unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "pp":
        return f"{value:.1f}%"
    return f"{value:,.1f}"


def verdict_text(verdict: Verdict) -> str:
    """Render one lab-sheet row as `baseline -> affinity (delta) mark`."""
    suffix = "pp" if verdict.unit == "pp" else "%"
    delta = (
        f"{verdict.delta:+.1f}{suffix}" if verdict.delta is not None else ""
    )
    return (
        f"{_fmt_value(verdict.baseline, verdict.unit)} → "
        f"{_fmt_value(verdict.affinity, verdict.unit)} ({delta})"
        f" {PASS_MARK[verdict.status]}"
    )


def judge(
    metric: str, affinity: Optional[float], baseline: Optional[float]
) -> Verdict:
    """Compare affinity vs baseline against the metric's reference range."""
    rule = KV_HIT_RULE if metric == "kv_hit_rate" else RULES[metric]
    unit_hint = "pp" if rule.unit == "pp" else "%"
    if rule.direction:
        arrow = "↓" if rule.direction < 0 else "↑"
        reference = (
            f"{arrow} 改善 ≥{rule.pass_pct:g}{unit_hint} 为 ✅，"
            f"恶化 >{rule.warn_pct:g}{unit_hint} 为 ❌"
        )
    else:
        reference = (
            f"≈ 持平（|Δ|≤{rule.pass_pct:g}% 为 ✅，"
            f"> {rule.warn_pct:g}% 为 ❌）"
        )
    if affinity is None or baseline is None:
        return Verdict(metric, rule.label, reference, baseline, affinity, None, rule.unit, NA)
    delta = (
        (affinity - baseline)
        if rule.unit == "pp"
        else ((affinity - baseline) / baseline * 100.0 if baseline else None)
    )
    if delta is None:
        return Verdict(metric, rule.label, reference, baseline, affinity, None, rule.unit, NA)
    if rule.direction == 0:  # flat expectations: smaller |delta| is better
        deviation = abs(delta)
        status = (
            PASS if deviation <= rule.pass_pct
            else WARN if deviation <= rule.warn_pct
            else FAIL
        )
    else:
        improved = -delta if rule.direction < 0 else delta
        if rule.unit == "pp":
            improved = affinity - baseline
        status = (
            PASS if improved >= rule.pass_pct
            else WARN if improved >= -rule.warn_pct
            else FAIL
        )
    return Verdict(metric, rule.label, reference, baseline, affinity, delta, rule.unit, status)


def overall_verdict(verdicts: List[Verdict], npu_moved: bool) -> str:
    """Overall conclusion: real affinity / partial / none / false affinity."""
    by_metric = {v.metric: v for v in verdicts}
    core = [by_metric[name] for name in CORE_FOUR if name in by_metric]
    if not core:
        return "➖ 数据不可得，无法判定"
    if all(v.status == PASS for v in core):
        return "✅ 算力亲和生效：核心四项（TTFT↓ Prefill↓ KV命中↑ E2E↓）同步改善"
    if all(v.status in (WARN, FAIL, NA) for v in core) and npu_moved:
        return "❌ 疑似假亲和：核心指标未改善，仅 NPU 资源指标变化"
    if any(v.status == PASS for v in core):
        return "⚠️ 部分改善：仅部分核心指标达标，见逐项判定"
    return "❌ 未体现亲和收益（核心指标持平或恶化）"


# -- engine & NPU side (optional sources) ----------------------------------------


_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")

# Per-URL "already warned" set so a dead/missing /metrics endpoint logs one
# WARNING per run instead of one per agent phase (24+ lines of noise).
_warned_metrics_urls: set = set()


def _warn_metrics_once(url: str, reason: str) -> None:
    """Log one WARNING per URL; later failures stay silent (N/A anyway)."""
    if url in _warned_metrics_urls:
        return
    _warned_metrics_urls.add(url)
    logger.warning(
        "metrics endpoint %s %s; engine-side metrics will be N/A", url, reason
    )


def fetch_prometheus(url: str) -> Optional[Dict[str, float]]:
    """GET a Prometheus text endpoint and parse ``name value`` lines."""
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # nosec B310
            body = response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        _warn_metrics_once(url, f"unreachable ({exc})")
        return None
    metrics: Dict[str, float] = {}
    for line in body.splitlines():
        match = _PROM_LINE.match(line.strip())
        if match:
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
    if not metrics:
        _warn_metrics_once(url, "returned no parseable metrics")
        return None
    return metrics


PROM_CACHE_HITS = "vllm:gpu_prefix_cache_hits_total"
PROM_CACHE_QUERIES = "vllm:gpu_prefix_cache_queries_total"
PROM_CACHE_USAGE = "vllm:gpu_cache_usage_perc"
PROM_LATENCY = "vllm:e2e_request_latency_seconds_count"

# vLLM renamed / added prefix-cache metrics across engines and versions
# (V0 vs V1, ``gpu_`` vs ``num_``, counters vs the V1 ``prefix_cache_hit_rate``
# gauge). Instead of hard-coding one name set, the deltas below discover
# whichever counters/gauges the engine actually exposes, so the same harness
# works against vLLM, vLLM-Ascend and gateway-passthrough deployments.
_PREFIX_HIT_RE = re.compile(r"prefix.*cache.*hit.*(total|count)", re.I)
_PREFIX_QUERY_RE = re.compile(r"prefix.*cache.*query.*(total|count)", re.I)
_PREFIX_MISS_RE = re.compile(r"prefix.*cache.*miss.*(total|count)", re.I)
_PREFIX_RATE_RE = re.compile(r"prefix.*cache.*hit_rate", re.I)
_CACHE_USAGE_RE = re.compile(r"(gpu_)?cache.*usage.*(perc|percent|ratio)", re.I)


def _pick_by_pattern(
    snapshots: List[Optional[Dict[str, float]]], pattern: "re.Pattern[str]"
) -> Optional[str]:
    """First metric name matching ``pattern`` across the snapshots."""
    for snapshot in snapshots:
        for name in snapshot or {}:
            if pattern.search(name):
                return name
    return None


def _rate_delta_from_gauge(
    before: Optional[Dict[str, float]], after: Optional[Dict[str, float]]
) -> Optional[float]:
    """V1 engines expose ``vllm:prefix_cache_hit_rate`` as a gauge: report
    the window's end value directly (no counter delta semantics)."""
    name = _pick_by_pattern([after, before], _PREFIX_RATE_RE)
    value = (after or {}).get(name) if name else None
    if value is None:
        return None
    return round(value * 100.0, 1) if value <= 1.0 else round(value, 1)


def cache_hit_rate_delta(
    before: Optional[Dict[str, float]], after: Optional[Dict[str, float]]
) -> Optional[float]:
    """Prefix-cache hit rate over the window between two snapshots (percent).

    Resolution order: a bare ``*hit_rate`` gauge (V1 engine) at face value;
    a ``hits`` + ``miss`` counter pair as ``hits / (hits + miss)``; a
    ``hits`` + ``queries`` counter pair (or the legacy
    ``gpu_prefix_cache_*`` names) as ``hits / queries``.
    """
    if not before or not after:
        return None
    if _pick_by_pattern([after, before], _PREFIX_RATE_RE) is not None:
        return _rate_delta_from_gauge(before, after)
    hit_name = _pick_by_pattern([after, before], _PREFIX_HIT_RE) or PROM_CACHE_HITS
    hits = after.get(hit_name, 0.0) - before.get(hit_name, 0.0)
    miss_name = _pick_by_pattern([after, before], _PREFIX_MISS_RE)
    if miss_name:
        missed = after.get(miss_name, 0.0) - before.get(miss_name, 0.0)
        total = hits + missed
        return round(hits / total * 100.0, 1) if total > 0 else None
    query_name = (
        _pick_by_pattern([after, before], _PREFIX_QUERY_RE) or PROM_CACHE_QUERIES
    )
    queries = after.get(query_name, 0.0) - before.get(query_name, 0.0)
    if queries <= 0:
        return None
    return round(hits / queries * 100.0, 1)


def cache_usage_peak(
    before: Optional[Dict[str, float]], after: Optional[Dict[str, float]]
) -> Optional[float]:
    """Peak KV-cache usage percentage across the two snapshots."""
    name = (
        _pick_by_pattern([before, after], _CACHE_USAGE_RE) or PROM_CACHE_USAGE
    )
    values = [
        snapshot.get(name)
        for snapshot in (before, after)
        if snapshot and snapshot.get(name) is not None
    ]
    return round(max(values), 3) if values else None


_KV_PAIR = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([0-9.]+)")


def sample_sidecar(command: str) -> Optional[Dict[str, float]]:
    """Run a user-supplied sampler command; parse ``key=value`` pairs.

    Example (NPU utilization + HBM via npu-smi on the engine host):

        --npu-cmd "ssh engine-host npu-smi info | awk '...'"  # prints util=57.5
    """
    if not command:
        return None
    try:
        output = subprocess.run(  # nosec B603 - explicit opt-in sampler
            command,
            shell=True,  # nosec B602
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    samples = {
        match.group(1): float(match.group(2)) for match in _KV_PAIR.finditer(output)
    }
    return samples or None
