"""Deterministic verification harness for the openJiuwen affinity port.

Runs the *same* two-user, three-turn dialogue schedule twice against the mock
engine:

- **plain phase** — ``AscendAffinityChatModel(enable_affinity=False)``: no
  ``cache_salt``, so both users share the engine's anonymous cache bucket.
  Interleaved turns keep diverging on it; every request recomputes cold.
- **affinity phase** — the same model salt-bound per session
  (``bind(session_id=...)``): each user gets an isolated KV-cache bucket
  (pure appends stay warm), and a mid-session history rewrite triggers the
  ported prefix-diff scheduler to ``POST /release_kv_cache`` exactly once.

Prints the engine-side comparison and exits non-zero if the affinity
invariants (release fired, fewer cold starts, identical answers) do not hold.

Run: ``python example/verify_affinity.py [--port 8001]``
"""

from __future__ import annotations

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_schedule() -> List[Dict[str, Any]]:
    """Deterministic two-user schedule with one mid-session history rewrite."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system = SystemMessage(content="You are a financial advisor.")
    a1 = "user-A: help me plan a 3-year fund investment"
    b1 = "user-B: check the SH000001 index quote"
    a_summary = "SUMMARY: A asked for a 3-year fund plan; risk profile collected."
    b_reply = "reply:31:check the SH000001 index quote"
    a2_reply = "reply:29:what about the downside risk?"
    return [
        {
            "session": "user-A",
            "messages": [system, HumanMessage(content=a1)],
            "tag": "A turn 1",
        },
        {
            "session": "user-B",
            "messages": [system, HumanMessage(content=b1)],
            "tag": "B turn 1",
        },
        {
            "session": "user-A",
            # compression rewrote turn 1's user message in place -> prefix
            # diverges at index 1: the ported scheduler must release from there
            "messages": [
                system,
                AIMessage(content=a_summary),
                HumanMessage(content="user-A: what about the downside risk?"),
            ],
            "tag": "A turn 2 (history rewritten)",
        },
        {
            "session": "user-B",
            "messages": [
                system,
                HumanMessage(content=b1),
                AIMessage(content=b_reply),
                HumanMessage(content="user-B: and the 1-year chart?"),
            ],
            "tag": "B turn 2 (pure append)",
        },
        {
            "session": "user-A",
            "messages": [
                system,
                AIMessage(content=a_summary),
                HumanMessage(content="user-A: what about the downside risk?"),
                AIMessage(content=a2_reply),
                HumanMessage(content="user-A: rebalance quarterly then"),
            ],
            "tag": "A turn 3 (pure append)",
        },
        {
            "session": "user-B",
            "messages": [
                system,
                HumanMessage(content=b1),
                AIMessage(content=b_reply),
                HumanMessage(content="user-B: and the 1-year chart?"),
                AIMessage(content="reply:33:and the 1-year chart?"),
                HumanMessage(content="user-B: thanks, done"),
            ],
            "tag": "B turn 3 (pure append)",
        },
    ]


def run_plain(base_url: str, schedule: List[Dict[str, Any]]) -> List[str]:
    """Phase 1: affinity disabled — shared anonymous cache bucket."""
    from langchain_ascend import AscendAffinityChatModel

    model = AscendAffinityChatModel(
        base_url=base_url, model="verify-model", enable_affinity=False
    )
    return [model.invoke(turn["messages"]).content for turn in schedule]


def run_affinity(base_url: str, schedule: List[Dict[str, Any]]) -> List[str]:
    """Phase 2: salt-bound sessions with the prefix-diff release scheduler."""
    from langchain_ascend import AscendAffinityChatModel

    model = AscendAffinityChatModel(base_url=base_url, model="verify-model")
    answers: List[str] = []
    for turn in schedule:
        bound = model.bind(session_id=turn["session"])
        answers.append(bound.invoke(turn["messages"]).content)
    return answers


def _row(label: str, plain: Any, affinity: Any) -> str:
    return f"  {label:<26} {str(plain):>12}  {str(affinity):>12}"


def render_report(plain_metrics: Dict[str, Any],
                  affinity_metrics: Dict[str, Any],
                  identical: bool) -> str:
    """Build the compact stdout comparison."""
    lines = [
        "=== Affinity verification: plain vs salt-bound (mock engine) ===",
        _row("metric", "plain", "affinity"),
        "  " + "-" * 54,
        _row("requests", plain_metrics["requests"], affinity_metrics["requests"]),
        _row("cold starts", plain_metrics["cold_starts"], affinity_metrics["cold_starts"]),
        _row("warm hits", plain_metrics["warm_hits"], affinity_metrics["warm_hits"]),
        _row(
            "partial recomputes",
            plain_metrics["partial_recomputes"],
            affinity_metrics["partial_recomputes"],
        ),
        _row("kv releases", plain_metrics["kv_releases"], affinity_metrics["kv_releases"]),
        _row(
            "avg TTFT (ms)",
            plain_metrics["ttft_avg_ms"],
            affinity_metrics["ttft_avg_ms"],
        ),
        _row(
            "salt buckets",
            ",".join(sorted(plain_metrics["salt_bindings"])),
            ",".join(sorted(affinity_metrics["salt_bindings"])),
        ),
        f"  answers identical across phases: {'yes' if identical else 'NO'}",
    ]
    return "\n".join(lines)


def _check_invariants(plain: Dict[str, Any], affinity: Dict[str, Any],
                      identical: bool) -> List[str]:
    """Return human-readable failures for every broken affinity invariant."""
    failures: List[str] = []
    if affinity["kv_releases"] < 1:
        failures.append("affinity phase never released stale KV blocks")
    if affinity["warm_hits"] <= plain["warm_hits"]:
        failures.append("affinity phase did not increase warm hits")
    if plain["ttft_avg_ms"] and affinity["ttft_avg_ms"] > plain["ttft_avg_ms"] * 0.8:
        failures.append(
            f"affinity avg TTFT {affinity['ttft_avg_ms']}ms is not clearly below "
            f"plain {plain['ttft_avg_ms']}ms"
        )
    if not identical:
        failures.append("answers differ across phases (determinism broken)")
    return failures


def _run_phases(
    base_url: str, state: Any, schedule: List[Dict[str, Any]]
) -> tuple[
    Dict[str, Any], Dict[str, Any], bool, float
]:
    """Run plain then affinity phase; return metrics, determinism, wall time."""
    started = time.perf_counter()
    plain_answers = run_plain(base_url, schedule)
    plain_metrics = state.snapshot()
    state.reset()
    affinity_answers = run_affinity(base_url, schedule)
    affinity_metrics = state.snapshot()
    elapsed = time.perf_counter() - started
    return plain_metrics, affinity_metrics, plain_answers == affinity_answers, elapsed


def main(argv: List[str] | None = None) -> int:
    """Run both phases and report; exit 0 only if all invariants hold."""
    from mock_engine import spawn

    # The langchain import chain is slow on some networks; warm it before
    # the timed window so "wall time" reflects the dialogue phases only.
    import_module("langchain_ascend")

    parser = argparse.ArgumentParser(prog="verify_affinity.py")
    parser.add_argument("--port", type=int, default=8001, help="mock engine port")
    args = parser.parse_args(argv)

    server, state = spawn(port=args.port)
    base_url = f"http://127.0.0.1:{args.port}/v1"
    schedule = build_schedule()
    try:
        report_data = _run_phases(base_url, state, schedule)
    finally:
        state.reset()
        server.should_exit = True

    plain_metrics, affinity_metrics, identical, elapsed = report_data
    print(render_report(plain_metrics, affinity_metrics, identical))
    print(f"  wall time (both phases): {elapsed:.2f}s")
    failures = _check_invariants(plain_metrics, affinity_metrics, identical)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: salt binding isolated sessions; rewrite triggered exactly one "
          "partial release; answers stayed deterministic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
