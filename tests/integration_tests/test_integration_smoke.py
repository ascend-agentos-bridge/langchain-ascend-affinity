"""Integration smoke tests — require real Ascend NPU hardware.

Skipped by default; enable with ``ASCEND_DEVICE_AVAILABLE=1`` and point
``ASCEND_ENGINE_URL`` at a MindIE / vLLM-Ascend OpenAI-compatible endpoint.

These tests exercise the end-to-end protocol surface that unit tests can
only mock: salt-bound round trips, SSE streaming, agent_hint management
methods, and the partial-release probe.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_ascend import AscendAffinityChatModel

pytestmark = pytest.mark.integration

_REASON = "Requires real Ascend NPU hardware (set ASCEND_DEVICE_AVAILABLE=1)"

_ENGINE_URL = os.environ.get("ASCEND_ENGINE_URL", "http://127.0.0.1:8000/v1")
_ENGINE_MODEL = os.environ.get("ASCEND_MODEL", "ascend-chat")


def _make_model(**kwargs: object) -> AscendAffinityChatModel:
    """Fresh model instance pointed at the configured engine."""
    return AscendAffinityChatModel(
        base_url=_ENGINE_URL,
        model=_ENGINE_MODEL,
        **kwargs,
    )


def _skip() -> bool:
    return not os.environ.get("ASCEND_DEVICE_AVAILABLE")


@pytest.mark.skipif(_skip, reason=_REASON)
def test_salt_bound_round_trip_on_hardware():
    """Pure appends keep the salt-bound prefix warm — no release expected."""
    model = _make_model()
    bound = model.bind(session_id="itest-session")
    first = bound.invoke([HumanMessage(content="ping")])
    # pure append keeps the salt-bound prefix warm — no release expected
    second = bound.invoke(
        [HumanMessage(content="ping"), HumanMessage(content="any advice?")]
    )
    assert isinstance(first.content, str)
    assert isinstance(second.content, str)
    stats = model.affinity_stats
    assert stats["affinity_requests"] >= 2
    assert stats["salt_bound_requests"] >= 2


@pytest.mark.skipif(_skip, reason=_REASON)
def test_streaming_round_trip_on_hardware():
    """The streaming flag aggregates SSE chunks into one message."""
    model = _make_model(streaming=True)
    bound = model.bind(session_id="itest-stream-session")
    result = bound.invoke([HumanMessage(content="count to three")])
    assert isinstance(result.content, str)
    assert result.content


@pytest.mark.skipif(_skip, reason=_REASON)
def test_agent_hint_management_on_hardware():
    """evict/offload/prefetch never raise and reach the management path.

    Engines that do not support agent_hint ignore the unknown fields, so the
    methods may return False — the smoke contract is: no exception and the
    management counters move.
    """
    model = _make_model(enable_agent_hint=True)
    session = "itest-hint-session"
    model.evict_kvc(session_id=session)
    model.offload_kvc(session_id=session)
    model.prefetch_kvc(session_id=session)
    stats = model.affinity_stats
    assert stats["management_requests"] >= 3


@pytest.mark.skipif(_skip, reason=_REASON)
def test_prefix_divergence_release_probe_on_hardware():
    """A rewritten history must trigger at least one partial-release attempt."""
    model = _make_model()
    bound = model.bind(session_id="itest-release-session")
    bound.invoke([HumanMessage(content="plan step one")])
    # rewrite the window (divergence) instead of appending
    bound.invoke([HumanMessage(content="plan step two, completely different")])
    stats = model.affinity_stats
    assert stats["releases_attempted"] >= 1
