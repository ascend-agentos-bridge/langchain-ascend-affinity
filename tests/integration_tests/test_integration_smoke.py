"""Integration smoke test — requires real Ascend NPU hardware.

Skipped by default; enable with ``ASCEND_DEVICE_AVAILABLE=1`` and point
``ASCEND_ENGINE_URL`` at a MindIE / vLLM-Ascend OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_ascend import AscendAffinityChatModel

pytestmark = pytest.mark.integration

_REASON = "Requires real Ascend NPU hardware (set ASCEND_DEVICE_AVAILABLE=1)"


@pytest.mark.skipif(not os.environ.get("ASCEND_DEVICE_AVAILABLE"), reason=_REASON)
def test_salt_bound_round_trip_on_hardware():
    model = AscendAffinityChatModel(
        base_url=os.environ.get("ASCEND_ENGINE_URL", "http://127.0.0.1:8000/v1"),
        model=os.environ.get("ASCEND_MODEL", "ascend-chat"),
    )
    bound = model.bind(session_id="itest-session")
    first = bound.invoke([HumanMessage(content="ping")])
    # pure append keeps the salt-bound prefix warm — no release expected
    second = bound.invoke(
        [HumanMessage(content="ping"), HumanMessage(content="any advice?")]
    )
    assert isinstance(first.content, str)
    assert isinstance(second.content, str)
