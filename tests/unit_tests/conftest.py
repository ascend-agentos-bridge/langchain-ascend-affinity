"""Shared pytest fixtures for langchain_ascend unit tests."""

from __future__ import annotations

import pytest

from langchain_ascend import AscendAffinityChatModel

_OK_RESPONSE = {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture
def chat_llm() -> AscendAffinityChatModel:
    """Model pointed at a fictional engine with a bound session."""
    return AscendAffinityChatModel(
        base_url="http://engine.test/v1",
        model="mindie-llama3",
        session_id="fixture-session",
    )


@pytest.fixture
def ok_response() -> dict:
    return _OK_RESPONSE
