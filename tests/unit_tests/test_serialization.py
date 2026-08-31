"""Wire-serialization regression tests.

The 2026-08-24 benchmark run failed every tool-calling turn of the affinity
agent with HTTP 501: streaming aggregation yields ``AIMessageChunk`` objects
whose ``type`` is the chunk class name, and the old serializer leaked that
name into the wire ``role`` (``"role": "AIMessageChunk"``), which vLLM-class
engines reject. These tests pin the role mapping for every chunk class.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)

from langchain_ascend.llms.serialization import SerializationMixin

_WIRE_ROLES = {"system", "user", "assistant", "tool", "function"}


def _aggregated_tool_call_chunk() -> AIMessageChunk:
    """What ``_generate_from_stream`` produces for one streamed tool call."""
    chunk = AIMessageChunk(content="")
    chunk = chunk + AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "get_customer_holdings",
                "args": "",
                "id": "call_1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )
    return chunk + AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "",
                "args": '{"customer_id": "C1003"}',
                "id": "",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )


class TestWireRoles:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (HumanMessage(content="hi"), "user"),
            (HumanMessageChunk(content="hi"), "user"),
            (AIMessage(content="ok"), "assistant"),
            (AIMessageChunk(content="ok"), "assistant"),
            (
                ToolMessage(content="{}", tool_call_id="call_1"),
                "tool",
            ),
            (ToolMessageChunk(content="{}", tool_call_id="call_1"), "tool"),
            (SystemMessage(content="sys"), "system"),
            (SystemMessageChunk(content="sys"), "system"),
        ],
    )
    def test_chunk_roles_never_leak_class_names(self, message, expected):
        assert SerializationMixin._wire_role(message) == expected

    def test_aggregated_tool_call_chunk_serializes_as_assistant(self):
        entry = SerializationMixin._serialize_message(_aggregated_tool_call_chunk())
        assert entry["role"] == "assistant"
        assert entry["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_customer_holdings",
                    "arguments": '{"customer_id": "C1003"}',
                },
            }
        ]

    def test_tool_message_keeps_tool_call_id(self):
        message = ToolMessage(
            content="{}", tool_call_id="call_9", name="get_fund_profile"
        )
        entry = SerializationMixin._serialize_message(message)
        assert entry["role"] == "tool"
        assert entry["tool_call_id"] == "call_9"

    def test_chat_message_legal_custom_role_preserved(self):
        message = ChatMessage(content="hi", role="assistant")
        assert SerializationMixin._wire_role(message) == "assistant"

    def test_chat_message_illegal_custom_role_clamped(self):
        # An illegal wire role is exactly what caused the 2026-08-24 501s;
        # custom roles that OpenAI-chat does not define are clamped.
        message = ChatMessage(content="hi", role="observer")
        assert SerializationMixin._wire_role(message) == "assistant"

    def test_unknown_type_clamps_to_legal_role(self):
        class _Exotic(AIMessage):
            @property
            def type(self) -> str:  # noqa: D102 - test shim
                return "totally-unknown"

        assert SerializationMixin._wire_role(_Exotic(content="x")) == "assistant"

    def test_message_zoo_never_produces_illegal_role(self):
        zoo: list = [
            HumanMessage(content="q"),
            AIMessage(content="a"),
            _aggregated_tool_call_chunk(),
            AIMessageChunk(content=""),
            ToolMessage(content="{}", tool_call_id="t"),
            SystemMessage(content="s"),
            HumanMessageChunk(content="q"),
            SystemMessageChunk(content="s"),
            ToolMessageChunk(content="{}", tool_call_id="t"),
            ChatMessage(content="c", role="assistant"),
        ]
        for message in zoo:
            assert SerializationMixin._wire_role(message) in _WIRE_ROLES


class TestUsageMetadata:
    def test_zero_cached_tokens_is_reported(self):
        usage: Dict[str, Any] = {
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "total_tokens": 104,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        metadata = SerializationMixin._usage_metadata(usage)
        assert metadata["input_token_details"] == {"cache_read": 0}

    def test_missing_details_omit_cache_read(self):
        metadata = SerializationMixin._usage_metadata(
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        )
        assert "input_token_details" not in metadata
