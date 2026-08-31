"""Message serialization and response parsing for the Ascend engine's
OpenAI-compatible wire format. Deterministic serialization is critical for
prefix-cache correctness.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

# Chunk classes (``AIMessageChunk`` etc.) subclass their base message, so the
# isinstance checks below also normalize them — their ``type`` attribute is the
# chunk class name, which must never leak into the wire ``role`` (engines
# reject unknown roles; observed as HTTP 501 on vLLM-class servers, which
# silently killed every tool-calling turn of streaming agents).
_ROLE_BY_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "chat": "assistant",
    "function": "function",
}
_WIRE_ROLES = frozenset(_ROLE_BY_TYPE.values())


class SerializationMixin:
    """Mixin with serialization helpers for messages, tool calls, and responses.

    The engine's prefix cache is keyed on the serialised wire shape, so the
    same serialization is used for both the request payload and the prefix
    diff.
    """

    # isinstance table: chunk classes subclass their base message, so one
    # entry per base class also normalizes every chunk variant.
    _ROLE_BY_CLASS = (
        (HumanMessage, "user"),
        (AIMessage, "assistant"),
        (ToolMessage, "tool"),
        (SystemMessage, "system"),
        (FunctionMessage, "function"),
    )

    @classmethod
    def _wire_role(cls, message: BaseMessage) -> str:
        """OpenAI wire role for a LangChain message (chunk-class safe).

        Resolution order: the isinstance table first (this normalizes
        ``AIMessageChunk`` → ``assistant`` and friends), then ``ChatMessage``'s
        own role, then the type-name map for exotic types — and as a last
        resort any unrecognized name is clamped to a legal wire role, so a
        class name can never leak onto the wire (the 2026-08-24 501 root
        cause).
        """
        for message_class, role in cls._ROLE_BY_CLASS:
            if isinstance(message, message_class):
                return role
        if isinstance(message, ChatMessage):
            custom = str(message.role or "")
            if custom in _WIRE_ROLES:
                return custom
            return "assistant"
        resolved = _ROLE_BY_TYPE.get(
            message.type, str(getattr(message, "role", "") or message.type)
        )
        return resolved if resolved in _WIRE_ROLES else "assistant"

    @staticmethod
    def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
        """Deterministic OpenAI-style serialization of a LangChain message."""
        entry: Dict[str, Any] = {
            "role": SerializationMixin._wire_role(message),
            "content": message.content
            if isinstance(message.content, str)
            else str(message.content),
        }
        for tool_call in getattr(message, "tool_calls", None) or []:
            entry.setdefault("tool_calls", []).append(
                {
                    "id": tool_call.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": json.dumps(tool_call.get("args", {})),
                    },
                }
            )
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        name = getattr(message, "name", None)
        if name:
            entry["name"] = name
        return entry

    @staticmethod
    def _parse_tool_calls(raw_calls: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse an OpenAI-compatible ``tool_calls`` array into LangChain shape."""
        tool_calls: List[Dict[str, Any]] = []
        for call in raw_calls:
            function = call.get("function", {})
            arguments = function.get("arguments")
            tool_calls.append(
                {
                    "name": function.get("name", ""),
                    "args": json.loads(arguments) if arguments else {},
                    "id": call.get("id", ""),
                    "type": "tool_call",
                }
            )
        return tool_calls

    @staticmethod
    def _usage_metadata(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Map an OpenAI-compatible ``usage`` block to LangChain usage metadata."""
        if not usage:
            return None
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        metadata: Dict[str, Any] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if cached is not None:
            metadata["input_token_details"] = {"cache_read": cached}
        return metadata

    def _parse_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """Convert an OpenAI chat-completion response to a LangChain ChatResult."""
        choice = (response.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        ai_message = AIMessage(
            content=message.get("content") or "",
            tool_calls=self._parse_tool_calls(message.get("tool_calls") or []),
        )
        usage_metadata = self._usage_metadata(response.get("usage"))
        if usage_metadata is not None:
            ai_message.usage_metadata = usage_metadata
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={"model": response.get("model", "")},
        )
