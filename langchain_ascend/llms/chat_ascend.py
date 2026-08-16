"""AscendAffinityChatModel: the openJiuwen agent-core affinity mechanism,
ported to LangChain as a drop-in BaseChatModel.

Per generation request this model does exactly what agent-core's
``InferenceAffinityModelClient`` + ``KVCacheManager`` do:

1. **Salt binding** — the request body carries ``cache_sharing: true`` and,
   when a session is bound, ``cache_salt: <session_id>`` (aligned with the
   native vLLM / vLLM-Ascend prefix-cache salt).
2. **Prefix-diff scheduling** — the outgoing ``(messages, tools)`` window is
   diffed against the previous window for that session. Pure appends (the
   normal agent loop) keep the prefix cache hot; rewritten history
   (``trim_messages``, summarization, compression) yields the first divergent
   index via :class:`~langchain_ascend.prefix_tracker.PrefixCacheTracker`.
3. **Partial release** — on divergence the model posts the *previous* window
   to ``POST {engine-root}/release_kv_cache`` with
   ``messages_released_index`` / ``tools_released_index`` so the engine drops
   only the stale KV blocks while keeping the valid prefix. Release failures
   are logged and never abort generation.

Session resolution order per call: per-call / ``bind(session_id=...)`` kwargs
→ best-effort ``run_manager.metadata`` → constructor ``session_id``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolCallChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_ascend.prefix_tracker import PrefixCacheTracker

logger = logging.getLogger(__name__)

_SESSION_KEYS = ("session_id", "session", "conversation_id")

_ROLE_BY_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "chat": "assistant",
}


def _new_affinity_stats() -> Dict[str, int]:
    """Fresh affinity counter dict (one per model instance)."""
    return {
        "affinity_requests": 0,
        "salt_bound_requests": 0,
        "releases_attempted": 0,
        "releases_failed": 0,
    }


def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
    """Deterministic OpenAI-style serialization of a LangChain message.

    The engine's prefix cache is keyed on this wire shape, so the same
    serialization is used for both the request payload and the prefix diff.
    """
    entry: Dict[str, Any] = {
        "role": _ROLE_BY_TYPE.get(message.type, message.type),
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


class AscendAffinityChatModel(BaseChatModel):
    """OpenAI-compatible chat model with agent-core compute affinity built in.

    Swap this in as the model of a ``langchain`` / ``langgraph`` /
    ``deepagents`` agent and every LLM call becomes cache-affine to the
    session — no callbacks, no handler wiring.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(
        default="ascend-chat", description="Model name advertised to the engine."
    )
    base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="OpenAI-compatible endpoint served by the Ascend engine.",
    )
    api_key: str = Field(default="EMPTY", description="API key for the endpoint.")
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    max_tokens: Optional[int] = Field(default=None)
    timeout: float = Field(default=30.0)
    session_id: Optional[str] = Field(
        default=None,
        description="Session id used as cache_salt when no per-call session "
        "is provided.",
    )
    enable_affinity: bool = Field(
        default=True,
        description="Inject cache_sharing/cache_salt and auto-release stale "
        "KV-Cache blocks. False turns this into a plain OpenAI-compatible "
        "client.",
    )
    release_endpoint: str = Field(
        default="/release_kv_cache",
        description="Partial KV-Cache release path on the engine "
        "(agent-core compatible). Empty string disables release requests.",
    )
    streaming: bool = Field(
        default=False,
        description="When True, invoke()/ainvoke() stream via SSE internally "
        "and aggregate the chunks, emitting on_llm_new_token callbacks "
        "(mirrors ChatOpenAI's streaming flag; enables real TTFT capture).",
    )

    _prefix_tracker: PrefixCacheTracker = PrivateAttr(default_factory=PrefixCacheTracker)
    _affinity_stats: Dict[str, int] = PrivateAttr(default_factory=_new_affinity_stats)

    @property
    def _llm_type(self) -> str:
        return "ascend-affinity-chat"

    @property
    def affinity_stats(self) -> Dict[str, int]:
        """Read-only copy of this instance's affinity counters."""
        return dict(self._affinity_stats)

    # -- tool calling --------------------------------------------------------

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        """Attach OpenAI-format tool schemas so agents can plan tool calls."""
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=openai_tools, **kwargs)

    # -- session resolution ---------------------------------------------------

    def _resolve_session_id(
        self,
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> Optional[str]:
        """Per-call kwargs first, then run metadata, then the constructor."""
        for key in _SESSION_KEYS:
            value = kwargs.get(key)
            if value:
                return str(value)
        metadata = getattr(run_manager, "metadata", None) or {}
        for key in _SESSION_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
        return self.session_id

    # -- transport -------------------------------------------------------------

    def _build_request(
        self, root: str, path: str, payload: Dict[str, Any]
    ) -> urllib.request.Request:
        """Build the JSON POST request for ``root + path``."""
        return urllib.request.Request(
            f"{root.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _post(self, root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``payload`` as JSON to ``root + path`` and return the body."""
        request = self._build_request(root, path, payload)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _engine_root(self) -> str:
        """Engine root URL (``base_url`` without the OpenAI ``/v1`` suffix)."""
        base = self.base_url.rstrip("/")
        return base[: -len("/v1")] if base.endswith("/v1") else base

    # -- affinity pipeline -------------------------------------------------------

    def _build_payload(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[Sequence[str]],
        tools: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [_serialize_message(message) for message in messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if stop:
            payload["stop"] = list(stop)
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = list(tools)
        return payload

    def _apply_affinity(
        self,
        session_id: Optional[str],
        message_dicts: List[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]],
        payload: Dict[str, Any],
    ) -> None:
        """Salt-bind the request and release stale KV blocks when enabled."""
        if not self.enable_affinity:
            return
        self._affinity_stats["affinity_requests"] += 1
        payload["cache_sharing"] = True
        if not session_id:
            return
        self._affinity_stats["salt_bound_requests"] += 1
        payload["cache_salt"] = session_id
        if not self.release_endpoint:
            self._prefix_tracker.update(session_id, message_dicts, tools)
            return
        plan = self._prefix_tracker.check_release_needed(session_id, message_dicts, tools)
        if plan is not None:
            self._affinity_stats["releases_attempted"] += 1
            release_payload: Dict[str, Any] = {
                "model": self.model,
                "cache_salt": session_id,
                "cache_sharing": True,
                "messages": plan.messages,
                "messages_released_index": plan.messages_released_index,
            }
            if plan.tools is not None and plan.tools_released_index is not None:
                release_payload["tools"] = plan.tools
                release_payload["tools_released_index"] = plan.tools_released_index
            try:
                self._post(self._engine_root(), self.release_endpoint, release_payload)
            except OSError as exc:
                self._affinity_stats["releases_failed"] += 1
                logger.warning(
                    "KV release request failed for session %s: %s", session_id, exc
                )
        self._prefix_tracker.update(session_id, message_dicts, tools)

    # -- response parsing ----------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw_calls: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        if cached:
            metadata["input_token_details"] = {"cache_read": cached}
        return metadata

    def _parse_chat_result(self, response: Dict[str, Any]) -> ChatResult:
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

    # -- generation ---------------------------------------------------------------

    def _prepare_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Shared prologue: session resolution + payload + affinity pipeline."""
        session_id = self._resolve_session_id(run_manager, kwargs)
        tools = kwargs.get("tools")
        payload = self._build_payload(messages, stop, tools)
        self._apply_affinity(session_id, payload["messages"], tools, payload)
        return payload

    def _request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> ChatResult:
        """Shared sync/async pipeline: affinity injection + one HTTP request."""
        payload = self._prepare_request(messages, stop, run_manager, kwargs)
        response = self._post(self.base_url, "/chat/completions", payload)
        return self._parse_chat_result(response)

    def _generate_from_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> ChatResult:
        """Aggregate the SSE stream so invoke() still returns one message."""
        final: Optional[AIMessageChunk] = None
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            final = chunk.message if final is None else final + chunk.message
        message = final if final is not None else AIMessageChunk(content="")
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"model": self.model},
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return self._generate_from_stream(messages, stop, run_manager, kwargs)
        return self._request(messages, stop, run_manager, kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.to_thread(
            functools.partial(self._generate, messages, stop, run_manager, **kwargs)
        )

    # -- streaming ----------------------------------------------------------------

    def _stream_events(
        self, payload: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """POST ``payload`` with SSE and yield each ``data:`` JSON event."""
        request = self._build_request(self.base_url, "/chat/completions", payload)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)

    @staticmethod
    def _tool_call_chunks_from_delta(
        delta: Dict[str, Any],
    ) -> List[ToolCallChunk]:
        """Convert an OpenAI streaming ``delta`` into tool-call chunks."""
        chunks: List[ToolCallChunk] = []
        for index, call in enumerate(delta.get("tool_calls") or []):
            function = call.get("function", {})
            chunks.append(
                ToolCallChunk(
                    name=function.get("name") or "",
                    args=function.get("arguments") or "",
                    id=call.get("id") or "",
                    index=call.get("index", index),
                )
            )
        return chunks

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the completion via SSE with the affinity pipeline applied."""
        payload = self._prepare_request(messages, stop, run_manager, kwargs)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        for event in self._stream_events(payload):
            usage_metadata = self._usage_metadata(event.get("usage"))
            if usage_metadata is not None:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", usage_metadata=usage_metadata)
                )
                continue
            choice = (event.get("choices") or [{}])[0]
            delta = dict(choice.get("delta") or {})
            content = delta.get("content") or ""
            tool_call_chunks = self._tool_call_chunks_from_delta(delta)
            if not content and not tool_call_chunks:
                continue
            if run_manager:
                run_manager.on_llm_new_token(content or "")
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content,
                    tool_call_chunks=tool_call_chunks,
                )
            )
