"""AscendAffinityChatModel: the openJiuwen agent-core affinity mechanism,
ported to LangChain as a drop-in BaseChatModel.

Per generation request this model does exactly what agent-core's
``InferenceAffinityModelClient`` + ``KVCacheManager`` do:

1. **Salt binding** — every request with a bound session carries
   ``cache_sharing: true`` + ``cache_salt: <session_id>`` (aligned with the
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

Stage A (2026-08) additionally supports the openjiuwen agent-core
``agent_hint`` lifecycle protocol (``session_id`` / ``parent_session_id`` +
``context_management`` ``evict/offload/prefetch``), opt-in via
``enable_agent_hint``: requests carry the identity fields, and the
``evict_kvc`` / ``offload_kvc`` / ``prefetch_kvc`` management methods match
agent-core's ``AscendAffinityModelClient`` field-for-field. Engines that
ignore unknown fields degrade safely.

Inference-then-manage (agent-core 75adc2b44e parity): pass
``agent_hint_manage={"action": ..., "target": ..., "start": ..., "end": ...}``
per invoke to carry ``context_management.manage_request=false`` edits on a
normal inference request, so the engine applies the edit after generation
atomically (e.g. evicting an ephemeral attachment tail).

Session resolution order per call: per-call / ``bind(session_id=...)`` kwargs
→ best-effort ``run_manager.metadata`` → constructor ``session_id``.

Implementation
--------------
The model is assembled from focused mixins to keep each concern
independently testable and reviewable:

* :class:`~langchain_ascend.llms.transport.TransportMixin` — HTTP POST & SSE
* :class:`~langchain_ascend.llms.serialization.SerializationMixin` — message
  serialization & response parsing
* :class:`~langchain_ascend.llms.affinity_pipeline.AffinityPipelineMixin` —
  salt binding, prefix-diff release, request preparation
* :class:`~langchain_ascend.llms.agent_hint.AgentHintMixin` — agent_hint
  lifecycle protocol (identity fields + KV-cache management methods)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_ascend.llms.affinity_pipeline import AffinityPipelineMixin
from langchain_ascend.llms.agent_hint import AgentHintMixin
from langchain_ascend.llms.serialization import SerializationMixin
from langchain_ascend.llms.transport import TransportMixin
from langchain_ascend.prefix_tracker import PrefixCacheTracker

logger = logging.getLogger(__name__)

_SESSION_KEYS = ("session_id", "session", "conversation_id")


def _new_affinity_stats() -> Dict[str, int]:
    """Fresh affinity counter dict (one per model instance)."""
    return {
        "affinity_requests": 0,
        "salt_bound_requests": 0,
        "releases_attempted": 0,
        "releases_failed": 0,
        "management_requests": 0,
        "management_failed": 0,
    }


class AscendAffinityChatModel(
    BaseChatModel,
    TransportMixin,
    SerializationMixin,
    AffinityPipelineMixin,
    AgentHintMixin,
):
    """OpenAI-compatible chat model with agent-core compute affinity built in.

    Swap this in as the model of a ``langchain`` / ``langgraph`` /
    ``deepagents`` agent and every LLM call becomes cache-affine to the
    session — no callbacks, no handler wiring.

    The implementation is assembled from focused mixins:
    :class:`TransportMixin`, :class:`SerializationMixin`,
    :class:`AffinityPipelineMixin`, :class:`AgentHintMixin`.
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
    enable_agent_hint: bool = Field(
        default=False,
        description="Opt-in to the openjiuwen agent-core agent_hint "
        "lifecycle protocol: requests carry session_id/parent_session_id "
        "identity, and evict_kvc/offload_kvc/prefetch_kvc management "
        "methods become available. Engines that ignore unknown fields "
        "degrade safely.",
    )
    idle_evict_after_seconds: float = Field(
        default=0.0,
        description="Auto-evict the session's KV cache after this many "
        "seconds of inactivity following a generation (0 = disabled). "
        "Requires enable_agent_hint; each new call cancels and re-arms the "
        "timer.",
    )
    streaming: bool = Field(
        default=False,
        description="When True, invoke()/ainvoke() stream via SSE internally "
        "and aggregate the chunks, emitting on_llm_new_token callbacks "
        "(mirrors ChatOpenAI's streaming flag; enables real TTFT capture).",
    )

    _prefix_tracker: PrefixCacheTracker = PrivateAttr(
        default_factory=PrefixCacheTracker
    )
    _affinity_stats: Dict[str, int] = PrivateAttr(default_factory=_new_affinity_stats)
    _idle_timer: Optional[threading.Timer] = PrivateAttr(default=None)
    _idle_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

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

    @staticmethod
    def _resolve_parent_session_id(
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> Optional[str]:
        """Parent session id (agent_hint lineage); defaults to the session id."""
        value = kwargs.get("parent_session_id")
        if value:
            return str(value)
        metadata = getattr(run_manager, "metadata", None) or {}
        value = metadata.get("parent_session_id")
        return str(value) if value else None

    # -- capability flag -------------------------------------------------------

    def supports_kv_cache_affinity(self) -> bool:
        """Capability flag consumed by lifecycle schedulers (agent-core parity)."""
        return self.enable_agent_hint

    # -- idle auto-evict -------------------------------------------------------

    def _cancel_idle_evict_locked(self) -> None:
        """Cancel the pending idle timer (caller holds the lock)."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _cancel_idle_evict(self) -> None:
        """Cancel any pending idle-evict timer (safe when disabled)."""
        with self._idle_lock:
            self._cancel_idle_evict_locked()

    def _schedule_idle_evict(self, session_id: Optional[str]) -> None:
        """Arm the idle-evict timer after a generation, if configured."""
        if not self.idle_evict_after_seconds or not session_id:
            return
        if not self.enable_agent_hint:
            return
        with self._idle_lock:
            self._cancel_idle_evict_locked()
            timer = threading.Timer(
                self.idle_evict_after_seconds,
                self._idle_evict_cb,
                args=(session_id,),
            )
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _idle_evict_cb(self, session_id: str) -> None:
        """Timer callback: best-effort session evict (never raises)."""
        with self._idle_lock:
            self._idle_timer = None
        self.evict_kvc(session_id=session_id)

    # -- generation ------------------------------------------------------------

    def _request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> ChatResult:
        """Shared sync/async pipeline: affinity injection + one HTTP request."""
        payload = self._prepare_request(messages, stop, run_manager, kwargs)
        response = self._post(self._chat_completions_url(), "", payload)
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
        session_id = self._resolve_session_id(run_manager, kwargs)
        self._cancel_idle_evict()
        if self.streaming:
            result = self._generate_from_stream(messages, stop, run_manager, kwargs)
        else:
            result = self._request(messages, stop, run_manager, kwargs)
        self._schedule_idle_evict(session_id)
        return result

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

    # -- streaming --------------------------------------------------------------

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
