"""Affinity pipeline mixin: payload construction, salt binding, KV release.

Extracted from :class:`~langchain_ascend.llms.chat_ascend.AscendAffinityChatModel`
to keep the main model class focused on the generation lifecycle."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class AffinityPipelineMixin:
    """Payload construction and compute-affinity pipeline.

    Requires (on the owning model instance):
      - ``_affinity_stats`` (:class:`dict`)
      - ``_prefix_tracker`` (:class:`~langchain_ascend.prefix_tracker.PrefixCacheTracker`)
      - ``_serialize_message`` (static helper, typically from
        :class:`~langchain_ascend.llms.serialization.SerializationMixin`)
      - ``_post`` / ``_engine_root`` (from
        :class:`~langchain_ascend.llms.transport.TransportMixin`)
      - ``_build_agent_hint`` (from
        :class:`~langchain_ascend.llms.agent_hint.AgentHintMixin`)
      - ``_resolve_session_id`` / ``_resolve_parent_session_id`` (from the model)
      - Pydantic fields: ``model``, ``temperature``, ``top_p``, ``max_tokens``,
        ``enable_affinity``, ``enable_agent_hint``, ``release_endpoint``.
    """

    # -- helpers for the type checker (these live on the owning BaseChatModel) --
    _affinity_stats: Dict[str, int]
    _prefix_tracker: Any
    model: str
    temperature: float
    top_p: float
    max_tokens: Optional[int]
    enable_affinity: bool
    enable_agent_hint: bool
    release_endpoint: str

    def _build_payload(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[Sequence[str]],
        tools: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Assemble the core chat-completions payload without affinity fields."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                self._serialize_message(message)  # type: ignore[attr-defined]
                for message in messages
            ],
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
        *,
        session_id: Optional[str],
        parent_session_id: Optional[str],
        message_dicts: List[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]],
        payload: Dict[str, Any],
        manage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Salt-bind the request and release stale KV blocks when enabled."""
        if not self.enable_affinity:
            return
        self._affinity_stats["affinity_requests"] += 1
        if not session_id:
            # No salt -> stay a plain OpenAI client. Sending cache_sharing
            # without a salt would put every anonymous request into one
            # shared cache bucket and risk cross-session KV pollution.
            logger.debug("affinity skipped: no session bound")
            return
        self._affinity_stats["salt_bound_requests"] += 1
        payload["cache_sharing"] = True
        payload["cache_salt"] = session_id
        logger.debug(
            "salt-bound request: session=%s parent_session=%s agent_hint=%s",
            session_id,
            parent_session_id or session_id,
            self.enable_agent_hint,
        )
        if self.enable_agent_hint:
            payload["agent_hint"] = self._build_agent_hint(  # type: ignore[attr-defined]
                session_id=session_id,
                parent_session_id=parent_session_id or session_id,
                action=manage.get("action", "evict") if manage else None,
                target=manage.get("target", "messages") if manage else "session",
                manage_request=False if manage else None,
                msg_start=manage.get("start") if manage else None,
                msg_end=manage.get("end") if manage else None,
                tools_start=manage.get("tools_start") if manage else None,
                tools_end=manage.get("tools_end") if manage else None,
                include_tools=bool(manage.get("include_tools", False))
                if manage
                else False,
            )
        if not self.release_endpoint:
            self._prefix_tracker.update(session_id, message_dicts, tools)
            return
        plan = self._prefix_tracker.check_release_needed(
            session_id, message_dicts, tools
        )
        if plan is not None:
            self._affinity_stats["releases_attempted"] += 1
            logger.debug(
                "prefix divergence for session %s: release messages at "
                "index %s, tools at %s",
                session_id,
                plan.messages_released_index,
                plan.tools_released_index,
            )
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
                self._post(  # type: ignore[attr-defined]
                    self._engine_root(),  # type: ignore[attr-defined]
                    self.release_endpoint,
                    release_payload,
                )
            except OSError as exc:
                self._affinity_stats["releases_failed"] += 1
                logger.warning(
                    "KV release request failed for session %s: %s", session_id, exc
                )
        self._prefix_tracker.update(session_id, message_dicts, tools)

    def _prepare_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Shared prologue: session resolution + payload + affinity pipeline."""
        session_id = self._resolve_session_id(run_manager, kwargs)  # type: ignore[attr-defined]
        parent_session_id = self._resolve_parent_session_id(  # type: ignore[attr-defined]
            run_manager, kwargs
        )
        tools = kwargs.get("tools")
        manage = kwargs.get("agent_hint_manage")
        payload = self._build_payload(messages, stop, tools)
        self._apply_affinity(
            session_id=session_id,
            parent_session_id=parent_session_id,
            message_dicts=payload["messages"],
            tools=tools,
            payload=payload,
            manage=manage if isinstance(manage, dict) else None,
        )
        return payload
