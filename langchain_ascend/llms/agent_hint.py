"""Agent-hint lifecycle mixin (openjiuwen agent-core AscendAffinity protocol).

Extracted from :class:`~langchain_ascend.llms.chat_ascend.AscendAffinityChatModel`
to keep the main model class focused on the generation lifecycle."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

_AGENT_HINT_ACTIONS = ("evict", "offload", "prefetch")
_AGENT_HINT_TARGETS = ("session", "messages", "tools")


class AgentHintMixin:
    """Agent-hint lifecycle protocol (openjiuwen agent-core field-for-field).

    Requires (on the owning model instance):
      - ``_affinity_stats`` (:class:`dict`)
      - ``_serialize_message`` (static, from :class:`SerializationMixin`)
      - ``_post`` / ``_chat_completions_url`` (from :class:`TransportMixin`)
      - ``_validate_action_target`` / ``_range_edit`` / ``_has_any_range`` /
        ``_build_target_edits`` / ``_build_agent_hint`` (static/class methods
        from this mixin)
      - Pydantic fields: ``model``, ``enable_agent_hint``.
    """

    # -- helpers for the type checker (these live on the owning BaseChatModel) --
    _affinity_stats: Dict[str, int]
    model: str
    enable_agent_hint: bool

    # -- validation & range helpers (staticmethods) ------------------------------

    @staticmethod
    def _validate_action_target(action: str, target: str) -> None:
        """Reject unknown lifecycle actions/targets (agent-core parity)."""
        if action not in _AGENT_HINT_ACTIONS:
            raise ValueError(f"unknown agent_hint action: {action}")
        if target not in _AGENT_HINT_TARGETS:
            raise ValueError(f"unknown agent_hint target: {target}")

    @staticmethod
    def _range_edit(
        action: str, target: str, start: Optional[int], end: Optional[int]
    ) -> Dict[str, Any]:
        """Half-open range edit; ``start < end`` required (agent-core parity)."""
        if start is None or end is None:
            raise ValueError(f"target={target} requires start and end")
        if start >= end:
            raise ValueError(f"target={target} half-open range requires start < end")
        return {"type": action, "target": target, "start": start, "end": end}

    @staticmethod
    def _has_any_range(**ranges: Optional[int]) -> bool:
        """True if any named range argument is not None."""
        return any(value is not None for value in ranges.values())

    @classmethod
    def _build_target_edits(  # pylint: disable=too-many-arguments  # agent-core parity
        cls,
        *,
        action: str,
        target: str,
        msg_start: Optional[int] = None,
        msg_end: Optional[int] = None,
        tools_start: Optional[int] = None,
        tools_end: Optional[int] = None,
        include_tools: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build context-management edits for one protocol target."""
        cls._validate_action_target(action, target)

        if target == "session":
            if cls._has_any_range(
                msg_start=msg_start,
                msg_end=msg_end,
                tools_start=tools_start,
                tools_end=tools_end,
            ):
                raise ValueError("target=session does not accept message/tool ranges")
            if include_tools:
                raise ValueError("target=session does not accept include_tools=True")
            return [{"type": action, "target": "session"}]

        if target == "messages":
            edits = [
                cls._range_edit(
                    action=action, target="messages", start=msg_start, end=msg_end
                )
            ]
            if include_tools:
                edits.append(
                    cls._range_edit(
                        action=action,
                        target="tools",
                        start=tools_start,
                        end=tools_end,
                    )
                )
            elif cls._has_any_range(tools_start=tools_start, tools_end=tools_end):
                raise ValueError(
                    "tools range requires include_tools=True or target=tools"
                )
            return edits

        if include_tools:
            raise ValueError("target=tools should not also set include_tools=True")
        if cls._has_any_range(msg_start=msg_start, msg_end=msg_end):
            raise ValueError("messages range is invalid for target=tools")
        return [
            cls._range_edit(
                action=action,
                target="tools",
                start=tools_start,
                end=tools_end,
            )
        ]

    @classmethod
    def _build_agent_hint(  # pylint: disable=too-many-arguments  # agent-core parity
        cls,
        *,
        session_id: str,
        parent_session_id: str,
        action: Optional[str] = None,
        target: str = "session",
        manage_request: Optional[bool] = None,
        msg_start: Optional[int] = None,
        msg_end: Optional[int] = None,
        tools_start: Optional[int] = None,
        tools_end: Optional[int] = None,
        include_tools: bool = False,
    ) -> Dict[str, Any]:
        """Build the agent_hint extension (agent-core field-for-field)."""
        hint: Dict[str, Any] = {
            "session_id": session_id,
            "parent_session_id": parent_session_id,
        }
        if action is None:
            return hint
        if not isinstance(manage_request, bool):
            raise ValueError("manage_request must be set when action is given")
        hint["context_management"] = {
            "manage_request": manage_request,
            "edits": cls._build_target_edits(
                action=action,
                target=target,
                msg_start=msg_start,
                msg_end=msg_end,
                tools_start=tools_start,
                tools_end=tools_end,
                include_tools=include_tools,
            ),
        }
        return hint

    # -- management request -----------------------------------------------------

    def _manage_kvc(  # pylint: disable=too-many-arguments  # agent-core parity
        self,
        action: str,
        *,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        target: str = "session",
        messages: Optional[Sequence[BaseMessage]] = None,
        msg_start: Optional[int] = None,
        msg_end: Optional[int] = None,
        tools_start: Optional[int] = None,
        tools_end: Optional[int] = None,
        include_tools: bool = False,
    ) -> bool:
        """One pure KV-cache management request; never raises on transport.

        Management requests are protocol peers of ``invoke`` and share the
        chat-completions endpoint with an ``agent_hint.context_management``
        block (agent-core ``AscendAffinityModelClient._manage_kvc`` parity).
        Failures are logged and counted, never fatal.
        """
        if not self.enable_agent_hint:
            logger.warning(
                "agent_hint disabled; ignoring %s_kvc for session %s",
                action,
                session_id,
            )
            return False
        self._validate_action_target(action, target)
        if not session_id:
            logger.warning("agent_hint %s requires session_id", action)
            return False
        if target != "session" and not messages:
            raise ValueError(f"messages is required for target={target}")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": (
                []
                if target == "session"
                else [
                    self._serialize_message(message)  # type: ignore[attr-defined]
                    for message in messages or []
                ]
            ),
            "stream": False,
            "agent_hint": self._build_agent_hint(
                session_id=session_id,
                parent_session_id=parent_session_id or session_id,
                action=action,
                target=target,
                manage_request=True,
                msg_start=msg_start,
                msg_end=msg_end,
                tools_start=tools_start,
                tools_end=tools_end,
                include_tools=include_tools,
            ),
        }
        self._affinity_stats["management_requests"] += 1
        try:
            self._post(  # type: ignore[attr-defined]
                self._chat_completions_url(), "", payload  # type: ignore[attr-defined]
            )
        except OSError as exc:
            self._affinity_stats["management_failed"] += 1
            logger.warning(
                "agent_hint %s request failed for session %s: %s",
                action,
                session_id,
                exc,
            )
            return False
        return True

    # -- public management methods -----------------------------------------------

    def evict_kvc(
        self,
        *,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        target: str = "session",
        **kwargs: Any,
    ) -> bool:
        """Evict the session's KV cache (agent-core ``evict_kvc`` parity)."""
        return self._manage_kvc(
            "evict",
            session_id=session_id,
            parent_session_id=parent_session_id,
            target=target,
            **kwargs,
        )

    def offload_kvc(
        self,
        *,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        target: str = "session",
        **kwargs: Any,
    ) -> bool:
        """Offload the session's KV cache (agent-core ``offload_kvc`` parity)."""
        return self._manage_kvc(
            "offload",
            session_id=session_id,
            parent_session_id=parent_session_id,
            target=target,
            **kwargs,
        )

    def prefetch_kvc(
        self,
        *,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        target: str = "session",
        **kwargs: Any,
    ) -> bool:
        """Prefetch the session's KV cache (agent-core ``prefetch_kvc`` parity)."""
        return self._manage_kvc(
            "prefetch",
            session_id=session_id,
            parent_session_id=parent_session_id,
            target=target,
            **kwargs,
        )
