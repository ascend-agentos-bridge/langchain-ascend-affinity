"""Client-side prefix-diff algorithm for KV-Cache affinity.

Ported from openJiuwen agent-core's ``KVCacheManager`` scheduling logic:
before every LLM call, compare the context window actually sent to the
engine (messages + tools) against the previous one for the same session
and locate the first divergent index.

- Purely appended turns (the normal ReAct/tool loop) keep the prefix
  valid: no release is needed and the engine's KV-Cache stays a hit.
- Rewritten history (``trim_messages``, summarization, context
  compression, offloading) invalidates the suffix from the divergence
  point onwards: a ``release_kv_cache`` hint carrying
  ``messages_released_index`` / ``tools_released_index`` lets the engine
  drop only the stale blocks while keeping the still-valid prefix.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ReleasePlan:
    """Result of a prefix diff: what (if anything) to release.

    Attributes:
        messages: The *previous* window's messages, replayed to the engine
            so it can recompute the block layout before truncating.
        messages_released_index: First stale message index (0-based).
        tools: The *previous* window's tool schemas, when tools diverged.
        tools_released_index: First stale tool index (0-based).
    """

    messages: List[Dict[str, Any]]
    messages_released_index: int
    tools: Optional[List[Dict[str, Any]]] = None
    tools_released_index: Optional[int] = None


def _first_divergence(
    previous: Optional[Sequence[Any]], current: Optional[Sequence[Any]]
) -> Optional[int]:
    """Return the first index where ``previous`` no longer prefixes ``current``.

    ``None`` means the previous sequence is a strict prefix (purely appended
    content) — nothing to release.
    """
    prev = list(previous or [])
    curr = list(current or [])
    for idx in range(min(len(prev), len(curr))):
        if prev[idx] != curr[idx]:
            return idx
    return None


class PrefixCacheTracker:
    """Tracks per-session context windows and detects prefix invalidation.

    Thread-safe. Usage pattern (mirrors agent-core's KVCacheManager)::

        tracker = PrefixCacheTracker()
        plan = tracker.check_release_needed(
            session_id, curr_messages, curr_tools
        )
        if plan is not None:
            backend.release_cache(session_id, plan.messages,
                                  plan.messages_released_index,
                                  tools=plan.tools,
                                  tools_released_index=plan.tools_released_index)
        tracker.update(session_id, curr_messages, curr_tools)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_messages: Dict[str, List[Dict[str, Any]]] = {}
        self._last_tools: Dict[str, List[Dict[str, Any]]] = {}

    # -- inspection -----------------------------------------------------------

    def check_release_needed(
        self,
        session_id: str,
        curr_messages: Sequence[Dict[str, Any]],
        curr_tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[ReleasePlan]:
        """Diff the upcoming window against the last one sent for this session.

        Returns ``None`` when the cache is still valid (first observation of
        the session, or purely appended messages/tools); otherwise a
        :class:`ReleasePlan` pointing at the first stale index.
        """
        with self._lock:
            prev_messages = self._last_messages.get(session_id)
            prev_tools = self._last_tools.get(session_id)
            if prev_messages is None:
                return None  # first call for this session — snapshot only

        msg_idx = _first_divergence(prev_messages, curr_messages)
        tool_idx = _first_divergence(prev_tools, curr_tools)

        if msg_idx is None and tool_idx is None:
            return None  # pure append — prefix cache stays a hit

        with self._lock:
            stale_tools = list(prev_tools or [])
        return ReleasePlan(
            messages=list(prev_messages or []),
            messages_released_index=(
                msg_idx if msg_idx is not None else len(prev_messages or [])
            ),
            tools=stale_tools if tool_idx is not None else None,
            tools_released_index=tool_idx,
        )

    # -- bookkeeping ----------------------------------------------------------

    def update(
        self,
        session_id: str,
        curr_messages: Sequence[Dict[str, Any]],
        curr_tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Record the window that is about to be sent to the engine."""
        with self._lock:
            self._last_messages[session_id] = list(curr_messages or [])
            self._last_tools[session_id] = list(curr_tools or [])

    def clear(self, session_id: str) -> None:
        """Drop the tracked window (e.g. after the session was evicted)."""
        with self._lock:
            self._last_messages.pop(session_id, None)
            self._last_tools.pop(session_id, None)
