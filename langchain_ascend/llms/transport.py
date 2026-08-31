"""HTTP transport layer for the Ascend engine: URL construction, request building,
POST helpers, and SSE event streaming. Extracted from AscendAffinityChatModel to
keep the main class lean.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List

from langchain_core.messages import ToolCallChunk

logger = logging.getLogger(__name__)


class TransportMixin:
    """Mixin providing the HTTP/S request/response plumbing for the Ascend engine.

    Depends on the host class to provide ``base_url``, ``api_key``, and
    ``timeout`` as pydantic fields. All methods are callable through ``self``
    after the class hierarchy is resolved.
    """

    # -- helpers for the type checker (these live on the owning BaseChatModel) --
    _salt_degraded_sessions: set
    _affinity_stats: Dict[str, int]

    def _build_request(
        self, root: str, path: str, payload: Dict[str, Any]
    ) -> urllib.request.Request:
        """Build the JSON POST request for ``root + path``.

        Authentication is optional (agent-core parity): the ``Authorization``
        header is sent only when ``api_key`` is non-empty, so anonymous
        engines can be reached with ``api_key=""``.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{root.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    @staticmethod
    def _log_http_error(exc: urllib.error.HTTPError, url: str) -> None:
        """Log an engine HTTP error with its response body (diagnosability).

        The status code alone hides the actual rejection reason (unknown
        message role, malformed salt fields, template errors, ...). The body
        is read defensively — it may already be consumed or the connection
        gone — and is never re-raised from here.
        """
        try:
            body = exc.read(2048).decode("utf-8", errors="replace").strip()
        except OSError:
            body = ""
        logger.warning(
            "engine HTTP %s on %s%s",
            exc.code,
            url,
            f": {body[:512]}" if body else " (response body unavailable)",
        )

    def _post(self, root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``payload`` as JSON to ``root + path`` and return the body."""
        request = self._build_request(root, path, payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._log_http_error(exc, request.full_url)
            raise

    # -- salt-rejection degradation --------------------------------------------

    def _has_salt_fields(self, payload: Dict[str, Any]) -> bool:
        """True when the chat payload carries the affinity salt fields."""
        return "cache_sharing" in payload or "cache_salt" in payload

    def _degrade_salt(
        self, payload: Dict[str, Any], exc: urllib.error.HTTPError
    ) -> Dict[str, Any]:
        """Drop the salt fields, lock degradation for the session, and log.

        Called once when the engine actively rejects (HTTP 501, observed on
        MindIE-class servers) a salt-bound chat request. Every other field is
        kept, so the retried request goes through as a plain OpenAI call.
        Degradation is sticky **per session** (``_salt_degraded_sessions``):
        further requests of the rejected session skip salt binding entirely
        (``salt_degraded_requests`` counter), while other sessions on the
        same model instance keep their own binding — one broken session must
        not silently strip affinity from every other conversation.
        """
        degraded = dict(payload)
        degraded.pop("cache_sharing", None)
        degraded.pop("cache_salt", None)
        session_id = payload.get("cache_salt")
        if session_id:
            self._salt_degraded_sessions.add(str(session_id))
        self._affinity_stats["salt_degraded_requests"] += 1
        logger.warning(
            "engine rejected salt-bound request (HTTP %s): retrying without "
            "cache_sharing/cache_salt; salt binding disabled for session %s "
            "(other sessions keep their binding)",
            exc.code,
            session_id or "<unknown>",
        )
        return degraded

    def _post_salt_aware(
        self, root: str, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST a chat-completions payload; on HTTP 501 retry once without salt."""
        try:
            return self._post(root, path, payload)
        except urllib.error.HTTPError as exc:
            if exc.code == 501 and self._has_salt_fields(payload):
                return self._post(root, path, self._degrade_salt(payload, exc))
            raise

    def _chat_completions_url(self) -> str:
        """Chat-completions URL for origin, ``/v1`` base, or full endpoint.

        Mirrors agent-core ``AscendAffinityModelClient._chat_completions_url``
        (2026-08 vLLM joint-debugging fix).
        """
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _engine_root(self) -> str:
        """Engine root URL (``base_url`` without ``/v1`` / ``/chat/completions``)."""
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        if base.endswith("/v1"):
            return base[: -len("/v1")]
        return base

    def _open_stream(self, payload: Dict[str, Any]) -> Any:
        """Open the SSE response for one chat payload (salt-degrading on 501)."""
        request = self._build_request(self._chat_completions_url(), "", payload)
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            self._log_http_error(exc, request.full_url)
            if exc.code == 501 and self._has_salt_fields(payload):
                degraded = self._degrade_salt(payload, exc)
                request = self._build_request(
                    self._chat_completions_url(), "", degraded
                )
                return urllib.request.urlopen(request, timeout=self.timeout)
            raise

    def _stream_events(
        self, payload: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """POST ``payload`` with SSE and yield each ``data:`` JSON event."""
        response = self._open_stream(payload)
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
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
