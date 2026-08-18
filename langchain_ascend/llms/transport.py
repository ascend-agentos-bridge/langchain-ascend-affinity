"""HTTP transport layer for the Ascend engine: URL construction, request building,
POST helpers, and SSE event streaming. Extracted from AscendAffinityChatModel to
keep the main class lean.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Iterator, List

from langchain_core.messages import ToolCallChunk


class TransportMixin:
    """Mixin providing the HTTP/S request/response plumbing for the Ascend engine.

    Depends on the host class to provide ``base_url``, ``api_key``, and
    ``timeout`` as pydantic fields. All methods are callable through ``self``
    after the class hierarchy is resolved.
    """

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

    def _post(self, root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``payload`` as JSON to ``root + path`` and return the body."""
        request = self._build_request(root, path, payload)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

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

    def _stream_events(
        self, payload: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """POST ``payload`` with SSE and yield each ``data:`` JSON event."""
        request = self._build_request(self._chat_completions_url(), "", payload)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
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
