"""Mock Ascend inference engine for the affinity verification harness.

Simulates an OpenAI-compatible chat endpoint (``/v1/chat/completions``), the
agent-core partial-release channel (``/release_kv_cache``) and a metrics
endpoint (``/metrics``). KV blocks are bound to the request's ``cache_salt``
(falling back to ``session_id``, else an anonymous bucket):

- a request whose messages extend the cached blocks is **warm** (``ttft_warm_ms``);
- a request that diverges mid-prefix pays a **partial recompute**
  (``ttft_warm_ms + (n - match) * msg_penalty_ms``) — this is what the
  plugin's release call minimizes by truncating the stale suffix up front;
- a request with no reusable prefix is **cold** (``ttft_cold_ms``).

At most ``kv_slots`` sessions stay resident; admitting a new salt evicts the
least-recently-used one entirely (its next request is cold). Requests without
a salt share one anonymous bucket, so interleaved users keep diverging on it —
the exact failure mode salt binding exists to fix. Fully deterministic.

Env vars (defaults in parentheses): ``MOCK_TTFT_WARM_MS`` (20),
``MOCK_TTFT_COLD_MS`` (250), ``MOCK_MSG_PENALTY_MS`` (15),
``MOCK_TOKENS_PER_SEC`` (60), ``MOCK_KV_SLOTS`` (4).

Run standalone: ``python example/mock_engine.py`` (serves on 127.0.0.1:8000).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

DEFAULT_TTFT_WARM_MS = float(os.environ.get("MOCK_TTFT_WARM_MS", "20"))
DEFAULT_TTFT_COLD_MS = float(os.environ.get("MOCK_TTFT_COLD_MS", "250"))
DEFAULT_MSG_PENALTY_MS = float(os.environ.get("MOCK_MSG_PENALTY_MS", "15"))
DEFAULT_TOKENS_PER_SEC = float(os.environ.get("MOCK_TOKENS_PER_SEC", "60"))
DEFAULT_KV_SLOTS = int(os.environ.get("MOCK_KV_SLOTS", "4"))
DEFAULT_PORT = int(os.environ.get("MOCK_ENGINE_PORT", "8000"))


@dataclass
class EngineConfig:
    """Static engine parameters (echoed in snapshots)."""

    kv_slots: int = DEFAULT_KV_SLOTS
    ttft_warm_ms: float = DEFAULT_TTFT_WARM_MS
    ttft_cold_ms: float = DEFAULT_TTFT_COLD_MS
    msg_penalty_ms: float = DEFAULT_MSG_PENALTY_MS
    tokens_per_sec: float = DEFAULT_TOKENS_PER_SEC


@dataclass
class SessionBlocks:
    """Resident KV blocks of one salt-bound session."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    last_access: float = 0.0


@dataclass
class TemperatureCounts:
    """Per-temperature request classification counters."""

    warm: int = 0
    partial: int = 0
    cold: int = 0


@dataclass
class ReleaseCounts:
    """Partial-release channel counters."""

    requests: int = 0
    blocks: int = 0


@dataclass
class EngineMetrics:
    """Mutable engine-side metrics, grouped by concern."""

    requests: int = 0
    temperature: TemperatureCounts = field(default_factory=TemperatureCounts)
    lru_evictions: int = 0
    salt_bindings: Dict[str, int] = field(default_factory=dict)
    releases: ReleaseCounts = field(default_factory=ReleaseCounts)
    ttft_ms: List[float] = field(default_factory=list)


def _common_prefix_len(
    cached: List[Dict[str, Any]], incoming: List[Dict[str, Any]]
) -> int:
    match = 0
    for block, message in zip(cached, incoming):
        if block != message:
            break
        match += 1
    return match


class EngineState:
    """Thread-safe simulation of salt-bound KV-cache lifecycle."""

    def __init__(
        self,
        kv_slots: int = DEFAULT_KV_SLOTS,
        ttft_warm_ms: float = DEFAULT_TTFT_WARM_MS,
        ttft_cold_ms: float = DEFAULT_TTFT_COLD_MS,
        msg_penalty_ms: float = DEFAULT_MSG_PENALTY_MS,
        tokens_per_sec: float = DEFAULT_TOKENS_PER_SEC,
    ) -> None:
        self.config = EngineConfig(
            kv_slots=kv_slots,
            ttft_warm_ms=ttft_warm_ms,
            ttft_cold_ms=ttft_cold_ms,
            msg_penalty_ms=msg_penalty_ms,
            tokens_per_sec=tokens_per_sec,
        )
        self.reset()

    def reset(self) -> None:
        """Drop all sessions and metrics."""
        self._lock = threading.RLock()
        self.sessions: Dict[str, SessionBlocks] = {}
        self.metrics = EngineMetrics()

    # -- metrics ----------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Consistent view of engine metrics and session residency."""
        with self._lock:
            metrics = self.metrics
            samples = list(metrics.ttft_ms)
            avg = sum(samples) / len(samples) if samples else 0.0
            return {
                "engine": {
                    "kv_slots": self.config.kv_slots,
                    "ttft_warm_ms": self.config.ttft_warm_ms,
                    "ttft_cold_ms": self.config.ttft_cold_ms,
                    "msg_penalty_ms": self.config.msg_penalty_ms,
                },
                "requests": metrics.requests,
                "warm_hits": metrics.temperature.warm,
                "partial_recomputes": metrics.temperature.partial,
                "cold_starts": metrics.temperature.cold,
                "lru_evictions": metrics.lru_evictions,
                "salt_bindings": dict(metrics.salt_bindings),
                "kv_releases": metrics.releases.requests,
                "blocks_released": metrics.releases.blocks,
                "ttft_avg_ms": round(avg, 1),
                "resident_salts": sorted(self.sessions),
            }

    # -- slot management ----------------------------------------------------------

    def _admit(self, salt: str) -> SessionBlocks:
        """Return the session's blocks, evicting LRU sessions over budget."""
        session = self.sessions.get(salt)
        if session is not None:
            return session
        while len(self.sessions) >= self.config.kv_slots:
            lru_salt = min(self.sessions, key=lambda s: self.sessions[s].last_access)
            del self.sessions[lru_salt]
            self.metrics.lru_evictions += 1
        session = SessionBlocks()
        self.sessions[salt] = session
        return session

    def _classify(self, blocks: List[Dict[str, Any]],
                  messages: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Return (temperature, ttft_ms) for a request against cached blocks."""
        match = _common_prefix_len(blocks, messages)
        cfg = self.config
        if blocks and match == len(blocks):
            return "warm", cfg.ttft_warm_ms
        if match > 0:
            # recompute cost scales with the stale fraction of the prompt,
            # floored at the warm baseline
            fraction = (len(messages) - match) / len(messages)
            return "partial", max(cfg.ttft_warm_ms, cfg.ttft_cold_ms * fraction)
        return "cold", cfg.ttft_cold_ms

    # -- endpoints ----------------------------------------------------------------

    def handle_chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Serve one chat completion with salt-bound cache accounting."""
        messages = body.get("messages") or []
        salt = str(
            body.get("cache_salt")
            or body.get("session_id")
            or "anonymous"
        )
        with self._lock:
            session = self.sessions.get(salt)
            blocks = list(session.messages) if session else []
            temperature, ttft_ms = self._classify(blocks, messages)
            metrics = self.metrics
            metrics.requests += 1
            metrics.salt_bindings[salt] = metrics.salt_bindings.get(salt, 0) + 1
            if temperature == "warm":
                metrics.temperature.warm += 1
            elif temperature == "partial":
                metrics.temperature.partial += 1
            else:
                metrics.temperature.cold += 1
            session = self._admit(salt)
            session.messages = [dict(message) for message in messages]
            session.last_access = time.monotonic()
            metrics.ttft_ms.append(ttft_ms)
        last_user = next(
            (
                str(m.get("content"))
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        time.sleep(ttft_ms / 1000.0)
        return {
            "id": f"chatcmpl-{metrics.requests}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": f"reply:{len(last_user)}:{last_user[-24:]}"},
                    "finish_reason": "stop",
                }
            ],
        }

    def handle_release(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate a session's stale suffix, keeping the valid prefix.

        Agent-core compatible payload: previous window plus the first stale
        index. Blocks after the index are dropped; the prefix stays resident,
        so the next request only recomputes the released tail.
        """
        salt = str(body.get("cache_salt") or body.get("session_id") or "anonymous")
        messages = body.get("messages") or []
        released_index = int(body.get("messages_released_index", len(messages)))
        with self._lock:
            metrics = self.metrics
            metrics.releases.requests += 1
            metrics.releases.blocks += max(0, len(messages) - released_index)
            session = self.sessions.get(salt)
            kept = 0
            if session is not None:
                session.messages = session.messages[:released_index]
                session.last_access = time.monotonic()
                kept = len(session.messages)
        return {"status": "ok", "session_id": salt,
                "released_from": released_index, "blocks_kept": kept}


def create_app(state: EngineState) -> FastAPI:
    """Build the FastAPI application backed by ``state``."""
    app = FastAPI(title="mock-ascend-engine")

    @app.post("/v1/chat/completions")
    def chat_completions(body: Dict[str, Any]) -> Any:
        if body.get("stream"):
            return StreamingResponse(
                iter([json.dumps(state.handle_chat(body))]), media_type="text/event-stream"
            )
        return state.handle_chat(body)

    @app.post("/release_kv_cache")
    def release_kv_cache(body: Dict[str, Any]) -> Dict[str, Any]:
        return state.handle_release(body)

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        return state.snapshot()

    @app.post("/metrics/reset")
    def metrics_reset() -> Dict[str, Any]:
        state.reset()
        return {"status": "ok"}

    return app


def spawn(port: int = DEFAULT_PORT, **kwargs: Any) -> Tuple[Any, EngineState]:
    """Start the engine in a daemon thread; return (server, state)."""
    import uvicorn

    state = EngineState(**kwargs)
    server = uvicorn.Server(
        uvicorn.Config(create_app(state), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True, name="mock-engine")
    thread.start()
    _wait_until_ready(port)
    return server, state


def _wait_until_ready(port: int, timeout_s: float = 15.0) -> None:
    """Block until the engine answers /metrics or the timeout expires."""
    import urllib.request

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=1
            ) as response:
                response.read()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"mock engine failed to start on port {port}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(EngineState()), host="127.0.0.1", port=DEFAULT_PORT, log_level="warning"
    )
