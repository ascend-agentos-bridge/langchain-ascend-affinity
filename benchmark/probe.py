"""Engine probing: reachability, release endpoint, streaming, usage, salt+tools.

Split out of ``run_benchmark.py`` so the CLI stays under pylint's
too-many-lines budget (same pattern as ``reporting.py``). All probes are
read-only against the engine; the results decide which affinity features
the runner enables (``release_enabled`` / ``salt_enabled``) and feed the
best-effort engine identity block at the top of the report.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class EngineConfig:
    """Resolved engine access parameters."""

    base_url: str
    engine_root: str
    model: str
    api_key: str


def _http_json(
    url: str, *, headers: Dict[str, str], payload: Optional[Dict[str, Any]] = None
) -> Tuple[int, Any]:
    """Perform a GET/POST and return (status, parsed-json-or-raw-bytes)."""
    request = urllib.request.Request(url, headers=headers)
    if payload is not None:
        request.data = json.dumps(payload).encode("utf-8")
        request.add_header("Content-Type", "application/json")
        request.method = "POST"
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def _salt_tool_probe_payload(model: str) -> Dict[str, Any]:
    """The exact request shape that MindIE-class engines reject with 501:
    cache_sharing/cache_salt together with tool-call messages."""
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "用工具查询基金 F001 的档案。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_probe_1",
                        "type": "function",
                        "function": {
                            "name": "get_fund_profile",
                            "arguments": '{"fund_code": "F001"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_probe_1", "content": "{}"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_fund_profile",
                    "description": "查询基金档案",
                    "parameters": {
                        "type": "object",
                        "properties": {"fund_code": {"type": "string"}},
                        "required": ["fund_code"],
                    },
                },
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_sharing": True,
        "cache_salt": "bench-probe-salt",
    }


# -- engine identity (best-effort type/version fingerprint) ---------------------


def _is_html(body: str) -> bool:
    """True when a response body is an HTML page.

    Gateways with an SPA catch-all answer *any* unknown path with 200 +
    index.html (observed on models.ascend.huawei.com), which would make a
    naive "status == 200" check report endpoints that do not exist. Treat
    HTML bodies as "endpoint absent".
    """
    head = body.lstrip()[:512].lower()
    return head.startswith("<!doctype html") or "<html" in head


def _http_probe(url: str, headers: Dict[str, str]) -> Tuple[int, Optional[str], str]:
    """Cheap GET: (status, Server header, up-to-2KB body) — never raises."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            server = response.headers.get("Server")
            body = response.read(2048).decode("utf-8", errors="replace")
            return response.status, server, body
    except urllib.error.HTTPError as exc:
        server = exc.headers.get("Server") if exc.headers is not None else None
        return exc.code, server, ""
    except OSError:
        return -1, None, ""


def _parse_version(body: str) -> Optional[str]:
    """Extract a version string from a /version body (JSON or plain text)."""
    text = body.strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                text = str(parsed.get("version") or text)
        except json.JSONDecodeError:
            pass
    match = re.match(r"^[vV]?([0-9][0-9A-Za-z.\-]*)", text.strip())
    return match.group(1) if match else None


def _classify_engine(identity: Dict[str, Any]) -> str:
    """Best-effort engine family from the collected signals.

    Resolution order: an explicit ``mindie`` / ``vllm`` marker in the Server
    header or version string beats endpoint heuristics; a working ``/version``
    endpoint strongly suggests vLLM / vLLM-Ascend (MindIE-class servers
    answer 404); a working ``/health`` alone leaves the family open.
    """
    haystack = " ".join(
        [
            str(identity.get("server_header") or ""),
            str(identity.get("version") or ""),
        ]
    ).lower()
    if "mindie" in haystack:
        return "MindIE"
    if "vllm" in haystack:
        return "vLLM / vLLM-Ascend family"
    if identity.get("version_endpoint"):
        return "vLLM-family (serves /version)"
    if identity.get("health"):
        return "OpenAI-compatible (serves /health)"
    return "unknown"


def probe_identity(engine: EngineConfig) -> Dict[str, Any]:
    """Best-effort engine type/version fingerprint; never blocks the run.

    Signals: ``GET /version`` (vLLM-family, JSON ``{"version": ...}``),
    ``GET /health``, ``GET /`` and the HTTP ``Server`` response header —
    all at the engine root, all capped at 2KB with an 8s timeout. Every
    failure degrades to ``None``/``False``; ``engine_type`` falls back to
    ``"unknown"``.
    """
    root = engine.engine_root.rstrip("/")
    identity: Dict[str, Any] = {
        "engine_type": None,
        "version": None,
        "version_endpoint": False,
        "health": False,
        "server_header": None,
    }
    for path, key in (("/version", "version_endpoint"), ("/health", "health")):
        status, server, body = _http_probe(
            f"{root}{path}", {"Authorization": f"Bearer {engine.api_key}"}
        )
        if status != 200 or _is_html(body):
            continue
        identity[key] = True
        if server:
            identity["server_header"] = server
        if key == "version_endpoint":
            identity["version"] = _parse_version(body)
    if identity["server_header"] is None:
        _, server, _ = _http_probe(f"{root}/", {"Authorization": f"Bearer {engine.api_key}"})
        identity["server_header"] = server
    identity["engine_type"] = _classify_engine(identity)
    return identity


def probe_engine(engine: EngineConfig) -> Dict[str, Any]:
    """Probe reachability, model list, release endpoint and streaming."""
    auth = {"Authorization": f"Bearer {engine.api_key}"}
    probe: Dict[str, Any] = {"base_url": engine.base_url, "model": engine.model}
    try:
        status, data = _http_json(f"{engine.base_url.rstrip('/')}/models", headers=auth)
        if status != 200:
            raise OSError(f"models endpoint returned {status}")
        entries = data.get("data") if isinstance(data, dict) else None
        probe["reachable"] = True
        probe["model_listed"] = engine.model in [
            entry.get("id") for entry in (entries or [])
        ]
    except OSError as exc:
        probe["reachable"] = False
        probe["error"] = str(exc)
        return probe
    probe["identity"] = probe_identity(engine)
    release_status, release_body = _http_json(
        f"{engine.engine_root.rstrip('/')}/release_kv_cache",
        headers=auth,
        payload={
            "model": engine.model,
            "cache_salt": "bench-probe",
            "cache_sharing": True,
            "messages": [{"role": "user", "content": "ping"}],
            "messages_released_index": 0,
        },
    )
    probe["release_endpoint"] = (
        release_status not in (404, 405) and not _is_html(str(release_body))
    )
    try:
        stream_status, stream_body = _http_json(
            f"{engine.base_url.rstrip('/')}/chat/completions",
            headers=auth,
            payload={
                "model": engine.model,
                "messages": [{"role": "user", "content": "回复：好"}],
                "max_tokens": 8,
                "stream": True,
            },
        )
        probe["streaming"] = stream_status == 200 and "data:" in str(stream_body)
    except OSError:
        probe["streaming"] = False
    try:
        usage_status, usage_body = _http_json(
            f"{engine.base_url.rstrip('/')}/chat/completions",
            headers=auth,
            payload={
                "model": engine.model,
                "messages": [{"role": "user", "content": "回复：好"}],
                "max_tokens": 8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        # vLLM-family engines emit a final SSE event carrying a top-level
        # "usage" block only when stream_options.include_usage is honored.
        probe["stream_usage"] = usage_status == 200 and '"usage"' in str(usage_body)
    except OSError:
        probe["stream_usage"] = False
    try:
        # MindIE-class engines actively reject (HTTP 501) requests that carry
        # cache_sharing/cache_salt together with tool-call messages, which
        # would abort every tool-using task of the affinity agent. Probe the
        # exact combination up front so the runner can disable salt binding
        # (salt_enabled=False) and keep the tool tasks executable.
        salt_status, salt_body = _http_json(
            f"{engine.base_url.rstrip('/')}/chat/completions",
            headers=auth,
            payload=_salt_tool_probe_payload(engine.model),
        )
        probe["salt_tool_calls"] = (
            salt_status == 200 and not _is_html(str(salt_body))
        )
    except OSError:
        probe["salt_tool_calls"] = False
    return probe
