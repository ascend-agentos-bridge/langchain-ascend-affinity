"""AscendAffinityChatModel contract tests (openJiuwen affinity port)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List
from urllib.error import HTTPError

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_ascend import AscendAffinityChatModel

_TOOL_RESPONSE: Dict[str, Any] = {
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_quote",
                            "arguments": '{"ticker": "SH000001"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "model": "mindie-llama3",
}


def _patch_post(model: AscendAffinityChatModel, mocker, response=None) -> Any:
    """Patch the HTTP layer; return the mock call recorder."""
    return mocker.patch.object(
        model, "_post", return_value=response or {"choices": [{"message": {"content": "ok"}}]}
    )


class TestRequestContract:
    def test_bound_session_carries_salt(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert payload["cache_sharing"] is True
        assert payload["cache_salt"] == "fixture-session"

    def test_no_session_stays_plain_client(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test/v1")
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert "cache_sharing" not in payload
        assert "cache_salt" not in payload

    def test_affinity_disabled_plain_payload(self, chat_llm, mocker):
        chat_llm.enable_affinity = False
        post = _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert "cache_sharing" not in payload
        assert "cache_salt" not in payload

    def test_session_resolution_priority(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        bound = chat_llm.bind(session_id="per-call-session")
        bound.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert payload["cache_salt"] == "per-call-session"

    def test_session_from_run_manager_metadata(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test/v1")
        post = _patch_post(model, mocker)
        manager = SimpleNamespace(metadata={"session_id": "meta-session"})
        model._generate(
            [HumanMessage(content="hi")],
            run_manager=manager,  # type: ignore[arg-type]
        )
        payload = post.call_args[0][2]
        assert payload["cache_salt"] == "meta-session"

    def test_chat_endpoint_url_and_auth(self, chat_llm, mocker):
        fake = mocker.MagicMock()
        fake.read.return_value = b'{"choices": [{"message": {"content": "ok"}}]}'
        fake.__enter__.return_value = fake
        urlopen = mocker.patch("urllib.request.urlopen", return_value=fake)
        chat_llm.invoke([HumanMessage(content="hi")])
        request = urlopen.call_args[0][0]
        assert request.full_url == "http://engine.test/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer EMPTY"

    def test_base_payload_fields(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert payload["model"] == "mindie-llama3"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["temperature"] == 0.7


class TestSchedulingFidelity:
    def test_pure_append_loop_never_releases(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        window: List[Any] = [HumanMessage(content="q")]
        for answer in ["a1", "a2", "a3"]:
            chat_llm.invoke(window)
            window = window + [AIMessage(content=answer), HumanMessage(content="next")]
        release_calls = [
            call for call in post.call_args_list if call[0][1] == "/release_kv_cache"
        ]
        assert not release_calls

    def test_rewritten_history_releases_previous_window(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="old answer")])
        chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="SUMMARY")])
        release_calls = [
            call for call in post.call_args_list if call[0][1] == "/release_kv_cache"
        ]
        assert len(release_calls) == 1
        root, _path, payload = release_calls[0][0]
        assert root == "http://engine.test"
        assert payload["cache_salt"] == "fixture-session"
        assert payload["cache_sharing"] is True
        assert payload["model"] == "mindie-llama3"
        assert payload["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "old answer"},
        ]
        assert payload["messages_released_index"] == 1

    def test_tool_divergence_releases_tool_indices(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        tools_v1 = [{"type": "function", "function": {"name": "lookup"}}]
        tools_v2 = [{"type": "function", "function": {"name": "search"}}]
        chat_llm.bind(tools=tools_v1).invoke([HumanMessage(content="hi")])
        chat_llm.bind(tools=tools_v2).invoke([HumanMessage(content="hi")])
        release_calls = [
            call for call in post.call_args_list if call[0][1] == "/release_kv_cache"
        ]
        assert len(release_calls) == 1
        payload = release_calls[0][0][2]
        assert payload["tools"] == tools_v1
        assert payload["tools_released_index"] == 0

    def test_sessions_are_isolated(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        chat_llm.bind(session_id="s1").invoke([HumanMessage(content="a")])
        chat_llm.bind(session_id="s2").invoke([HumanMessage(content="z")])
        chat_llm.bind(session_id="s1").invoke([HumanMessage(content="a")])
        release_calls = [
            call for call in post.call_args_list if call[0][1] == "/release_kv_cache"
        ]
        assert not release_calls


class TestReleaseTransport:
    def test_release_failure_does_not_break_generation(self, chat_llm, mocker):
        ok = {"choices": [{"message": {"content": "ok"}}]}
        responses: List[Any] = [
            ok,  # invoke #1: chat (first window, no release yet)
            OSError("engine busy"),  # invoke #2: release (must be swallowed)
            ok,  # invoke #2: chat
        ]

        def fake_post(root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            result = responses.pop(0)
            if isinstance(result, OSError):
                raise result
            return result

        mocker.patch.object(chat_llm, "_post", side_effect=fake_post)
        chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="old")])
        result = chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="new")])
        assert result.content == "ok"
        assert not responses

    def test_empty_release_endpoint_skips_requests(self, chat_llm, mocker):
        chat_llm.release_endpoint = ""
        post = _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="old")])
        chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="new")])
        roots = [call[0][0] for call in post.call_args_list]
        assert roots == [
            "http://engine.test/v1/chat/completions",
            "http://engine.test/v1/chat/completions",
        ]

    async def test_agenerate_applies_affinity(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        await chat_llm.ainvoke([HumanMessage(content="hi"), AIMessage(content="old")])
        await chat_llm.ainvoke([HumanMessage(content="hi"), AIMessage(content="new")])
        release_calls = [
            call for call in post.call_args_list if call[0][1] == "/release_kv_cache"
        ]
        assert len(release_calls) == 1


class TestToolCalling:
    def test_bind_tools_converts_and_sends(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        bound = chat_llm.bind_tools(
            [{"name": "lookup_quote", "description": "d", "parameters": {"type": "object"}}]
        )
        bound.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert payload["tools"][0]["function"]["name"] == "lookup_quote"

    def test_tool_calls_parsed_into_ai_message(self, chat_llm, mocker):
        _patch_post(chat_llm, mocker, response=_TOOL_RESPONSE)
        result = chat_llm.invoke([HumanMessage(content="hi")])
        assert result.content == ""
        assert result.tool_calls == [
            {
                "name": "lookup_quote",
                "args": {"ticker": "SH000001"},
                "id": "call_1",
                "type": "tool_call",
            }
        ]

    def test_llm_type_and_export(self, chat_llm):
        assert chat_llm._llm_type == "ascend-affinity-chat"
        assert json.loads(json.dumps(chat_llm.model)) == chat_llm.model


class _FakeSSE:
    """urlopen stub serving an OpenAI SSE body built from JSON events."""

    def __init__(self, payloads: List[Dict[str, Any]]) -> None:
        self._lines = [f"data: {json.dumps(p)}".encode() for p in payloads]
        self._lines.append(b"data: [DONE]")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _content_event(text: str) -> Dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def _patch_sse(model: AscendAffinityChatModel, mocker, payloads) -> Any:
    """Patch urlopen to serve ``payloads`` as SSE; return call recorder."""
    return mocker.patch(
        "urllib.request.urlopen", return_value=_FakeSSE(payloads)
    )


class TestStreaming:
    def test_stream_yields_content_chunks(self, chat_llm, mocker):
        urlopen = _patch_sse(
            chat_llm, mocker, [_content_event("Hel"), _content_event("lo")]
        )
        chunks = list(chat_llm.stream([HumanMessage(content="hi")]))
        assert "".join(c.text for c in chunks) == "Hello"
        request = urlopen.call_args[0][0]
        payload = json.loads(request.data)
        assert payload["stream"] is True
        assert payload["cache_sharing"] is True
        assert payload["cache_salt"] == "fixture-session"

    def test_stream_assembles_tool_call_deltas(self, chat_llm, mocker):
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {"name": "lookup_quote", "arguments": ""},
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"ticker": "A"}'},
                                }
                            ]
                        },
                    }
                ]
            },
        ]
        _patch_sse(chat_llm, mocker, events)
        chunks = list(chat_llm.stream([HumanMessage(content="hi")]))
        assert chunks[0].tool_call_chunks[0]["name"] == "lookup_quote"
        full = chunks[0] + chunks[1]
        assembled = full.tool_calls[0]
        assert assembled["name"] == "lookup_quote"
        assert assembled["args"] == {"ticker": "A"}
        assert assembled["id"] == "call_9"
        assert assembled["type"] == "tool_call"

    def test_stream_skips_role_only_delta(self, chat_llm, mocker):
        events = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            _content_event("hi"),
        ]
        _patch_sse(chat_llm, mocker, events)
        chunks = list(chat_llm.stream([HumanMessage(content="q")]))
        # the framework appends one empty finalization chunk; only real
        # deltas may carry content
        assert [c.text for c in chunks if c.text] == ["hi"]

    def test_stream_notifies_run_manager(self, chat_llm, mocker):
        _patch_sse(chat_llm, mocker, [_content_event("Hel"), _content_event("lo")])
        captured: List[str] = []

        def record(token: str) -> None:
            captured.append(token)

        manager = SimpleNamespace(on_llm_new_token=record)
        list(chat_llm._stream([HumanMessage(content="q")], run_manager=manager))
        assert captured == ["Hel", "lo"]

    async def test_astream_streams_tokens(self, chat_llm, mocker):
        _patch_sse(chat_llm, mocker, [_content_event("a"), _content_event("b")])
        parts = [c.text async for c in chat_llm.astream([HumanMessage(content="q")])]
        assert "".join(parts) == "ab"


class TestStreamingRoute:
    """invoke/ainvoke must keep the run manager (metadata -> cache salt)."""

    def test_should_stream_false_for_generation(self, chat_llm):
        assert chat_llm._should_stream(async_api=False) is False
        assert chat_llm._should_stream(async_api=True) is False

    def test_should_stream_true_for_explicit_stream_api(self, chat_llm):
        assert chat_llm._should_stream(async_api=False, stream=True) is True
        assert chat_llm._should_stream(async_api=True, stream=True) is True

    def test_generate_with_stream_kwarg_streams_internally(self, chat_llm, mocker):
        """Explicit stream=True stays on the internal streaming path so the
        run manager (and its metadata) is never dropped."""
        events = [
            _content_event("Hel"),
            _content_event("lo"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        ]
        mocker.patch.object(chat_llm, "_stream_events", return_value=iter(events))
        post = _patch_post(chat_llm, mocker)
        manager = SimpleNamespace(
            metadata={"session_id": "meta-session"}, on_llm_new_token=lambda *a, **k: None
        )
        result = chat_llm._generate(
            [HumanMessage(content="hi")],
            run_manager=manager,  # type: ignore[arg-type]
            stream=True,
        )
        assert result.generations[0].message.content == "Hello"
        assert post.call_count == 0  # streaming path, not _post

    async def test_agenerate_uses_sync_run_manager(self, chat_llm, mocker):
        """The worker thread must receive the sync counterpart of the async
        run manager so on_llm_new_token stays thread-safe and metadata (the
        session salt) is still visible."""

        class FakeAsyncManager:
            def __init__(self) -> None:
                self.sync = SimpleNamespace(
                    metadata={"session_id": "async-meta-session"},
                    on_llm_new_token=lambda *a, **k: None,
                )

            def get_sync(self):
                return self.sync

        post = _patch_post(chat_llm, mocker)
        await chat_llm._agenerate(
            [HumanMessage(content="hi")], run_manager=FakeAsyncManager()  # type: ignore[arg-type]
        )
        payload = post.call_args[0][2]
        assert payload["cache_salt"] == "async-meta-session"

    def test_stream_usage_propagates_to_recorder(self, mocker):
        """A streamed usage chunk must surface on the aggregated message."""
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", streaming=True
        )
        events = [
            _content_event("Hel"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                    "prompt_tokens_details": {"cached_tokens": 15},
                },
            },
        ]
        mocker.patch.object(model, "_stream_events", return_value=iter(events))
        message = model.invoke([HumanMessage(content="q")])
        assert message.usage_metadata["input_tokens"] == 20
        assert message.usage_metadata["input_token_details"] == {"cache_read": 15}


class TestAffinityStats:
    def test_counters_track_bind_and_failed_release(self, chat_llm, mocker):
        ok = {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            if path == "/release_kv_cache":
                raise OSError("engine busy")
            return ok

        mocker.patch.object(chat_llm, "_post", side_effect=fake_post)
        chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="old")])
        chat_llm.invoke([HumanMessage(content="hi"), AIMessage(content="new")])
        assert chat_llm.affinity_stats == {
            "affinity_requests": 2,
            "salt_bound_requests": 2,
            "releases_attempted": 1,
            "releases_failed": 1,
            "management_requests": 0,
            "management_failed": 0,
            "salt_degraded_requests": 0,
        }

    def test_stats_property_returns_copy(self, chat_llm, mocker):
        _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi")])
        snapshot = chat_llm.affinity_stats
        snapshot["affinity_requests"] = 999
        assert chat_llm.affinity_stats["affinity_requests"] == 1

    def test_disabled_affinity_keeps_zero_counters(self, chat_llm, mocker):
        chat_llm.enable_affinity = False
        _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi")])
        assert chat_llm.affinity_stats == {
            "affinity_requests": 0,
            "salt_bound_requests": 0,
            "releases_attempted": 0,
            "releases_failed": 0,
            "management_requests": 0,
            "management_failed": 0,
            "salt_degraded_requests": 0,
        }


class TestSaltRejectionDegradation:
    """MindIE-class engines reject salt-bound tool-call requests with HTTP 501
    (observed on the 2026-08-20 real-engine run: 18 failed tool tasks while the
    no-salt 0818 run had zero). The client must retry once without the salt
    fields and then stop binding salt for the instance."""

    @staticmethod
    def _http_501() -> HTTPError:
        return HTTPError(
            "http://engine.test/v1/chat/completions", 501, "Not Implemented", None, None
        )

    def test_501_retries_without_salt_then_disables_binding(self, chat_llm, mocker):
        ok = {"choices": [{"message": {"content": "ok"}}]}
        calls: List[Dict[str, Any]] = []

        def fake_post(root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            calls.append(payload)
            if len(calls) == 1:
                raise self._http_501()
            return ok

        mocker.patch.object(chat_llm, "_post", side_effect=fake_post)
        result = chat_llm.invoke([HumanMessage(content="hi")])
        assert result.content == "ok"
        assert "cache_salt" in calls[0]  # first attempt was salt-bound
        assert "cache_sharing" not in calls[1]  # retry dropped the salt fields
        assert "cache_salt" not in calls[1]
        assert chat_llm.affinity_stats["salt_degraded_requests"] == 1
        # subsequent calls skip salt binding entirely
        chat_llm.invoke([HumanMessage(content="again")])
        assert "cache_salt" not in calls[2]
        assert "cache_sharing" not in calls[2]
        assert chat_llm.affinity_stats["salt_bound_requests"] == 1

    def test_501_retry_failure_raises(self, chat_llm, mocker):
        def fake_post(root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            raise self._http_501()

        mocker.patch.object(chat_llm, "_post", side_effect=fake_post)
        with pytest.raises(HTTPError):
            chat_llm.invoke([HumanMessage(content="hi")])
        assert chat_llm.affinity_stats["salt_degraded_requests"] == 1

    def test_non_501_error_never_degrades(self, chat_llm, mocker):
        def fake_post(root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            raise HTTPError(
                "http://engine.test/v1/chat/completions", 503, "Unavailable", None, None
            )

        mocker.patch.object(chat_llm, "_post", side_effect=fake_post)
        with pytest.raises(HTTPError):
            chat_llm.invoke([HumanMessage(content="hi")])
        assert chat_llm.affinity_stats["salt_degraded_requests"] == 0
        assert chat_llm._salt_degraded is False  # pylint: disable=protected-access

    def test_plain_request_501_propagates_untouched(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test/v1")

        def fake_post(root: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            raise self._http_501()

        mocker.patch.object(model, "_post", side_effect=fake_post)
        with pytest.raises(HTTPError):
            model.invoke([HumanMessage(content="hi")])
        assert model.affinity_stats["salt_degraded_requests"] == 0

    def test_streaming_501_retries_without_salt(self, chat_llm, mocker):
        urlopen = mocker.patch(
            "urllib.request.urlopen",
            side_effect=[
                self._http_501(),
                _FakeSSE([_content_event("Hel"), _content_event("lo")]),
            ],
        )
        chunks = list(chat_llm.stream([HumanMessage(content="hi")]))
        assert "".join(c.text for c in chunks) == "Hello"
        assert urlopen.call_count == 2
        degraded_payload = json.loads(urlopen.call_args_list[1][0][0].data)
        assert "cache_sharing" not in degraded_payload
        assert "cache_salt" not in degraded_payload
        assert chat_llm.affinity_stats["salt_degraded_requests"] == 1

    def test_salt_enabled_false_keeps_pipeline_but_skips_salt(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", session_id="s", salt_enabled=False
        )
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert "cache_sharing" not in payload
        assert "cache_salt" not in payload
        assert model.affinity_stats["affinity_requests"] == 1
        assert model.affinity_stats["salt_bound_requests"] == 0


class TestUsagePassthrough:
    def test_usage_maps_to_usage_metadata(self, chat_llm, mocker):
        response = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        }
        _patch_post(chat_llm, mocker, response)
        message = chat_llm.invoke([HumanMessage(content="hi")])
        assert message.usage_metadata == {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "input_token_details": {"cache_read": 80},
        }

    def test_missing_usage_leaves_metadata_none(self, chat_llm, mocker):
        _patch_post(chat_llm, mocker)
        message = chat_llm.invoke([HumanMessage(content="hi")])
        assert message.usage_metadata is None

    def test_streaming_invoke_aggregates_chunks_and_usage(self, mocker):
        from langchain_core.callbacks import BaseCallbackHandler

        class _TokenCollector(BaseCallbackHandler):
            def __init__(self) -> None:
                self.tokens: List[str] = []

            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                if token:  # framework also emits empty finalization tokens
                    self.tokens.append(token)

        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", streaming=True
        )
        events = [
            _content_event("Hel"),
            _content_event("lo"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        ]
        collector = _TokenCollector()
        mocker.patch.object(model, "_stream_events", return_value=iter(events))
        message = model.invoke(
            [HumanMessage(content="q")], config={"callbacks": [collector]}
        )
        assert message.content == "Hello"
        assert message.usage_metadata == {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
        }
        assert collector.tokens == ["Hel", "lo"]
