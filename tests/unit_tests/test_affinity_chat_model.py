"""AscendAffinityChatModel contract tests (openJiuwen affinity port)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

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

    def test_no_session_keeps_sharing_without_salt(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test/v1")
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert payload["cache_sharing"] is True
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
        from types import SimpleNamespace

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
        paths = [call[0][1] for call in post.call_args_list]
        assert paths == ["/chat/completions", "/chat/completions"]

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
