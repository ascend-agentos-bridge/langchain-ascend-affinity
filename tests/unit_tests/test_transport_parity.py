"""Transport parity with openjiuwen agent-core (2026-08 vLLM fix)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from langchain_ascend import AscendAffinityChatModel


def _patch_post(model: AscendAffinityChatModel, mocker: Any) -> Any:
    return mocker.patch.object(
        model, "_post", return_value={"choices": [{"message": {"content": "ok"}}]}
    )


class TestChatCompletionsUrl:
    def test_v1_base_url(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test/v1")
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        assert post.call_args[0][0] == "http://engine.test/v1/chat/completions"

    def test_origin_base_url_appends_v1(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test")
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        assert post.call_args[0][0] == "http://engine.test/v1/chat/completions"

    def test_full_endpoint_url_used_as_is(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1/chat/completions"
        )
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        assert post.call_args[0][0] == "http://engine.test/v1/chat/completions"

    def test_engine_root_strips_chat_path(self):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1/chat/completions"
        )
        assert model._engine_root() == "http://engine.test"
        model = AscendAffinityChatModel(base_url="http://engine.test/v1")
        assert model._engine_root() == "http://engine.test"
        model = AscendAffinityChatModel(base_url="http://engine.test")
        assert model._engine_root() == "http://engine.test"


class TestOptionalAuth:
    def test_auth_sent_when_api_key_set(self, chat_llm, mocker):
        fake = mocker.MagicMock()
        fake.read.return_value = b'{"choices": [{"message": {"content": "ok"}}]}'
        fake.__enter__.return_value = fake
        urlopen = mocker.patch("urllib.request.urlopen", return_value=fake)
        chat_llm.invoke([HumanMessage(content="hi")])
        request = urlopen.call_args[0][0]
        assert request.get_header("Authorization") == "Bearer EMPTY"

    def test_auth_omitted_when_api_key_empty(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test/v1", api_key="")
        fake = mocker.MagicMock()
        fake.read.return_value = b'{"choices": [{"message": {"content": "ok"}}]}'
        fake.__enter__.return_value = fake
        urlopen = mocker.patch("urllib.request.urlopen", return_value=fake)
        model.invoke([HumanMessage(content="hi")])
        request = urlopen.call_args[0][0]
        assert request.get_header("Authorization") is None


class TestCapabilityFlag:
    def test_supports_kv_cache_affinity_reflects_enable_agent_hint(self):
        assert AscendAffinityChatModel(base_url="http://e/v1").supports_kv_cache_affinity() is False
        assert (
            AscendAffinityChatModel(
                base_url="http://e/v1", enable_agent_hint=True
            ).supports_kv_cache_affinity()
            is True
        )
