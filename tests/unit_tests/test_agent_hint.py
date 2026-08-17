"""Unit tests for the agent_hint lifecycle protocol (stage A)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from langchain_ascend import AscendAffinityChatModel


def _patch_post(model: AscendAffinityChatModel, mocker: Any) -> Any:
    return mocker.patch.object(
        model, "_post", return_value={"choices": [{"message": {"content": "ok"}}]}
    )


class TestAgentHintIdentity:
    def test_disabled_by_default(self, chat_llm, mocker):
        post = _patch_post(chat_llm, mocker)
        chat_llm.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert "agent_hint" not in payload

    def test_identity_injected_when_enabled(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        post = _patch_post(model, mocker)
        model.invoke(
            [HumanMessage(content="hi")],
            config={"metadata": {"session_id": "s1"}},
        )
        payload = post.call_args[0][2]
        assert payload["agent_hint"] == {
            "session_id": "s1",
            "parent_session_id": "s1",
        }

    def test_parent_session_override(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        post = _patch_post(model, mocker)
        model.invoke(
            [HumanMessage(content="hi")],
            config={"metadata": {"session_id": "s1", "parent_session_id": "team-1"}},
        )
        payload = post.call_args[0][2]
        assert payload["agent_hint"]["parent_session_id"] == "team-1"

    def test_no_session_no_agent_hint(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert "agent_hint" not in payload

    def test_respects_affinity_disabled(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1",
            enable_agent_hint=True,
            enable_affinity=False,
        )
        post = _patch_post(model, mocker)
        model.invoke([HumanMessage(content="hi")])
        payload = post.call_args[0][2]
        assert "agent_hint" not in payload
        assert "cache_salt" not in payload


class TestManagementMethods:
    def test_evict_session_payload(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        post = _patch_post(model, mocker)
        assert model.evict_kvc(session_id="s1") is True
        root, path, payload = post.call_args[0]
        assert root == "http://engine.test/v1/chat/completions"
        assert path == ""
        assert payload["messages"] == []
        hint = payload["agent_hint"]
        assert hint["session_id"] == "s1"
        assert hint["parent_session_id"] == "s1"
        assert hint["context_management"] == {
            "manage_request": True,
            "edits": [{"type": "evict", "target": "session"}],
        }

    def test_offload_and_prefetch(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        post = _patch_post(model, mocker)
        assert model.offload_kvc(session_id="s1") is True
        payload = post.call_args[0][2]
        assert payload["agent_hint"]["context_management"]["edits"] == [
            {"type": "offload", "target": "session"}
        ]
        assert model.prefetch_kvc(session_id="s1") is True
        payload = post.call_args[0][2]
        assert payload["agent_hint"]["context_management"]["edits"] == [
            {"type": "prefetch", "target": "session"}
        ]

    def test_management_requires_enabled(self, mocker):
        model = AscendAffinityChatModel(base_url="http://engine.test/v1")
        post = _patch_post(model, mocker)
        assert model.evict_kvc(session_id="s1") is False
        post.assert_not_called()

    def test_management_failure_non_fatal(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        mocker.patch.object(model, "_post", side_effect=OSError("engine busy"))
        assert model.evict_kvc(session_id="s1") is False
        assert model.affinity_stats["management_requests"] == 1
        assert model.affinity_stats["management_failed"] == 1

    def test_messages_target_payload(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        post = _patch_post(model, mocker)
        ok = model.evict_kvc(
            session_id="s1",
            target="messages",
            messages=[HumanMessage(content="hi")],
            msg_start=1,
            msg_end=3,
            include_tools=True,
            tools_start=0,
            tools_end=1,
        )
        assert ok is True
        payload = post.call_args[0][2]
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        edits = payload["agent_hint"]["context_management"]["edits"]
        assert edits == [
            {"type": "evict", "target": "messages", "start": 1, "end": 3},
            {"type": "evict", "target": "tools", "start": 0, "end": 1},
        ]

    def test_validation_errors(self, mocker):
        model = AscendAffinityChatModel(
            base_url="http://engine.test/v1", enable_agent_hint=True
        )
        _patch_post(model, mocker)
        with pytest.raises(ValueError, match="target=session"):
            model.evict_kvc(session_id="s1", msg_start=0, msg_end=2)
        with pytest.raises(ValueError, match="messages is required"):
            model.evict_kvc(session_id="s1", target="messages")
        with pytest.raises(ValueError, match="unknown agent_hint target"):
            model.evict_kvc(session_id="s1", target="bogus")
        with pytest.raises(ValueError, match="requires start and end"):
            model.evict_kvc(session_id="s1", target="messages",
                            messages=[HumanMessage(content="hi")])
