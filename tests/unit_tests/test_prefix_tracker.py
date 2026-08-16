"""Prefix-diff scheduling algorithm tests (ported from agent-core KVCacheManager)."""

from __future__ import annotations

from langchain_ascend.prefix_tracker import PrefixCacheTracker, ReleasePlan, _first_divergence


def _msgs(*contents):
    return [{"role": "user", "content": c} for c in contents]


def _tools(*names):
    return [{"type": "function", "function": {"name": n}} for n in names]


class TestFirstDivergence:
    def test_none_for_identical_sequences(self):
        assert _first_divergence(_msgs("a", "b"), _msgs("a", "b")) is None

    def test_none_for_pure_append(self):
        assert _first_divergence(_msgs("a"), _msgs("a", "b")) is None

    def test_returns_first_mismatch_index(self):
        assert _first_divergence(_msgs("a", "b", "c"), _msgs("a", "X", "c")) == 1

    def test_shrinked_current_is_not_a_rewrite(self):
        # previous longer than current = trimmed tail, not a divergence;
        # unused tail blocks are reclaimed by the engine itself.
        assert _first_divergence(_msgs("a", "b"), _msgs("a")) is None

    def test_empty_inputs(self):
        assert _first_divergence(None, None) is None
        assert _first_divergence([], []) is None


class TestCheckReleaseNeeded:
    def test_first_observation_never_releases(self):
        tracker = PrefixCacheTracker()
        assert tracker.check_release_needed("s", _msgs("a")) is None

    def test_pure_append_keeps_cache_hot(self):
        tracker = PrefixCacheTracker()
        tracker.update("s", _msgs("a", "b"))
        assert tracker.check_release_needed("s", _msgs("a", "b", "c")) is None

    def test_rewritten_message_releases_from_divergence(self):
        tracker = PrefixCacheTracker()
        tracker.update("s", _msgs("a", "b", "c"))
        plan = tracker.check_release_needed("s", _msgs("a", "X", "c"))
        assert plan is not None
        assert plan.messages_released_index == 1
        assert plan.messages == _msgs("a", "b", "c")
        assert plan.tools is None and plan.tools_released_index is None

    def test_shrunk_history_does_not_release(self):
        # Matches agent-core semantics: a shorter window (trim) is not a
        # rewrite — vLLM prefix blocks are content-hashed, so unused tail
        # blocks are reclaimed by the engine itself.
        tracker = PrefixCacheTracker()
        tracker.update("s", _msgs("a", "b", "c"))
        assert tracker.check_release_needed("s", _msgs("a")) is None

    def test_tools_divergence_triggers_release(self):
        tracker = PrefixCacheTracker()
        tracker.update("s", _msgs("a"), _tools("lookup", "calc"))
        plan = tracker.check_release_needed("s", _msgs("a"), _tools("lookup", "search"))
        assert plan is not None
        assert plan.tools == _tools("lookup", "calc")
        assert plan.tools_released_index == 1
        # messages unchanged -> released at len(prev), keeping prefix valid
        assert plan.messages_released_index == 1

    def test_tools_pure_append_does_not_release(self):
        tracker = PrefixCacheTracker()
        tracker.update("s", _msgs("a"), _tools("lookup"))
        assert tracker.check_release_needed("s", _msgs("a"), _tools("lookup", "calc")) is None

    def test_sessions_are_isolated(self):
        tracker = PrefixCacheTracker()
        tracker.update("s1", _msgs("a"))
        tracker.update("s2", _msgs("z"))
        assert tracker.check_release_needed("s1", _msgs("a")) is None
        assert tracker.check_release_needed("s2", _msgs("q")) is not None


class TestBookkeeping:
    def test_clear_resets_session(self):
        tracker = PrefixCacheTracker()
        tracker.update("s", _msgs("a"))
        tracker.clear("s")
        assert tracker.check_release_needed("s", _msgs("a")) is None

    def test_update_replaces_window(self):
        tracker = PrefixCacheTracker()
        tracker.update("s", _msgs("a"))
        tracker.update("s", _msgs("a", "b"))
        # diff against the newest window: rewriting "b" now diverges at 1
        plan = tracker.check_release_needed("s", _msgs("a", "X"))
        assert plan is not None
        assert plan.messages_released_index == 1


class TestReleasePlanShape:
    def test_plan_is_immutable(self):
        import dataclasses
        import pytest

        plan = ReleasePlan(messages=_msgs("a"), messages_released_index=0)
        assert dataclasses.is_dataclass(plan)
        with pytest.raises(Exception):
            plan.messages_released_index = 5
