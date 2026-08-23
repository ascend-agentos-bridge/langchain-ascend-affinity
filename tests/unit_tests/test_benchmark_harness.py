# pylint: disable=too-many-lines
"""Benchmark harness unit tests: metrics engine, tasks, oj adapter glue."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position  # repo-root path hack
from benchmark import tasks as bench_tasks
from benchmark.metrics import (
    NA,
    PASS,
    CallMetrics,
    aggregate,
    cache_hit_rate_delta,
    cache_usage_peak,
    fetch_prometheus,
    judge,
    median_metrics,
    overall_verdict,
    sample_sidecar,
    usage_field,
    verdict_text,
)
from benchmark.oj_adapter import OJCallCollector
from benchmark.probe import probe_identity
from benchmark.run_benchmark import (
    EngineConfig,
    LlmCallRecord,
    TTFTRecorder,
    build_agents,
    configure_logging,
    probe_engine,
    records_to_metrics,
    rotate,
    resolve_agents,
    run_agent_phase,
    run_benchmark,
)


def _call(
    ttft: Optional[float],
    e2e: float,
    prompt: Optional[int] = 100,
    completion: Optional[int] = 50,
    cached: Optional[int] = None,
) -> CallMetrics:
    return CallMetrics(
        agent="a",
        task_id="t",
        round_idx=0,
        ttft_ms=ttft,
        e2e_ms=e2e,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
    )


class TestAggregate:
    def test_tpot_and_hit_rate(self):
        metrics = aggregate(
            [
                _call(100.0, 1100.0, prompt=1000, completion=101, cached=500),
                _call(200.0, 1200.0, prompt=500, completion=101, cached=250),
            ]
        )
        assert metrics.llm_calls == 2
        assert metrics.streamed_calls == 2
        assert metrics.ttft_mean_ms == 150.0
        # (1100-100)/100 + (1200-200)/100 = 10+10 ms per output token
        assert metrics.tpot_mean_ms == 10.0
        assert metrics.kv_hit_rate == 50.0  # 750 / 1500
        assert metrics.prefill_per_call == 750.0
        assert metrics.decode_per_call == 101.0

    def test_no_usage_degrades_to_none(self):
        metrics = aggregate([_call(None, 900.0, prompt=None, completion=None)])
        assert metrics.ttft_mean_ms is None
        assert metrics.kv_hit_rate is None
        assert metrics.decode_tps is None

    def test_median_across_rounds(self):
        rounds = [
            aggregate([_call(100.0, 1000.0)]),
            aggregate([_call(300.0, 3000.0)]),
            aggregate([_call(200.0, 2000.0)]),
        ]
        median = median_metrics(rounds)
        assert median.ttft_mean_ms == 200.0
        assert median.e2e_mean_ms == 2000.0
        assert median.llm_calls == 1

    def test_median_averages_token_totals(self):
        rounds = [
            aggregate([_call(100.0, 1000.0, prompt=100, completion=40)]),
            aggregate([_call(200.0, 2000.0, prompt=150, completion=50)]),
            aggregate([_call(300.0, 3000.0, prompt=200, completion=60)]),
        ]
        median = median_metrics(rounds)
        assert median.prefill_tokens == 150  # (100+150+200)/3
        assert median.decode_tokens == 50    # (40+50+60)/3
        assert median.llm_calls == 1

    def test_median_empty_returns_defaults(self):
        median = median_metrics([])
        assert median.llm_calls == 0
        assert median.ttft_mean_ms is None

    def test_median_single_round_passthrough(self):
        single = aggregate([_call(100.0, 1000.0)])
        assert median_metrics([single]) is single


class TestVerdicts:
    def test_ttft_pass_on_drop(self):
        verdict = judge("ttft_mean_ms", 540.0, 812.0)
        assert verdict.status == PASS
        assert verdict.delta == pytest.approx(-33.5, abs=0.1)

    def test_ttft_fail_on_rise(self):
        assert judge("ttft_mean_ms", 900.0, 812.0).status == "FAIL"

    def test_flat_metric_passes_when_unchanged(self):
        assert judge("tpot_mean_ms", 10.2, 10.0).status == PASS  # |+2%|
        assert judge("tpot_mean_ms", 11.5, 10.0).status == "WARN"  # |+15%|
        assert judge("tpot_mean_ms", 15.0, 10.0).status == "FAIL"  # |+50%|

    def test_kv_hit_rate_uses_percentage_points(self):
        assert judge("kv_hit_rate", 40.0, 25.0).status == PASS  # +15pp
        assert judge("kv_hit_rate", 40.0, 25.0).delta == pytest.approx(15.0)
        assert judge("kv_hit_rate", 24.0, 25.0).status == "WARN"  # -1pp
        assert judge("kv_hit_rate", 24.0, 25.0).delta == pytest.approx(-1.0)
        assert judge("kv_hit_rate", 20.0, 25.0).status == "FAIL"

    def test_missing_data_is_na(self):
        assert judge("ttft_mean_ms", None, 100.0).status == NA
        assert judge("ttft_mean_ms", 100.0, None).status == NA

    def test_overall_real_affinity(self):
        verdicts = [
            judge("ttft_mean_ms", 5.0, 10.0),
            judge("prefill_per_call", 90.0, 100.0),
            judge("kv_hit_rate", 40.0, 20.0),
            judge("e2e_mean_ms", 90.0, 100.0),
        ]
        assert "生效" in overall_verdict(verdicts, npu_moved=False)

    def test_overall_false_affinity_alert(self):
        verdicts = [
            judge("ttft_mean_ms", 10.1, 10.0),
            judge("prefill_per_call", 100.0, 100.0),
            judge("kv_hit_rate", 20.5, 20.0),
            judge("e2e_mean_ms", 100.2, 100.0),
        ]
        assert "假亲和" in overall_verdict(verdicts, npu_moved=True)

    def test_overall_partial(self):
        verdicts = [
            judge("ttft_mean_ms", 5.0, 10.0),
            judge("prefill_per_call", 100.0, 100.0),
            judge("kv_hit_rate", 21.0, 20.0),
            judge("e2e_mean_ms", 100.0, 100.0),
        ]
        assert "部分" in overall_verdict(verdicts, npu_moved=False)

    def test_verdict_text_rendering(self):
        verdict = judge("ttft_mean_ms", 540.0, 812.0)
        text = verdict_text(verdict)
        assert "→" in text
        assert "33.5%" in text
        assert "✅" in text

    def test_verdict_text_na_and_pp(self):
        na = judge("ttft_mean_ms", None, 812.0)
        assert "n/a" in verdict_text(na)
        assert verdict_text(na).endswith("➖")
        pp = judge("kv_hit_rate", 40.0, 25.0)
        assert "+15.0pp" in verdict_text(pp)  # percentage points unit
        assert "40.0%" in verdict_text(pp)

    def test_judge_baseline_zero_is_na(self):
        verdict = judge("ttft_mean_ms", 100.0, 0.0)
        assert verdict.status == NA
        assert verdict.delta is None

    def test_overall_no_core_metrics(self):
        assert "不可得" in overall_verdict([], npu_moved=False)

    def test_overall_no_improvement(self):
        verdicts = [
            judge("ttft_mean_ms", 10.1, 10.0),
            judge("prefill_per_call", 100.0, 100.0),
            judge("kv_hit_rate", 20.5, 20.0),
            judge("e2e_mean_ms", 100.2, 100.0),
        ]
        assert "未体现" in overall_verdict(verdicts, npu_moved=False)


class TestEngineSources:
    def test_prometheus_parse_and_cache_delta(self, mocker):
        body = (
            b'vllm:gpu_prefix_cache_hits_total 100\n'
            b'vllm:gpu_prefix_cache_queries_total{model="m"} 200\n'
            b'vllm:gpu_cache_usage_perc 0.55\n'
            b'# HELP noise\n'
        )
        fake = mocker.MagicMock()
        fake.__enter__.return_value.read.return_value = body
        mocker.patch("urllib.request.urlopen", return_value=fake)
        metrics = fetch_prometheus("http://engine/metrics") or {}
        assert metrics["vllm:gpu_prefix_cache_hits_total"] == 100.0
        after = dict(metrics)
        after["vllm:gpu_prefix_cache_hits_total"] = 180.0
        after["vllm:gpu_prefix_cache_queries_total"] = 340.0
        assert cache_hit_rate_delta(metrics, after) == 57.1
        assert cache_usage_peak(metrics, after) == 0.55

    def test_prometheus_empty_url_returns_none(self):
        assert fetch_prometheus("") is None

    def test_sidecar_key_value_parse(self):
        samples = sample_sidecar('echo "util=57.5 hbm=42.1"')
        assert samples == {"util": 57.5, "hbm": 42.1}

    def test_sidecar_disabled_or_bad_returns_none(self):
        assert sample_sidecar("") is None

    def test_prometheus_connection_error_returns_none(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("engine down"))
        assert fetch_prometheus("http://engine/metrics") is None

    def test_prometheus_failure_warns_once_per_url(self, mocker, caplog):
        """A missing /metrics endpoint must log one WARNING per URL instead of
        failing silently 24 times per run."""
        url = "http://engine.test/metrics-unique-1"
        mocker.patch("urllib.request.urlopen", side_effect=OSError("engine down"))
        with caplog.at_level(logging.WARNING, logger="benchmark.metrics"):
            assert fetch_prometheus(url) is None
            assert fetch_prometheus(url) is None
        warnings = [r for r in caplog.records if "metrics endpoint" in r.message]
        assert len(warnings) == 1
        assert url in warnings[0].message

    def test_prometheus_empty_body_warns_once(self, mocker, caplog):
        url = "http://engine.test/metrics-unique-2"
        fake = mocker.MagicMock()
        fake.__enter__.return_value.read.return_value = b"# HELP no metrics at all\n"
        mocker.patch("urllib.request.urlopen", return_value=fake)
        with caplog.at_level(logging.WARNING, logger="benchmark.metrics"):
            assert fetch_prometheus(url) is None
        assert any("no parseable metrics" in r.message for r in caplog.records)

    def test_prometheus_bad_number_line_ignored(self, mocker):
        fake = mocker.MagicMock()
        fake.__enter__.return_value.read.return_value = (
            b"vllm:gpu_prefix_cache_hits_total 1e+\n"
        )
        mocker.patch("urllib.request.urlopen", return_value=fake)
        assert fetch_prometheus("http://engine/metrics") is None

    def test_cache_delta_missing_snapshot(self):
        assert cache_hit_rate_delta(None, {"a": 1.0}) is None
        assert cache_hit_rate_delta({}, {}) is None

    def test_cache_delta_no_queries(self):
        before = {
            "vllm:gpu_prefix_cache_hits_total": 10.0,
            "vllm:gpu_prefix_cache_queries_total": 0.0,
        }
        after = {
            "vllm:gpu_prefix_cache_hits_total": 10.0,
            "vllm:gpu_prefix_cache_queries_total": 0.0,
        }
        assert cache_hit_rate_delta(before, after) is None

    def test_cache_delta_v1_gauge_names(self):
        """V1 engines expose a hit-rate gauge instead of hit/query counters."""
        before = {"vllm:prefix_cache_hit_rate": 0.42}
        after = {"vllm:prefix_cache_hit_rate": 0.57}
        assert cache_hit_rate_delta(before, after) == 57.0
        # a gauge reporting 0-100 directly is passed through at face value
        raw = {"vllm:prefix_cache_hit_rate": 10.0}
        assert cache_hit_rate_delta(raw, {"vllm:prefix_cache_hit_rate": 80.0}) == 80.0

    def test_cache_delta_alt_counter_names(self):
        before = {
            "vllm:prefix_cache_hits_total": 100.0,
            "vllm:prefix_cache_miss_requests_total": 100.0,
        }
        after = {
            "vllm:prefix_cache_hits_total": 160.0,
            "vllm:prefix_cache_miss_requests_total": 100.0,
        }
        assert cache_hit_rate_delta(before, after) == 100.0

    def test_cache_usage_peak_alt_name(self):
        before = {"vllm:cache_usage_percent": 0.4}
        after = {"vllm:cache_usage_percent": 0.75}
        assert cache_usage_peak(before, after) == 0.75

    def test_usage_field_accepts_dict_and_namespace(self):
        assert usage_field({"input_tokens": 7, "output_tokens": 3}, "input_tokens") == 7
        assert usage_field({"input_tokens": 7}, "output_tokens") is None
        assert usage_field(SimpleNamespace(input_tokens=9), "input_tokens") == 9
        assert usage_field(None, "input_tokens") is None

    def test_sidecar_command_error_returns_none(self, mocker):
        mocker.patch("subprocess.run", side_effect=OSError("no sampler"))
        assert sample_sidecar("npu-smi") is None


class TestTasksBaseline:
    def test_synthetic_customers_present(self):
        holdings = bench_tasks.holdings_of("C2001")
        assert holdings["risk_profile"] == "进取型"
        assert bench_tasks.holdings_of("C2025")["customer_id"] == "C2025"

    def test_risk_math_deterministic(self):
        assert bench_tasks.portfolio_risk(70, 20, 10)["rating"] == "激进"
        assert "error" in bench_tasks.portfolio_risk(50, 40, 20)

    def test_fingerprint_stable(self):
        assert bench_tasks.task_fingerprint() == bench_tasks.task_fingerprint()
        assert bench_tasks.task_fingerprint(
            True
        ) != bench_tasks.task_fingerprint(False)

    def test_longrun_task_shape(self):
        task = bench_tasks.load_longrun_tasks()[0]
        assert task.category == "longrun"
        assert "C2001" in task.turns[0] and "C2025" in task.turns[0]

    def test_fund_profile_missing(self):
        profile = bench_tasks.fund_profile_of("F999")
        assert "error" in profile

    def test_risk_ratings_balanced_and_conservative(self):
        assert bench_tasks.portfolio_risk(40, 30, 30)["rating"] == "平衡"
        assert bench_tasks.portfolio_risk(10, 40, 50)["rating"] == "稳健"

    def test_tool_wrappers(self):
        holdings = bench_tasks.get_customer_holdings.func("C2001")
        assert "error" not in holdings
        fund = bench_tasks.get_fund_profile.func("F001")
        assert "error" not in fund
        risk = bench_tasks.compute_portfolio_risk.func(70, 20, 10)
        assert risk["rating"] == "激进"

    def test_build_tools(self):
        tools = bench_tasks.build_tools()
        assert len(tools) == 3
        assert {tool.name for tool in tools} == {
            "get_customer_holdings",
            "get_fund_profile",
            "compute_portfolio_risk",
        }


class TestOJCollector:
    @pytest.mark.asyncio
    async def test_before_after_pair_records_usage(self):
        collector = OJCallCollector("oj-affinity")
        collector.bind_task("task-1", 0)
        usage = SimpleNamespace(input_tokens=120, output_tokens=45, input_token_details=None)
        ctx = SimpleNamespace(
            inputs=SimpleNamespace(response=SimpleNamespace(usage_metadata=usage))
        )
        await collector.before_model_call(ctx)
        await collector.after_model_call(ctx)
        record = collector.records[0]
        assert record.task_id == "task-1"
        assert record.prompt_tokens == 120
        assert record.completion_tokens == 45
        assert record.ttft_ms is None
        assert record.e2e_ms >= 0.0

    @pytest.mark.asyncio
    async def test_warmup_records_are_dropped(self):
        collector = OJCallCollector("oj-affinity")
        ctx = SimpleNamespace(
            inputs=SimpleNamespace(response=SimpleNamespace(usage_metadata=None))
        )
        await collector.before_model_call(ctx)
        await collector.after_model_call(ctx)
        collector.bind_task("real", 0)
        await collector.before_model_call(ctx)
        await collector.after_model_call(ctx)
        collector.drop_warmup()
        assert [r.task_id for r in collector.records] == ["real"]


class TestRunnerGlue:
    def test_rotate_cancels_order_bias(self):
        names = ["a", "b", "c", "d"]
        assert rotate(names, 0) == names
        assert rotate(names, 1) == ["b", "c", "d", "a"]
        assert rotate(names, 5) == ["b", "c", "d", "a"]

    def test_resolve_agents(self):
        assert resolve_agents("all") == [
            "lc-baseline",
            "lc-affinity",
            "oj-baseline",
            "oj-affinity",
        ]
        assert resolve_agents("lc") == ["lc-baseline", "lc-affinity"]
        assert resolve_agents("lc-affinity,oj-affinity") == [
            "lc-affinity",
            "oj-affinity",
        ]

    def test_records_skip_warmup(self):
        records = [
            LlmCallRecord("w", "a", "warmup", 0, 10.0, 50.0),
            LlmCallRecord("r", "a", "task", 0, 10.0, 50.0),
        ]
        converted = records_to_metrics(records)
        assert [c.task_id for c in converted] == ["task"]

    def test_recorder_end_extracts_usage(self):
        recorder = TTFTRecorder("lc-affinity", 0)
        run_id = uuid4()
        recorder.on_llm_start({}, ["hi"], run_id=run_id, metadata={})
        recorder.on_llm_new_token("x", run_id=run_id)
        response = SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=SimpleNamespace(
                            usage_metadata=SimpleNamespace(
                                input_tokens=7,
                                output_tokens=3,
                                input_token_details=None,
                            )
                        )
                    )
                ]
            ]
        )
        recorder.on_llm_end(response, run_id=run_id)
        record: LlmCallRecord = recorder.calls[0]
        assert record.prompt_tokens == 7
        assert record.completion_tokens == 3
        assert record.ttft_ms is not None
        assert record.e2e_ms >= record.ttft_ms

    def test_recorder_end_without_start_ignored(self):
        recorder = TTFTRecorder("a", 0)
        recorder.on_llm_end(SimpleNamespace(generations=[]), run_id=uuid4())
        assert not recorder.calls

    def test_recorder_error_cleans_active_entry(self, caplog):
        """Failed calls (e.g. engine HTTP 501s) never reach on_llm_end; the
        recorder must drop the active entry instead of leaking it."""
        recorder = TTFTRecorder("lc-affinity", 0)
        run_id = uuid4()
        recorder.on_llm_start(
            {},
            ["hi"],
            run_id=run_id,
            metadata={
                "session_id": "bench-s1",
                "bench_agent": "lc-affinity",
                "bench_task": "t1",
            },
        )
        recorder.on_llm_new_token("x", run_id=run_id)
        with caplog.at_level(logging.WARNING, logger="benchmark.run_benchmark"):
            recorder.on_llm_error(
                RuntimeError("HTTP Error 501: Not Implemented"), run_id=run_id
            )
        assert not recorder._active  # pylint: disable=protected-access
        assert not recorder.calls
        assert any(
            "[llm]" in r.message and "failed" in r.message for r in caplog.records
        )

    def test_recorder_extracts_dict_usage_metadata(self):
        """usage_metadata is a TypedDict at runtime; getattr would drop it."""
        recorder = TTFTRecorder("lc-affinity", 0)
        run_id = uuid4()
        recorder.on_llm_start({}, ["hi"], run_id=run_id, metadata={})
        recorder.on_llm_new_token("x", run_id=run_id)
        response = SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=SimpleNamespace(
                            usage_metadata={
                                "input_tokens": 120,
                                "output_tokens": 40,
                                "total_tokens": 160,
                                "input_token_details": {"cache_read": 90},
                            }
                        )
                    )
                ]
            ]
        )
        recorder.on_llm_end(response, run_id=run_id)
        record: LlmCallRecord = recorder.calls[0]
        assert record.prompt_tokens == 120
        assert record.completion_tokens == 40
        assert record.cached_tokens == 90

    def test_oj_collector_extracts_dict_usage(self):
        collector = OJCallCollector("oj-affinity")
        collector.bind_task("task-1", 0)
        usage = {
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
            "input_token_details": {"cache_read": 100},
        }
        ctx = SimpleNamespace(
            inputs=SimpleNamespace(response=SimpleNamespace(usage_metadata=usage))
        )

        async def run() -> None:
            await collector.before_model_call(ctx)
            await collector.after_model_call(ctx)

        asyncio.run(run())
        record = collector.records[0]
        assert record.prompt_tokens == 120
        assert record.completion_tokens == 45
        assert record.cached_tokens == 100

    def test_probe_engine_reports_stream_usage(self, mocker):
        engine = EngineConfig(
            base_url="http://engine.test/v1",
            engine_root="http://engine.test",
            model="dsv4-0731",
            api_key="empty",
        )
        responses = [
            (200, {"data": [{"id": "dsv4-0731"}]}),
            (404, "not found"),
            (200, 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]'),
            (
                200,
                'data: {"choices":[]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":9}}\n\ndata: [DONE]',
            ),
            (200, 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]'),
        ]
        mocker.patch(
            "benchmark.probe._http_json",
            side_effect=responses,
        )
        mocker.patch("benchmark.probe._http_probe", return_value=(-1, None, ""))
        probe = probe_engine(engine)
        assert probe["reachable"] is True
        assert probe["model_listed"] is True
        assert probe["release_endpoint"] is False
        assert probe["streaming"] is True
        assert probe["stream_usage"] is True
        assert probe["salt_tool_calls"] is True

    def test_probe_engine_stream_usage_missing(self, mocker):
        engine = EngineConfig(
            base_url="http://engine.test/v1",
            engine_root="http://engine.test",
            model="m",
            api_key="",
        )
        responses = [
            (200, {"data": [{"id": "m"}]}),
            (200, "{}"),
            (200, 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]'),
            (200, 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]'),
            (501, "Not Implemented"),
        ]
        mocker.patch("benchmark.probe._http_json", side_effect=responses)
        mocker.patch("benchmark.probe._http_probe", return_value=(-1, None, ""))
        probe = probe_engine(engine)
        assert probe["stream_usage"] is False
        assert probe["streaming"] is True
        assert probe["salt_tool_calls"] is False

    def test_probe_engine_salt_tool_calls_rejected(self, mocker):
        """MindIE-class engines 501 salt + tool messages: probe must surface
        it so the runner can pre-disable salt binding."""
        engine = EngineConfig(
            base_url="http://engine.test/v1",
            engine_root="http://engine.test",
            model="m",
            api_key="",
        )
        responses = [
            (200, {"data": [{"id": "m"}]}),
            (404, "not found"),
            (200, 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]'),
            (
                200,
                'data: {"choices":[],"usage":{"prompt_tokens":9}}\n\ndata: [DONE]',
            ),
            (501, "Not Implemented"),
        ]
        mocker.patch("benchmark.probe._http_json", side_effect=responses)
        mocker.patch("benchmark.probe._http_probe", return_value=(-1, None, ""))
        probe = probe_engine(engine)
        assert probe["salt_tool_calls"] is False

    def test_probe_engine_salt_probe_timeout_safe(self, mocker):
        engine = EngineConfig(
            base_url="http://engine.test/v1",
            engine_root="http://engine.test",
            model="m",
            api_key="",
        )
        responses = [
            (200, {"data": [{"id": "m"}]}),
            (404, "not found"),
            (200, 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]'),
            (200, 'data: {"choices":[],"usage":{"prompt_tokens":9}}\n\ndata: [DONE]'),
        ]
        real: List[Any] = []

        def flaky(url, *, headers, payload=None):
            if len(real) < 4:
                real.append(url)
                status, body = responses[len(real) - 1]
                return status, body
            raise OSError("timeout on salt probe")

        mocker.patch("benchmark.probe._http_json", side_effect=flaky)
        mocker.patch("benchmark.probe._http_probe", return_value=(-1, None, ""))
        probe = probe_engine(engine)
        assert probe["salt_tool_calls"] is False


class TestProbeHttp:
    """Drive the real HTTP layer (only urllib is mocked) so _http_json's
    JSON / raw-body / HTTPError branches are exercised, not just the
    probe_engine orchestration."""

    @staticmethod
    def _engine() -> EngineConfig:
        return EngineConfig(
            base_url="http://engine.test/v1",
            engine_root="http://engine.test",
            model="m",
            api_key="",
        )

    def test_probe_engine_real_http_happy_path(self, mocker):
        import io
        from urllib.error import HTTPError

        engine = self._engine()
        bodies = [
            b'{"data": [{"id": "m"}]}',  # models (JSON)
            b'data: {"choices":[]}\n\ndata: [DONE]',  # streaming SSE
            b'data: {"choices":[],"usage":{"prompt_tokens":9}}\n\ndata: [DONE]',
            b'data: {"choices":[]}\n\ndata: [DONE]',  # salt probe
        ]

        def fake_urlopen(request, timeout=20):
            url = request.full_url
            if url.endswith("/release_kv_cache"):
                raise HTTPError(url, 404, "Not Found", None, io.BytesIO(b"not found"))
            if url.rstrip("/").endswith(("/version", "/health")) or url.endswith("/"):
                fake = mocker.MagicMock()
                fake.status = 404
                fake.read.return_value = b"not found"
                fake.headers = {"Server": "uvicorn"}
                fake.__enter__.return_value = fake
                return fake
            fake = mocker.MagicMock()
            fake.status = 200
            fake.read.return_value = bodies.pop(0)
            fake.headers = {"Server": "uvicorn"}
            fake.__enter__.return_value = fake
            return fake

        mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)
        probe = probe_engine(engine)
        assert probe["reachable"] is True
        assert probe["model_listed"] is True
        assert probe["release_endpoint"] is False
        assert probe["streaming"] is True
        assert probe["stream_usage"] is True
        assert probe["salt_tool_calls"] is True
        # identity: /version and /health are 404 on this fake engine
        assert probe["identity"]["version_endpoint"] is False
        assert probe["identity"]["server_header"] == "uvicorn"
        assert probe["identity"]["engine_type"] == "unknown"

    def test_probe_engine_models_down_marks_unreachable(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("no route"))
        probe = probe_engine(self._engine())
        assert probe["reachable"] is False
        assert probe["error"] == "no route"

    def test_probe_engine_stream_usage_salt_failures_safe(self, mocker):
        engine = self._engine()
        bodies = [b'{"data": [{"id": "m"}]}']

        def fake_urlopen(request, timeout=20):
            url = request.full_url
            if url.endswith("/release_kv_cache"):
                fake = mocker.MagicMock()
                fake.status = 404
                fake.read.return_value = b"not found"
                fake.headers = {}
                fake.__enter__.return_value = fake
                return fake
            if url.rstrip("/").endswith(("/version", "/health")) or url.endswith("/"):
                fake = mocker.MagicMock()
                fake.status = 404
                fake.read.return_value = b"not found"
                fake.headers = {}
                fake.__enter__.return_value = fake
                return fake
            if bodies:
                fake = mocker.MagicMock()
                fake.status = 200
                fake.read.return_value = bodies.pop(0)
                fake.headers = {}
                fake.__enter__.return_value = fake
                return fake
            raise OSError("streaming endpoint down")

        mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)
        probe = probe_engine(engine)
        assert probe["streaming"] is False
        assert probe["stream_usage"] is False
        assert probe["salt_tool_calls"] is False
        assert probe["identity"]["engine_type"] == "unknown"

    def test_probe_identity_vllm_family_via_version_endpoint(self, mocker):
        """vLLM-family engines serve GET /version with a JSON body."""
        engine = self._engine()

        def fake_urlopen(request, timeout=20):
            url = request.full_url
            if url.endswith("/version"):
                fake = mocker.MagicMock()
                fake.status = 200
                fake.read.return_value = b'{"version": "0.10.1"}'
                fake.headers = {"Server": "uvicorn"}
                fake.__enter__.return_value = fake
                return fake
            if url.endswith("/health"):
                fake = mocker.MagicMock()
                fake.status = 200
                fake.read.return_value = b"OK"
                fake.headers = {"Server": "uvicorn"}
                fake.__enter__.return_value = fake
                return fake
            raise OSError("not part of the identity probe")

        mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)
        identity = probe_identity(engine)
        # a working /version without an explicit "vllm" marker lands on the
        # vLLM-family heuristic; an explicit marker upgrades it (see below)
        assert identity["engine_type"] == "vLLM-family (serves /version)"
        assert identity["version"] == "0.10.1"
        assert identity["version_endpoint"] is True
        assert identity["health"] is True

    def test_classify_engine_markers(self):
        from benchmark.probe import _classify_engine

        assert _classify_engine({"server_header": "MindIE/3.0.0"}) == "MindIE"
        assert (
            _classify_engine({"version": "0.15.0", "server_header": "vllm-ascend"})
            == "vLLM / vLLM-Ascend family"
        )
        assert _classify_engine({"version_endpoint": True}) == "vLLM-family (serves /version)"
        assert _classify_engine({"health": True}) == "OpenAI-compatible (serves /health)"
        assert _classify_engine({}) == "unknown"

    def test_probe_identity_mindie_via_server_header(self, mocker):
        """MindIE-class servers answer /version with 404; a Server header
        marker still identifies the family."""
        engine = self._engine()

        def fake_urlopen(request, timeout=20):
            fake = mocker.MagicMock()
            fake.status = 404
            fake.read.return_value = b"not found"
            fake.headers = {"Server": "MindIE/3.0.0"}
            fake.__enter__.return_value = fake
            return fake

        mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)
        identity = probe_identity(engine)
        assert identity["engine_type"] == "MindIE"
        assert identity["version"] is None
        assert identity["version_endpoint"] is False
        assert identity["server_header"] == "MindIE/3.0.0"

    def test_probe_identity_all_failures_unknown(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("no route"))
        identity = probe_identity(self._engine())
        assert identity["engine_type"] == "unknown"
        assert identity["version"] is None
        assert identity["server_header"] is None

    def test_parse_version_accepts_json_and_plain(self):
        from benchmark.probe import _parse_version

        assert _parse_version('{"version": "v0.9.1"}') == "0.9.1"
        assert _parse_version("0.10.1") == "0.10.1"
        assert _parse_version("v0.15.0") == "0.15.0"
        assert _parse_version("{broken json") is None
        assert _parse_version("not a version") is None

    def test_http_probe_http_error_returns_status_and_server(self, mocker):
        from urllib.error import HTTPError

        from benchmark.probe import _http_probe

        exc = HTTPError(
            "http://engine.test/version",
            404,
            "Not Found",
            {"Server": "MindIE/3.0.0"},
            None,
        )
        mocker.patch("urllib.request.urlopen", side_effect=exc)
        status, server, body = _http_probe("http://engine.test/version", {})
        assert status == 404
        assert server == "MindIE/3.0.0"
        assert body == ""

    def test_probe_engine_models_non_200_marks_unreachable(self, mocker):
        engine = self._engine()

        def fake_urlopen(request, timeout=20):
            fake = mocker.MagicMock()
            fake.status = 500
            fake.read.return_value = b'{"error": "boom"}'
            fake.headers = {}
            fake.__enter__.return_value = fake
            return fake

        mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)
        probe = probe_engine(engine)
        assert probe["reachable"] is False
        assert "500" in probe["error"]

    def test_is_html_detects_spa_catch_all(self):
        from benchmark.probe import _is_html

        assert _is_html('<!doctype html>\n<html lang="en">\n  <head>...')
        assert _is_html("  <html><body>app</body></html>")
        assert not _is_html('{"data": [{"id": "m"}]}')
        assert not _is_html('data: {"choices":[]}\n\ndata: [DONE]')
        assert not _is_html("OK")

    def test_release_endpoint_html_means_absent(self, mocker):
        """Gateways with an SPA catch-all answer /release_kv_cache with 200 +
        index.html; that must NOT count as a release endpoint."""
        engine = self._engine()
        responses = [
            (200, {"data": [{"id": "m"}]}),
            (200, "<!doctype html><html><body>app</body></html>"),
            (200, 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]'),
            (
                200,
                'data: {"choices":[],"usage":{"prompt_tokens":9}}\n\ndata: [DONE]',
            ),
            (200, 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]'),
        ]
        mocker.patch("benchmark.probe._http_json", side_effect=responses)
        mocker.patch("benchmark.probe._http_probe", return_value=(-1, None, ""))
        probe = probe_engine(engine)
        assert probe["release_endpoint"] is False
        assert probe["salt_tool_calls"] is True

    def test_salt_tool_calls_html_response_false(self, mocker):
        engine = self._engine()
        responses = [
            (200, {"data": [{"id": "m"}]}),
            (404, "not found"),
            (200, 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]'),
            (
                200,
                'data: {"choices":[],"usage":{"prompt_tokens":9}}\n\ndata: [DONE]',
            ),
            (200, "<!doctype html><html><body>app</body></html>"),
        ]
        mocker.patch("benchmark.probe._http_json", side_effect=responses)
        mocker.patch("benchmark.probe._http_probe", return_value=(-1, None, ""))
        probe = probe_engine(engine)
        assert probe["salt_tool_calls"] is False

    def test_identity_html_endpoints_not_recognized(self, mocker):
        """SPA catch-all 200+HTML on /version and /health must not be read
        as working endpoints (models.ascend.huawei.com behaves this way)."""
        engine = self._engine()

        def fake_urlopen(request, timeout=20):
            fake = mocker.MagicMock()
            fake.status = 200
            fake.read.return_value = b"<!doctype html><html><body>app</body></html>"
            fake.headers = {"Server": "nginx/1.21.5"}
            fake.__enter__.return_value = fake
            return fake

        mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)
        identity = probe_identity(engine)
        assert identity["version_endpoint"] is False
        assert identity["health"] is False
        assert identity["version"] is None
        assert identity["server_header"] == "nginx/1.21.5"
        assert identity["engine_type"] == "unknown"


class TestWriteReports:
    """The JSON report must serialize TaskResult objects (regression: a
    real-engine run crashed with "Object of type TaskResult is not JSON
    serializable" and no report was written)."""

    @staticmethod
    def _engine() -> EngineConfig:
        return EngineConfig(
            base_url="http://engine.test/v1",
            engine_root="http://engine.test",
            model="m",
            api_key="",
        )

    def test_write_reports_serializes_task_results(self):
        import json as jsonlib

        from benchmark.reporting import write_reports
        from benchmark.run_benchmark import TaskResult

        engine = self._engine()
        tasks = [bench_tasks.load_tasks()[0]]
        data = {
            "summaries": {
                "lc-baseline": {
                    "median": {},
                    "per_round": [],
                    "affinity_stats": {},
                    "build_error": None,
                    "engine_windows": [],
                }
            },
            "results": [
                TaskResult(
                    agent="lc-baseline",
                    task_id="rebalance-C1001",
                    category="rebalance",
                    round_idx=0,
                    turn_e2e_ms=[100.0, 200.0],
                    keyword_hits=2,
                    keywords_total=3,
                    error=None,
                    final_reply="hello",
                )
            ],
            "llm_calls": [],
        }
        args = SimpleNamespace(rounds=1, max_parallel=1, turn_timeout=10, log_file=None)
        tmp_dir = Path(__file__).resolve().parents[2] / ".report-test-tmp"
        try:
            md_path = write_reports(
                report_dir=str(tmp_dir),
                engine=engine,
                probe={"identity": {"engine_type": "unknown"}},
                tasks=tasks,
                agent_names=["lc-baseline"],
                data=data,
                args=args,
                fingerprint="fp123",
            )
            json_path = md_path.with_suffix(".json")
            payload = jsonlib.loads(json_path.read_text(encoding="utf-8"))
            assert payload["results"][0]["agent"] == "lc-baseline"
            assert payload["results"][0]["turn_e2e_ms"] == [100.0, 200.0]
            assert "rebalance-C1001" in md_path.read_text(encoding="utf-8")
        finally:
            for path in tmp_dir.glob("benchmark_report_*.json"):
                path.unlink(missing_ok=True)
            for path in tmp_dir.glob("benchmark_report_*.md"):
                path.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass


class TestAgentBuilders:
    """Agent factories must keep the two benchmark arms comparable:
    baseline requests streamed usage; affinity honours the salt_enabled
    probe decision."""

    def test_baseline_model_requests_stream_usage(self):
        from benchmark.agents import build_baseline_model

        model = build_baseline_model(model="m", base_url="http://engine.test/v1")
        assert model.stream_usage is True
        assert model.streaming is True

    def test_affinity_model_salt_enabled_passthrough(self):
        from benchmark.agents import build_affinity_model

        disabled = build_affinity_model(
            model="m", base_url="http://engine.test/v1", salt_enabled=False
        )
        assert disabled.salt_enabled is False
        default = build_affinity_model(model="m", base_url="http://engine.test/v1")
        assert default.salt_enabled is True

    def test_affinity_model_release_enabled_passthrough(self):
        from benchmark.agents import build_affinity_model

        on = build_affinity_model(
            model="m", base_url="http://engine.test/v1", release_enabled=True
        )
        assert on.release_endpoint == "/release_kv_cache"
        off = build_affinity_model(
            model="m", base_url="http://engine.test/v1", release_enabled=False
        )
        assert off.release_endpoint == ""


class TestBuildResilience:
    def _engine(self) -> EngineConfig:
        return EngineConfig(
            base_url="http://engine.test/v1",
            engine_root="http://engine.test",
            model="m",
            api_key="",
        )

    def test_build_error_captured_and_phase_skipped(self, mocker):
        engine = self._engine()
        mocker.patch(
            "benchmark.oj_adapter.build_openjiuwen_agent",
            side_effect=RuntimeError("openjiuwen API drift"),
        )

        async def build() -> None:
            specs = await build_agents(["oj-baseline"], engine, 0, True)
            assert specs[0].agent is None
            assert "RuntimeError" in specs[0].build_error
            args = SimpleNamespace(npu_cmd="", max_parallel=1, turn_timeout=10)
            results, window = await run_agent_phase(specs[0], [], 0, args, "")
            assert results == []
            assert window.hit_rate_delta is None

        asyncio.run(build())

    def test_affinity_stats_accumulate_across_rounds(self, mocker):
        """Counters reset on each round's fresh model; the summary must sum
        them into a run-total instead of keeping only the last round."""
        engine = self._engine()
        args = SimpleNamespace(rounds=2, metrics_url=None, npu_cmd="")
        seen_salt_enabled: List[Optional[bool]] = []

        class FakeModel:
            affinity_stats = {"affinity_requests": 2, "salt_bound_requests": 1}

        class FakeRecorder:
            calls = []

        def lc_build(names, _engine, round_idx, release_enabled, salt_enabled):
            del _engine, round_idx, release_enabled
            seen_salt_enabled.append(salt_enabled)
            return [
                SimpleNamespace(
                    name=name,
                    kind="lc",
                    agent=object(),
                    recorder=FakeRecorder(),
                    collector=None,
                    affinity_source=FakeModel() if name == "lc-affinity" else None,
                    build_error=None,
                )
                for name in names
            ]

        def skip_phase(spec, tasks, round_idx, args_, metrics_url):
            del spec, tasks, round_idx, args_, metrics_url
            return [], SimpleNamespace(
                hit_rate_delta=None, cache_usage_peak=None, npu_samples=[]
            )

        mocker.patch("benchmark.run_benchmark.build_agents", side_effect=lc_build)
        mocker.patch("benchmark.run_benchmark.run_agent_phase", side_effect=skip_phase)
        data = asyncio.run(
            run_benchmark(
                engine,
                [],
                ["lc-affinity"],
                args,
                release_enabled=False,
                salt_enabled=False,
            )
        )
        stats = data["summaries"]["lc-affinity"]["affinity_stats"]
        assert stats["affinity_requests"] == 4  # 2 rounds x 2
        assert stats["salt_bound_requests"] == 2
        # the engine-probe decision reaches every round's model build
        assert seen_salt_enabled == [False, False]

    def test_build_error_surfaces_in_summary(self, mocker):
        engine = self._engine()
        args = SimpleNamespace(rounds=1, metrics_url=None, npu_cmd="")

        def bad_build(names, _engine, round_idx, release_enabled, salt_enabled):
            del _engine, round_idx, release_enabled, salt_enabled
            return [
                SimpleNamespace(
                    name=names[0],
                    kind="oj",
                    agent=None,
                    recorder=None,
                    collector=SimpleNamespace(records=[], drop_warmup=lambda: None),
                    affinity_source=None,
                    build_error="ImportError: no openjiuwen",
                )
            ]

        def skip_phase(spec, tasks, round_idx, args_, metrics_url):
            del spec, tasks, round_idx, args_, metrics_url
            return [], SimpleNamespace(
                hit_rate_delta=None, cache_usage_peak=None, npu_samples=[]
            )

        mocker.patch("benchmark.run_benchmark.build_agents", side_effect=bad_build)
        mocker.patch("benchmark.run_benchmark.run_agent_phase", side_effect=skip_phase)
        data = asyncio.run(
            run_benchmark(engine, [], ["oj-baseline"], args, release_enabled=False)
        )
        assert data["summaries"]["oj-baseline"]["build_error"] == "ImportError: no openjiuwen"


class TestRunLogging:
    def test_configure_logging_writes_file(self):
        # tmp_path lives under the OS temp dir, which the dev sandbox denies;
        # use a workspace-local dir and clean it up afterwards.
        import benchmark.run_benchmark as runner

        tmp_dir = Path(__file__).resolve().parents[2] / ".log-test-tmp"
        tmp_dir.mkdir(exist_ok=True)
        log_file = tmp_dir / "bench.log"
        try:
            configure_logging("INFO", str(log_file))
            runner.logger.info("hello-bench-mark")
            assert "hello-bench-mark" in log_file.read_text(encoding="utf-8")
        finally:
            configure_logging("INFO")  # closes the file handler
            log_file.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

    def test_recorder_logs_call_line_with_salt_flag(self, caplog):
        recorder = TTFTRecorder("lc-affinity", 0)
        run_id = uuid4()
        recorder.on_llm_start(
            {},
            ["hi"],
            run_id=run_id,
            metadata={
                "session_id": "bench-s1",
                "bench_agent": "lc-affinity",
                "bench_task": "t1",
            },
        )
        recorder.on_llm_new_token("x", run_id=run_id)
        response = SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=SimpleNamespace(
                            usage_metadata={
                                "input_tokens": 7,
                                "output_tokens": 3,
                                "total_tokens": 10,
                                "input_token_details": {"cache_read": 5},
                            }
                        )
                    )
                ]
            ]
        )
        with caplog.at_level(logging.INFO, logger="benchmark.run_benchmark"):
            recorder.on_llm_end(response, run_id=run_id)
        messages = [r.message for r in caplog.records if "[llm]" in r.message]
        assert messages and "salt=yes" in messages[0]
        assert "prompt=7 comp=3 cached=5" in messages[0]
        assert "t1" in messages[0]

    def test_recorder_logs_salt_no_without_session(self, caplog):
        recorder = TTFTRecorder("lc-baseline", 0)
        run_id = uuid4()
        recorder.on_llm_start({}, ["hi"], run_id=run_id, metadata={})
        recorder.on_llm_new_token("x", run_id=run_id)
        response = SimpleNamespace(
            generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=None))]]
        )
        with caplog.at_level(logging.INFO, logger="benchmark.run_benchmark"):
            recorder.on_llm_end(response, run_id=run_id)
        messages = [r.message for r in caplog.records if "[llm]" in r.message]
        assert messages and "salt=no" in messages[0]
