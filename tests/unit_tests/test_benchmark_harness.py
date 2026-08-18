"""Benchmark harness unit tests: metrics engine, tasks, oj adapter glue."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
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
    verdict_text,
)
from benchmark.oj_adapter import OJCallCollector
from benchmark.run_benchmark import (
    LlmCallRecord,
    TTFTRecorder,
    records_to_metrics,
    rotate,
    resolve_agents,
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
