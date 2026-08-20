"""openJiuwen (agent-core) agent pair for the 4-agent benchmark matrix.

Builds two ``ReActAgent`` advisors over the same financial tools and a
semantically equivalent prompt as the deepagents pair:

- ``oj-baseline``: provider ``OpenAI``, ``enable_kv_cache_release=False``
- ``oj-affinity``: provider ``InferenceAffinity`` (salt binding + context
  engine KV release), ``enable_kv_cache_release=True``

Per-LLM-call metrics come from ``BEFORE_MODEL_CALL`` / ``AFTER_MODEL_CALL``
callbacks: wall E2E plus ``usage_metadata`` token counts (prompt / decode /
cache-read when exposed). agent-core callbacks carry no token-level events,
so TTFT/TPOT render as N/A on the lab sheet for openJiuwen agents.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List

from benchmark.metrics import CallMetrics, usage_field

logger = logging.getLogger(__name__)

OJ_ADVISOR_INSTRUCTIONS = """你是一名严谨的中文投资顾问智能体，服务于零售客户。

工作准则：
1. 涉及客户持仓、基金档案、组合风险评分时，必须先调用对应工具核实，禁止凭空编造数字。
2. 回答使用中文，结论清晰；给出配置建议时说明理由与风险。
3. 客户更正此前的需求时，以更正后的内容为准重新推理。
4. 不提供保证收益的表述，必要时提示市场风险。"""


class OJCallCollector:
    """agent-core model-call hooks -> per-call metric records."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.records: List[CallMetrics] = []
        self._starts: List[float] = []
        self._task_id = "warmup"
        self._round_idx = 0

    def bind_task(self, task_id: str, round_idx: int) -> None:
        """Tag subsequent records with the running task/round."""
        self._task_id = task_id
        self._round_idx = round_idx

    def drop_warmup(self) -> None:
        """Discard records collected before the first real task."""
        count = sum(1 for r in self.records if r.task_id == "warmup")
        if count:
            self.records = [r for r in self.records if r.task_id != "warmup"]

    async def before_model_call(self, ctx: Any) -> None:
        """Record the model-call start (strictly nested per ReAct loop)."""
        self._starts.append(time.perf_counter())

    async def after_model_call(self, ctx: Any) -> None:
        """Compute E2E and extract usage from the response message."""
        if not self._starts:
            return
        e2e_ms = round((time.perf_counter() - self._starts.pop()) * 1000.0, 1)
        response = getattr(getattr(ctx, "inputs", None), "response", None)
        usage = getattr(response, "usage_metadata", None)
        details = (
            usage.get("input_token_details")
            if isinstance(usage, dict)
            else getattr(usage, "input_token_details", None)
        )
        self.records.append(
            CallMetrics(
                agent=self.agent_name,
                task_id=self._task_id,
                round_idx=self._round_idx,
                ttft_ms=None,
                e2e_ms=e2e_ms,
                prompt_tokens=usage_field(usage, "input_tokens"),
                completion_tokens=usage_field(usage, "output_tokens"),
                cached_tokens=usage_field(details, "cache_read"),
            )
        )
        logger.info(
            "[llm] r%s %s %s ttft=n/a e2e=%.0fms prompt=%s comp=%s cached=%s",
            self._round_idx,
            self.agent_name,
            self._task_id,
            e2e_ms,
            usage_field(usage, "input_tokens"),
            usage_field(usage, "output_tokens"),
            usage_field(details, "cache_read"),
        )


def build_oj_tools() -> List[Any]:
    """The benchmark tools wrapped with openJiuwen's ``@tool`` decorator."""
    import json

    from openjiuwen.core.foundation.tool import tool as oj_tool  # pylint: disable=import-error  # optional proprietary dep

    from benchmark import tasks

    @oj_tool(
        name="get_customer_holdings",
        description="查询客户当前持仓明细（资产、类型、市值、风险等级）。",
    )
    def holdings(customer_id: str) -> str:
        """customer_id: 客户编号，例如 C1001。"""
        return json.dumps(tasks.holdings_of(customer_id), ensure_ascii=False)

    @oj_tool(
        name="get_fund_profile",
        description="查询基金档案（类型、费率、近一年收益与最大回撤）。",
    )
    def fund_profile(fund_code: str) -> str:
        """fund_code: 基金代码，可选 F001/F002/F003/F004。"""
        return json.dumps(tasks.fund_profile_of(fund_code), ensure_ascii=False)

    @oj_tool(
        name="compute_portfolio_risk",
        description="按股/债/现金占比计算组合风险评分（0-100，越高越激进）。",
    )
    def portfolio_risk(equity_pct: float, bond_pct: float, cash_pct: float) -> str:
        """equity_pct/bond_pct/cash_pct: 三类资产的百分比数字，合计 100。"""
        return json.dumps(
            tasks.portfolio_risk(equity_pct, bond_pct, cash_pct), ensure_ascii=False
        )

    return [holdings, fund_profile, portfolio_risk]


async def build_openjiuwen_agent(
    *,
    affinity: bool,
    model: str,
    base_url: str,
    api_key: str,
    collector: OJCallCollector,
    max_iterations: int = 60,
) -> Any:
    """Build one openJiuwen ReActAgent advisor (async: callback registry)."""
    from openjiuwen.core.single_agent import (  # pylint: disable=import-error  # optional proprietary dep
        AgentCard,
        ReActAgent,
        ReActAgentConfig,
    )
    from openjiuwen.core.single_agent.rail.base import AgentCallbackEvent  # pylint: disable=import-error  # optional proprietary dep

    label = "affinity" if affinity else "baseline"
    agent = ReActAgent(
        card=AgentCard(
            name=f"bench-oj-{label}",
            description="financial advisor benchmark agent (openJiuwen)",
        )
    )
    config = (
        ReActAgentConfig()
        .configure_model_client(
            provider="InferenceAffinity" if affinity else "OpenAI",
            api_key=api_key,
            api_base=base_url,
            model_name=model,
        )
        .configure_prompt_template(
            [{"role": "system", "content": OJ_ADVISOR_INSTRUCTIONS}]
        )
        .configure_max_iterations(max_iterations)
    )
    config.configure_context_engine(
        max_context_message_num=None,
        default_window_round_num=None,
        enable_kv_cache_release=affinity,
    )
    agent.configure(config)
    for oj_tool_instance in build_oj_tools():
        agent.add_tool(oj_tool_instance)
    await agent.register_callback(
        AgentCallbackEvent.BEFORE_MODEL_CALL, collector.before_model_call
    )
    await agent.register_callback(
        AgentCallbackEvent.AFTER_MODEL_CALL, collector.after_model_call
    )
    return agent
