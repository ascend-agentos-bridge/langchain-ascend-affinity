"""Financial advisor task set for the affinity benchmark.

Eight multi-turn advisor-customer dialogues across four categories. Half of
the tasks include one client-side history rewrite (the user revises an
earlier message and the stale AI reply is dropped), which is exactly the
prefix-divergence pattern the affinity model must detect and release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.tools import tool

_HOLDINGS: Dict[str, Dict[str, Any]] = {
    "C1001": {
        "customer_id": "C1001",
        "customer_name": "张伟",
        "risk_profile": "平衡型",
        "total_assets": 1000000,
        "positions": [
            {"asset": "沪深300指数基金", "type": "权益", "value": 600000},
            {"asset": "中债总指数基金", "type": "债券", "value": 300000},
            {"asset": "现金管理货币基金", "type": "现金", "value": 100000},
        ],
    },
    "C1002": {
        "customer_id": "C1002",
        "customer_name": "李娜",
        "risk_profile": "稳健型",
        "total_assets": 500000,
        "positions": [
            {"asset": "中证红利低波ETF", "type": "权益", "value": 150000},
            {"asset": "利率债基金", "type": "债券", "value": 250000},
            {"asset": "同业存单基金", "type": "现金", "value": 100000},
        ],
    },
    "C1003": {
        "customer_id": "C1003",
        "customer_name": "王强",
        "risk_profile": "进取型",
        "total_assets": 800000,
        "positions": [
            {"asset": "科创50ETF", "type": "权益", "value": 560000},
            {"asset": "可转债基金", "type": "债券", "value": 160000},
            {"asset": "货币基金", "type": "现金", "value": 80000},
        ],
    },
    "C1004": {
        "customer_id": "C1004",
        "customer_name": "赵敏",
        "risk_profile": "保守型",
        "total_assets": 300000,
        "positions": [
            {"asset": "短债基金", "type": "债券", "value": 210000},
            {"asset": "货币基金", "type": "现金", "value": 90000},
        ],
    },
}

_FUND_PROFILES: Dict[str, Dict[str, Any]] = {
    "F001": {
        "fund_code": "F001",
        "fund_name": "华安优选成长混合",
        "fund_type": "偏股混合",
        "management_fee": "1.50%",
        "custody_fee": "0.25%",
        "max_drawdown_1y": "-18.6%",
        "return_1y": "12.4%",
    },
    "F002": {
        "fund_code": "F002",
        "fund_name": "易方达安心回报债券",
        "fund_type": "债券型",
        "management_fee": "0.70%",
        "custody_fee": "0.10%",
        "max_drawdown_1y": "-3.2%",
        "return_1y": "4.8%",
    },
    "F003": {
        "fund_code": "F003",
        "fund_name": "天弘中证500指数A",
        "fund_type": "指数型",
        "management_fee": "0.50%",
        "custody_fee": "0.10%",
        "max_drawdown_1y": "-22.1%",
        "return_1y": "9.1%",
    },
    "F004": {
        "fund_code": "F004",
        "fund_name": "招商产业精选股票",
        "fund_type": "股票型",
        "management_fee": "1.20%",
        "custody_fee": "0.20%",
        "max_drawdown_1y": "-25.8%",
        "return_1y": "15.7%",
    },
}


@tool
def get_customer_holdings(customer_id: str) -> dict:
    """查询客户当前持仓明细（资产、类型、市值、风险等级）。

    Args:
        customer_id: 客户编号，例如 C1001。
    """
    holdings = _HOLDINGS.get(customer_id)
    if holdings is None:
        return {"error": f"未找到客户 {customer_id}"}
    return holdings


@tool
def get_fund_profile(fund_code: str) -> dict:
    """查询基金档案（类型、管理费、托管费、近一年收益与最大回撤）。

    Args:
        fund_code: 基金代码，可选 F001/F002/F003/F004。
    """
    profile = _FUND_PROFILES.get(fund_code)
    if profile is None:
        return {"error": f"未找到基金 {fund_code}"}
    return profile


@tool
def compute_portfolio_risk(equity_pct: float, bond_pct: float, cash_pct: float) -> dict:
    """按股/债/现金占比计算组合风险评分（0-100，越高越激进）。

    Args:
        equity_pct: 权益占比（百分比数字）。
        bond_pct: 债券占比（百分比数字）。
        cash_pct: 现金占比（百分比数字）。
    """
    total = equity_pct + bond_pct + cash_pct
    if abs(total - 100.0) > 0.5:
        return {"error": "股/债/现金占比之和必须等于 100"}
    score = min(100.0, equity_pct * 0.9 + bond_pct * 0.3 + cash_pct * 0.05)
    if score >= 60:
        rating = "激进"
    elif score >= 35:
        rating = "平衡"
    else:
        rating = "稳健"
    return {"risk_score": round(score, 1), "rating": rating}


@dataclass(frozen=True)
class FinanceTask:
    """One advisor-customer dialogue.

    ``edit_replaces_turn`` (optional ``j``) marks that before turn ``j + 1``
    the client rewrites the user message of turn ``j`` to
    ``edit_replacement`` and drops the stale AI reply after it — the
    history-rewrite pattern that must trigger a partial KV release.
    """

    task_id: str
    category: str
    customer_id: str
    turns: List[str]
    expected_keywords: List[str] = field(default_factory=list)
    edit_replaces_turn: int = -1
    edit_replacement: str = ""


def build_tools() -> List[Any]:
    """The deterministic in-memory tool set shared by both agents."""
    return [get_customer_holdings, get_fund_profile, compute_portfolio_risk]


def load_tasks() -> List[FinanceTask]:
    """The eight benchmark tasks (stable ids, no engine access needed)."""
    return [
        FinanceTask(
            task_id="rebalance-C1001",
            category="rebalance",
            customer_id="C1001",
            turns=[
                "我是C1001，请先查一下我目前的持仓结构，告诉我股债现金各占多少比例。",
                "市场波动加大，我想把权益仓位从目前水平降到40%，债券提高到50%，请给出具体调仓步骤。",
                "请用风险计算工具算一下调整后的组合风险评分，并说明与调整前的差别。",
                "最后请把本次调仓方案整理成一份清单，注明卖出和买入的方向。",
            ],
            expected_keywords=["40%", "债券", "调仓"],
            edit_replaces_turn=1,
            edit_replacement="市场波动加大，我想把权益仓位降到35%，债券提高到55%（不是40/50），请给出具体调仓步骤。",
        ),
        FinanceTask(
            task_id="rebalance-C1003",
            category="rebalance",
            customer_id="C1003",
            turns=[
                "我是C1003，帮我看看当前持仓，我担心回撤太大。",
                "我想把权益占比降到50%以内，现金留足15%，其余配债券，给出调仓建议。",
                "调仓后组合风险评分是多少？适合我的进取型定位吗？",
            ],
            expected_keywords=["权益", "债券", "风险"],
            edit_replaces_turn=-1,
            edit_replacement="",
        ),
        FinanceTask(
            task_id="risk-C1002",
            category="risk",
            customer_id="C1002",
            turns=[
                "我是C1002，帮我做一次风险测评，先查我的持仓。",
                "我今年38岁，投资期限10年，能接受本金最大亏损10%。",
                "重新说明一下：我能接受的最大亏损其实只有5%，投资期限改为5年，请重新评估我的风险等级。",
                "根据新的风险等级，我现在的持仓需要怎么调整？",
            ],
            expected_keywords=["稳健", "债券", "调整"],
            edit_replaces_turn=2,
            edit_replacement="重新说明一下：我能接受的最大亏损只有5%，投资期限5年，重新评估。",
        ),
        FinanceTask(
            task_id="risk-C1004",
            category="risk",
            customer_id="C1004",
            turns=[
                "我是C1004，请查询我的持仓并给出风险评分。",
                "我完全不能接受亏损，这笔钱两年内要用于购房首付。",
                "那按我的情况，权益类资产应该保留多少？",
            ],
            expected_keywords=["保守", "现金", "债券"],
            edit_replaces_turn=-1,
            edit_replacement="",
        ),
        FinanceTask(
            task_id="compare-F001-F002",
            category="compare",
            customer_id="C1001",
            turns=[
                "请帮我对比基金F001和F002，查一下两边的费率、近一年收益和最大回撤。",
                "我更在意回撤控制而不是收益，这两只更适合哪只？说明理由。",
                "更正一下：我其实更在意长期收益，能承受20%以内的回撤，结论会变吗？",
                "给出最终的配置建议比例。",
            ],
            expected_keywords=["回撤", "收益", "建议"],
            edit_replaces_turn=2,
            edit_replacement="更正一下：我更在意长期收益，能承受20%以内回撤，重新给结论。",
        ),
        FinanceTask(
            task_id="compare-F003-F004",
            category="compare",
            customer_id="C1003",
            turns=[
                "对比一下F003和F004这两只基金的费率与风险收益特征。",
                "我是进取型投资者，二选一你推荐哪只？为什么？",
                "如果各配一半，组合风险评分怎么算？",
            ],
            expected_keywords=["推荐", "风险", "费率"],
            edit_replaces_turn=-1,
            edit_replacement="",
        ),
        FinanceTask(
            task_id="market-rate-qa",
            category="market",
            customer_id="C1002",
            turns=[
                "最近市场利率下行对我的债券基金有什么影响？",
                "利率下行环境下，我该增持长久期利率债还是短债？",
                "这个判断和我稳健型的风险定位冲突吗？",
                "总结一下你的三条核心建议。",
            ],
            expected_keywords=["利率", "债券", "建议"],
            edit_replaces_turn=-1,
            edit_replacement="",
        ),
        FinanceTask(
            task_id="market-policy-qa",
            category="market",
            customer_id="C1004",
            turns=[
                "如果出台更积极的资本市场政策，我的保守型组合需要调整吗？",
                "政策利好通常先传导到哪类资产？",
                "我最多愿意把10%的资产转为权益，可行吗？",
            ],
            expected_keywords=["权益", "政策", "保守"],
            edit_replaces_turn=-1,
            edit_replacement="",
        ),
    ]
