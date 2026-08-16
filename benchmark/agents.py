"""Agent factories: baseline (native ChatOpenAI) vs affinity plugin.

Both agents are built with ``deepagents.create_deep_agent`` over the same
tool set and the same advisor instructions — the ONLY variable is the chat
model, which keeps the benchmark single-variable by construction.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from benchmark.tasks import build_tools

ADVISOR_INSTRUCTIONS = """你是一名严谨的中文投资顾问智能体，服务于零售客户。

工作准则：
1. 涉及客户持仓、基金档案、组合风险评分时，必须先调用对应工具核实，禁止凭空编造数字。
2. 回答使用中文，结论清晰；给出配置建议时说明理由与风险。
3. 客户更正此前的需求时，以更正后的内容为准重新推理。
4. 不提供保证收益的表述，必要时提示市场风险。"""


def build_baseline_model(
    *, model: str, base_url: str, api_key: str = "EMPTY", timeout: float = 120.0
) -> Any:
    """Native LangChain OpenAI chat model against the same engine."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        timeout=timeout,
    )


def build_affinity_model(
    *, model: str, base_url: str, api_key: str = "EMPTY", timeout: float = 120.0
) -> Any:
    """Affinity chat model (salt binding + prefix diff + partial release).

    The per-task cache salt is delivered at invoke time via run metadata
    (``session_id``), which the affinity model resolves per call.
    """
    from langchain_ascend import AscendAffinityChatModel

    return AscendAffinityChatModel(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        timeout=timeout,
    )


def build_agent(llm: Any) -> Any:
    """Wrap either model into an identical deepagents advisor."""
    return create_deep_agent(
        model=llm, tools=build_tools(), system_prompt=ADVISOR_INSTRUCTIONS
    )
