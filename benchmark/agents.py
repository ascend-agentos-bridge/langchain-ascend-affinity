"""Agent factories: baseline (native ChatOpenAI) vs affinity plugin.

Both agents are built with ``deepagents.create_deep_agent`` over the same
tool set and the same advisor instructions — the ONLY variable is the chat
model, which keeps the benchmark single-variable by construction.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent  # pylint: disable=import-error  # benchmark/requirements.txt

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
    """Native LangChain OpenAI chat model against the same engine.

    ``streaming=True`` makes invoke() aggregate an SSE stream internally,
    emitting on_llm_new_token callbacks so client-side TTFT is measurable
    (identical sampling on both sides of the pair). ``stream_usage=True``
    requests ``stream_options.include_usage`` so token metrics (prefill /
    decode / TPOT / KV hit rate) are comparable with the affinity side —
    without it, custom-base_url ChatOpenAI silently drops usage by default.
    """
    from langchain_openai import ChatOpenAI  # pylint: disable=import-error  # benchmark/requirements.txt

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        timeout=timeout,
        streaming=True,
        stream_usage=True,
    )


def build_affinity_model(
    *,
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    release_enabled: bool = True,
    salt_enabled: bool = True,
) -> Any:
    """Affinity chat model (salt binding + prefix diff + partial release).

    The per-task cache salt is delivered at invoke time via run metadata
    (``session_id``), which the affinity model resolves per call.

    ``release_enabled=False`` (engine probe found no ``/release_kv_cache``)
    disables release requests while keeping salt binding and prefix tracking,
    so engines without the agent-core endpoint don't collect 404 noise.

    ``salt_enabled=False`` (engine probe ``salt_tool_calls`` rejected
    salt-bound tool-call requests) keeps the pipeline counters but never
    injects ``cache_sharing``/``cache_salt``, so tool-calling agents run as a
    plain OpenAI client instead of failing with engine HTTP 501s.
    """
    from langchain_ascend import AscendAffinityChatModel

    return AscendAffinityChatModel(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        timeout=timeout,
        streaming=True,
        release_endpoint="/release_kv_cache" if release_enabled else "",
        salt_enabled=salt_enabled,
    )


def build_agent(llm: Any) -> Any:
    """Wrap either model into an identical deepagents advisor."""
    return create_deep_agent(
        model=llm, tools=build_tools(), system_prompt=ADVISOR_INSTRUCTIONS
    )
