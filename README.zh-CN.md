# langchain-ascend-affinity

[English](README.md) | [简体中文](README.zh-CN.md)

将 openJiuwen agent-core 的昇腾算力亲和机制移植到 LangChain 生态。
换一个模型对象，用 **langchain**、**langgraph** 或 **deepagents** 构建的智能体
即可获得对推理引擎的缓存亲和能力——无需回调、无需挂载 handler。

## 工作原理

`AscendAffinityChatModel` 在每次 LLM 调用中做三件事：

1. **salt 绑定** —— 每个请求携带 `cache_sharing: true` +
   `cache_salt: <session_id>`，每个会话获得独立的 KV-Cache 桶。
2. **前缀差异调度** —— 发出的 `(messages, tools)` 窗口与会话上一个窗口
   逐条对比；纯追加（正常智能体循环）保持前缀缓存命中，历史被改写时
   精确定位第一个分歧下标。
3. **部分释放** —— 发生分歧时，模型把失效窗口 POST 到
   `{engine}/release_kv_cache`，引擎只丢弃失效的 KV 块。释放失败仅告警，
   绝不阻断生成。

## 安装

```bash
pip install langchain-ascend-affinity
# 依赖 langchain-core >=1.0；无其他运行时依赖
```

## 快速开始（LangChain 1.x）

**所有会话共用一个模型实例**——session 随每次 `invoke` 调用传递
（运行元数据 → `cache_salt`），一个智能体即可服务多用户：

```python
from langchain_ascend import AscendAffinityChatModel

llm = AscendAffinityChatModel(
    base_url="http://127.0.0.1:8000/v1",  # MindIE / vLLM-Ascend 服务地址
    model="Qwen3-32B",
)
```

### langchain

```python
from langchain.agents import create_agent

agent = create_agent(
    llm.bind_tools([lookup_quote, calculator]),
    system_prompt="你是一名理财顾问。",
)
result = agent.invoke(
    {"messages": [("user", "帮我规划一笔3年期基金定投")]},
    config={"metadata": {"session_id": "user-123"}},  # 每会话独立的 salt
)
```

<details><summary>langgraph</summary>

```python
from langgraph.graph import END, START, MessagesState, StateGraph

def advise(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("advise", advise)
graph.add_edge(START, "advise")
graph.add_edge("advise", END)
app = graph.compile()
app.invoke(
    {"messages": [("user", "查询 SH000001 并给出建议")]},
    config={"metadata": {"session_id": "user-123"}},
)
```

</details>

<details><summary>deepagents</summary>

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=llm,
    tools=[lookup_quote, calculator],
    system_prompt="你是一名理财顾问。",
)
result = agent.invoke(
    {"messages": [("user", "调研指数基金并起草方案")]},
    config={"metadata": {"session_id": "user-123"}},
)
```

deepagents 运行中的上下文编辑（摘要会改写历史消息）正是前缀差异调度器的
用武之地：每次改写精确触发一次部分释放。

</details>

## 配置

| 字段 | 默认值 | 说明 |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` | 引擎地址；支持裸 origin、带 `/v1` 基址或完整 `/chat/completions` 端点三种形态 |
| `model` | `ascend-chat` | 通告给引擎的模型名 |
| `session_id` | `None` | 单会话应用的兜底 salt |
| `enable_affinity` | `True` | `False` = 普通 OpenAI 兼容客户端 |
| `release_endpoint` | `/release_kv_cache` | 引擎根路径下的部分释放端点；置 `""` 禁用 |
| `timeout` / `api_key` / `temperature` / `top_p` / `max_tokens` | — | 常规请求选项（匿名引擎设 `api_key=""`） |

每次调用的会话解析顺序：逐调用 / `bind(session_id=...)` 参数 → 运行
元数据（`config={"metadata": {"session_id": ...}}`，多会话服务推荐，
可穿透智能体/图层层传递）→ 构造参数 `session_id`（兜底）。未绑定会话时
模型不发送任何亲和字段，保持普通 OpenAI 客户端。

高级选项：`enable_agent_hint` / `idle_evict_after_seconds` 可选启用
agent_hint 生命周期协议（身份字段 + evict / offload / prefetch）——
详见 [COMPATIBILITY.zh-CN.md](COMPATIBILITY.zh-CN.md)。

## 引擎支持

可对接任何 OpenAI 兼容引擎；亲和收益取决于引擎是否消费亲和字段
（`cache_salt`、`/release_kv_cache`）。降级始终安全：引擎忽略未知字段时
请求退化为普通 OpenAI 调用，释放失败仅产生非致命告警。

哪个引擎版本支持什么（MindIE、vLLM-Ascend）、完整接口契约、与
openjiuwen agent-core 的协议兼容、以及 LLM 网关透传行为，**集中维护**在
[COMPATIBILITY.zh-CN.md](COMPATIBILITY.zh-CN.md)。引擎的实际行为由基准
测试显式暴露（release 端点探测、`affinity_stats`）——见
[benchmark/PRINCIPLES.md](benchmark/PRINCIPLES.md)。

## 验证

真实亲和收益需在真实昇腾引擎上度量（MindIE / vLLM-Ascend 前缀缓存统计）。
无硬件环境下，单元测试覆盖完整协议契约（salt 注入、前缀差异、释放调度与
传输）：

```bash
python -m pytest tests/unit_tests
```

## 基准测试

面向真实引擎的单变量基准测试位于 [benchmark/](benchmark/)：两个完全相同的
`deepagents` 投顾智能体跑同一套金融任务集——baseline 用原生 `ChatOpenAI`，
实验组用 `AscendAffinityChatModel`。度量真实 TTFT（每次 LLM 调用首 token）
与亲和行为（salt 绑定、部分释放）：

```bash
python benchmark/run_benchmark.py --setup \
  --engine-url http://<引擎地址>:<端口>/v1 --model <模型名> \
  --api-key <API密钥>
```

配置与报告解读见 [benchmark/README.zh-CN.md](benchmark/README.zh-CN.md)。

## 开发

```bash
python -m pytest tests/unit_tests   # 覆盖率门槛：90%
python scripts/quality_gate.py      # pylint 10.00/10 + 单元测试
```

规格驱动的设计文档见
[openspec/](openspec/projects/affinity-core/proposals/openjiuwen-affinity-port/)。
