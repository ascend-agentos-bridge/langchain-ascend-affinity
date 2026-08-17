# langchain-ascend-affinity

[English](README.md) | [简体中文](README.zh-CN.md)

将 openJiuwen agent-core 的昇腾算力亲和机制完整移植到 LangChain 生态。
换一个模型对象，用 **langchain**、**langgraph** 或 **deepagents** 构建的智能体
即可获得对推理引擎的缓存亲和能力——无需回调、无需挂载 handler。

## 工作原理

`AscendAffinityChatModel` 在每次 LLM 调用中做的事，与 agent-core 的
`InferenceAffinityModelClient` + `KVCacheManager` 完全一致：

1. **salt 绑定** —— 每个 `/v1/chat/completions` 请求携带
   `cache_sharing: true` + `cache_salt: <session_id>`（对齐 vLLM /
   vLLM-Ascend 原生前缀缓存 salt），每个会话获得独立的 KV-Cache 桶，
   不再互相踩踏共享缓存。
2. **前缀差异调度** —— 发出的 `(messages, tools)` 窗口与会话上一个窗口逐条
   对比。纯追加（正常智能体循环）保持前缀缓存命中；历史被改写
   （`trim_messages`、摘要压缩、deepagents 上下文编辑）时精确定位第一个
   分歧下标。
3. **部分释放** —— 发生分歧时，模型把上一个窗口 POST 到
   `{engine}/release_kv_cache`，携带 `messages_released_index` /
   `tools_released_index`（与 agent-core 逐字节兼容），引擎只丢弃失效的
   KV 块、保留有效前缀常驻。释放失败仅告警，绝不阻断生成。

## 安装

```bash
pip install langchain-ascend-affinity
# 依赖 langchain-core >=0.3（推荐 1.x）；无其他运行时依赖
```

## 快速开始（LangChain 1.x）

公共部分——**所有会话共用一个模型实例**；session 随每次 `invoke` 调用
传递（运行元数据 → `cache_salt`），一个智能体即可服务多用户，无需
按会话维护实例：

```python
from langchain_ascend import AscendAffinityChatModel

llm = AscendAffinityChatModel(
    base_url="http://127.0.0.1:8000/v1",  # MindIE / vLLM-Ascend 服务地址
    model="Qwen3-32B",
)

config = {"metadata": {"session_id": "user-123"}}  # 每会话独立的 salt
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
    config={"metadata": {"session_id": "user-123"}},
)
```

### langgraph

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

### deepagents

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
用武之地：每次改写精确触发一次部分释放，有效前缀保持缓存常驻。

## 配置

| 字段 | 默认值 | 说明 |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` | OpenAI 兼容引擎地址 |
| `model` | `ascend-chat` | 通告给引擎的模型名 |
| `session_id` | `None` | 未逐调用绑定会话时的兜底 salt（仅单会话应用建议使用） |
| `enable_affinity` | `True` | `False` = 普通 OpenAI 兼容客户端 |
| `release_endpoint` | `/release_kv_cache` | 部分释放路径；置 `""` 禁用 |
| `timeout` / `api_key` / `temperature` / `top_p` / `max_tokens` | — | 常规请求选项 |

每次调用的会话解析顺序：逐调用 / `bind(session_id=...)` 参数 → 运行
元数据（`config={"metadata": {"session_id": ...}}`，多会话服务推荐，
可穿透智能体/图层层传递）→ 构造参数 `session_id`（兜底）。

## 引擎接口要求

`AscendAffinityChatModel` 可对接任何 OpenAI 兼容引擎，但亲和收益取决于
以下接口契约。

**必需基线（任何引擎都能跑）**

| 接口 | 要求 |
|---|---|
| `POST {base_url}/chat/completions` | OpenAI 兼容 `messages`；工具调用智能体需支持 `tools`；测 TTFT 需支持 `stream`（SSE） |
| 认证 | `Authorization: Bearer <api_key>` |

**亲和契约（决定有没有收益）**

| 本库发送什么 | 引擎应做什么 | 缺失/被忽略时的行为 |
|---|---|---|
| 请求体携带 `cache_sharing: true` | 允许该会话加入前缀缓存共享 | 无收益，无害 |
| 请求体携带 `cache_salt: <session_id>` | 按 vLLM-Ascend 前缀缓存 salt 语义：同 salt 会话获得独立 KV 桶，不被无关请求的 LRU 驱逐 | 退回共享 LRU 桶——无隔离、无收益 |
| 检测到前缀被改写时 `POST {engine-root}/release_kv_cache`，携带 `model`、`cache_salt`、`cache_sharing`、`messages`、`messages_released_index`（以及可选的 `tools`、`tools_released_index`） | 按 agent-core 兼容的部分释放语义：从释放下标起丢弃脏块，保留有效前缀 | 记入 `releases_failed` 计数并告警；改写频繁的智能体失去释放收益 |

降级始终安全：符合规范的网关会忽略未知字段，释放失败仅产生非致命告警，
模型作为普通 OpenAI 客户端照常工作。引擎的实际行为由基准测试显式暴露
（release 端点探测、`affinity_stats`、疑似假亲和警报）——见
[benchmark/PRINCIPLES.md](benchmark/PRINCIPLES.md)。

## 验证

真实亲和收益需在真实昇腾引擎上度量（MindIE / vLLM-Ascend 前缀缓存统计）。
无硬件环境下，单元测试覆盖完整协议契约（salt 注入、前缀差异、释放调度与传输）：

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
