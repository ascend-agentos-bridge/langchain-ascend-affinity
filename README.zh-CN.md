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
| `salt_enabled` | `True` | `False` = 保留管线计数但不再注入 `cache_sharing` / `cache_salt`（用于拒绝 salt+工具调用请求的引擎，见下） |
| `release_endpoint` | `/release_kv_cache` | 引擎根路径下的部分释放端点；置 `""` 禁用 |
| `enable_agent_hint` | `False` | 可选启用 agent_hint 生命周期协议（身份字段 + `evict` / `offload` / `prefetch` 管理方法） |
| `idle_evict_after_seconds` | `0` | 生成后空闲多少秒自动驱逐会话 KV 缓存（`0` = 关闭；需 `enable_agent_hint`） |
| `streaming` | `False` | `invoke()` / `ainvoke()` 内部经 SSE 流式聚合，触发 `on_llm_new_token` 回调 |
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
请求退化为普通 OpenAI 调用，释放失败仅产生非致命告警。引擎**主动拒绝**
亲和字段时（据 agent-core 联调经验：`cache_salt` + 工具调用消息
组合返回 HTTP 501，MindIE 类服务器，本仓库未独立复核）同样自动处理——
请求去掉 salt 字段重试一次，随后**该 session** 的 salt 绑定被禁用
（`salt_degraded_requests` 计数 + WARNING 日志，其他 session 不受影响），
工具调用型智能体以普通 OpenAI 客户端继续工作。

哪个引擎版本支持什么（MindIE、vLLM-Ascend）、完整接口契约、与
openjiuwen agent-core 的协议兼容、以及 LLM 网关透传行为，**集中维护**在
[COMPATIBILITY.zh-CN.md](COMPATIBILITY.zh-CN.md)。引擎的实际行为由基准
测试显式暴露（release 端点探测、`affinity_stats`）——见
[benchmark/PRINCIPLES.md](benchmark/PRINCIPLES.md)。

## 可观测性

每个模型实例通过 `affinity_stats` 暴露只读计数器字典——与基准测试报告的
指标一致：

| 键 | 含义 |
|---|---|
| `affinity_requests` | 进入亲和管线的请求数 |
| `salt_bound_requests` | 实际与会话完成 salt 绑定的请求数 |
| `salt_degraded_requests` | 引擎拒绝 salt 绑定请求（HTTP 501）后去掉 salt 字段重试的请求数；此后该 session 禁用 salt 绑定（其他 session 不受影响） |
| `releases_attempted` | 已发送的部分 KV 释放请求数 |
| `releases_failed` | 释放失败数（从不致命） |
| `management_requests` | 已发送的 agent_hint `evict` / `offload` / `prefetch` 请求数 |
| `management_failed` | 管理请求失败数（从不致命） |

```python
stats = model.affinity_stats
# {"affinity_requests": 3, "salt_bound_requests": 3, ...}
```

在 `DEBUG` 日志级别，模型记录每次 salt 绑定与前缀分叉释放决策（会话 id、
释放下标）；失败始终以 WARNING 记录。

## 异步与 agent_hint 用法

`ainvoke` 走完全相同的亲和管线；agent_hint 管理方法是 `invoke` 的协议对等
体（与 agent-core 同名同语义）：

```python
import asyncio

from langchain_core.messages import HumanMessage

from langchain_ascend import AscendAffinityChatModel


async def main() -> None:
    model = AscendAffinityChatModel(
        base_url="http://127.0.0.1:8000/v1",
        enable_agent_hint=True,  # 可选启用生命周期协议
    )
    reply = await model.ainvoke(
        [HumanMessage(content="hello")],
        config={"metadata": {"session_id": "s1"}},
    )
    print(reply.content)

    # 生命周期管理，与 agent-core 方法语义一致
    model.evict_kvc(session_id="s1")
    model.offload_kvc(session_id="s1")
    model.prefetch_kvc(session_id="s1")


asyncio.run(main())
```

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

### 离线 / 无外网环境打包

如果基准测试要跑在**封闭无互联网**的昇腾集群上，推荐的方式是在一台
可联网的构建机上打出完整的自包含上传包。`scripts/build_benchmark.py`
可以把源码树、版本钉死的 requirements、安装脚本以及（可选）目标平台
的全部 wheel 一起打成 zip。

完整说明（跨平台 wheel 下载、Linux/Windows 安装脚本、包内容速查、
故障排查）见 [OFFLINE_PACKAGING.zh-CN.md](OFFLINE_PACKAGING.zh-CN.md)。
在能联网的构建机上一条命令完成打包：

```bash
python scripts/build_benchmark.py --with-wheels --with-installers --zip
```

生成的 zip 经 scp / U 盘 搬到封闭机后，运行包内的
`install_offline.sh`（Linux）或 `install_offline.ps1`（Windows）
即可完成部署。

## 开发

```bash
python -m pytest tests/unit_tests   # 覆盖率门槛：90%
python scripts/quality_gate.py      # pylint 10.00/10 + 单元测试
```

规格驱动的设计文档见
[openspec/](openspec/projects/affinity-core/proposals/openjiuwen-affinity-port/)。
