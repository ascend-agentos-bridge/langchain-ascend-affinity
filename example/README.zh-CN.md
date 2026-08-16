# 验证示例：openJiuwen 算力亲和移植

一套无需昇腾硬件的验证工具，证明移植的算力亲和机制真实生效：
[verify_affinity.py](verify_affinity.py) 把**完全相同**的确定性“双用户 × 三轮
对话”调度对着模拟昇腾引擎（[mock_engine.py](mock_engine.py)）跑两遍：

- **plain 阶段** —— `AscendAffinityChatModel(enable_affinity=False)`：不带
  `cache_salt`，两个用户共享同一个匿名 KV-Cache 桶，交错轮次不断造成前缀
  分歧（6 次请求中 5 次付出部分/全量重算）。
- **affinity 阶段** —— 同一个模型按会话绑定 salt（`bind(session_id=...)`）：
  每个用户获得独立的缓存桶（6 次中 4 次命中热缓存），且会话中途的历史改写
  触发移植的前缀差异调度器，精确发出 1 次 `POST /release_kv_cache`。

## 运行

```bash
pip install -r example/requirements.txt
python example/verify_affinity.py          # 输出 PASS + 对比表

# 可选项
python example/verify_affinity.py --port 8001
MOCK_TTFT_COLD_MS=500 python example/verify_affinity.py   # 放大差距
```

## 如何读输出

```text
metric                        plain    affinity
requests                          6           6
cold starts                       1           2
warm hits                         0           4
partial recomputes                5           0
kv releases                       0           1
avg TTFT (ms)                 189.6        96.7
salt buckets                anonymous  user-A,user-B
answers identical across phases: yes
```

- **warm hits / partial recomputes**：salt 绑定隔离了会话缓存，纯追加轮次
  直接命中热缓存，而不是对着失效前缀重算。
- **kv releases**：调度器检测到被改写的历史消息后，通知引擎只释放失效后缀
  （payload 与 agent-core 完全兼容），保留有效前缀常驻。
- **answers identical**：引擎与调度均为确定性，两阶段答案逐字节一致——对比
  是公平的。

任一不变量被打破（未触发释放、热命中无增益、TTFT 未明显下降、答案不一致）
脚本都会以非零码退出，因此它同时可用作冒烟测试。

## 模拟引擎

`mock_engine.py` 提供 `/v1/chat/completions`、`/release_kv_cache` 与
`/metrics`，按 `cache_salt` 绑定 KV 块，并按缓存温度计价 TTFT
（warm / partial·按失效比例折算 / cold）。可用环境变量调参：
`MOCK_TTFT_WARM_MS`（20）、`MOCK_TTFT_COLD_MS`（250）、
`MOCK_KV_SLOTS`（4）、`MOCK_ENGINE_PORT`（8000）。详见模块 docstring。
