# 离线基准测试打包指南

[English](OFFLINE_PACKAGING.md) | [简体中文](OFFLINE_PACKAGING.zh-CN.md)

本文说明如何将 `langchain-ascend-affinity` 核心库与其 benchmark harness
一起打成**自包含、完全离线可用**的软件包，以便上传到**封闭无互联网**
的环境（例如部署昇腾 NPU 集群的安全机房）执行基准测试。

---

## 1. 适用场景

以下情况请使用本打包流程：

- 目标环境**没有外网**（无法访问 PyPI、内网镜像未开通或因策略被拒）。
- 需要**字节精确、可复现**的依赖版本——避免半年后有人 `pip install`
  时被意外升级。
- 需要**零交互**安装——上传后一条命令就能安装并跑起来。

以下情况**不需要**使用本流程：

- 目标环境有 pip 访问权限（PyPI 或公司镜像）→ 按 `benchmark/README.md`
  正常流程即可。
- 只验证核心库、不跑 4-agent 基准。

---

## 2. 快速开始（一条命令）

```bash
# 在仓库根目录执行：
python scripts/build_benchmark.py \
  --with-wheels \
  --with-installers \
  --zip \
  --output-dir ./ascend-benchmark-offline
```

执行后产出：

```
ascend-benchmark-offline/
├── langchain_ascend/          # 核心库（替换一个模型对象即生效）
├── benchmark/                 # 4-agent 基准 harness + 任务集 + 指标
├── scripts/                   # quality_gate.py、build_benchmark.py
├── wheels/                    # 预下载的 .whl 文件
├── requirements.txt           # 锁定版本的依赖清单
├── pyproject.toml             # 包元信息（可编辑安装）
├── install_offline.sh         # Linux 安装脚本（Bash）
├── install_offline.ps1        # Windows 安装脚本（PowerShell）
├── README.md                  # 项目快速开始（英文）
├── README.zh-CN.md            # 项目快速开始（中文）
├── AGENTS.md / LICENSE
└── OFFLINE_PACKAGING.zh-CN.md # 本文档
```

加了 `--zip` 会在输出目录旁同步产出 `ascend-benchmark-offline.zip`，
直接上传即可。

### CLI 参数一览

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--output-dir DIR` | `./ascend-benchmark-offline` | 输出目录 |
| `--zip` | 关 | 额外压缩为 `<dir-name>.zip` |
| `--with-wheels` | 关 | 打包完成后调用 `pip download` 下载 wheels 到 `wheels/` |
| `--with-installers` | 关 | 写入 `install_offline.sh` 和 `.ps1` 安装脚本 |
| `--wheel-platform PLATFORM` | `current` | wheel 目标平台。`current` = 与构建机相同。跨平台详见第 3 节。 |
| `--wheel-python-version VER` | 自动探测 | 例如 CPython 3.11 填 `311`，与 `--wheel-platform` 配合做跨平台。 |
| `--index-url URL` | 系统默认 | 覆盖下载 wheels 用的 PyPI 索引（例如公司镜像）。 |
| `--no-clean` | 关 | 重建前**不**清空输出目录（保留已下载的 wheels，避免重复下载）。 |

---

## 3. 跨平台 wheel 下载

### 为什么 wheel 分平台

一次完整 build 约 110 个包，其中约 80 个是**纯 Python**
（`py3-none-any.whl`）→ **任何操作系统/CPU 都能用**。

其余约 30 个是**C / Rust / Cython 扩展包**，编译产物绑定特定
「操作系统 × Python 版本 × CPU 架构」三元组。常见名单：

- `pydantic-core`（Rust）— pydantic 校验核心
- `numpy`（C）— 数值数组
- `orjson` / `ujson` / `msgpack`（C/Rust）— 高性能 JSON
- `tokenizers` / `tiktoken`（Rust）— BPE 分词
- `pillow`（C）— 图片 I/O
- `lxml`（C）— XML 解析
- `regex`、`rpds-py`、`zstandard`、`httptools` 等

### 三种典型场景

#### 场景 A — 构建机 == 目标机（同 OS / Python / 架构）

```bash
python scripts/build_benchmark.py --with-wheels --with-installers --zip
```

直接完事。`--wheel-platform current` 是默认值。

#### 场景 B — Windows 上构建，Linux 上部署（非常常见）

两条路：

**路径 1 — 推荐：找一台有网的 Linux 机器重打。**

与目标机同架构（x86_64，或 ARM 昇腾服务器用 aarch64）、同 Python
版本（推荐 3.11+）：

```bash
# 在 Linux 机器上执行：
git clone <本仓库>
python scripts/build_benchmark.py --with-wheels --with-installers --zip
# 把 ascend-benchmark-offline.zip 传到封闭环境即可。
```

**路径 2 — Windows 上交叉下载。**

速度慢一些，且华为内部镜像等可能没有全平台索引（典型表现：
pip 报 `No matching distribution`）。建议显式指定官方 PyPI：

```bash
# 目标：Linux x86_64 + CPython 3.11
python scripts/build_benchmark.py `
  --with-wheels `
  --with-installers `
  --zip `
  --wheel-platform linux_x86_64 `
  --wheel-python-version 311 `
  --index-url https://pypi.org/simple/
```

ARM 昇腾（aarch64）：

```bash
python scripts/build_benchmark.py `
  --with-wheels `
  --with-installers `
  --zip `
  --wheel-platform linux_aarch64 `
  --wheel-python-version 311 `
  --index-url https://pypi.org/simple/
```

如果个别 Rust 包（`pydantic-core`、`tokenizers`）交叉下载失败，
就走路径 1——临时开一台云 Linux 小机（1C/2G 足够），10 分钟搞定。

#### 场景 C — 构建机本身也完全无网

1. 在**任何有网的机器**上拉取 wheel：
   ```bash
   pip download -r requirements.txt -d wheels \
     --only-binary=:all: --platform linux_x86_64 --python-version 311
   ```
2. 手动把 `wheels/` 拷到包输出目录中。
3. `install_offline.sh` 会通过 `--find-links wheels` 自动从本地安装。

---

## 4. 封闭环境安装流程

### Linux（Bash）— 典型昇腾 NPU 宿主机

```bash
unzip ascend-benchmark-offline.zip
cd ascend-benchmark-offline
bash install_offline.sh
```

安装器做四件事：
1. 校验 Python ≥ 3.11（`deepagents 0.7.6` 强制要求）。
2. 从 `wheels/` 离线装依赖（`--no-index --find-links=wheels`，绝不出网）。
3. 以 editable 模式安装本地 `langchain-ascend-affinity` 包。
4. 验证 import 链，打印完整状态。

然后跑基准：

```bash
python benchmark/run_benchmark.py \
  --engine-url http://<engine-host>:<port>/v1 \
  --model <model-name> \
  --api-key <api-key>
```

### Windows（PowerShell）— 例如 LM Studio + ascend-sim 开发站

```powershell
Expand-Archive ascend-benchmark-offline.zip
cd ascend-benchmark-offline
.\install_offline.ps1
```

### 安装故障排查

**问题：「No matching distribution found」出现在某个 C 扩展包上。**

根因：`wheels/` 里的 wheel 与目标平台不匹配。典型提示：

```
ERROR: Could not find a version that satisfies the requirement numpy==2.3.5
(from versions: none)
```

修复：按第 3 节场景 B 路径 1，在匹配目标平台的机器上重打包；
或手动把对应 `.whl` 文件补进 `wheels/`。

**问题：`pydantic-core` / `tokenizers` 报「Metadata-generation-failed」。**

根因：二进制 wheel 没匹配上，pip 退化为源码编译，但是 maturin / rustc
没装。修复同上——补对正确的平台二进制 wheel。

**问题：`openjiuwen` 缺失。**

`openjiuwen` 是华为内部私有包，不在 PyPI 上。安装器会报缺失；
基准运行器会自动跳过 `oj-baseline` 和 `oj-affinity` 两个 agent。
LangChain 侧的 `lc-baseline` / `lc-affinity` 仍然正常运行。
如需跑全部四个 agent，从内部索引装：

```bash
pip install openjiuwen --index-url https://your-internal-index/simple/
```

---

## 5. 包内容结构速查

```
ascend-benchmark-offline/
├── langchain_ascend/
│   ├── __init__.py                       # 导出：AscendAffinityChatModel
│   ├── prefix_tracker.py                 # 前缀差异调度器
│   └── llms/
│       ├── chat_ascend.py                # 主模型（盐绑定 + 局部释放）
│       ├── affinity_pipeline.py          # 管道阶段
│       ├── agent_hint.py                 # evict/offload/prefetch（显式启用）
│       ├── serialization.py              # 请求/响应序列化
│       └── transport.py                  # HTTP 传输 + 重试
├── benchmark/
│   ├── run_benchmark.py                  # 入口（setup → probe → 运行）
│   ├── agents.py                         # lc-baseline + lc-affinity 构建
│   ├── oj_adapter.py                     # oj-baseline + oj-affinity 构建
│   ├── tasks.py                          # 金融任务集 + 工具 + 长轮任务
│   ├── metrics.py                        # TTFT/E2E/KV命中率/TPOT/解码聚合
│   ├── probe.py                          # 引擎探测（释放端点 / 盐兼容性 / 流式 usage）
│   ├── reporting.py                      # JSON + Markdown 化验报告
│   ├── requirements.txt                  # 仅基准测试依赖子集
│   ├── PRINCIPLES.md                     # 基准方法论
│   ├── README.md / README.zh-CN.md       # 基准快速开始（双语）
│   └── reports/                          # 报告输出目录（初始为空）
├── scripts/
│   ├── build_benchmark.py                # 本打包工具
│   └── quality_gate.py                   # Pylint + pytest 质量门禁
├── wheels/                               # 110 个预下载 wheels（约 62 MB）
├── requirements.txt                      # 锁定版本
├── pyproject.toml                        # 可编辑安装用
├── install_offline.sh                    # Linux 安装脚本
├── install_offline.ps1                   # Windows 安装脚本
├── README.md / README.zh-CN.md           # 项目快速开始（双语）
├── AGENTS.md                             # 项目维护约定
└── LICENSE
```

---

## 6. 升级 / 重建

代码变更后重新打包：

```bash
# 1. 修改代码
# 2. 若版本有漂移，重新安装基准依赖
pip install -r benchmark/requirements.txt

# 3. 重新生成包（含 wheels）
python scripts/build_benchmark.py --with-wheels --with-installers --zip --no-clean

# 4. 上传新 zip
```

**说明**：`--no-clean` 保留已有的 `wheels/`，不再重下 62 MB 已验证过的
wheel。要从零开始重建（包括清理过期 wheel）就去掉这个参数。

---

## 7. FAQ

**Q：只跑 lc-* agents 不用 openjiuwen，包怎么瘦身？**

直接打包就行。`oj_adapter.py` 不影响运行（缺失依赖自动跳过）；
运行时传 `--agents lc` 即可。`openjiuwen` wheel 从未被打进包，
所以包体积不会因为去掉它而缩小。

**Q：包有 62 MB，能再小吗？**

可以。按收益排序：

1. 删掉 `transformers`、`tokenizers`、`huggingface_hub` —— 核心库和
   benchmark 都**不直接用**，只是被 `langchain-core` 的可选依赖拉进来的。
   从 `requirements.txt` 和 `wheels/` 里删掉它们。可省约 12 MB。
2. 删掉 `pillow`、`lxml`、`openpyxl`、`reportlab`、`python-docx`、
   `python-pptx`、`pdfplumber` 等文档处理包。再省约 8 MB。
3. 用 `pip install --dry-run --report` 找未使用的传递依赖。
4. 只保留 `--agents lc` 的话，可再删 `langchain-anthropic`、
   `langchain-google-genai`。

**Q：最小可用包多大？**

只打核心库（不跑基准）：

```
langchain_ascend/ + requirements.txt
  -> langchain-core, pydantic, typing_extensions, annotated-types,
     jiter, anyio, sniffio, idna, certifi, h11, h2, httpx, httpcore
  ≈ 8 个 wheel，< 5 MB
```

**Q：安装器能自动跑 quality_gate.py 吗？**

默认不跑。质量门禁跑**单元测试**，验证库在当前平台上的正确性——
在封闭环境如果 wheel 只匹配了别的平台，会误报误导。确认 wheels 匹配
后手动执行：

```bash
python scripts/quality_gate.py
```
