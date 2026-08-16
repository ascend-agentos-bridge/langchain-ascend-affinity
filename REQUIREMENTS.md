SPEC-01: 架构与项目结构设计 (Architecture Spec)
项目名称: langchain-ascend-affinity
核心目标: 为 LangChain 提供无侵入的昇腾算力亲和插件，通过 Agent Hint 机制实现 KV Cache 的主动卸载、预取和驱逐。
架构模式: 策略模式 (Strategy Pattern)，支持 torch_npu, mindspore, cann 三种底层后端动态切换。

目录结构规范:
langchain-ascend/
├── pyproject.toml               # 采用 Poetry 管理依赖
├── langchain_ascend/
│   ├── __init__.py
│   ├── callbacks/               # 算力亲和生命周期钩子
│   │   ├── __init__.py
│   │   └── compute_affinity.py  # 核心：AscendAffinityCallbackHandler
│   ├── backends/                # 底层接口适配器 (Strategy Pattern)
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAscendBackend 抽象类
│   │   ├── torch_npu_impl.py    # 基于 PyTorch-NPU 的实现
│   │   ├── mindspore_impl.py    # 基于 MindSpore 的实现
│   │   └── cann_impl.py         # 基于 CANN (ACL) 的实现
│   ├── llms/                    # 对齐 LangChain BaseChatModel 的扩展
│   │   ├── __init__.py
│   │   └── chat_ascend.py       # AscendChatLLM (透传 agent_hint)
├── tests/
│   ├── unit_tests/
│   └── integration_tests/
└── README.md


SPEC-02: 核心功能实现 (Core Implementation Spec)
1. 后端适配器工厂 (langchain_ascend/backends/)

需求: 定义 BaseAscendBackend 抽象类，包含三个虚方法：offload_cache(session_id), prefetch_cache(session_id), evict_cache(session_id)。

动态路由: 在 __init__.py 中实现工厂方法 get_backend(backend_name: str)，根据环境变量 ASCEND_BACKEND 或初始化参数动态加载 torch_npu, mindspore 或 cann 模块。如果缺失依赖库，抛出友好的 ImportError，提示用户安装相应的昇腾 SDK。

2. 算力亲和生命周期管理 (langchain_ascend/callbacks/compute_affinity.py)

需求: 实现 AscendAffinityCallbackHandler，继承自 langchain_core.callbacks.BaseCallbackHandler。

状态机映射:

on_tool_start: 获取当前 session_id，调用 backend.offload_cache(session_id)（工具执行期间，释放大模型显存）。

on_tool_end: 调用 backend.prefetch_cache(session_id)（工具执行完毕，提前拉回 KV Cache 准备下一次生成）。

on_chain_end / on_agent_finish: 调用 backend.evict_cache(session_id)（任务彻底结束，清空缓存）。

异步支持: 必须同时实现异步回调（on_tool_start_async 等）。

3. LLM 客户端包装 (langchain_ascend/llms/chat_ascend.py)

需求: 继承自 BaseChatModel。在底层 _generate / _agenerate 方法中，自动捕获上下文中配置的 session_id，并将其封装为 JSON 格式的 agent_hint，通过 HTTP Payload 透传给后端的推理引擎（如 MindIE vLLM 兼容层）。

📂 SPEC-03: 单元测试规范 (Testing Spec)
测试框架: 统一使用 pytest。

硬件 Mock 要求: 考虑到 GitHub Actions 或常规 CI 环境没有昇腾 NPU，必须使用 unittest.mock 或 pytest-mock 对 torch_npu, mindspore, cann 的底层调用进行全面 Mock。

核心测试用例:

test_backend_factory_routing: 测试环境变量不同配置下，能否正确加载对应的底层实现类。

test_affinity_callback_lifecycle: 测试 AscendAffinityCallbackHandler 在完整的 LangChain Agent 执行流中（思考 -> 工具调用 -> 返回），是否按正确顺序触发了 offload -> prefetch -> evict 信号。

覆盖率要求: 核心逻辑覆盖率需达到 90% 以上。

📂 SPEC-04: 开发者指导说明书 (Documentation Spec)
在项目根目录生成一份 README.md，要求言简意赅，包含以下四个核心章节：

🚀 Introduction (简介): 一句话说明：为 LangChain 生态带来类似 openJiuwen 的昇腾算力亲和能力，实现 KV Cache 显存主动管理，降低多智能体场景 TTFT（首字延迟）。

📦 Installation (安装): 提供 pip 安装命令，并说明如何根据硬件环境安装可选依赖（如 pip install langchain-ascend[torch_npu]）。

⚙️ Backend Configuration (配置切换): 代码示例，展示如何通过 os.environ["ASCEND_BACKEND"] = "mindspore" 或代码参数一键切换三大底层引擎。

⚡ Quick Start (快速开始): 给出一个完整的 15 行极简代码示例，展示如何将 AscendAffinityCallbackHandler 注入到 LangGraph 或传统 AgentExecutor 中。

📂 SPEC-05: Git 提交流程与自动化 (Git Workflow Spec)
开发及测试通过后，智能体需执行以下流程完成代码推送：

初始化仓库: 在工作空间执行 git init，创建 .gitignore（忽略 __pycache__, .pytest_cache, .venv 等）。

获取认证信息: 智能体需在当前交互对话中向用户询问 GitHub 仓库地址 (Repository URL) 以及 Personal Access Token (PAT)，如果用户已在环境变量提供，则直接读取。

Commit 规范: 严格遵守 Conventional Commits 规范，例如：

feat: implement AscendAffinityCallbackHandler for compute-affinity

feat: add dynamic backend routing for torch_npu, mindspore, cann

docs: create quick start guide in README

推送代码: 使用 git remote add origin https://<TOKEN>@[github.com/](https://github.com/)<USERNAME>/langchain-ascend.git 绑定远程仓库，并执行 git push -u origin main