# LangChain 集成收录 PR 材料

目标：把 `langchain-ascend-affinity` 登记到 LangChain 官方集成体系（下载表），
使 LangChain 用户可发现本集成。

## 官方规则（2026-08 抓取 docs.langchain.com）

> New integrations are not accepted as PRs to any langchain-ai repository. All
> new integrations must be published as independent packages to PyPI. The only
> PR you should open to a langchain-ai repo is to list your published package
> in the docs: either a YAML row for the download table, or a hosted guide if
> you meet the eligibility criteria.
>
> — https://docs.langchain.com/oss/python/contributing/integrations-langchain

本仓库已是独立包形态（`pyproject.toml` + P1 的 PyPI 发布流水线），满足前提。
默认收录方式（月下载 <50k）：在 `scripts/data/integration_external_docs.yaml`
加一行。参照先例：langchain-ai/docs PR #5511（AlphaAI）、#5494（Coalent）。

## 目标仓库与分支

- 仓库：`langchain-ai/docs`（**注意：默认分支是 `main`**，不是 langchain 主
  仓库的 master；文档已从 langchain-ai/langchain 迁出）
- 分支：main
- 收录文件：`scripts/data/integration_external_docs.yaml`（数据源）、
  `packages.yml`（包注册表）、`src/snippets/oss/python-chat-downloads.mdx`
  （渲染下载表，由 `scripts/refresh_integration_downloads.py` 生成/刷新，
  按 AlphaAI 先例随 PR 手动带上）

## 精确改动（3 个文件）

### 1. `scripts/data/integration_external_docs.yaml`

在 `python:` → `chat:` 列表末尾（当前锚点：`- name: ChatKinetica` 条目之后）
追加：

```yaml
  - name: ChatAscendAffinity
    pypi: langchain-ascend-affinity
    docs_url: https://github.com/ascend-agentos-bridge/langchain-ascend-affinity
    stream: true
    tool_calling: true
    structured_output: false
    multimodal: false
```

能力字段依据（与仓库实现核对）：

| 字段 | 值 | 依据 |
|---|---|---|
| `stream` | true | `_stream` SSE 流式 + `streaming` 聚合标志 |
| `tool_calling` | true | `bind_tools`（OpenAI 格式 tool schemas） |
| `structured_output` | false | 未特化 `with_structured_output` |
| `multimodal` | false | 纯文本 |

### 2. `packages.yml`

在 `packages:` 列表（按字母序，`langchain-` 前缀区）追加：

```yaml
- name: langchain-ascend-affinity
  name_title: AscendAffinity
  repo: ascend-agentos-bridge/langchain-ascend-affinity
  js: "n/a"
```

（`downloads` / `downloads_updated_at` 由 docs 仓库 weekly workflow
`update-package-downloads.yml` 自动填充，参照 AlphaAI 先例不手写。）

### 3. `src/snippets/oss/python-chat-downloads.mdx`

在表尾（锚点：`| [`TokenMix`](...) | ... |` 行之后、`</div>` 之前）追加：

```mdx
| [`ChatAscendAffinity`](https://github.com/ascend-agentos-bridge/langchain-ascend-affinity) | <span data-sort-value="2">✅</span> | <span data-sort-value="2">✅</span> | <span data-sort-value="1">❌</span> | <span data-sort-value="1">❌</span> | <span data-sort-value="-1">N/A</span> |
```

（能力列序：stream / tool_calling / structured_output / multimodal；
新包无下载数据 → `N/A`。）

## PR 标题与描述

标题：`docs: list langchain-ascend-affinity as external chat model integration`

描述：

```
## What

List the standalone `langchain-ascend-affinity` package (AscendAffinityChatModel)
in the external integration download tables (chat model component).

## Why

Ports the openJiuwen agent-core Ascend compute-affinity mechanism to LangChain:
cache_salt session binding + prefix-diff scheduling + partial KV-Cache release
through `POST /release_kv_cache`, plus the agent_hint lifecycle protocol
(evict/offload/prefetch, opt-in). Works as a drop-in `BaseChatModel` for
langchain / langgraph / deepagents agents — no callbacks, no handler wiring.

## Checklist

- [x] Package is a standalone PyPI-eligible package (`langchain-ascend-affinity`)
- [x] `stream` / `tool_calling` verified against the implementation
- [x] Rows added to `scripts/data/integration_external_docs.yaml` and
      `packages.yml`; chat downloads snippet refreshed (AlphaAI precedent #5511)
```

## 提交流程

方式 A（gh CLI，推荐）——`gh auth login` 完成认证后：

```powershell
gh repo fork langchain-ai/docs --clone
cd docs
git checkout -b docs/list-langchain-ascend-affinity
# 应用上述 3 处改动
git add scripts/data/integration_external_docs.yaml packages.yml src/snippets/oss/python-chat-downloads.mdx
git commit -m "docs: list langchain-ascend-affinity as external chat model integration"
git push -u origin docs/list-langchain-ascend-affinity
gh pr create --repo langchain-ai/docs --title "docs: list langchain-ascend-affinity as external chat model integration" --body "..."
```

方式 B（token）：`$env:GH_TOKEN="<PAT>"` 后同上（fork/push/PR 用 gh 或 git）。

## 前置：PyPI 发布

`pypi` 字段与下载表 badge 指向 `langchain-ascend-affinity`（当前 PyPI 未占用，
2026-08 检查 404）。建议在 PR 前完成发布：

- 本地：`poetry build && python -m twine upload dist/*`（需 PyPI token）
- 或 GitHub：在仓库 Settings → Secrets 配置 `PYPI_API_TOKEN` 后打 tag
  （`.github/workflows/release.yml` 自动构建发布，PyPI job 仅在
  `PYPI_API_TOKEN` 存在时执行）

若暂不发布 PyPI，docs_url 用 GitHub 链接仍符合官方优先序（partner docs >
GitHub > PyPI），但下载表 badge 会显示 404，维护者可能要求先发布。
