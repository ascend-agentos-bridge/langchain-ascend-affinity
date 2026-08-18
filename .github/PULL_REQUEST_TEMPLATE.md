## PR 描述

<!-- 简要描述这个 PR 做了什么，关联哪个 Issue（如有）。-->

## 变更类型

- [ ] feat: 新功能
- [ ] fix: 缺陷修复
- [ ] docs: 文档变更
- [ ] test: 测试变更
- [ ] chore: 构建/工具/CI 变更
- [ ] refactor: 重构（不改变公开 API）

## 质量门禁检查

- [ ] `python scripts/quality_gate.py` 通过（pylint 10.00/10 + 单元测试覆盖率 ≥90%）
- [ ] 新增 Python 文件已 `git add`（确保 pylint 扫描到）
- [ ] `README.md` 与 `README.zh-CN.md` 已同步（若涉及 API/配置变更）
- [ ] 若涉及引擎兼容性事实变更，`COMPATIBILITY.md` / `COMPATIBILITY.zh-CN.md` 已
      交叉比对上游来源后更新
- [ ] 提交遵循 Conventional Commits
- [ ] 公开 API 变更在 `CHANGELOG.md` 中记录（[Unreleased] 区块）

## 协议字段对齐检查（若涉及 agent_hint / 亲和字段）

- [ ] 与 openjiuwen agent-core `AscendAffinityModelClient` 字段级一致
- [ ] 引擎不支持时安全降级（非致命、可观测）
- [ ] 管理路径失败仅计数/告警，不中断生成

## 补充说明

<!-- 任何评审者需要知道的信息 -->
