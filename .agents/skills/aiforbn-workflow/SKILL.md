---
name: aiforbn-workflow
description: 维护 aiforbn 的代码、文档、项目 skills、agent contract 和研究 artifacts，按改动选择验证范围。Proposal 或 Overleaf 交付使用专门的 proposal skill。
---

# AI-for-BN 维护

从 `AGENTS.md` 和 `HANDOFF.md` 定位当前任务；修改模块时再读其 `AGENTS.md` 与 `PY_FILES_SUMMARY.md`。环境、命令和结果判读见根目录 `TESTING.md`；`main.py --emit-agent-commands` 提供实际验证档位，避免复制长期易漂移的命令清单。

## 按任务取用

- 文档、skill、入口元数据：选择 `architecture_doc_skill_edit`。
- 公共 API、模型或模块逻辑：选择 `module_logic_edit`，同步最近的公开函数说明。
- 科学流程/artifact 行为：选择 `scientific_pipeline_edit`；只有交付需要新结果时才完整运行 `main.py`。
- UI：选择 `ui_edit`。启动接线变化时，可按 `SERVICES.md` 补一次有时限的本机 HTTP 检查。
- 明确授权的 research-plan/Overleaf 任务：使用 `$aiforbn-overleaf-proposal`。两项 skill 分开保留，避免普通代码任务加载远程文档流程。

## 项目经验与边界

- `HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`；`human_docs/` 是用户拥有的只读上下文，只有明确的人类文档任务才可修改指定内容。
- 整体评估模型可以使用结构特征，formula-only screening 不可以。排名是低置信度跟进优先级；未松弛结构和测试通过都不是发现或物理验证。
- Artifact 是否可用取决于 provenance 和实际输出摘要校验，不能只看文件存在。具体发布约束见 `docs/HANDOFF.md` 和模块 API 文档。
- 优先沿现有公共 API 复用；保留项目特有的路径/来源校验。是否提取通用函数取决于实际复用和行为兼容性，不自动扩展到其他仓库。
- 先辨认已有 dirty 改动，再验证和提交自己的字节。保留必要的安全说明，重复流程用链接集中到根文档；历史维护细节交给 Git。
