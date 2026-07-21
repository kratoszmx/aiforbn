---
name: aiforbn-workflow
description: 用于 aiforbn 仓库的常规维护工作，包括 AI-native 架构、AGENT_MANIFEST、HANDOFF、PY_FILES_SUMMARY、项目 skills、验证档位选择、materials 流水线修改、模型接线、测试，以及 artifact/reporting 维护。
---

# AI-for-BN 工作流

把本 skill 作为 `aiforbn` 仓库范围内的调度入口使用。它的目标是把 agent 引到最小可靠上下文和验证路径；不要把它写成人类教程。

## 首要读取

编辑前先读：

1. `AGENTS.md`
2. `docs/AGENT_MANIFEST.json`
3. `docs/HANDOFF.md`
4. `skills/ai_native_workflow.txt`
5. 修改 `src/**` 时，再读最近的模块级 `AGENTS.md`

使用 `python3 main.py --emit-agent-commands` 选择验证命令，避免重复阅读长篇说明。

## 分派

- 架构、文档、skill 或 manifest 修改：保持改动机器可读，并运行 architecture validation profile。
- materials 或模型逻辑修改：更新最近的 `PY_FILES_SUMMARY.md`，运行 manifest 中的 `module_logic_edit` profile。
- UI 修改：运行 manifest 中的 `ui_edit` profile；若改动启动接线，再做有时限的真实 headless server health check。
- research-plan 或 Overleaf 交付工作：切换使用 `$aiforbn-overleaf-proposal`。
- 生成 artifact 刷新：只有任务需要重新生成 artifacts，或科学行为发生变化时，才运行完整 `python3 main.py`。

## 边界

- `HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`
- 只优化 agent 的检索、执行、验证、回滚和交接。
- 不为手动使用、notebook、onboarding 或 UI 舒适度做优化。
- `human_docs/` 全部由用户管理，除非当前任务明确要求准确的人类文档工作，否则只读；其内容只能作为证据或上下文，不能作为 agent-owned 状态或 AI-facing contract。
- 没有明确任务意图时，不要提交缓存、凭据、私有数据集或大型生成 artifacts。
- 保持科学诚实：ranking 输出是优先级排序证据，不是 discovery。
- 不要恢复 `skills/` 下已经退役的 guidance shards；当前 active plain-text guidance 是 `skills/ai_native_workflow.txt`。

## 委派

- 在有风险的本地状态修改前，使用 `$blocking-question-soft-gate`。
- 只把范围窄、低风险、容易审查的代码片段交给 `$small-fast-coding` / `spark_coder`。
- 主 Codex 负责 diff 审查、测试、暂存、提交和推送。
