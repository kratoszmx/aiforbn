---
name: aiforbn-workflow
description: 维护 aiforbn（AI for Science）的代码、文档、项目 skills、agent contract 和研究产物；区分合作方原型与历史带隙流程，按改动选择验证。Proposal/Overleaf 交付使用专门 skill。
---

# AI for Science 维护

从 `AGENTS.md` 和 `HANDOFF.md` 定位任务；修改模块时再读最近的 `AGENTS.md` 与 `PY_FILES_SUMMARY.md`。共用函数、环境/测试、服务生命周期分别见根目录 `COMMON_FUNCTIONS.md`、`TESTING.md`、`SERVICES.md`。

## 选择工作流与验证

用已确认的 quant 解释器先运行 `main.py --verify-agent-contract`，再以 `main.py --emit-agent-commands` 选择足够的验证范围。命令索引的 `requires` / `provides` 描述实际覆盖；环境与命令示例集中在 `TESTING.md`。

| 改动 | 验证档位 / 入口 |
| --- | --- |
| 文档、skill、入口元数据 | `architecture_doc_skill_edit` |
| 公共 API、模型、模块逻辑或合作方 API/网页/监测 | `module_logic_edit`；合作方的重点测试见 `TESTING.md` |
| 科学流程或产物行为 | `scientific_pipeline_edit`；交付需要新结果时才完整重算 |
| Streamlit 展示 | `ui_edit`；启动接线变更时参考 `SERVICES.md` 的限时 HTTP 检查 |
| 明确授权的 research-plan/Overleaf 交付 | 使用 `$aiforbn-overleaf-proposal` |

合作方原型的当前用法、方法与版本证据见 `docs/research/separator_prototype/INDEX.md`。`main.py` 和它的 dry-run 属于历史带隙流程，不能代替合作方测试。根目录旧 `skills/` 已退役；有效的运行指导集中在本 skill。

## 项目经验与边界

- `HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`；`human_docs/` 是用户拥有的只读上下文，只有明确的人类文档任务才可修改指定内容。
- 合作方的 BN 隔膜、水性浆料和电解液证据保持独立；历史带隙标签不进入这些任务。缺失测量条件保留缺失，回放中的节省不是前瞻实验或 BN 收益。
- 历史流程的整体评估模型可以使用结构特征，formula-only screening 不可以。排名与未松弛结构不是发现或物理验证；产物可用性取决于 provenance 与实际输出摘要校验。
- 沿现有公开 API 复用并保留项目路径/来源校验；跨仓库提取取决于实际复用和兼容性。保留 `main.py` 的线性调用脉络，公开接口变化同步最近的函数说明。
- 先辨认已有 dirty 改动，再验证和提交本次字节。写入/发布前检查与读取时检查各有失败边界；精简时保留它们。当前状态集中在 `HANDOFF.md`，历史证据留给版本报告与 Git。
