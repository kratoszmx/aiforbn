---
name: aiforbn-overleaf-proposal
description: 处理 aiforbn 中明确授权的 research-plan/Overleaf 文档任务，包括 LaTeX 编译、source/PDF 同步与协作者权限核验。
---

# Proposal 与 Overleaf 交付

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`。从根 `AGENTS.md` 和任务指定文件定位交付范围；`human_docs/` 的授权只覆盖本次明确要求的文档，不延伸到其他研究材料。

## 实用经验

- 先确定本次主 `.tex`、`.bib`、输出 PDF 和远程项目，避免把较新的文件名当成已批准版本。
- 当前 v18 同时使用 `fontspec` 与 `xeCJK`，应选 XeLaTeX。只有 `fontspec` 时也可能使用 LuaLaTeX；按实际包依赖选编译器，不把某份 proposal 的选择泛化到全部 LaTeX 文档。参见 [Overleaf 编译器说明](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler) 与 [xeCJK 示例](https://www.overleaf.com/learn/latex/Japanese)。
- 项目 URL 本身不能证明可编辑；通过 member privilege，或任务已授权的受控更新核实。发送协作邀请或改变权限需要对应任务授权。
- 优先使用已加载的 Overleaf 文字接口；也可使用可用的文字 HTTP/DOM/Git 路径。某个 connector 缺失不等于全部工作阻断，先完成可独立验证的本地部分并说明远程证据缺口。
- 上传后的文件树可能延迟。需要精确交付证明时，下载 source ZIP，比对文件名、大小及 checksum/字节；单看上传提示不足以证明内容一致。
- 临时下载放在任务临时目录或确认已忽略的路径。当前仓库不整体忽略 `artifacts/`，不要默认那里可以安全放原始下载。
- 机构/个人背景只总结任务必要部分。编译日志、PDF 文本和文件校验可提供文字证据；不默认需要视觉检查。

交付时列明本地 source/PDF、编译结果、远程同步与权限证据，以及尚未完成的部分。Git 同步沿用当前会话规则，只暂存本次获授权的文档产物；本地成功与远程成功分别报告。
