---
name: aiforbn-overleaf-proposal
description: 用于 aiforbn 仓库的 research-plan 和 Overleaf 交付工作，包括 LaTeX proposal 编辑、XeLaTeX/fontspec/xeCJK 编译器检查、Overleaf source/PDF 同步、协作者权限验证，以及最终 PDF/source 交接。
---

# AI-for-BN Overleaf Proposal

仅在 `aiforbn` 的 research-plan 或 Overleaf 交付任务中使用本 skill。

## 范围

本 skill 覆盖：

- proposal LaTeX 源文件维护
- Overleaf 编译器和协作者检查
- PDF 上传或 source ZIP 对比
- 项目读写权限验证
- proposal 交付后的 Git 同步

不要把本 skill 用于普通 materials pipeline 或模型代码修改。

## 已知坑点

- 如果 proposal 使用 `fontspec` 或 `xeCJK`，Overleaf 必须使用 XeLaTeX 编译；pdfLaTeX 会在 `fontspec` 路径失败。
- 可编辑性来自 member privileges，而不是单纯拥有 project-management URL。
- PDF 上传到 Overleaf 后，文件树可能有延迟；如果需要精确上传证明，下载 source ZIP 并逐字节比较 PDF。
- Research-plan 文件可能包含敏感的机构或个人上下文；只总结任务需要的内容。

## 工作流

1. 读取仓库根目录的 `AGENTS.md` 和当前 proposal 任务上下文。
2. 编辑前确认准确的本地 proposal source 和 output 文件。
3. 需要远程 Overleaf source 检查、ZIP 下载、Git mirror 工作或 browser-session fallback 时，使用 `$mcp-overleaf`。
4. 用文字证据验证编译器选择和协作者权限。
5. 除非任务明确要求追踪原始下载，否则把 Overleaf 原始下载放在 ignored temporary 或 artifact 路径。
6. commit 或 push 前，使用 `$git-sync`，并且只 stage 有意的 proposal artifacts。

## 验证

- 本地 source 和生成 PDF 的文件名明确。
- 出现 `fontspec` 或 `xeCJK` 时，已检查 XeLaTeX 要求。
- Overleaf 写权限由 member privilege 或成功的受控更新证明。
- 在需要精确性时，通过文件大小、checksum 或逐字节比较验证上传/下载 artifacts。
