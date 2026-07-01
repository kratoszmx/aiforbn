---
name: aiforbn-overleaf-proposal
description: Use for /Users/zmx/Projects/aiforbn research-plan and Overleaf delivery work, including LaTeX proposal editing, XeLaTeX/fontspec/xeCJK compiler checks, Overleaf source/PDF sync, collaborator privilege verification, and final PDF/source handoff.
---

# AI-for-BN Overleaf Proposal

Use this skill only for research-plan or Overleaf delivery tasks in `aiforbn`.

## Scope

This skill covers:

- proposal LaTeX source maintenance
- Overleaf compiler and collaborator checks
- PDF upload or source ZIP comparison
- project read/write access verification
- Git sync after proposal delivery

Do not use it for ordinary materials pipeline or model code changes.

## Known Pitfalls

- If the proposal uses `fontspec` or `xeCJK`, Overleaf must compile with XeLaTeX. pdfLaTeX will fail on the `fontspec` path.
- Editability comes from member privileges, not from possession of a project-management URL alone.
- After uploading a PDF to Overleaf, the file tree may lag. Verify by downloading the source ZIP and comparing the PDF byte-for-byte when exact upload proof matters.
- Research-plan files may contain sensitive institutional or personal context. Summarize only what the task needs.

## Workflow

1. Read `/Users/zmx/Projects/aiforbn/AGENTS.md` and the current proposal task context.
2. Identify the exact local proposal source and output files before editing.
3. Use `$mcp-overleaf` when remote Overleaf source inspection, ZIP download, Git mirror work, or browser-session fallback is needed.
4. Verify compiler choice and collaborator privilege with textual evidence.
5. Keep raw Overleaf downloads in ignored temporary or artifact paths unless the task explicitly asks to track them.
6. Before commit or push, use `$git-sync` and stage only intentional proposal artifacts.

## Validation

- Local source and generated PDF names are explicit.
- XeLaTeX requirement is checked when `fontspec` or `xeCJK` is present.
- Overleaf write access is proven by member privilege or a successful controlled update.
- Uploaded or downloaded artifacts are verified by file size, checksum, or byte comparison when exactness matters.
