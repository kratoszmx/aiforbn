# AGENTS.md

This repository is maintained for autonomous AI agents. Do not treat it as an app, tutorial, or manual workflow.

## Purpose

`aiforbn` is a research-grade AI-for-BN demo project.

The project combines literature/research planning, materials data pipelines, structure generation, model experiments, reporting, and demo artifacts. Treat `.agents/skills/aiforbn-workflow/SKILL.md` and `.agents/skills/aiforbn-overleaf-proposal/SKILL.md` as the repo-scoped Codex skills. Treat `skills/ai_native_workflow.txt` as the compact project runtime guidance. Treat `assets/deep-research-report.md` as research context and `assets/poc_workflow_brief.txt` as an editable coding plan rather than a fixed contract.

## AI-Native Working Mode

- Optimize structure, names, docs, scripts, and state records for agent search, execution, verification, rollback, and handoff.
- Do not optimize this repository for onboarding or manual operation.
- Prefer `AGENTS.md` as the root entry point. Root `README.md` should not be introduced unless an external platform requires it.
- Treat `docs/AGENT_MANIFEST.json` plus `python3 main.py --verify-agent-contract` as the machine-readable project contract and first inspection command.
- Use `python3 main.py --emit-agent-commands` when choosing the smallest sufficient validation profile for a change.
- The section layout in this file is guidance, not a fixed process. If an agent invents a better workflow, record the reason in this file or a nearby state file before relying on it.

## Directory Map

- `src/materials/`: materials data, feature building, candidate screening, ranking, structure artifacts, and reporting logic.
- `src/runtime/`: shared runtime schemas and IO helpers.
- `src/template/`: template utilities.
- `src/torch_models/`: neural model components and experiments.
- `src/ui/`: optional artifact viewer; not a primary operation surface.
- `src/tests/`: cross-module tests.
- `tasks/`: task-specific research and implementation areas.
- `docs/`: agent handoff notes, machine-readable state, project reports, and research plans.
- `assets/`: prompts, deep research report, and proof-of-concept planning notes.
- `data/`: raw and processed project data.
- `artifacts/`: generated research/demo artifacts; check sensitivity and reproducibility before committing new files.
- `.agents/skills/`: repo-scoped Codex `SKILL.md` files that trigger only for this project scope.
- `skills/`: compact project runtime guidance; only `ai_native_workflow.txt` should remain active.

## Current State

- Several subtrees already have local `AGENTS.md` and `PY_FILES_SUMMARY.md`; keep them aligned when changing public modules or task boundaries.
- `docs/research_plan/` is a sensitive research-plan bundle already present in the current tracked history. Do not edit, expand, or re-stage similar generated proposal files unless the task explicitly asks for research-plan work.
- `docs/AGENT_MANIFEST.json` records the AI-native contract for entrypoints, module boundaries, validation commands, and safety boundaries.
- Legacy `skills/*_skill.txt`, `skills/template.txt`, and `skills/workflow.txt` are retired; their still-current instructions are consolidated into `skills/ai_native_workflow.txt`, `.agents/skills/`, this file, or module-local `AGENTS.md`.
- `.DS_Store`, caches, local environment files, and generated scratch outputs should remain untracked.

## Safety Boundary

- Do not commit private datasets, credentials, unpublished external documents, local caches, or large generated artifacts without checking task intent.
- Be careful with research-plan documents and professor/user feedback files; summarize rather than exposing unnecessary personal or institutional detail.
- Prefer text, code, structured data, and reproducible scripts over notebook-only or visual-only workflows.

## Validation

- Use the conda `quant` environment by default.
- Run `python3 main.py --emit-agent-commands` when selecting a validation profile.
- Run `python3 main.py --verify-agent-contract` before larger architecture/workflow edits.
- Run `python3 main.py --dry-run` for fast config / feature / model wiring checks.
- Run focused pytest tests for the touched module when possible.
- If changing public functions or module boundaries, update the nearest `PY_FILES_SUMMARY.md`.
- UI/demo changes should still have a text-verifiable path through tests, logs, or generated structured output.

## Git

- Current branch is `main`.
- Main remote is `origin`.
- Commit only intentional source, tests, research docs, and agent-maintenance documentation.
- Keep unrelated untracked research-plan bundles unstaged unless explicitly requested.
