# AGENTS.md

This repository is maintained for AI agents. Do not treat it as a human-facing app or tutorial.

## Purpose

`aiforbn` is a research-grade AI-for-BN demo project.

The project combines literature/research planning, materials data pipelines, structure generation, model experiments, reporting, and UI/demo artifacts. Treat `skill.txt` and `skills/` as project-specific guidance, and treat `assets/poc_workflow_brief.txt` as an editable coding plan rather than a fixed contract.

## AI-Native Working Mode

- Optimize structure, names, docs, scripts, and state records for agent search, execution, verification, rollback, and handoff.
- Do not optimize this repository for non-technical human onboarding. 見微 is not expected to run this project manually.
- Prefer `AGENTS.md` as the root entry point. Root `README.md` should not be introduced unless an external platform requires it.
- The section layout in this file is guidance, not a fixed process. If an agent invents a better workflow, record the reason in this file or a nearby state file before relying on it.

## Directory Map

- `src/materials/`: materials data, feature building, candidate screening, ranking, structure artifacts, and reporting logic.
- `src/runtime/`: shared runtime schemas and IO helpers.
- `src/template/`: template utilities.
- `src/torch_models/`: neural model components and experiments.
- `src/ui/`: Streamlit demo UI.
- `src/tests/`: cross-module tests.
- `tasks/`: task-specific research and implementation areas.
- `docs/`: handoff notes, project reports, research plans, and user-facing Chinese summaries.
- `assets/`: prompts, deep research report, and proof-of-concept planning notes.
- `data/`: raw and processed project data.
- `artifacts/`: generated research/demo artifacts; check sensitivity and reproducibility before committing new files.
- `skills/`: project-specific agent guidance.

## Current State

- Several subtrees already have local `AGENTS.md` and `PY_FILES_SUMMARY.md`; keep them aligned when changing public modules or task boundaries.
- `docs/research_plan/` currently contains untracked research-plan source and rendered files. Do not stage them unless the task explicitly asks to preserve that research-plan bundle.
- `.DS_Store`, caches, local environment files, and generated scratch outputs should remain untracked.

## Safety Boundary

- Do not commit private datasets, credentials, unpublished external documents, local caches, or large generated artifacts without checking task intent.
- Be careful with research-plan documents and professor/user feedback files; summarize rather than exposing unnecessary personal or institutional detail.
- Prefer text, code, structured data, and reproducible scripts over notebook-only or visual-only workflows.

## Validation

- Use the conda `quant` environment by default.
- Run focused pytest tests for the touched module when possible.
- If changing public functions or module boundaries, update the nearest `PY_FILES_SUMMARY.md`.
- UI/demo changes should still have a text-verifiable path through tests, logs, or generated structured output.

## Git

- Current branch is `main`.
- Main remote is `origin`.
- Commit only intentional source, tests, research docs, and agent-maintenance documentation.
- Keep unrelated untracked research-plan bundles unstaged unless explicitly requested.
