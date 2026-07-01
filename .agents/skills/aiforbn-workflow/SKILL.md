---
name: aiforbn-workflow
description: Use for routine work in /Users/zmx/Projects/aiforbn, including AI-native project maintenance, materials pipeline changes, model/demo wiring, project documentation, validation planning, and handoff updates. Trigger when the task is inside aiforbn or mentions AI-for-BN repo workflow, AGENT_MANIFEST, HANDOFF, PY_FILES_SUMMARY, materials, torch_models, or project-specific skills/*.txt guidance.
---

# AI-for-BN Workflow

Use this skill for normal `aiforbn` repository work. It is a project-specific routing layer, not a replacement for the repository's own `AGENTS.md` files.

## First Reads

Before changing files, read the current task-relevant entrypoints:

1. `/Users/zmx/Projects/aiforbn/AGENTS.md`
2. `/Users/zmx/Projects/aiforbn/docs/AGENT_MANIFEST.json`
3. `/Users/zmx/Projects/aiforbn/docs/HANDOFF.md`
4. The nearest module `AGENTS.md` under `src/` when touching a module.
5. The relevant existing plain-text project guidance under `/Users/zmx/Projects/aiforbn/skills/`.

Use `skills/ai_native_workflow.txt` as the main project workflow summary. Read other `skills/*.txt` only when relevant to the task.

## Project Boundaries

- Treat the repository as an AI-agent-operated research codebase, not a human tutorial.
- Keep generated artifacts, caches, private datasets, credentials, and local runtime files out of commits unless the task explicitly asks.
- Do not edit or re-stage research-plan bundles under `docs/research_plan/` unless the task is explicitly research-plan work.
- Preserve scientific honesty: ranking and screening outputs are prioritization evidence, not discovery claims.

## Validation

Prefer the lightest verification that proves the touched behavior:

1. For larger workflow or architecture edits, run `python3 main.py --verify-agent-contract`.
2. For wiring checks, run `python3 main.py --dry-run`.
3. For module changes, run focused tests around the touched module.
4. If public functions, commands, or module boundaries change, update the nearest `PY_FILES_SUMMARY.md`.

Use conda `quant` by default, following the repo instructions.

## Related Global Skills

Use global skills when the task crosses their scope:

- `$ai-native-projects` for broad AI-native project structure and MCP terminology.
- `$git-sync` for commit/push decisions.
- `$blocking-question-soft-gate` before risky local changes.
