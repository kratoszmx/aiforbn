---
name: aiforbn-workflow
description: Use for routine work in /Users/zmx/Projects/aiforbn, including AI-native architecture, AGENT_MANIFEST, HANDOFF, PY_FILES_SUMMARY, project skills, validation-profile selection, materials pipeline changes, model wiring, tests, and artifact/reporting maintenance.
---

# AI-for-BN Workflow

Use this as the repo-scoped dispatcher for `aiforbn`. It routes agents to the smallest reliable context and validation path. Do not treat it as a tutorial.

## First Reads

Before edits:

1. `/Users/zmx/Projects/aiforbn/AGENTS.md`
2. `/Users/zmx/Projects/aiforbn/docs/AGENT_MANIFEST.json`
3. `/Users/zmx/Projects/aiforbn/docs/HANDOFF.md`
4. `/Users/zmx/Projects/aiforbn/skills/ai_native_workflow.txt`
5. The nearest module `AGENTS.md` when touching `src/**`

Use `python3 main.py --emit-agent-commands` to choose validation commands without rereading long prose.

## Dispatch

- Architecture/docs/skill/manifest edits: keep changes machine-readable and run the architecture validation profile.
- Materials or model logic edits: update the nearest `PY_FILES_SUMMARY.md`, run focused tests, then `python3 -m pytest -q src` when dependencies are available.
- Research-plan or Overleaf delivery work: switch to `$aiforbn-overleaf-proposal`.
- Generated artifact refresh: only run full `python3 main.py` when the task needs regenerated artifacts or scientific behavior changed.

## Boundaries

- Optimize for agent search, execution, verification, rollback, and handoff only.
- Do not optimize for manual use, notebooks, onboarding, or UI comfort.
- Do not edit `docs/research_plan/` unless the task is explicitly proposal/research-plan work.
- Do not commit caches, credentials, private datasets, or large generated artifacts without explicit task intent.
- Preserve scientific honesty: ranking output is prioritization evidence, not discovery.
- Do not restore retired guidance shards under `skills/`; the active plain-text guidance is `skills/ai_native_workflow.txt`.

## Delegation

- Use `$blocking-question-soft-gate` before risky local state changes.
- Use `$small-fast-coding` / `spark_coder` only for narrow, low-risk, easily reviewed code slices.
- Main Codex owns diff review, tests, staging, commit, and push.
