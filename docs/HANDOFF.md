# Agent Handoff

## Current state

- Project: `aiforbn`, a research-grade AI-for-BN demonstration repository maintained for autonomous agents.
- Default environment: conda `quant`; run from the repository root.
- `HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`
- No active repository blocker is recorded. Start new work from a reproduced defect or an explicit scientific-delivery request, not from historical maintenance chronology.
- Git history is the forensic record for completed maintenance rounds; this file records only current operational truth.

## Scientific contract

- The design space is bounded and BN-centered. Candidate ranking is low-confidence formula-level follow-up prioritization, not open-ended material discovery.
- Stage 1 screening must use candidate-compatible formula-only features. The overall evaluation model may use lightweight structure-aware features and may differ from the screening model.
- Formula-stage stability, application relevance, and directness are conservative proxies. Structure-dependent claims begin only after a structure hypothesis exists and passes the applicable checks.
- Current structure outputs are deterministic, unrelaxed prototypes and validation-ready handoff evidence. They are not structure relaxation, experimental synthesis, thermodynamic-stability proof, direct-gap proof, or discovery evidence.
- Grouped-by-formula, BN formula/family holdout, BN-vs-non-BN, uncertainty, domain-support, novelty, and action-label diagnostics are evidence layers; none promotes a formula-stage result into a structure or experimental claim.
- Default model/config truth belongs to `src/config.py`. Fractional-attention, sparse-attention, and Roost-like implementations remain experimental rather than default-mainline evidence; historical pilot results do not justify a GPU-success claim.

The machine-readable v18 anchors, non-claims, and deliverable chain are canonical in `docs/AGENT_MANIFEST.json`. Read the human research context only when the task requires it and never treat it as agent-owned state.

## Authority map

| Need | Canonical source |
|---|---|
| Repository entry, ownership, safety | `AGENTS.md` |
| Entrypoints, modules, dependencies, profiles, v18 boundaries | `docs/AGENT_MANIFEST.json` |
| Routine execution and profile selection | `.agents/skills/aiforbn-workflow/SKILL.md` and `skills/ai_native_workflow.txt` |
| Proposal/Overleaf delivery | `.agents/skills/aiforbn-overleaf-proposal/SKILL.md` |
| Public Python callables and signatures | root and nearest module `PY_FILES_SUMMARY.md` |
| Runtime defaults | `src/config.py` |
| Historical changes and rollback | Git commits and diffs |

Do not duplicate exact commands, dependency lists, public signatures, or round narratives here when their canonical owner above is machine-checkable.

## Architecture and artifact truth

- `main.py` remains the linear, agent-traceable pipeline entrypoint.
- Production ownership is `runtime`, `materials`, `torch_models`, and `ui`; manifest records for `tests` and `template` are non-production contract surfaces.
- Allowed production dependencies are `runtime -> []`, `torch_models -> []`, `ui -> [runtime]`, and `materials -> [runtime, torch_models]`.
- Runtime reuses the stable `myutils` filesystem and JSON APIs behind project-specific path, identity, and human-document guards. Dependency availability is proven by timeout-bounded isolated imports whose successes are cached only for an opaque import-environment, owner-module, and manifest-owned consumer-target identity; ordered preloads mirror actual project import context. Project-specific artifact, dependency-contract, and AST semantics remain local.
- Successful artifact publication uses v2 source/config/dataset/output identity, commits actual published bytes by relative path and SHA-256, and writes the completion marker last. Missing, malformed, mismatched, legacy, or uncommitted-known output state is non-current and must not render report content.
- Structure summary, writer, and viewer roles derive from the shared runtime contract. Writer semantic preflight happens before mutation; the viewer independently validates persisted state.

## Safety boundaries

- Everything under `human_docs/` is user-owned and read-only unless the current task explicitly requests human-document work. Do not edit, move, regenerate, stage, normalize, reclassify, or copy its contents into agent-owned history.
- Treat `data/` and `artifacts/` as scientific/generated state. Do not regenerate or commit them during ordinary maintenance; run the full pipeline only when the task explicitly requires refreshed artifacts and review the resulting diff.
- Do not commit credentials, private datasets, caches, local environment state, or scratch outputs.
- An installed local package, authorization state, or historical artifact is not current project truth unless it is declared by the machine contract and verified in the active environment.

## Validated checkpoint

Latest validated tree (2026-07-24):

- agent contract, nine-field command-index parity, and ordered pytest target rendering: `ok`, 0 errors, 0 warnings;
- dry-run pipeline wiring: passed;
- emitted architecture/docs focused profile: 513 passed;
- cache-disabled collection/full `src` suite: 1068 collected, 1068 passed;
- manifest pytest non-vacuity regression: all three declared commands reject zero-call exit-0 runs while preserving partial, collect-only, failure, interrupt, and no-test outcomes;
- warning classification: one upstream PyTorch nested-tensor prototype warning, no project warning regression;
- Streamlit AppTest: 104 passed;
- bounded loopback renderer: health 200, root 200, clean shutdown, zero remaining listener;
- both repo skills valid; external-cache compile, diff, residue, and protected-tree checks clean.

These checks do not regenerate scientific artifacts. Select commands from `python3 main.py --emit-agent-commands`; do not copy this snapshot forward after behavior or test inventory changes without rerunning the exact affected checks.

## Resume and recovery

1. Read `AGENTS.md`, `docs/AGENT_MANIFEST.json`, this file, `skills/ai_native_workflow.txt`, and the relevant repo skill.
2. Run `python3 main.py --emit-agent-commands` and `python3 main.py --verify-agent-contract`.
3. Classify the task and select the smallest emitted validation profile. For `src/**` work, also read the nearest module `AGENTS.md` and `PY_FILES_SUMMARY.md`.
4. Keep scientific regeneration separate from code/docs maintenance. Use artifact provenance rather than file presence to judge currentness.
5. Before handoff, review the complete diff, validate the frozen tree, stage only intentional paths, synchronize the existing remotes, and prove clean ref equality.

There is no separate active chronology or archive document. Use `git log -- docs/HANDOFF.md` and the relevant commit diff for forensic recovery.
