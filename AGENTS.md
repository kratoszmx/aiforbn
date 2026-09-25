# aiforbn — agent entrypoint

`aiforbn` is a research PoC for boron-nitride (BN) themed materials screening. It loads 2D-material data, predicts band gaps, evaluates formula/family holdouts, ranks formula-only candidates, and builds deterministic unrelaxed structure prototypes for follow-up. Ranking and prototype generation do not establish discovery, stability, synthesizability, or a direct band gap.

## First useful run

Run from the repository root with Conda `quant` and the local `myutils` checkout available:

```sh
conda run -n quant python -c 'import sys; print(sys.executable)'
conda run -n quant python main.py --verify-agent-contract
conda run -n quant python main.py --emit-agent-commands
conda run -n quant python main.py --dry-run
```

The first command should resolve inside `envs/quant`. [TESTING.md](TESTING.md) explains interpreter/PATH diagnosis, the `MYUTILS_ROOT` override, validation profiles, and results. Inspection emits JSON; dry-run checks config, candidate features, and model construction without training or rewriting research outputs. It can clear caches and create configured runtime directories.

For an authorized artifact refresh, `conda run -n quant python main.py` runs the complete pipeline using [src/config.py](src/config.py). A raw-cache miss can download the dataset; the run writes data caches and `artifacts/`. Inspect provenance before interpreting existing results.

## Find the right context

| Need | Entry |
| --- | --- |
| Current progress and next work | [HANDOFF.md](HANDOFF.md) |
| Scientific/publication boundaries; deferred work | [docs/HANDOFF.md](docs/HANDOFF.md) |
| Commands, dependencies, module ownership, profiles | [docs/AGENT_MANIFEST.json](docs/AGENT_MANIFEST.json) |
| Shared helpers, imports, inputs/outputs | [COMMON_FUNCTIONS.md](COMMON_FUNCTIONS.md) |
| Python callable index | [docs/PY_FILES_SUMMARY.md](docs/PY_FILES_SUMMARY.md), then the module's summary |
| Tests, including each child module | [TESTING.md](TESTING.md) |
| Services, optional viewer, MCP ownership | [SERVICES.md](SERVICES.md) |
| Maintenance decisions | [.agents/skills/aiforbn-workflow/SKILL.md](.agents/skills/aiforbn-workflow/SKILL.md) |
| Authorized proposal/Overleaf work | [.agents/skills/aiforbn-overleaf-proposal/SKILL.md](.agents/skills/aiforbn-overleaf-proposal/SKILL.md) |
| Compact runtime routing | [skills/ai_native_workflow.txt](skills/ai_native_workflow.txt) |

The two `docs/` index paths remain because runtime validation and public-surface tests consume them. Root documents provide short task-oriented entrypoints; module summaries own detailed API behavior. Git history holds completed maintenance chronology.

## Ownership and working boundaries

- `HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`. Everything under `human_docs/` is user-owned context. Editing, moving, deleting, regenerating, or staging it requires an explicit task for those documents. General repository maintenance does not grant that scope.
- Keep credentials, private datasets, environment files, caches, and scratch output out of commits. Review generated data/artifacts against the authorized research task before staging them.
- Preserve unrelated worktree edits and stage only owned paths or hunks. Work on `main` and verify the existing remote refs after synchronization.
- Prefer text and reproducible commands for execution, verification, rollback, and handoff. `AGENTS.md` is the entrypoint; a root README or manual onboarding layer is unnecessary.
- Read the nearest module `AGENTS.md` when touching `src/**`. Cross-module calls use documented public APIs and the manifest's dependency directions; underscore-prefixed helpers remain internal.
- When changing helpers, consult local `utils.py` and the `myutils` API for relevant reuse. Move code between repositories only when task scope and behavioral compatibility justify it; project-specific policy stays local.
- Keep `main.py` traceable as a linear pipeline. Update the nearest `PY_FILES_SUMMARY.md` when public callables or module boundaries change, and choose validation through the emitted command index.

## Source map

`src/materials/` owns data, features, models, evaluation, screening and report/structure artifacts; `src/runtime/` owns config, guarded IO, schemas and agent inspection; `src/torch_models/` owns sklearn-style neural regressors; `src/ui/` is an optional artifact viewer. `src/tests/` covers entrypoints and cross-module contracts; `src/template/` is a starting point for new modules. Local tests sit under each production module's `tests/` directory.
