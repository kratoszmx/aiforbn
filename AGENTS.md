# AI for Science — agent entrypoint

The public-facing name is **AI for Science**; the local directory, repository and internal identifiers remain `aiforbn`. Choose the flow before running commands:

| Flow | Purpose and entry |
| --- | --- |
| Partner separator/dispersion prototype | Source-backed formulation lookup, four bounded numerical tasks, model explanation and next-experiment selection. Start at the [prototype index](docs/research/separator_prototype/INDEX.md); service lifecycle is in [SERVICES.md](SERVICES.md). |
| Historical BN band-gap PoC | `main.py` loads 2D-material data, evaluates formula/family holdouts, ranks formula-only candidates and builds unrelaxed prototypes. Ranking/prototypes do not establish discovery, stability, synthesizability or a direct gap. |

Keep BN separator, aqueous and electrolyte cohorts separate. Historical band-gap labels do not enter partner tasks; replay savings are conditional retrospective estimates, not prospective laboratory validation or BN gains.

## First useful run

Run from the repository root with Conda `quant` and the local `myutils` checkout available:

```sh
conda run -n quant python -c 'import sys; print(sys.executable)'
conda run -n quant python main.py --verify-agent-contract
conda run -n quant python main.py --emit-agent-commands
conda run -n quant python main.py --dry-run
```

The first command should resolve inside `envs/quant`. [TESTING.md](TESTING.md) explains interpreter/PATH diagnosis, the `MYUTILS_ROOT` override, validation profiles, and results. Inspection emits JSON; the band-gap dry-run checks config, candidate features, and model construction without training or rewriting research outputs. It can clear caches and create configured runtime directories. Partner API, browser and monitor checks are listed separately in `TESTING.md`.

For an authorized artifact refresh, `conda run -n quant python main.py` runs the complete pipeline using [src/config.py](src/config.py). A raw-cache miss can download the dataset; the run writes data caches and `artifacts/`. Inspect provenance before interpreting existing results.

## Find the right context

| Need | Entry |
| --- | --- |
| Current progress and next work | [HANDOFF.md](HANDOFF.md) |
| Proposed research next steps after the teacher meeting | [docs/research/next_steps_en.md](docs/research/next_steps_en.md); user copy: [human_docs/next_steps_zh.md](human_docs/next_steps_zh.md) |
| Recording/transcript provenance and research source leads | [official_docs/INDEX.md](official_docs/INDEX.md) |
| Scientific/publication boundaries; deferred work | [docs/HANDOFF.md](docs/HANDOFF.md) |
| Commands, dependencies, module ownership, profiles | [docs/AGENT_MANIFEST.json](docs/AGENT_MANIFEST.json) |
| Shared helpers, imports, inputs/outputs | [COMMON_FUNCTIONS.md](COMMON_FUNCTIONS.md) |
| Python callable index | [docs/PY_FILES_SUMMARY.md](docs/PY_FILES_SUMMARY.md), then the module's summary |
| Tests, including each child module | [TESTING.md](TESTING.md) |
| Services, optional viewer, MCP ownership | [SERVICES.md](SERVICES.md) |
| Maintenance decisions | [.agents/skills/aiforbn-workflow/SKILL.md](.agents/skills/aiforbn-workflow/SKILL.md) |
| Authorized proposal/Overleaf work | [.agents/skills/aiforbn-overleaf-proposal/SKILL.md](.agents/skills/aiforbn-overleaf-proposal/SKILL.md) |

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

`src/materials/` owns data, features, models, evaluation, screening, report/structure artifacts and separator evidence/inference; `src/runtime/` owns config, guarded IO, schemas and agent inspection; `src/torch_models/` owns sklearn-style neural regressors; `src/ui/` owns the optional artifact viewer and the partner separator web app. `src/tests/` covers entrypoints and cross-module contracts; `src/template/` is a starting point for new modules. Local tests sit under each production module's `tests/` directory.
