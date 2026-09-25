# Materials

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned read-only context unless the task explicitly authorizes work on those documents.

Owns dataset normalization, candidates, features, model selection, benchmarks, screening, summaries, artifact publication and structure follow-up. Its project dependencies are the public `runtime` and `torch_models` surfaces.

- Formula-only candidate screening cannot consume structure-aware features. Keep overall evaluation and screening model identities separate.
- Keep source/seed/edit-plan/output identity checks and publication-before-marker ordering; see the artifact writer's public summary for the detailed contract.
- `common.py`, `ranking_tables.py`, `structure_artifacts.py`, `structure_helpers.py` and `utils.py` are internal surfaces.
- Extract shared pipeline logic into documented functions here when useful, preserving the linear trace in `main.py`.

Public API: [PY_FILES_SUMMARY.md](PY_FILES_SUMMARY.md). Shared reuse and ownership guidance: [root AGENTS.md](../../AGENTS.md) and [COMMON_FUNCTIONS.md](../../COMMON_FUNCTIONS.md).

Validation: [TESTING.md](../../TESTING.md); focused target from the repository root: `conda run -n quant python -m pytest -q src/materials/tests`.
