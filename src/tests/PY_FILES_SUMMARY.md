# tests module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

`src/tests` is a pytest coverage directory, not a production API module.

## Public callable surface

- No stable external callable functions are exposed for production use.

## What lives here

- `test_config.py`
  - Validates the real `src/config.py` defaults, including that mandatory formula screening has no false top-level `screening.enabled` master switch while its explicit nested gates remain independently testable.
- `test_main.py`
  - Covers the complete top-level orchestration branch, `--dry-run`, and all JSON control-plane commands, including requested-path `--write-agent-state`, relocated checkout basenames, and operation without runtime `myutils` imports.
- `test_public_surfaces.py`
  - Verifies explicit cross-module imports are documented, production and root-entrypoint imports follow public/private boundaries, every implemented control flag is manifested, and every module/root documented symbol and callable signature exists in its declared file; root-summary callable parsing must be nonempty for `main.py`, runtime, materials, and UI.
- `utils.py`
  - Currently exposes no public helper functions.

## Notes

- Production code must not import from `src/tests`.
- Keep shared test helpers private unless a clearly reusable test-only helper emerges.
