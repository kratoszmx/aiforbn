# template module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This directory is the module template used when a new top-level module is created.

## Public callable surface

- No public functions are currently exposed from `utils.py`.

## Files in this template

- `AGENTS.md`
  - Rules and maintenance guidance that every copied module should refine.
- `PY_FILES_SUMMARY.md`
  - The place where each concrete module documents its public callable functions and classes.
- `utils.py`
  - Module-local reusable helpers, if any.

## Notes

- When you copy this template into a new module, update this file immediately to describe the new module's real public surface.
- Internal helper notes belong in `AGENTS.md`, not here.
