# Module template

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned read-only context unless the task explicitly authorizes work on those documents.

This directory supplies a starting shape for a new module; it has no production callable surface.

- Adapt `AGENTS.md`, `PY_FILES_SUMMARY.md` and `utils.py` to the actual module instead of copying generic rules unchanged.
- Record public callables, module dependencies, test locations and the relevant manifest changes together.
- Keep project-wide rules in root `AGENTS.md`; add only module-specific context here.

Public API: [PY_FILES_SUMMARY.md](PY_FILES_SUMMARY.md). Shared reuse and ownership guidance: [root AGENTS.md](../../AGENTS.md) and [COMMON_FUNCTIONS.md](../../COMMON_FUNCTIONS.md).

Validation: [TESTING.md](../../TESTING.md); this template has no separate suite, so validate a new module's behavior and public-surface/manifest integration.
