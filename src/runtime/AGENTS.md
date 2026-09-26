# Runtime

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned read-only context unless the task explicitly authorizes work on those documents.

Owns trusted config loading, guarded IO/cache handling, provenance, schemas and agent inspection. It has no production dependency on other project modules.

- Keep filesystem/provenance policy in project guards; call `myutils` IO and digest APIs directly after validation. JSON writing uses the shared writer's normalization and staging; explicit multi-output preflight remains `validate_json_payload`.
- Public functions live in `io_utils.py` and `agent_state.py`; shared schemas/role constants live in `schema.py`. Bootstrap paths and underscore-prefixed helpers are implementation details.
- The output guard, writer preflight and viewer assessment have separate failure boundaries; retain each when simplifying.

Public API: [PY_FILES_SUMMARY.md](PY_FILES_SUMMARY.md). Shared reuse and ownership guidance: [root AGENTS.md](../../AGENTS.md) and [COMMON_FUNCTIONS.md](../../COMMON_FUNCTIONS.md).

Validation: [TESTING.md](../../TESTING.md); focused target from the repository root: `conda run -n quant python -m pytest -q src/runtime/tests`.
