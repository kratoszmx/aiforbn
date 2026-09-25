# Entrypoint and cross-module tests

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned read-only context unless the task explicitly authorizes work on those documents.

`src/tests` covers config, main orchestration, public APIs and manifest-driven validation. It is not a production API.

- Production code does not import test helpers.
- Keep shared test helpers private unless a reusable test-only surface is needed.
- `conftest.py` owns `src/` import setup, cache cleanup and declared-command non-vacuity checks. Preserve the difference between a collected test and a passed call.

Public API: [PY_FILES_SUMMARY.md](PY_FILES_SUMMARY.md). Shared reuse and ownership guidance: [root AGENTS.md](../../AGENTS.md) and [COMMON_FUNCTIONS.md](../../COMMON_FUNCTIONS.md).

Validation: [TESTING.md](../../TESTING.md); focused target from the repository root: `conda run -n quant python -m pytest -q src/tests`.
