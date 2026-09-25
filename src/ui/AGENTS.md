# Artifact viewer

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned read-only context unless the task explicitly authorizes work on those documents.

Owns Streamlit display, with `runtime` as its only project production dependency.

- `streamlit_app.py::render_streamlit_app` is the public entrypoint. Read artifacts here; scientific computation belongs to `materials`.
- Validate persisted provenance, committed bytes and role/path identity before displaying report content. A writer preflight does not replace this check.
- Use text-verifiable AppTest coverage; startup checks and process lifecycle are documented in root `SERVICES.md`.

Public API: [PY_FILES_SUMMARY.md](PY_FILES_SUMMARY.md). Shared reuse and ownership guidance: [root AGENTS.md](../../AGENTS.md) and [COMMON_FUNCTIONS.md](../../COMMON_FUNCTIONS.md).

Validation: [TESTING.md](../../TESTING.md); focused target from the repository root: `conda run -n quant python -m pytest -q src/ui/tests/test_streamlit_app.py`.
