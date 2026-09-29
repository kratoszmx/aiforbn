# Artifact viewer

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned read-only context unless the task explicitly authorizes work on those documents.

Owns the Streamlit artifact viewer and the separator partner web demonstration. Project dependencies are `runtime` and the documented separator APIs in `materials`.

- `streamlit_app.py::render_streamlit_app` is the public entrypoint. Read artifacts here; scientific computation belongs to `materials`.
- Validate persisted provenance, committed bytes and role/path identity before displaying report content. A writer preflight does not replace this check.
- Use text-verifiable AppTest coverage; startup checks and process lifecycle are documented in root `SERVICES.md`.
- `separator_app.py` serves the AI for Science partner site, curated public datasets and explicit downloads. Anonymous numerical analysis accepts bounded structured inputs, expires with the deployment, and enforces a persisted 100-new-analysis rolling daily budget. Keep provider identity, credentials and diagnostics private under ignored `.runtime/`; the public interface labels estimates and observations and uses readable tables.

Public API: [PY_FILES_SUMMARY.md](PY_FILES_SUMMARY.md). Shared reuse and ownership guidance: [root AGENTS.md](../../AGENTS.md) and [COMMON_FUNCTIONS.md](../../COMMON_FUNCTIONS.md).

Validation: [TESTING.md](../../TESTING.md); focused target from the repository root: `conda run -n quant python -m pytest -q src/ui/tests`. This includes Streamlit, the separator API/browser and the response monitor; the emitted `ui_render_smoke` command covers only Streamlit.
