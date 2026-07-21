# ui module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This file lists the stable public functions that external code may call from `ui`.
Anything underscore-prefixed or omitted here should be treated as internal.

## streamlit_app.py

- `render_streamlit_app()`
  - Render the Streamlit artifact viewer for the generated project outputs.
  - Includes BN model-role evidence, default-vs-BN-centered rank-stability evidence,
    and the unrelaxed structure follow-up handoff report; absent optional artifacts are skipped.
  - Resolves the configured artifact root and summary-declared execution paths, classifies local provenance as current/stale/unverified, and never marks a missing core bundle or unreadable provenance/summary/manifest current; those identity JSON failures and empty-schema CSVs produce text warnings instead of renderer failures.

## utils.py

- No public functions are currently exposed.

## tests/

- `test_streamlit_app.py`
  - Covers every declared artifact section, completion/provenance failure states, configured/nested paths, and empty-schema CSV handling through a fake renderer while verifying the supported `width='stretch'` dataframe contract.
  - Runs the app through Streamlit's real `AppTest` renderer so import and render failures remain text-verifiable.
