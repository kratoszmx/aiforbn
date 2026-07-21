# ui module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This file lists the stable public functions that external code may call from `ui`.
Anything underscore-prefixed or omitted here should be treated as internal.

## streamlit_app.py

- `render_streamlit_app()`
  - Render the Streamlit artifact viewer for the generated project outputs.
  - Includes BN model-role evidence, default-vs-BN-centered rank-stability evidence,
    and the unrelaxed structure follow-up handoff report; absent optional artifacts are skipped.
  - Resolves the configured artifact root and summary-declared execution paths, verifies v2 committed output bytes before rendering a committed bundle, skips non-current or uncommitted known outputs, and never marks missing/changed/malformed committed content current; JSON/CSV read failures produce text warnings instead of renderer failures.

## utils.py

- No public functions are currently exposed.

## tests/

- `test_streamlit_app.py`
  - Covers every declared artifact section, completion/provenance/content-mutation states, configured/nested paths, unrelated-extra tolerance, and malformed JSON/CSV handling while verifying the supported `width='stretch'` dataframe contract.
  - Runs the app through Streamlit's real `AppTest` renderer, including the mixed-bundle matrix, so import and render failures remain text-verifiable.
