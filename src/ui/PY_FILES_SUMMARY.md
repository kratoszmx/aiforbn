# ui module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This file lists the stable public functions that external code may call from `ui`.
Anything underscore-prefixed or omitted here should be treated as internal.

## separator_app.py

- `create_separator_app(data_path=DATASET_PATH, runtime_dir=None, model_executable=None)`
  - Construct the AI for Science partner web app: source-verified BN, aqueous-slurry and electrolyte data, public tables/CSV exports, readable source locations and sequential-replay comparison. Five structured tasks cover water-based slurry viscosity, BNNT conductivity, CA/BN thickness, liquid-electrolyte conductivity and next-experiment selection. Anonymous requests obey expiry, one-call concurrency, exact-input/configuration caching and a persisted default 100-new-analysis rolling 24-hour limit; failures return no fabricated prediction. The provider model is trusted local configuration and never public response metadata. The public page supports persisted light/dark preference, task-specific estimated waits and three user-facing sections.

## streamlit_app.py

- `render_streamlit_app()`
  - Render the Streamlit artifact viewer for the generated project outputs.
  - Includes BN model-role evidence, default-vs-BN-centered rank-stability evidence,
    and the unrelaxed structure follow-up handoff report; absent optional artifacts are skipped.
  - Resolves the configured artifact root and execution paths, then accepts a summary declaration only when it identifies that same guarded file. Wrong-shaped nested summary objects, fixed/cross-role relabeling, invalid suffixes, missing paths, and paths absent from the v2 commitment fail closed; absent/null/empty containers retain a valid configured baseline without reviving stale outputs.
  - Shares the canonical three-role summary/config/suffix/default mapping and fixed-report filename contract with the structure builder and materials writer while independently validating persisted or post-publication state.
  - Verifies v2 committed output bytes and renders report content only after the viewer's final assessment is current with a concrete committed-path set. Missing, changed, malformed, legacy, incomplete, or uncommitted-known bundles stay non-green and render no report tables; JSON/CSV read failures produce text warnings instead of renderer failures.

## utils.py

- No public functions are currently exposed.

## tests/

- Run `conda run -n quant python -m pytest -q src/ui/tests/test_streamlit_app.py` from the repository root. [TESTING.md](../../TESTING.md) covers prerequisites/profiles; [SERVICES.md](../../SERVICES.md) covers optional loopback startup, health and shutdown.
- `test_streamlit_app.py`
  - Covers the source-derived fixed/dynamic render inventory, completion/provenance/content-mutation states, configured/nested path transitions, nested object-shape matrices, guarded file-identity and role matching, unrelated-extra tolerance, and malformed JSON/CSV handling while verifying the supported `width='stretch'` dataframe contract.
  - Runs the app through Streamlit's real `AppTest` renderer, including asymmetric BN slice/family prediction states and malformed/legacy/non-current provenance suppression, so import and render failures remain text-verifiable.

## separator_monitor.py

- `HTTP_INTERVAL = 300`, `MODEL_INTERVAL = 21600`: HTTP / new-provider-call cadence.
- `monitor_status(runtime_dir=RUNTIME, *, now=None)`: project a deployment-bound, fresh receipt to status/code/evidence fields; no network or model access. Unknown, failed and scheduled-expiry states remain distinct.
- `check_separator_service(runtime_dir=RUNTIME, *, client=None, now=None)`: lock one monitor, check both origins/page, periodically make an uncached public numerical call, and atomically persist private minimal evidence. Uses the same public quota and inference lock; no repair or messages.
- CLI adds result-v1 `schema_version=1`; `--status` is Supervisor's read-only entry. Tests cover cached/malformed/failed responses, timing, changed configuration, in-progress/interrupted calls and verified expiry.

API v4 uses the PP linear reference for the displayed number with a separate
language-model explanation; private receipts retain the model's raw guess. The
planner returns one next recipe to match sequential replay. Dispersion literature
reviews enrich source descriptions without becoming PP training labels. New
aqueous numeric inference uses six matched patent cases; the other aqueous
cases remain separately identified references. Public evidence contains a paper
title and section instead of an XML locator. Model notice sentences are removed
from explanations without deleting tentative scientific wording or failure cases.
