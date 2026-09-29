# Validation and test guide

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

## Run all tests

Run from the repository root (the directory containing `main.py`). Use the existing Conda `quant` environment; dependencies are declared in [requirements.txt](requirements.txt) and [docs/AGENT_MANIFEST.json](docs/AGENT_MANIFEST.json). Runtime also needs a local `myutils` checkout containing `file_utils/filesystem.py` and `file_utils/json_io.py`. If it is not ancestor-adjacent, set its root explicitly, for example `export MYUTILS_ROOT=/Users/zmx/Projects/myutils` on this host.

The browser suite uses installed Google Chrome through Playwright's `chrome` channel, with headless DOM/text checks and no screenshots. It starts its own temporary loopback server. Automated tests replace provider calls with fixtures; they need neither a live deployment nor model authentication.

```sh
conda run -n quant python -c 'import sys; print(sys.executable)'
conda run -n quant python main.py --verify-agent-contract
conda run -n quant python main.py --emit-agent-commands
conda run -n quant python main.py --dry-run
conda run --no-capture-output -n quant python -m pytest -q -ra src
```

The interpreter should be inside `envs/quant`. On this host, PATH ordering can make `conda run -n quant python3` resolve to Homebrew Python even though `CONDA_PREFIX` says `quant`. Use the verified `python` executable above or its absolute path. In emitted commands, `python3` means that same environment's interpreter; retain the command's arguments and pytest targets when substituting it. Missing imports under the wrong interpreter do not justify installing packages.

`--no-capture-output` streams progress; default Conda capture hides it until completion. Agent-state cases repeatedly analyze production source, so a full run can take tens of minutes. Give large fixture parameters short explicit pytest IDs so individual-case selection stays within command-line limits.

Contract success is exit 0 with `validation.status == "ok"` and no errors. Inspect warnings separately. Dependency probes run isolated imports with per-probe and aggregate time bounds; missing modules, import failure, timeout and exhausted probe budget describe different failures. The command index is the authority for exact targets and each profile's `requires`/`provides` coverage.

## Choose the smallest sufficient profile

| Change | Emitted profile | Checks |
| --- | --- | --- |
| Docs, project skills, handoff, manifest | `architecture_doc_skill_edit` | Contract, dry-run, focused regression |
| Public functions, dependencies, module/model/artifact logic, separator partner workflow | `module_logic_edit` | Contract, dry-run, focused regression, full `src` suite |
| Scientific pipeline or generated output behavior | `scientific_pipeline_edit` | Contract, dry-run, full `src` suite; full pipeline only when fresh research outputs are required |
| Streamlit imports, rendering or artifact display | `ui_edit` | Contract, focused regression, real Streamlit AppTest |

The commands above cover all profiles. For a smaller change, use the emitted profile's commands with the verified interpreter. The full `src` suite includes the focused regression and UI targets; do not repeat them separately. `ui_edit` / `ui_render_smoke` cover Streamlit only; separator API, browser and monitor changes need the partner tests below and the `module_logic_edit` profile.

Dry-run uses tiny in-memory data, checks candidate generation and feature/model compatibility, and constructs models. It clears project caches and may create runtime directories; it does not train models, download the real dataset, or republish research artifacts. `conftest.py` already clears project caches before pytest, so a separate pre-test cleanup is normally redundant.

## Child-module test map

These are diagnostic/focused commands; use the full emitted profile for a change whose scope requires it.

| Target after `conda run -n quant python -m pytest -q` | Coverage |
| --- | --- |
| `src/tests` | Config defaults, main orchestration/control flags, public API signatures/import boundaries, validation-command non-vacuity |
| `src/runtime/tests` | Schemas, guarded config/IO/cache paths, provenance, manifest/skills/dependency inspection, active-skill profile routing and retired-directory rejection |
| `src/materials/tests` | Dataset/cache/download fixtures, features/splits/models, BN diagnostics, publication/structure contracts, separator/aqueous/electrolyte evidence and experiment selection |
| `src/torch_models/tests` | Invalid-input, fit-state, device-policy and ensemble-seed contracts; actual fit/predict integration is in `src/materials/tests` |
| `src/ui/tests` | Streamlit AppTest; separator API/quota/cache/expiry; headless Chrome forms/languages/themes; response-monitor state and receipt privacy |
| `src` | All of the above; `src/template` has no separate test suite |

File-level maps: [materials](src/materials/PY_FILES_SUMMARY.md#tests), [runtime](src/runtime/PY_FILES_SUMMARY.md), [UI](src/ui/PY_FILES_SUMMARY.md#tests). Data tests stub download responses; a pass is not a live JARVIS availability check. Model tests do not establish scientific quality or GPU success. Live service checks and lifecycle are separate procedures in [SERVICES.md](SERVICES.md).

Test preparation stays beside its consumers: tiny CPU model configuration and the fake separator process are local fixtures/helpers. Keep writer, provenance and viewer rejection tests separate because they guard different entrypoints. No additional shared-helper package or test runner is needed.

## Read results and avoid false proof

- A pass belongs to the tested source/docs tree and interpreter. Record command, scope and result; do not carry old pass counts forward after edits.
- Pytest success is exit 0 with passed tests and no failures/errors. `-ra` explains skips/xfails; `--durations=20` can identify slow tests when needed. Case-alias filesystem tests may skip on case-sensitive hosts; Torch skips mean its model integration was not exercised, and the contract check still requires the declared dependency.
- The three manifest-owned pytest command target sets require at least one passed non-xfail test call. All-skipped, all-xfailed and non-strict XPASS-only exit-0 runs fail this guard; partial target selection and collect-only retain native pytest behavior. The subprocess regression uses only pytest's built-in plugins in a temporary project, with a 30-second limit per invocation.
- Focused success does not prove the full suite. Full-suite success does not refresh historical research outputs or prove physical validity.
- `python main.py` is a research run, not a routine test: it can download data, train the configured model grid, and replace data/artifact outputs. Use it only for the task's required recomputation.
- For worktrees with unrelated edits, validate a candidate containing exactly the intended committed bytes, then preserve unrelated changes during staging.

## Separator partner workflow

Focused diagnostic command:

```sh
conda run --no-capture-output -n quant python -m pytest -q -ra \
  src/materials/tests/test_separator_prototype.py \
  src/materials/tests/test_aqueous_slurry.py \
  src/materials/tests/test_experiment_planning.py \
  src/ui/tests/test_separator_app.py \
  src/ui/tests/test_separator_browser.py \
  src/ui/tests/test_separator_monitor.py
```

The full `src` command already includes these six files:

| Area | Important coverage |
| --- | --- |
| Evidence and numerical estimates | Source hashes/literal readings, 20 BN records, 12 aqueous cases, 38 electrolyte compositions / 125 readings; loading/domain limits, held-out-label isolation, five coating settings |
| Experiment selection | Candidate-label rejection, unseen-outcome mutation, all 1,200 published replay traces, one reproduced seed and grouped numerical errors |
| Provider process | Default/configurable model, answer schema/citations, forbidden tool events, private bounded diagnostics, timeout/interruption cleanup and no retries |
| API | Anonymous structured inputs, independent training-only PP reference, 100-request quota and refusal, persisted caches, expiry, concurrency, invalid/non-finite requests and private provider failures |
| Browser | Four forms and next-round selection at mobile/desktop widths; three languages, input persistence, themes, readable tables and original source excerpts |
| Monitor | Fresh versus cached/model-free output, failure/busy/stale/configuration-change receipts and verified HTTP 410 expiry |

These remain separate evidence cohorts: only six aqueous cases support the viscosity series; PP uses one study. Fixture success does not prove live model responses, public-host health or Supervisor integration. Software test counts are not independent scientific cases.

Only when a research-artifact refresh is explicitly required, reproduce the experiment-selection comparison with:

```sh
PYTHONPATH=src conda run --no-capture-output -n quant python -m materials.experiment_planning
```

This writes comparison artifacts after freezing the protocol. The 100 starts repeat one measured pool; bootstrap intervals do not establish cross-study reliability or actual laboratory time saved.
