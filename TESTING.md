# Validation and test guide

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

## Run all tests

Run from the repository root. Use the existing Conda `quant` environment; dependency declarations are in [requirements.txt](requirements.txt) and [docs/AGENT_MANIFEST.json](docs/AGENT_MANIFEST.json). Runtime also needs a local `myutils` checkout containing `file_utils/filesystem.py` and `file_utils/json_io.py`. If it is not ancestor-adjacent, set `MYUTILS_ROOT` to that checkout's root.

```sh
conda run -n quant python -c 'import sys; print(sys.executable)'
conda run -n quant python main.py --verify-agent-contract
conda run -n quant python main.py --emit-agent-commands
conda run -n quant python main.py --dry-run
conda run -n quant python -m pytest -q -ra src
```

The interpreter should be inside `envs/quant`. On this host, PATH ordering can make `conda run -n quant python3` resolve to Homebrew Python even though `CONDA_PREFIX` says `quant`. Use the verified `python` executable above or its absolute path. In emitted commands, `python3` means that same environment's interpreter; retain the command's arguments and pytest targets when substituting it. Missing imports under the wrong interpreter do not justify installing packages.

Contract success is exit 0 with `validation.status == "ok"` and no errors. Inspect warnings separately. Dependency probes run isolated imports with per-probe and aggregate time bounds; missing modules, import failure, timeout and exhausted probe budget describe different failures. The command index is the authority for exact targets and each profile's `requires`/`provides` coverage.

## Choose the smallest sufficient profile

| Change | Emitted profile | Checks |
| --- | --- | --- |
| Docs, project skills, handoff, manifest | `architecture_doc_skill_edit` | Contract, dry-run, focused regression |
| Public functions, dependencies, module/model/artifact logic | `module_logic_edit` | Contract, dry-run, focused regression, full `src` suite |
| Scientific pipeline or generated output behavior | `scientific_pipeline_edit` | Contract, dry-run, full `src` suite; full pipeline only when fresh research outputs are required |
| UI imports, rendering or artifact display | `ui_edit` | Contract, focused regression, real Streamlit AppTest |

The commands above cover all profiles. For a smaller change, use the emitted profile's commands with the verified interpreter. The full `src` suite includes the focused regression and UI targets; when running it, there is no need to execute those same pytest targets separately.

Dry-run uses tiny in-memory data, checks candidate generation and feature/model compatibility, and constructs models. It clears project caches and may create runtime directories; it does not train models, download the real dataset, or republish research artifacts. `conftest.py` already clears project caches before pytest, so a separate pre-test cleanup is normally redundant.

## Child-module test map

These are diagnostic/focused commands; use the full emitted profile for a change whose scope requires it.

| Target after `conda run -n quant python -m pytest -q` | Coverage |
| --- | --- |
| `src/tests` | Config defaults, main orchestration/control flags, public API signatures/import boundaries, validation-command non-vacuity |
| `src/runtime/tests` | Schemas, guarded config/IO/cache paths, provenance, manifest/skills/dependency inspection |
| `src/materials/tests` | Dataset/cache/download-failure fixtures, features and splits, model selection, BN diagnostics, ranking, publication and structure contracts |
| `src/torch_models/tests` | Invalid-input, fit-state, device-policy and ensemble-seed contracts; actual fit/predict integration is in `src/materials/tests` |
| `src/ui/tests/test_streamlit_app.py` | Real Streamlit AppTest, provenance suppression, artifact roles, malformed/missing content |
| `src` | All of the above; `src/template` has no separate test suite |

Materials coverage is split across `test_data.py`, `test_bn_filter.py`, `test_features_pipeline.py`, `test_diagnostic_edge_cases.py`, `test_reporting.py`, and `test_structure_execution_contracts.py`. Data tests stub download responses; a pass is not a live JARVIS availability check. Model tests do not establish scientific quality or GPU success. UI tests prove renderer behavior, while a startup-wiring change can additionally use the bounded loopback procedure in [SERVICES.md](SERVICES.md).

Test preparation stays beside its consumers: model integration cases share a private tiny CPU configuration helper, and each model has its own pytest case. Keep writer, provenance and viewer rejection tests separate because they guard different entrypoints. No additional shared-helper package or test runner is needed.

## Read results and avoid false proof

- A pass belongs to the tested source/docs tree and interpreter. Record command, scope and result; do not carry old pass counts forward after edits.
- Pytest success is exit 0 with passed tests and no failures/errors. `-ra` explains skips/xfails; `--durations=20` can identify slow tests when needed. Case-alias filesystem tests may skip on case-sensitive hosts; Torch skips mean its model integration was not exercised, and the contract check still requires the declared dependency.
- The three manifest-owned pytest command target sets require at least one passed non-xfail test call. All-skipped, all-xfailed and non-strict XPASS-only exit-0 runs fail this guard; partial target selection and collect-only retain native pytest behavior. The subprocess regression uses only pytest's built-in plugins in a temporary project, with a 30-second limit per invocation.
- Focused success does not prove the full suite. Full-suite success does not refresh historical research outputs or prove physical validity.
- `python main.py` is a research run, not a routine test: it can download data, train the configured model grid, and replace data/artifact outputs. Use it only for the task's required recomputation.
- For worktrees with unrelated edits, validate a candidate containing exactly the intended committed bytes, then preserve unrelated changes during staging.
