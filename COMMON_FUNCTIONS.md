# Shared functions and API routing

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

This is a flat source tree, not an installed package. `main.py` and `conftest.py` add `src/` to the import path. A standalone caller can set `PYTHONPATH=src` from the repository root, then import from the concrete module:

```sh
PYTHONPATH=src conda run -n quant python -c 'from runtime.io_utils import load_config; print(load_config("src/config.py")["project"]["name"])'
```

## Shared runtime helpers

Definitions and full signatures are in [src/runtime/io_utils.py](src/runtime/io_utils.py) and [its public summary](src/runtime/PY_FILES_SUMMARY.md).

| Callable | Input → output / effect |
| --- | --- |
| `load_config` | Trusted Python config path → `CONFIG` dict; rejects executable state under `human_docs/` |
| `validate_runtime_output_path` | Path plus optional parent/kind/alias constraints → guarded canonical `Path` |
| `ensure_runtime_dirs` | Config plus project root → preflight all runtime directories, then create them |
| `configure_matplotlib_cache` | `MPLCONFIGDIR` or safe temporary default → guarded `Path`, also exported to the environment |
| `clear_project_cache` | Existing project root → remove real cache directories within scope, preserve protected paths and symlinked caches |
| `read_json_file` | JSON path → decoded payload |
| `make_json_safe` | NumPy/pandas/path-like values → JSON-safe values |
| `validate_json_payload` | Payload and serialization options → `None` on success, exception on invalid serialization; no write |
| `write_json_file` | Payload, path and serialization options → guarded write after serialization preflight |
| `build_artifact_provenance` | Config, dataset manifest, successfully published paths → v2 source/config/dataset/output identity dict |
| `assess_artifact_provenance` | Stored provenance plus current config/dataset/root → assessment dict (`current`, `stale`, `unverified`) |

The project wrappers reuse `myutils/file_utils/filesystem.py` and `json_io.py`. Runtime locates an ancestor-adjacent `myutils` checkout or uses `MYUTILS_ROOT`. The shared API entrypoint is `/Users/zmx/Projects/myutils/docs/PUBLIC_API.md`; retain project path, human-document, and provenance guards when adopting shared functions.

## Module API map

| Owner | Import/use | Contract |
| --- | --- | --- |
| Agent inspection | `runtime.agent_state`: manifest/layout/state/command-index functions; project root → JSON-ready dicts or serialized state | [runtime summary](src/runtime/PY_FILES_SUMMARY.md) |
| Schemas | `runtime.schema`: dataset/material schemas and fixed/dynamic artifact role constants | [runtime summary](src/runtime/PY_FILES_SUMMARY.md) |
| Scientific workflow | `materials.data`, `candidate_space`, `feature_building`, `modeling`, `selection`, `benchmarking`, `screening`, `summary`, `artifacts`, `plots`, `structure_execution` | [materials summary](src/materials/PY_FILES_SUMMARY.md), [callable inputs/outputs](docs/PY_FILES_SUMMARY.md) |
| Neural models | Usually `materials.modeling.make_model`; direct `torch_models` imports when needed, numeric matrix/target → fitted regressor/predictions | [model summary](src/torch_models/PY_FILES_SUMMARY.md) |
| Artifact display | `ui.streamlit_app.render_streamlit_app()` in Streamlit → text-verifiable view of a current committed bundle | [UI summary](src/ui/PY_FILES_SUMMARY.md) |

`utils.py` files currently expose no public API. Runtime's private path helpers enforce project-specific boundaries; the other module utility files are empty template slots. `src/tests/` and `src/template/` expose no production callables. Dependencies and exact symbol/signature checks are covered by the manifest and public-surface tests.
