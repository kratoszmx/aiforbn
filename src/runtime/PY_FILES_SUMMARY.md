# runtime module public surface

This file lists the stable public functions and classes that other modules may import from `runtime`.
Anything underscore-prefixed or omitted here should be treated as internal.

## io_utils.py

- `load_config(path)`
  - Load a Python config file and return its top-level `CONFIG` dict.
- `ensure_runtime_dirs(cfg)`
  - Create the configured runtime directories needed by the project.
- `clear_project_cache(project_root_path='.')`
  - Delete Python/cache artifacts for an existing project root; treats already-removed cache paths as a successful concurrent cleanup.
- `read_json_file(path)`
  - Read JSON through the shared `myutils/file_utils/json_io.py` helper.
- `write_json_file(payload, path, ...)`
  - Write JSON through the shared `myutils/file_utils/json_io.py` helper.
- `make_json_safe(value)`
  - Convert numpy/pandas/path-like values into JSON-serializable objects through `myutils`.

## agent_state.py

- `load_agent_manifest(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Load the checked-in machine-readable AI-native manifest.
- `validate_agent_layout(project_root_path='.', manifest=None)`
  - Validate required agent-facing files, source-of-truth surfaces, command surfaces, module contracts, exact v18 source-file and ordered deliverable-chain alignment, validation profiles, dependency imports, and known layout warnings.
- `build_agent_state(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable agent-state payload used by `main.py --emit-agent-state` and `main.py --verify-agent-contract`.
- `build_agent_command_index(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable command index, including v18 research-plan alignment, used by `main.py --emit-agent-commands`.
- `agent_state_to_json(state)`
  - Serialize an agent-state payload for stdout or logs.
- `write_agent_state(state, path)`
  - Write an agent-state payload to a JSON file.

## schema.py

- `DatasetManifest`
  - Pydantic-style schema for normalized dataset-manifest metadata.
- `MaterialRecord`
  - Schema for a normalized material record payload.

## utils.py

- No public functions are currently exposed.
