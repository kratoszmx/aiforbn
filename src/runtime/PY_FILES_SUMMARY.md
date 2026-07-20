# runtime module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This file lists the stable public functions and classes that other modules may import from `runtime`.
Anything underscore-prefixed or omitted here should be treated as internal.

## io_utils.py

- `load_config(path)`
  - Load a Python config file and return its top-level `CONFIG` dict.
- `validate_runtime_output_path(path, project_root_path=None, *, required_parent_path=None, reject_leaf_symlink=False, expected_output_kind=None)`
  - Return the canonical path used by writers after enforcing canonical/declared human-doc exclusion, optional configured-root containment, leaf kind and symlink rules, directory-only parent chains, and hardlink rejection. An alternate declared root cannot weaken the canonical guard.
- `ensure_runtime_dirs(cfg, project_root_path='.')`
  - Preflight every configured runtime directory, then create them together; invalid file leaves or parent chains fail without partial directory creation.
- `clear_project_cache(project_root_path='.')`
  - Delete real Python/cache directories for an existing project root while preserving every real or symlinked path under user-owned `human_docs/`; rejects roots that resolve inside the canonical human-doc tree before discovery, safely skips cache-directory symlinks, and treats already-removed cache paths as a successful concurrent cleanup.
- `read_json_file(path)`
  - Read JSON through the shared `myutils/file_utils/json_io.py` helper.
- `write_json_file(payload, path, ...)`
  - Write JSON through the shared `myutils/file_utils/json_io.py` helper after enforcing the runtime output and filesystem-alias boundary.
- `make_json_safe(value)`
  - Convert numpy/pandas/path-like values into JSON-serializable objects through `myutils`.

## agent_state.py

- `load_agent_manifest(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Load the checked-in machine-readable AI-native manifest.
- `validate_agent_layout(project_root_path='.', manifest=None)`
  - Validate required agent-facing files, exact command mappings, the human-document ownership policy and its declared instruction-surface markers, source-of-truth surfaces, the exact six-module path/role/public-surface/local-utils/dependency contracts, local instruction paths, exact v18 source-file and ordered deliverable-chain alignment, validation profiles, dependency imports, and known layout warnings.
- `build_agent_state(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable agent-state payload used by `main.py --emit-agent-state` and `main.py --verify-agent-contract`.
- `build_agent_command_index(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable command index, including module dependencies, the human-document policy, and v18 research-plan alignment, used by `main.py --emit-agent-commands`.
- `agent_state_to_json(state)`
  - Serialize an agent-state payload for stdout or logs.
- `write_agent_state(state, path)`
  - Write an agent-state payload to a JSON file while refusing runtime-state output under the canonical or state-declared user-owned `human_docs/` and existing file leaves with multiple hard links; the payload cannot redirect the canonical guard.

## schema.py

- `DatasetManifest`
  - Pydantic-style schema for normalized dataset-manifest metadata, including the target column that defines processed-cache identity.
- `MaterialRecord`
  - Schema for a normalized material record payload.

## utils.py

- No public functions are currently exposed.
