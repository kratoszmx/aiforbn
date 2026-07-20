# runtime module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This file lists the stable public functions and classes that other modules may import from `runtime`.
Anything underscore-prefixed or omitted here should be treated as internal.

## io_utils.py

- `load_config(path)`
  - Compile a trusted Python config without emitting bytecode, return its top-level `CONFIG` dict, and reject user-owned `human_docs/` as executable configuration state.
- `validate_runtime_output_path(path, project_root_path=None, *, required_parent_path=None, reject_leaf_symlink=False, expected_output_kind=None)`
  - Return the canonical path used by writers after enforcing canonical/declared human-doc exclusion, optional configured-root containment, leaf kind and symlink rules, directory-only parent chains, and hardlink rejection. An alternate declared root cannot weaken the canonical guard.
- `configure_matplotlib_cache()`
  - Canonicalize and guard `MPLCONFIGDIR`, then return and export the exact safe path that Matplotlib and JARVIS may use for dependency caches.
- `ensure_runtime_dirs(cfg, project_root_path='.')`
  - Preflight every configured runtime directory, then create them together; invalid file leaves or parent chains fail without partial directory creation.
- `clear_project_cache(project_root_path='.')`
  - Delete real Python/cache directories for an existing project root while preserving every real or symlinked path under user-owned `human_docs/`; rejects any symlink component in the caller-supplied root, skips discovered paths that escape the canonical root, safely skips cache-directory symlinks, and treats already-removed cache paths as a successful concurrent cleanup.
- `read_json_file(path)`
  - Read JSON through the shared `myutils/file_utils/json_io.py` helper.
- `write_json_file(payload, path, ...)`
  - Serialize and encode-check before parent creation, then write through the shared `myutils/file_utils/json_io.py` helper using the canonical guarded path.
- `make_json_safe(value)`
  - Convert numpy/pandas/path-like values into JSON-serializable objects through `myutils`.

## agent_state.py

- `load_agent_manifest(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Load the checked-in machine-readable AI-native manifest.
- `validate_agent_layout(project_root_path='.', manifest=None)`
  - Validate required agent-facing files, exact command and validation-profile mappings, exact active project-skill and retired-guidance records, the human-document ownership policy and its declared instruction-surface markers, source-of-truth surfaces, the exact six-module contracts, local instruction paths, v18 alignment, dependency imports, and known layout warnings.
- `build_agent_state(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable agent-state payload used by `main.py --emit-agent-state` and `main.py --verify-agent-contract`.
- `build_agent_command_index(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable command index, including module dependencies, the human-document policy, and v18 research-plan alignment, used by `main.py --emit-agent-commands`.
- `agent_state_to_json(state)`
  - Serialize an agent-state payload for stdout or logs.
- `write_agent_state(state, path)`
  - Serialize before parent creation, then write an agent-state payload while refusing runtime-state output under any canonical, state-declared, or filesystem-equivalent user-owned `human_docs/` path and any multi-hardlink leaf.

## schema.py

- `DatasetManifest`
  - Pydantic-style schema for normalized dataset-manifest metadata, including the target column that defines processed-cache identity.
- `MaterialRecord`
  - Schema for a normalized material record payload.

## utils.py

- No public functions are currently exposed.
- Internal pure-stdlib helpers centralize filesystem identity/descendant and symlink-component checks for the two runtime guard surfaces; the project-specific human-document policy remains in this repository rather than `myutils`.
