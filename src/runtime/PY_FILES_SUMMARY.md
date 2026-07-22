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
  - Treat unset or blank `MPLCONFIGDIR` as the safe temporary default, then canonicalize, guard, return, and export the exact path used by Matplotlib and JARVIS.
- `ensure_runtime_dirs(cfg, project_root_path='.')`
  - Preflight every configured runtime directory, then create them together; invalid file leaves or parent chains fail without partial directory creation.
- `build_artifact_provenance(cfg, dataset_manifest=None, *, published_output_paths, project_root_path=None)`
  - Build local-only artifact provenance from the current source revision/dirty state, canonical effective-config and dataset-manifest hashes, and stable artifact-relative SHA-256 commitments for the supplied successfully published files; missing Git identity degrades to explicit unknown values.
- `assess_artifact_provenance(provenance, cfg, dataset_manifest=None, *, project_root_path=None)`
  - Classify a stored bundle as `current`, `stale`, or `unverified` using stable local source/config/dataset/output identity; legacy or malformed markers, missing/schema-invalid dataset manifests, and missing/unreadable/changed committed outputs never assess current.
- `validate_json_payload(payload, ...)`
  - Apply the same JSON-safety and serialization contract as `write_json_file` without creating or replacing a path, for multi-output preflight.
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
  - Validate required agent-facing files; exact validation command/scope/capability records; profile reachability including mandatory dependency capabilities; repo-skill trigger frontmatter and repo-local `$skill` reference resolution; bidirectional normalized requirements/manifest specifier parity; source-derived external-import ownership (including evaluation-scope, binding-position, branch, loop-cycle, short-circuit, conditional-expression, and ordered context-manager-aware `importlib`/`__import__` aliases; ambiguous late-bound owner sets; declared `global`/`nonlocal` targets; match and starred-target shadows; precise comprehension shadows; fail-closed nonliteral dynamic names and delegated wrappers; relative-local exclusion; and identity-resolved direct literal calls to the unique `main.py` `_bind_missing` loader); distribution/module identity; direct/backend and core/scientific/UI/test consumer constraints; local `myutils` owner records; blocking dependency availability probes; exact retired-guidance and six-module records; the human-document policy; local instruction paths; stable v18 boundaries; and known layout warnings.
- `build_agent_state(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable agent-state payload used by `main.py --emit-agent-state` and `main.py --verify-agent-contract`.
- `build_agent_command_index(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`
  - Build the live JSON-serializable command index, including module dependencies, the human-document policy, and v18 research-plan alignment, used by `main.py --emit-agent-commands`.
- `agent_state_to_json(state)`
  - Serialize an agent-state payload for stdout or logs.
- `write_agent_state(state, path)`
  - Serialize before parent creation, then write an agent-state payload while refusing runtime-state output under any canonical, state-declared, or filesystem-equivalent user-owned `human_docs/` path and any multi-hardlink leaf.

## schema.py

- `FIXED_REPORT_ARTIFACT_NAMES`
  - Canonical fixed report filename set shared by materials collision preflight and UI persisted-state role validation; configurable structure-execution outputs may not relabel these files.
- `STRUCTURE_EXECUTION_OUTPUT_ROLES`
  - Canonical three-role mapping shared by structure building, materials publication, and UI persisted-state validation: viewer artifact key, experiment-summary field, configured path field, required suffix, and canonical default filename.
- `DatasetManifest`
  - Pydantic-style schema for normalized dataset-manifest metadata, including the target column that defines processed-cache identity.
- `MaterialRecord`
  - Schema for a normalized material record payload.

## utils.py

- No public functions are currently exposed.
- Internal pure-stdlib helpers centralize filesystem identity/descendant and symlink-component checks for the two runtime guard surfaces; the project-specific human-document policy remains in this repository rather than `myutils`.
