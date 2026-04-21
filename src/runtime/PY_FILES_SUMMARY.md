# runtime module public surface

This file lists the stable public functions and classes that other modules may import from `runtime`.
Anything underscore-prefixed or omitted here should be treated as internal.

## io_utils.py

- `load_config(path)`
  - Load a Python config file and return its top-level `CONFIG` dict.
- `ensure_runtime_dirs(cfg)`
  - Create the configured runtime directories needed by the project.
- `clear_project_cache(project_root_path='.')`
  - Delete Python/cache artifacts for the project root.

## schema.py

- `DatasetManifest`
  - Pydantic-style schema for normalized dataset-manifest metadata.
- `MaterialRecord`
  - Schema for a normalized material record payload.

## utils.py

- No public functions are currently exposed.
