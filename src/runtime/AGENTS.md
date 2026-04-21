# Module Instructions for Codex

## Essential check before working

Common module-level utilities are stored in `utils.py`. Before starting work each time, you must check:

- Whether `utils.py` contains sufficiently general-purpose functions. If so, move them to an appropriate location under `~/projects/myutils`, and update the relevant `PY_FILES_SUMMARY.md` files accordingly.
- `~/projects/myutils/PY_FILES_SUMMARY.md` to determine whether there are already useful functions that can be reused directly for the current task.

This check must not be removed and must be performed every time.

## Template rules that apply to this module

- This module must stay independent, complete, and non-subordinate.
- Cross-module calls must go through documented public functions or classes only.
- `utils.py` stores reusable helpers that are local to this module. If a helper becomes general enough for multiple modules or projects, move it to `~/projects/myutils`.
- Public callable functions and classes belong in `PY_FILES_SUMMARY.md`. Internal helpers, especially underscore-prefixed ones, should be documented here only when future maintainers need guidance.

## Runtime-specific guidance

- `runtime` is the shared runtime-support module. It is allowed to expose configuration loading, runtime-directory preparation, cache clearing, and schema contracts.
- Keep the public surface small and stable. External callers should use the documented functions in `io_utils.py` and the documented schema classes in `schema.py`.
- Do not expose `sys.path` bootstrapping details as part of the public contract. Treat that logic as implementation detail unless a deliberate interface change is needed.
- No other module should import underscore-prefixed names from `runtime`.
