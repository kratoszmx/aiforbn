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

## Materials-specific guidance

- `materials` is the main business module. It owns dataset normalization, candidate generation, feature building, model selection, benchmarking, screening, artifact writing, summary building, and structure-execution handoff logic.
- Cross-module dependencies are intentionally narrow: call only the public runtime helpers from `runtime` and the documented regressor classes from `torch_models`.
- Do not import underscore-prefixed names from other modules. Inside `materials` itself, underscore-prefixed helpers are internal implementation details and must not be treated as stable external API.
- The following files are implementation-heavy internal surfaces, not public API surfaces: `common.py`, `ranking_tables.py`, `structure_artifacts.py`, and `structure_helpers.py`.
- Keep `main.py` linear. If you refactor shared business logic out of `main.py`, move it into documented public functions inside this module instead of introducing wrapper façades.
