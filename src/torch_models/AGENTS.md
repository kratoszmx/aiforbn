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

## Torch-model-specific guidance

- `torch_models` exposes model classes, not a broad utility grab bag.
- The documented public surface is the regressor classes in `base.py`, `ensemble.py`, `attention.py`, `sparse_attention.py`, and `roost_like.py`.
- Underscore-prefixed helpers in `base.py` are internal implementation details for this module only.
- External business logic should normally instantiate these models through documented public call sites, especially `materials.modeling.make_model(...)`, unless a direct class import is explicitly warranted.
