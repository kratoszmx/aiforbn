# Module Instructions for Codex

## Essential check before working

Common module-level utilities are stored in `utils.py`. Before starting work each time, you must check:

- Whether `utils.py` contains sufficiently general-purpose functions. If so, move them to an appropriate location under `~/projects/myutils`, and update the relevant `PY_FILES_SUMMARY.md` files accordingly.
- `~/projects/myutils/PY_FILES_SUMMARY.md` to determine whether there are already useful functions that can be reused directly for the current task.

This check must not be removed and must be performed every time.

## Template rules that apply to every module

- `src/template` is the module template. If you create a new module, copy this template first and then fill in the module-specific details.
- Each module must stay independent, complete, and non-subordinate. Cross-module calls should go through documented public functions or classes only.
- `AGENTS.md` stores the module-specific rules, implementation guidance, and any internal details that future Codex runs must follow.
- `utils.py` stores reusable helpers that are local to the module. If a helper becomes general enough for multiple modules or projects, move it to `~/projects/myutils`.
- Public callable functions and classes belong in `PY_FILES_SUMMARY.md`. Internal helpers, especially underscore-prefixed ones, should be documented here only when future maintainers need guidance.

## Template-specific note

- Keep this template generic. Do not put project-specific implementation details here unless they should be inherited by every new module copied from this template.
