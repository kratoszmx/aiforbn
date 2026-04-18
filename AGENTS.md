# AGENTS.md

Repository-local guidance for coding agents working on `ai_for_bn`.

## Read first
Before coding, reread these files in this order:
1. `skills.txt`
2. `../myutils/generalskill.txt`
3. `../myutils/skills.txt`
4. `HANDOFF.md`
5. `chat/老師回覆.txt`

Do not edit project skill files yourself.

## Project goal
This repo is a PhD-level **AI for BN** research PoC, not just an app demo.
Prioritize methodological credibility over cosmetic pipeline changes.

Current standing priorities:
- keep evaluation honest
- keep BN central in screening logic, not just in report wording
- strengthen candidate-validation credibility
- preserve transparent baselines and controls
- avoid overclaiming discovery

## Hard constraints
- Keep `main.py` linear, readable, and notebook-friendly.
- Use the existing `quant` environment only. Do not create a new Python environment.
- Before tests or batch runs, clear cache via `../myutils/file_utils.delete_cache('.')` or equivalent.
- Before any commit, both of these must pass:
  - `pytest -q`
  - `/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python main.py`
- Prefer reusing `../myutils` helpers when they are truly generic.
- Keep important functionality comprehensively tested, but do not chase trivial helper coverage.

## Research honesty rules
- Treat formula-only screening as **formula-level ranking/demo evidence**, not structure-level discovery.
- Do not present analog evidence, support scores, or disagreement heuristics as direct stability proof.
- Preserve baseline controls such as `basic_formula_composition`.
- Keep the distinction explicit between:
  - best overall evaluation model
  - best candidate-compatible screening model
- If a change would make the claims sound stronger than the evidence, do not do it.

## Candidate-screening guidance
Prefer upgrades that improve real methodological credibility, for example:
- better candidate-space design
- stronger validation grounding
- clearer applicability-domain logic
- better uncertainty treatment
- more defensible BN-centered evidence layers

Deprioritize cosmetic renaming, ornamental metadata, or leaderboard-style presentation work.

## Documentation ownership
OpenClaw owns the final wording of human-facing docs.
Unless explicitly told otherwise, do **not** edit these files in Codex-driven coding runs:
- `给见微的说明.md`
- `项目汇报.md`
- `HANDOFF.md`
- `PY_FILES_SUMMARY.md`

Code, tests, and internal technical notes are fair game when requested.

## Operational rules
- Use English when communicating with coding agents or writing AI-facing technical notes.
- Avoid low-value full reruns when an incremental check is enough.
- If you stall for 5 to 10 minutes without meaningful progress, stop and surface the blocker.
- If environment, web, package, or API access blocks progress, report it immediately instead of hanging.
- Keep diffs focused. Do not touch unrelated files.
- Do not include `skills.txt` edits in commits unless the human explicitly requested that.

## Current interpretation of success
A good change in this repo usually does at least one of these:
- reduces evaluation leakage or overclaiming
- makes BN more central to actual screening logic
- makes candidate ranking more defensible
- improves evidence quality for advisor-facing reporting
- keeps the whole repo runnable and honestly documented
