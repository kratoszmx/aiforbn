# Current project state

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

The executable PoC covers dataset normalization, grouped evaluation, BN diagnostics, formula-only ranking, uncertainty/abstention, prototype handoff, and deterministic first-pass structure generation. The generated structures are unrelaxed. No synthesis, stability, direct-gap, or validated-discovery result follows from a successful software test.

- Runtime defaults: [src/config.py](src/config.py). Experimental attention/Roost-like models remain outside the default model sweep.
- Current artifact evidence: inspect the v2 `artifact_provenance.json` and committed output digests before using results. At the 2026-09-25 audit, this checkout has no such completion marker; its checked-in research outputs are historical, not a freshly validated run.
- Next research work should start from an explicit experiment/validation question. Ordinary documentation maintenance does not require recomputing the dataset, training, or rewriting artifacts.
- File/JSON digest reuse is implemented: runtime calls `myutils.sha256_file` and `sha256_json(make_json_safe(...))` directly. Byte-compatibility coverage is in `src/runtime/tests/test_io_utils.py`; project path/provenance guards remain local.

## Verification

Use [TESTING.md](TESTING.md) and the emitted command index for a fresh check. Older test counts are historical evidence in Git, not proof for the current tree. The 2026-09-25 documentation audit preserves existing runtime implementation work separately; its final validation scope is recorded with the documentation commit.

This file owns the short status view. [docs/HANDOFF.md](docs/HANDOFF.md) owns operational/scientific boundaries, [COMMON_FUNCTIONS.md](COMMON_FUNCTIONS.md) routes API use, and [SERVICES.md](SERVICES.md) describes the optional viewer and absence of project-owned daemons/MCP servers.
