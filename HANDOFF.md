# Current project state

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

The executable PoC covers dataset normalization, grouped evaluation, BN diagnostics, formula-only ranking, uncertainty/abstention, prototype handoff, and deterministic first-pass structure generation. The generated structures are unrelaxed. No synthesis, stability, direct-gap, or validated-discovery result follows from a successful software test.

- Runtime defaults: [src/config.py](src/config.py). Experimental attention/Roost-like models remain outside the default model sweep.
- Current artifact evidence: inspect the v2 `artifact_provenance.json` and committed output digests before using results. At the 2026-09-25 audit, this checkout has no such completion marker; its checked-in research outputs are historical, not a freshly validated run.
- Next research work should start from an explicit experiment/validation question. Ordinary documentation maintenance does not require recomputing the dataset, training, or rewriting artifacts.
- File/JSON digest reuse is implemented: runtime calls `myutils.sha256_file` and `sha256_json(make_json_safe(...))` directly. Byte-compatibility coverage is in `src/runtime/tests/test_io_utils.py`; project path/provenance guards remain local.

## Verification

Use [TESTING.md](TESTING.md) and the emitted command index for a fresh check. Older test counts are historical evidence in Git, not proof for the current tree. The 2026-09-25 documentation audit preserves existing runtime implementation work separately; its final validation scope is recorded with the documentation commit.

This file owns the short status view. [docs/HANDOFF.md](docs/HANDOFF.md) owns operational/scientific boundaries, [COMMON_FUNCTIONS.md](COMMON_FUNCTIONS.md) routes API use, and [SERVICES.md](SERVICES.md) describes the optional viewer, separator web deployment and absence of a project-owned MCP server.

## AI for Science partner prototype — 2026-09-28 revision 2

Public branding is AI for Science; repository and folder names remain aiforbn. Anonymous access replaces invitations, and the persisted rolling limit is 100 new analyses. The site offers readable formulation tables, three numerical tasks and next-experiment selection. BN source facts (20 records across four studies) remain separate from the new Clio liquid-electrolyte data (125 measurements grouped into 38 compositions). See [the current report](docs/research/separator_prototype/revision_2_report.md), [replay protocol](docs/research/separator_prototype/replay_protocol.json), and [service lifecycle](SERVICES.md).

In the frozen measured-pool replay, finding three formulations at ≥13 mS/cm takes 9.32 experiments with random selection versus 7.69 with the adaptive forest, a 17.5% conditional reduction over 100 paired starts. The nearest-formulation method takes 7.75; its difference is not clear. This is one electrolyte study, not prospective lab savings or a BN result. The old one-case PP/Astra benchmark remains historical; raw BNNT inference is now capped at 0.3 mg/cm² because the paper reports a decrease beyond that loading.

The pre-existing builtin-getattr alias changes in five runtime/documentation paths remain separately owned. Historical band-gap artifacts and human_docs are untouched. Deployment URL, expiry, private provider evidence and old invitation backup live under ignored `.runtime/separator/`. Next scientific work is compatible independent BN measurements and a prospective same-budget comparison with the partner's actual decision workflow.

## September 28 revision 3

See [current audit](docs/research/separator_prototype/revision_3_report.md) and
[weekly draft](docs/research/separator_prototype/weekly_update_draft_zh.md). The
website is simplified; the selector returns one next recipe, matching replay.
PP values use the linear reference plus model explanation; this matches, does
not beat, linear regression. Two additional full-text dispersion reviews improve
meeting relevance but provide no new matched PP validation. The actual meeting
focus remains aqueous BN dispersion, viscosity and coating performance.

A per-user monitor checks HTTP every five minutes and fresh public model response
every six hours; Supervisor reads its bounded evidence in daily reports. Initial
real response took 21.275 seconds. Monitor installation/stop and scope are in
SERVICES.md. No partner message was sent.
