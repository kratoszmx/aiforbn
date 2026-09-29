# Current project state — AI for Science

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`.

## Partner prototype

Current implementation: [revision 5](docs/research/separator_prototype/revision_5_report.md), with [revision 4](docs/research/separator_prototype/revision_4_report.md) owning the aqueous-method and meeting-coverage detail. [Prototype index](docs/research/separator_prototype/INDEX.md) separates current use from dated evidence; [partner walkthrough](docs/research/separator_prototype/partner_demo_zh.md) gives the shortest operating path.

- AI for Science is the public name; directory, repository and internal identifiers remain `aiforbn`.
- The anonymous site has four numerical forms, public source tables/downloads and one-next-experiment selection. Simplified Chinese is the default, with persistent English, Traditional Chinese and light/dark choices.
- Evidence stays separate: 20 BN records across four studies; 12 aqueous patent cases, of which six support one viscosity series; 125 electrolyte readings grouped into 38 compositions. PP still uses one study. See the index for source manifests and evaluation limits.
- Model explanation, local numerical estimates and sequential replay are implemented. Partner-data import, persistent laboratory histories, automatic retraining and prospective savings validation remain future work. Particle-size mixtures, dispersant selection and matched timed-shrinkage prediction also remain research gaps.
- Next work: agree a compatible partner protocol, acquire independent measurements, freeze recommendations, then compare against the partner's usual selection under an equal experiment budget. The [weekly draft](docs/research/separator_prototype/weekly_update_draft_zh.md) is unsent.

Service lifecycle and monitor evidence are in [SERVICES.md](SERVICES.md). Deployment URL, expiry, quotas, receipts and provider diagnostics belong to ignored `.runtime/separator/`; dated reports do not establish current availability. Repository documentation work needs no new paid model call or service restart.

## Historical band-gap workflow

`main.py` covers dataset normalization, grouped evaluation, BN diagnostics, formula-only ranking and deterministic unrelaxed structures. It provides no synthesis, stability, direct-gap or validated-discovery proof. Defaults live in [src/config.py](src/config.py); attention/Roost-like models remain experimental and outside the default sweep. Historical labels and artifacts do not validate the partner tasks.

Assess stored outputs through v2 `artifact_provenance.json` and committed output digests before interpreting them. Ordinary maintenance does not refresh these research outputs. Runtime reuses `myutils` filesystem/JSON/digest APIs while retaining project path and provenance guards.

## Maintenance and verification

[AGENTS.md](AGENTS.md) is the entrypoint; [COMMON_FUNCTIONS.md](COMMON_FUNCTIONS.md), [TESTING.md](TESTING.md) and [docs/HANDOFF.md](docs/HANDOFF.md) own API routing, validation and detailed scientific/publication boundaries. The active maintenance skill is `.agents/skills/aiforbn-workflow/SKILL.md`; the root `skills/` directory is retired. Use Git and versioned reports for previous maintenance and delivery evidence, then inspect the current worktree before claiming a fresh pass.
