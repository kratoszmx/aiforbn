# AI for Science partner prototype

This is the current entrypoint for the separator/dispersion partner flow. `main.py` belongs to the separate historical band-gap workflow.

| Need | Read |
| --- | --- |
| Try the website | [Partner walkthrough](partner_demo_zh.md); current URL/expiry are in local `.runtime/separator/` |
| Current delivery and language/architecture behavior | [Revision 5](revision_5_report.md) |
| Aqueous method, meeting coverage and scientific gaps | [Revision 4](revision_4_report.md) |
| Source definitions and limitations | [BN dictionary](data_dictionary.md), [evidence cards](evidence_cards.md), [failure cases](error_cases.md) |
| Reproduce software checks / operate services | [TESTING.md](../../../TESTING.md), [SERVICES.md](../../../SERVICES.md) |
| Planned research / unsent communication | [Research plan](../next_steps_en.md), [weekly draft](weekly_update_draft_zh.md) |

## Current evidence and task boundaries

| Cohort | Source of truth | Supported use |
| --- | --- | --- |
| BN separators | [Dataset](../../../data/separators/dataset.json): 20 records, four studies | Source retrieval; bounded PP conductivity reference plus model explanation; same-process CA thickness interpolation |
| Aqueous recipes | [Manifest](../../../data/dispersion/aqueous_sources.json): 12 patent cases | Six matched cases support viscosity interpolation; controls and BN/PI references stay separate |
| Liquid electrolytes | [Source manifest](../../../official_docs/experiment_planning/source_manifest.json): 125 readings, 38 compositions | Conductivity estimation and one-next-recipe selection; [replay protocol](replay_protocol.json) / [results](workflow_evaluation.json) |

These are distinct material systems. Missing test conditions remain unknown. PP has one study; viscosity checks use the same six-point series; electrolyte savings are conditional same-pool replay estimates. None establishes independent BN accuracy, prospective laboratory savings or arbitrary-formulation prediction. Partner history/import/retraining and matched shrinkage prediction remain future work.

## Dated delivery evidence

[Phase 1](phase_1_report.md) records the first source package. [Phase 2](phase_2_report.md), [initial task](model_task.md) and [initial verification](verification.md) preserve the frozen PP evaluation and original deployment. [Revision 2](revision_2_report.md) adds anonymous access and electrolyte replay; [revision 3](revision_3_report.md) records the PP numerical-reference/one-recipe changes and response monitor. Their old invitation, quota, recipe-range and service observations describe those versions; use the current guides above for operation. Unique source and evaluation evidence remains useful, so it is retained rather than treated as disposable scratch output.
