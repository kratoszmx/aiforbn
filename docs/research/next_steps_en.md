# BN separator research: execution steps

Updated: 2026-09-27. Owner: one MPhil researcher. User counterpart: [Chinese execution plan](../../human_docs/next_steps_zh.md).

The user's latest decisions govern this plan: **work on BN separators first; start from public data without expecting partner data; develop validation routes ourselves while progressing; preserve v18 and iterate these next-steps documents.** Steps follow dependencies, with no calendar deadlines. Substep IDs match the Chinese plan for progress reporting.

Objective: predict the likely prepared BN separator material and its properties from ingredients, quantities and processing conditions; then recommend candidates for verification and test whether this reduces unproductive experiments.

## Step 1: implement preparation-outcome and property prediction

The user's reactants-to-products example becomes an ingredient/condition-to-material-outcome task. Use a separate reaction-product task for an actual chemical transformation. Dispersion, coating and physical assembly require formulation/process representations; they do not necessarily produce a new molecule. One accessible BN composite-separator study explicitly reports physical binding in its system. [Primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11596189/)

1. **1.1 Collect public data.** Search original BN separator papers, supplementary tables, public repositories and patent examples. Capture formulations, procedures, controls, measured outcomes and reported failures. Start with BN-coated polypropylene (PP) separators; keep other BN separator substrates as separate cohorts. Partner data are optional future additions. Starting sources:
   - [BN/graphene-coated PP separators](https://pubmed.ncbi.nlm.nih.gov/33010580/): the abstract confirms a relevant experimental study; obtain full text or supplementary material before extracting complete recipes. Full-text access is not established by the abstract.
   - [Calcium-alginate-fibre/BN separators](https://pmc.ncbi.nlm.nih.gov/articles/PMC11596189/): accessible primary full text for schema development and a separate substrate cohort, not interchangeable PP data.
   - [Open Reaction Database](https://open-reaction-database.org/about): an organic-reaction resource to inspect for relevant modification reactions. Its [schema](https://docs.open-reaction-database.org/en/latest/schema.html) records inputs, conditions, outcomes and provenance. Inspect actual coverage before adopting records; ORD availability does not establish BN separator coverage.
2. **1.2 Build traceable records.** Retain source URL/DOI, version and table/row locator; study, sample, formulation, batch and measurement-series IDs; BN morphology/size and surface treatment; substrate, binder, solvent and amounts with their mass/volume basis; mixing, coating and drying conditions; thickness; test protocol; target, unit, uncertainty and controls. Distinguish specified settings from post-preparation measurements. Preserve original values, missingness, duplicate links and extraction confidence. Record access and redistribution permissions separately. Missing results do not establish failed experiments.
3. **1.3 Fix the initial model contract.** Operational starting choice: BN-coated PP battery separators, initially seeking thermal shrinkage under specified temperature, duration and measurement definition as the primary target. Collect ionic transport, electrolyte wetting and mechanical properties separately. This is a researcher's starting choice, not a user-specified substrate or an established dataset. If the audit supports another quantitative separator endpoint better, document the change within BN separators and freeze it before model comparison. Viscosity/dispersion can be process subtasks; they do not replace the separator application. Preserve area versus linear shrinkage definitions and keep incompatible test protocols separate.
4. **1.4 Implement the language-model outcome workflow.** Use an available existing model and retrieval over curated sources. Return structured preparation/material outcomes, reaction products where applicable, supporting records and missing conditions. Keep reaction examples and evaluation separate from coating records. Build reproducible inference before deciding on fine-tuning. Do not invent reaction equations for physical coating.
5. **1.5 Add measured-property prediction.** Establish a simple baseline and a suitable tabular predictor trained on measured labels. Combine it with language-model retrieval and material descriptions. Mark outputs as reported evidence, prediction or unsupported/unknown. Generated values are not experimental labels. Use inputs available at the decision point: measured final thickness is not a pre-fabrication input unless separately predicted or explicitly specified as a design value.

Deliver: source inventory, data dictionary, first curated dataset, input/output specification and reproducible examples. Count independent sources/formulations separately from measurement rows. Completion establishes an operational flow; Step 2 establishes predictive evidence. If numeric labels are insufficient, deliver the retrieval/outcome component, continue acquisition, and mark the numeric predictor incomplete.

## Step 2: benchmark and improve the model

1. **2.1 Reserve independent evaluation groups.** Link duplicate publications/patents and reused datasets; group studies, formulations, batches and repeated measurements for the intended generalization claim. Keep test answers out of retrieval, examples and prompts. Public papers may be in language-model pretraining data: disclose that limitation and distinguish retrospective from prospective evidence.
2. **2.2 Compare meaningful baselines.** Include a training-only mean/median, similar-formulation retrieval, simple regression and the selected predictor. Compare the language model with and without retrieval. Report property MAE/RMSE in original units and per-source performance. Score material outcomes on source support, material/product identity and condition completeness; compare actual reaction predictions with independently reported products using an appropriate chemical representation. Keep the scores separate.
3. **2.3 Improve from observed errors.** Diagnose units, missing process variables, mixed substrates, sparse support and model failures. Compare incremental effects of condition features, retrieval and justified fine-tuning. Keep preprocessing, selection and tuning within training data. A final test used to revise the method becomes development data and requires another independent evaluation.
4. **2.4 Characterize support and uncertainty.** Inspect errors by system and supported input range. Calibrate intervals on suitable held-out training-side data and assess coverage/width independently when group counts permit. With few groups, report exploratory performance and instability; do not promise fixed accuracy or coverage under arbitrary source shift.

Deliver: benchmark table, group definitions/results, error cases, support limits and improvement log. Reliable improvement over a meaningful baseline supports quantitative recommendations. Negative or inconclusive results direct further data/method work and remain reportable.

## Step 3: generate and rank BN separator formulations

1. **3.1 Define the candidate domain.** Use reported examples to bound BN form/modification, loading, binder and processing choices. Identify adjustable parameters and fixed conditions.
2. **3.2 Generate and screen candidates.** Use source-grounded language-model proposals, then property predictors and explicit constraints. Display shrinkage, ionic transport, thickness, mechanics and cost separately. Essential missing evidence remains visible.
3. **3.3 Reuse v18's decision principles.** Rank on target relevance, uncertainty, domain support, novelty and verification cost. Preserve action meanings `control`, `priority`, `explore`, `hold`; calibrate rules for separators rather than copying band-gap thresholds. Check ranking sensitivity to reasonable weights and model variation.
4. **3.4 Produce formulation cards.** Give ingredients/quantity bases, processing, predicted properties/ranges, sources, unresolved fields, controls and proposed measurements. Unsupported suggestions remain exploratory, not effective formulations.

Deliver: candidate table, ranking rationale, controls and formulation cards. Counts follow evidence and verification capacity; v18's crystal-family/prototype quotas are not separator acceptance criteria.

## Step 4: independently validate and update

1. **4.1 Start with public-data validation.** Freeze prediction and selection rules before retrospective evaluation on unused sources. Compare with similar-formulation retrieval, simple models or random selection under the same conditions. Acquire new sources if existing data were used for development. Report missing outcomes and publication bias; retrospective replay covers observed candidates only.
2. **4.2 Establish relevant computational checks.** Select a named applicable material/interface or transport method for a specific question and check it against known cases. State which property it supports. Existing band-gap predictions and unrelaxed crystal prototypes do not validate separator processing or cell performance. If no applicable simulator is available, progress through independent data evaluation and experimental preparation.
3. **4.3 Prepare an experimental route ourselves.** Starting alongside Step 1, identify university facilities or external testing services from public information. Prepare sample/control requirements, test definitions, repeat measurements, equipment and quotation requirements. Bring a concrete test package to resource/cost discussions. Do not wait for assumed commitments from meeting participants.
4. **4.4 Obtain prospective measurements when resources are arranged.** Freeze formulations and evaluation rules before testing. Record successes, failures and protocol deviations. Evaluate the frozen model before adding results to the next version.
5. **4.5 Test the cost-reduction objective.** Compare the candidate tests needed to meet a prespecified property requirement under the same pool/protocol. Account for actual time/expenditure when prospective experiments exist. Retrospective estimates are simulated selection savings, not realized laboratory cost savings.

Deliver: independent validation report, experimental handoff, resource/cost options and, when acquired, prospective results with model updates. Report retrospective, computational and physical validation separately. Only measured results complete physical validation; lacking them does not block the other work.

## Step 5: consolidate reproducible results and manuscript material

1. **5.1 Freeze a reproducible release.** Record source/data versions, group splits, configurations, evaluation commands and candidate history.
2. **5.2 Build a minimal demonstration.** Show inputs → predicted preparation outcome/properties → ranked candidates → evidence and validation status.
3. **5.3 Establish the supported contribution.** Compare prior work and use ablations to test condition-aware prediction, cross-source generalization, uncertainty-guided selection or experimental efficiency. Base claims on results, not the presence of a language model.
4. **5.4 Write the report and manuscript draft.** Assemble the problem, data, methods, comparisons, validation and limitations continuously. Separate predicted candidates from observations.

Deliver: reproducible research package, demonstration, technical report and manuscript draft.

## Continuity with v18 and the meeting

Preserve [v18 source](../../human_docs/research_plan/ai_for_bn_research_plan_v18.tex), bibliography and PDF byte-for-byte at their current paths. Its original UV/wide-gap and dielectric aims remain there. This execution plan extends its methods to the user-selected separator application; it does not amend the formal proposal or treat band gaps as separator targets.

| v18 element | Execution here |
| --- | --- |
| Provenance-aware BN data | Step 1: formulation, process, measurement and outcome records |
| Grouped benchmarking and BN diagnostics | Step 2: source/formulation evaluation and system-specific errors |
| Uncertainty, support, ranking and action labels | Step 3: evidence-aware formulation prioritization |
| Structure handoff and validation | Step 4: formulation/process/test handoff, with structures for applicable computational checks |
| Technical report and demonstration | Step 5: reproducible evidence, demonstration and manuscript material |

The [reviewed transcript](../../official_docs/meetings/2026-09-09_bn_research/transcripts/transcript_reviewed_zh.txt) supports starting with separators at 01:13:38–01:13:52, public-data dispersion/modification work at 02:12:38–02:17:17, and checking data/model outputs at 02:48:06–02:48:21. Speaker identities and some terms remain transcription inferences. The user's latest decisions resolve the former scope/data/validation questions. Provenance is in the [archive index](../../official_docs/INDEX.md).

## Progress reporting

For each report, identify the actual substep, linked artifact/version, observed result against its comparator, unresolved issue and attempted remedy, and next concrete action. Count studies, independent formulations and measurement rows separately. Mark work planned, in progress or completed according to evidence. Do not invent completed counts or set weekly completion deadlines.

## Decisions and implementation boundaries

No unanswered user decision blocks starting. BN separators, public data, self-organized validation and preservation of v18 are settled. Make and record routine substrate/endpoint/model choices from evidence instead of returning the previous four questions to the user.

Later decisions concern concrete expenditure (paid compute/data, materials or commissioned tests, with costs and alternatives), outgoing communications or collaboration commitments (prepare the content first), and authorship/submission/data-release arrangements. This document revision sends no messages and purchases no services.

Use existing text/table/metadata tools in Conda `quant`; retain reusable permitted sources under `official_docs/` with provenance. Figure-only values remain unextracted until text/table data are obtained. No multimodal processing or new package installation without the user's authorization. Keep private meeting material locally excluded as documented in the source index.

The executable repository is a band-gap/prototype PoC, not an implemented separator formulation model. Later code work must use semantically appropriate public APIs, dedicated formulation schemas and relevant tests; preserve existing datasets/artifacts and v18 manifest anchors. This plan requires no new subproject.

Current delivery is the revised bilingual plan only. No new separator training corpus, trained model or experiment is claimed. Next research action: **1.1, collect original public BN separator sources and supplementary tables**, while recording the initial contract in 1.3.
