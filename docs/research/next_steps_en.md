# BN separator research: staged delivery and early reporting

Updated: 2026-09-27. [User-facing Chinese plan](../../human_docs/next_steps_zh.md). Governing decisions: BN separators first; public data without waiting for partner data; develop validation routes ourselves; preserve v18. Specify the first two phases, and progressively refine later phases from evidence and partner feedback.

Objective: ingredients/formulation/process → material outcome and property prediction → candidate selection → validation, with experimental-efficiency claims tested against evidence. Every phase produces something the partner can inspect, check or operate. A phase ends on inspectable deliverables, not a promised week number.

## Phase overview

| Phase | Partner-visible delivery | Review question |
| --- | --- | --- |
| 1. Public data and evidence examples | Source inventory, pilot records, source-to-record cards, known-case checks and a first model-task recommendation | Where do the data come from, what is usable, and how can it be checked? |
| 2. Inspectable minimum prototype | Runnable demonstration or complete operation record, fixed examples, prediction/measurement comparison and failure cases | What does it do, is it checkable, and does it improve on simple methods? |
| 3. Jointly refine needs and validate candidates | Formulation cards, validation package and candidate results/status | Which proposals matter to the partner and deserve verification? |
| 4. Iterate and consolidate research | Version comparisons, reproducible package, technical report and manuscript/demo material | Which improvements and research claims are supported? |

## Phase 1: deliver a public-data evidence package

1. **1.1 Inventory public sources.** Initial effort targets: screen 10–15 relevant sources and seek at least three original studies reporting formulations and test conditions. Record DOI/URL, system, full-text/supplement access, extractable content and permitted use. Distinguish abstract-only, figure-only and inaccessible sources. Prioritize primary papers, supplements, public repositories and patent examples; identify duplicated/reused experimental data.
2. **1.2 Produce source-to-record cards.** Start with 5–10 traceable formulations/samples. Each row points to a source and page/table/paragraph. Make three representative cards displaying original evidence, normalized fields and the check result. Preserve values, units, quantity bases, extraction uncertainty and missingness; generated numbers are not observations.
3. **1.3 Assemble dataset v1.** Work toward 20–30 independent formulations/samples in the first package; count repeated measurements separately. Include substrate, BN form/size/treatment, binder, solvent, ratios, mixing/coating/drying, thickness, measured endpoints, test conditions, controls and reported failures. Separate designed settings from post-fabrication observations. Define compatible cohorts, missingness, duplicates and license/access state. These are collection targets, not guaranteed yields or statistical sufficiency thresholds.
4. **1.4 Check known cases.** Select 2–3 source-supported comparisons with controls. Recalculate reproducible quantities or ratios and check that extracted records preserve the published comparison under matching conditions. Deliver calculation steps and discrepancy tables. This is evidence/data reconstruction, not laboratory replication. Keep incompatible definitions, including linear versus area shrinkage, separate.
5. **1.5 Recommend the first model task.** Deliver a one-page contract: inputs, output, material cohort, evaluation data and rationale. Start searching BN-coated PP separators, with shrinkage and ionic-transport measures as candidate endpoints. Choose an endpoint from actual comparable coverage; separate other substrates. The application is fixed, but substrate/endpoint choices can evolve after the phase review. If partner feedback is unavailable, proceed with the best-supported documented working choice.

Partner package: filterable source inventory and records, dictionary, three evidence cards, 2–3 known-case checks, model-task recommendation and a short phase report. Suggested future files are `source_inventory.csv`, `pilot_records.csv`, `data_dictionary.md`, `evidence_cards.md`, `known_case_checks.md`, `phase_1_report.md`; these are planned outputs, not files asserted to exist today. A spreadsheet export can accompany CSV for partner convenience.

Acceptance: each retained value has source location and conditions; compatible and incompatible observations are identified; at least one complete source → record → check chain can be demonstrated. Report actual counts and attempted gap-filling when collection targets are missed; never pad counts or invent data. Ask the partner to react to concrete cases and prioritize relevant outputs, without making a reply a prerequisite for ongoing work.

## Phase 2: deliver a checkable minimum prototype

1. **2.1 Freeze the initial task and evaluation data.** Define one cohort, primary output and measurement context before model comparison. Link same-source duplicates and group by study/formulation/batch for the intended generalization claim. Reserve unused evaluation data; keep test answers out of examples, retrieval and prompts. Document possible language-model pretraining exposure to public papers.
2. **2.2 Build the smallest complete flow.** Accept BN/other ingredients, ratios and processing conditions; return supporting published recipes, likely preparation outcomes, supported property predictions and source links. Use an existing language model for retrieval/structured descriptions and measured labels for property baselines. Add reactants-to-products prediction only for a defined chemical transformation; represent coating/dispersion as formulation/process tasks. Exclude inputs only known after obtaining the predicted outcome.
3. **2.3 Prepare three fixed demonstration cases.** Show a source-backed known case, an unused-data prediction, and an unsupported input that should receive an explicit limitation. Deliver exact inputs, outputs, evidence, reference answers and check results. Provide a runnable prototype or complete operation record; label any unfinished capability rather than simulating its completion.
4. **2.4 Compare and diagnose.** Compare similar-formulation lookup, simple statistics/regression and the selected method. Report prediction-versus-measurement rows, errors in physical units, source-level results and failures. Check citation fidelity and hallucinated missing fields separately from predictive error. Fit preprocessing, tune models and calibrate uncertainty within training data. Too few independent groups permit exploratory results only. If numeric labels remain insufficient, deliver the working retrieval component and identify the unfinished numeric task with a concrete data remedy.

Partner package: prototype, three demonstration cases, baseline/evaluation table, error cases, supported/unsupported capability list and phase report. Suggested outputs: a runnable demo entrypoint, `demo_cases`, `prediction_vs_measurement.csv`, `baseline_comparison`, `error_cases.md`, `phase_2_report.md`; final placement follows the repository's existing ownership rules.

Acceptance: fixed cases are reproducible, outputs distinguish evidence from predictions, and comparison with a simple method is available. Functional readiness and predictive reliability are distinct conclusions. Use the demonstration to learn whether inputs are available to the partner, outputs support decisions and which improvement matters most. Use this feedback to specify Phase 3.

## First four weekly reports

Prepare one report per reporting week in this content sequence. Phase completion depends on actual deliverables; these are not promises to finish whole phases within a fixed number of weeks. Each report includes an inspectable partial result even if an anticipated model/data task is incomplete.

| Report | Work focus | Concrete partner-visible material | Verification and review |
| --- | --- | --- | --- |
| R1: first weekly report | Source inventory and first extraction | Screening inventory targeting 10–15 sources, first 5–10 valid samples/formulations, three evidence cards; actual counts and gaps shown | Walk through one original value, normalized unit and condition; distinguish full text from abstract-only access |
| R2: second weekly report | Dataset cleaning and known-case checks | Dataset v1/dictionary, duplicate/missingness summary, 2–3 checks and first model-task recommendation; include Phase 1 review when its acceptance is met | Which records are comparable, and which evidence will test the model? |
| R3: third weekly report | Minimum prototype and examples | Runnable prototype or complete operation record, three demonstration types and supported/unsupported feature list | Which outputs are retrieved facts versus predictions, and can the example be checked again? |
| R4: fourth weekly report | Independent comparison and feedback | Prediction/measurement comparison, baselines, failure cases and feedback/decision log; include Phase 2 review when acceptance is met | What improved, what failed, what remains unevaluable and which issue deserves the next phase? |

Every weekly report contains: linked versioned delivery, actual counts, at least one evidence example, problems and attempted remedies, next concrete delivery, and one focused request for partner feedback. Separate sources, independent formulations/samples and measurement counts. Record unreceived feedback as pending, not agreement. Feedback informs work progressively rather than becoming a blanket stop condition.

Every phase report follows: problem addressed → artifact demonstration → data provenance → validation method/results → remaining limitations → proposed next work and specific partner feedback. Keep reports and artifacts together so progress remains inspectable after a meeting. R1–R4 are planned reporting contents, not completed reports or invented results.

## Phase 3: jointly focus requirements and validate candidates

Keep details provisional until the first two phases supply evidence and feedback.

1. **3.1 Select one valuable optimization problem.** Translate the emerging need into adjustable inputs, target property and constraints. Produce a small candidate set with source-backed formulation cards, uncertainty and v18-style `control`, `priority`, `explore`, `hold` actions. Determine detailed search space and ranking rules from observed support.
2. **3.2 Choose and execute applicable validation.** Use unused public outcomes first. Computational checks require a named method, relevant physics and known-case validation. For physical tests, prepare samples, controls, measurements, capacity and costs; investigate facilities in parallel with early data work. Set candidate/replicate counts when resources and the actual question are known. Freeze recommendations before observing new outcomes.

Partner delivery and reports: candidate cards, validation package and per-candidate evidence/result/status table. Weekly reports track supported and rejected proposals, incomplete checks and the next adjustment. Phase review decides whether to change recipes, acquire data or revise the method. Keep retrospective, computational and physical evidence separate; old band-gap predictions and unrelaxed prototypes cannot validate separator fabrication or cell performance.

## Phase 4: iterate and establish research results

1. **4.1 Update data and models.** Incorporate new observations in a versioned cycle while retaining independent evaluation. Assess selection improvement and tests, time or cost required to achieve the same defined objective. Retrospective replay covers observed candidates and gives estimated selection savings, not realized experimental savings.
2. **4.2 Establish the contribution.** Select the manuscript question, comparisons, extension scope and demonstration from evidence. Do not promise universal cross-system generalization or publication before results.

Partner delivery and reports: version comparisons, successes/failures, reproducible research package, technical report and manuscript/demo material. Weekly reports show new results; phase reports delimit supported conclusions. Specify detailed targets later as the project and partner needs become clearer.

## Provenance and verifiability throughout

Every measured value needs a source locator, original value/unit, test conditions, extraction method and version. Every claimed result needs its check: original-source/known-case verification early, independent data in the prototype, applicable computation or physical tests later. A language model grading its own answer is not independent validation. Keep facts, predictions and unresolved items visibly distinct in every delivery.

Use existing Conda `quant` text/table tools. Figure-only values stay unextracted until a text/table source is found; no multimodal processing. Keep reusable permitted source documents under `official_docs/` with provenance. Public availability and redistribution permission are separate fields; private partner/meeting materials retain their exclusions.

Starting leads: [BN/graphene-coated PP separators](https://pubmed.ncbi.nlm.nih.gov/33010580/) and [calcium-alginate-fibre/BN separator full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11596189/). Phase 1 must establish actual extractable, comparable data coverage; these references are not a completed training corpus.

## Continuity and decisions

Keep [v18 source](../../human_docs/research_plan/ai_for_bn_research_plan_v18.tex), bibliography and PDF unchanged. Carry forward its provenance-aware data, grouped evaluation, uncertainty-aware ranking, verification handoff and report chain across the four phases. Preserve its original UV/wide-gap/dielectric proposal without treating existing band-gap outputs as separator labels.

The [meeting transcript](../../official_docs/meetings/2026-09-09_bn_research/transcripts/transcript_reviewed_zh.txt) discusses starting from separators/data at 01:13:38–01:13:52, dispersion/modification/public data at 02:12:38–02:17:17, and judging generated outputs at 02:48:06–02:48:21. Speaker/term transcription uncertainty remains; the user's latest staged-delivery request governs this revision.

No unanswered user decision blocks early work. Learn the partner's preferred properties, available inputs, acceptable errors and validation conditions by showing deliveries. Prepare concrete costs/alternatives before requesting paid resources, and actual content before authorization for outgoing messages, commitments or submission. This planning revision sends no partner communications.

Delivery update, 2026-09-28: phases 1 and 2 now have a public separator evidence package and runnable demonstration alongside the original band-gap pipeline. See the [phase 1 report](separator_prototype/phase_1_report.md), [phase 2 report](separator_prototype/phase_2_report.md) and [service instructions](../../SERVICES.md). The initial numeric comparison is one same-study held-out formulation and establishes no advantage over simple regression or laboratory savings. Historical research artifacts and v18 remain unchanged. Next research work is to acquire compatible independent studies and obtain partner feedback on the demonstrated inputs and outputs before defining phase 3 candidates. R1–R4 remain a reporting sequence, not four already-completed weeks.
