# BN modification and dispersion: next-step execution plan

Date: 2026-09-27. Status: proposed feasibility plan, prepared from v18 and the meeting transcript. It does not record a supervisor-approved pivot, allocated experimental resources, or a changed runtime contract.

User-facing counterpart: [Chinese plan](../../human_docs/next_steps_zh.md). Source access and literature leads: [official_docs index](../../official_docs/INDEX.md). Owner: one MPhil researcher, currently the sole project executor. Use relative weeks from the actual start date; confirm the remaining degree time independently of the two-year grant duration in v18.

## Evidence and interpretation

Preserve the existing [v18 source](../../human_docs/research_plan/ai_for_bn_research_plan_v18.tex), bibliography and PDF at their current paths. Read the [reviewed transcript](../../official_docs/meetings/2026-09-09_bn_research/transcripts/transcript_reviewed_zh.txt) with its speaker and terminology caveats. The [source manifest](../../official_docs/meetings/2026-09-09_bn_research/SOURCE_MANIFEST.json) records byte identity; it does not verify the truth of statements in the recording.

| Evidence | Supported interpretation | Not established |
| --- | --- | --- |
| 00:52:48–00:53:49 | An AI entry point and publication-oriented work are desired; formulation improvement is suggested | Publication novelty or acceptance |
| 00:57:06–00:57:43; 01:13:38–01:14:23 | An experimental contact and data/validation collaboration are discussed; separator work is proposed | Delivered data, confirmed identity/spelling of the experimental contact, a booked experiment, or publication permission |
| 01:23:03–01:24:58 | Predictions should be checkable before expensive experiments; simulation is discussed | An available validated simulator, an 80–90% success guarantee, or permission to replace physical evidence with model output |
| 02:12:38–02:17:17 | Later discussion narrows the initial task to dispersion/modification, public-source data and a small polymer-related workflow | A final matrix, chemical recipe, numerical target or a compatible training set |
| 02:23:54–02:24:16; 02:48:06–02:48:21 | Start by organizing data and defining how to judge outputs | A commitment to train a foundation model first |

Working interpretation: separator applications are the application context, while dispersion/modification is the smaller proposed entry task. Their exact experimental connection remains unresolved. Do not silently equate epoxy, polyurethane or silicone examples with the eventual separator slurry. Record competing interpretations until a material-system definition or data sample resolves them. Do not expand uncertain ASR terms such as agent names, functional groups or product codes into a definite formulation.

The transcript's informal publication discussion at 01:25:12–01:25:28 is not a formal degree requirement. Commercial, regulatory, supply-chain and military anecdotes are outside this plan and are not adopted as facts.

## Relationship to v18 and the current repository

| v18 element | Reusable principle | Required change for the proposed task |
| --- | --- | --- |
| Provenance-aware BN data | Source identity, units, compatible labels, missingness | Observe formulation/process/test records instead of only composition and crystal-property rows |
| Grouped formula/family evaluation | Test genuinely unseen groups; keep selection separate from evaluation | Group by original study, laboratory batch and formulation/measurement series as appropriate |
| Uncertainty, support and action labels | Report uncertainty; abstain outside support | Recalibrate against the selected experimental label and its measurement conditions |
| Formula ranking and structure handoff | Trace each recommendation to evidence and a reviewer | Handoff a constrained formulation/process proposal with test conditions; crystal prototypes cannot validate it |
| UV/wide-gap and dielectric targets | Preserve the approved historical proposal | Do not reuse band-gap targets, 30-family/5–10-structure quotas or the old budget as new-task acceptance criteria |

No new subproject or production module is created by this document task. Do not repurpose `main.py`, change the v18 manifest anchors, mix experimental labels into the 2D-material caches, or regenerate research artifacts just to implement this plan. Any later implementation should use the existing module rules and public APIs where semantics actually match, with dedicated data schemas and tests for new behaviour. Avoid wrappers and speculative abstractions.

The current repository contains a BN-themed band-gap PoC and unrelaxed structure prototypes. Checked-in scientific artifacts are historical and lack the v2 completion marker; software checks do not turn them into fresh results. This planning task ran no full scientific pipeline and makes no new model-performance claim.

## Primary question and scope

Proposed question: within one explicit BN dispersion/modification system, can condition-aware data curation and a modest predictive method improve one measured property prediction over simple baselines under a defensible independent-group evaluation, while identifying unsupported inputs?

The first candidate endpoint is viscosity under specified temperature and shear conditions, only if relevant data and partner needs support it. A specified sedimentation/dispersion-stability measurement is an alternative; select one endpoint after the source audit. Viscosity, suspension stability, cured-composite thermal conductivity, separator shrinkage and cell performance must not be pooled as interchangeable labels.

Treat the baseline implementation as a feasibility result. Potential research contributions, still requiring literature comparison, include handling cross-source measurement conditions, uncertainty under source shift, or a later constrained experiment-selection policy. LLM retrieval/extraction can assist data work; generated answers and synthetic labels are not experimental observations. Do not assume that using BN or training another regressor supplies novelty; existing PAO/hBN viscosity ML is already documented in the [source index](../../official_docs/INDEX.md).

Keep foundation-model training, unrestricted recipe generation, cross-polymer generalization, production-line control, intelligent sensing separators and unrelated industry directions outside the initial six-week scope. Experimental suggestions remain subject to materials review.

## Phase A: days 1–14, data feasibility before model commitment

| Work item | Output | Acceptance / stop condition |
| --- | --- | --- |
| A1, days 1–2: problem charter | One matrix/system, controllable inputs, one measured endpoint, intended decision, test protocol, reviewer and resource needs | Unknowns explicit; do not invent a matrix or target from garbled transcript terms |
| A2, days 1–7: partner data request preparation | Request 3–5 concrete public sources and a few original experimental rows, including controls/failures, with test metadata; identify data and validation contacts | Draft only until messaging is authorized; oral willingness is not data access or a scheduled experiment |
| A3, days 3–7: bounded source audit | Aim to screen 10–20 original sources and extract a pilot of 30–50 traceable records | Effort targets, not guaranteed yields or statistical sufficiency. Report independent studies, batches and formulations separately from repeated measurement rows |
| A4, days 8–10: comparability audit | Duplicates, units, concentration bases, missingness, endpoint/protocol compatibility and use permissions | Each retained target has an original source location; keep incompatible domains separate and missing data missing |
| A5, days 11–14: go/no-go | A short feasibility report, a candidate data specification and an evaluation design | Model work requires meaningful compatible labels and a defensible training/selection/evaluation arrangement; otherwise deliver gaps and a narrower task |

Proposed future outputs, not files claimed to exist today: `problem_charter.md`, `source_inventory.csv`, `data_dictionary.md`, `pilot_records.csv`, `feasibility_report.md` under a task-scoped research directory. Keep private partner data in an explicitly excluded location. Decide permanent dataset placement only when its scope, access and schema are known.

Minimum data dictionary:

| Group | Fields to request/extract |
| --- | --- |
| Traceability | Source ID, DOI/URL or partner record, source version/hash, page/table/row, extraction method, original value and unit, reviewer, permitted use |
| Independent units | Study/lab, formulation ID, batch ID, sample ID, replicate/measurement-series ID, duplicate linkage |
| Ingredients | BN morphology/grade/size and size definition; surface treatment; matrix/polymer grade; solvent; additive identity; supplier where relevant |
| Amounts | Component quantities, loading and its mass/volume basis, solids fraction, conversion evidence; no wt%/vol% conversion without required densities |
| Processing | Mixing/sonication/milling method, duration, energy or speed where recorded, temperature, pH and other relevant conditions |
| Targets | Property name, value/unit, measurement method, temperature, shear condition, elapsed time, uncertainty/replicates, controls and failed runs |
| Review state | Missing fields, transcription/terminology uncertainty, inclusion/exclusion reason, incompatible protocol/domain flag |

Preserve both measured values and normalized representations; apply missing-data or scaling transformations only within training folds. If an input would be available only after measuring the target, exclude it from the deployable predictor. Inspect whether source identity merely proxies a label range.

Use existing text/table/metadata tools in Conda `quant`. Save reusable lawful source documents under `official_docs/` with origin and retrieval metadata when acquired. For figure-only values, mark unavailable and seek text tables or original spreadsheets; do not render images, run multimodal extraction or guess numbers. No additional package installation without the user's approval.

## Phase B: weeks 3–4, baseline and honest evaluation

Proceed only after Phase A identifies a coherent dataset. Freeze the endpoint, units, group definition and primary metric before tuning. Start with a training-only mean/median baseline, regularized regression and one suitable nonlinear tabular method. Choose complexity from the available independent observations, not the raw row count.

Keep repeated measurements and near-duplicate recipes together. Separate original studies where testing cross-source use; use batch or formulation holdout for an explicitly narrower within-lab question. Where enough independent groups exist, reserve untouched groups and perform model selection within the training side. With too few groups, report exploratory leave-group-out results and their instability, without claiming validated generalization or calibration.

Report MAE/RMSE in the endpoint's units, per-group errors, relevant residual patterns and support failures. Quantify uncertainty at the independent-group level when the sample size permits it; report limitations when it does not. Estimate intervals/calibration on held-out calibration data within the training side, then assess coverage and width on untouched evaluation data. Do not promise nominal coverage under arbitrary source shift. If ranking is justified, assess meaningful small-K stability and sensitivity, not overlap dominated by selecting nearly the whole candidate set.

Promotion condition: credible improvement over a meaningful baseline on the intended unseen groups, with uncertainty and stability adequate for the decision. Define the minimum practically useful improvement with the materials reviewer using measurement repeatability and decision costs before viewing the final test results. If improvement is absent or indeterminate, document that result and revisit data or scope; do not conceal it with a larger model sweep.

## Phase C: weeks 5–6, bounded validation

Requires an identified reviewer, an explicit feasible parameter region, and agreed validation capacity. Prepare a small set of suggestions (for example 3–5 for discussion, not a fixed promised experimental count), appropriate controls and a protocol. Agree replicate counts, measurements, cost, turnaround and stopping conditions with the experimental owner before work starts.

Freeze suggestions before observing new results. A new batch/lab or prospective measurement is stronger evidence than a retrospective same-source split; report exactly which was obtained. If only existing holdouts are available, label the delivery retrospective. A proposed simulator must have a named implementation, applicable physics, parameter inputs and comparison against relevant measurements before its output is considered validation evidence. A language model judging its own suggestion is not validation.

Deliver a traceable dataset/specification, baseline comparison, limitations and a next validation decision. A materials-facing claim requires the corresponding physical measurement; no synthesis, battery-safety, application-readiness or discovery claim follows from a passing code test.

## Dependencies and fallback decisions

| Dependency | Proposed owner / evidence needed | If absent |
| --- | --- | --- |
| Degree priorities, allowed v18 pivot, remaining timeline | User and supervisor; explicit scope decision | Continue reversible public-source audit; do not revise formal proposal or promise a grant/degree deliverable |
| Material-system and endpoint definition | Collaboration contact plus an identified materials reviewer | Keep competing scopes separate; no quantitative recommendation |
| Partner data and permitted use | Data owner; received sample/schema and usage conditions | Audit public data; do not claim partner availability |
| Enough comparable independent observations | Researcher; completed source/data audit | Narrow the endpoint/system or deliver a provenance-backed knowledge/data-gap study |
| Experimental or validated simulation access | Named validation owner; protocol and capacity | Limit work to retrospective evaluation and label the gap |
| Novel contribution | Researcher with supervisor; comparison to prior work | Treat the baseline as groundwork, not a publication guarantee |

At day 14 choose one main path. If there are only qualitative descriptions or incompatible labels, pause numerical performance promises and retain source-backed retrieval/data organization as an explicitly narrower outcome. If that is insufficient for the degree goal, discuss restoring the v18 track or another small problem. Do not silently run two main projects in parallel.

## Immediate next action and completion state

Next research action: draft A1 and A2 using the four questions at the end of the Chinese plan, then audit concrete public sources while partner answers are pending. Preparing this plan does not itself authorize sending messages, consuming paid APIs, installing packages or executing experiments.

Completed in this documentation task: archive 13 source files with matching hashes; preserve v18's three files; relocate uncertain historical material under `human_docs/deprecated/`; remove the explicitly requested context/README files and obsolete placeholders/prompts; write bilingual plans and source/recovery indexes. No training corpus or new model result is asserted. The local source files are ignored by Git and absent from a fresh clone unless transferred separately.
