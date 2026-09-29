# Frozen initial task (historical)

The text below preserves the initial frozen PP task and evaluation. Current task domains, numerical methods, languages and operating limits are in the [prototype index](INDEX.md). In particular, the current raw-BNNT range is ≤0.3 mg/cm²; the historical exploratory range below is not the live API contract.

Working application: a BN separator formulation assistant. The public component
accepts structured ingredients, BN loading, binder ratio, drying conditions and
electrolyte identity. It returns source records, condition checks, simple numeric
baselines and, for the narrow supported protocol, a GPT-6 Astra hypothesis.

Primary numeric output: ionic conductivity in mS/cm for a PP separator coated with
raw/purified BNNT using PVDF/NMP, BN:PVDF 4:1, sonication for one hour, stirring
overnight and vacuum drying at 50°C for 24 hours. The electrolyte is 1 M LiTFSI
with 1 wt% LiNO3 in DOL:DME 1:1 v/v. Measurement temperature is unspecified in the
available source. The exploratory request range is 0–0.5 mg/cm², with zero loading
allowed only for the bare control. Purified BNNT is a preparation change with no
training label of its own; predictions for it are expressly unvalidated.

The interface additionally retrieves CA, CNF and solid-polymer records, but rejects
numeric transfer to these systems, unknown solvents, different electrolytes,
specified temperatures, unsupported loadings or changed process conditions.
Coating is a formulation/process task, not a reactant-to-product prediction.

Evidence coverage decides the initial endpoint: PP shrinkage values were not
available as sufficient matching text labels. We therefore retain the plan's PP
ionic-transport task and explicitly report insufficient training coverage instead
of merging incompatible systems. The old band-gap dataset is not used.

Evaluation is frozen as two training formulations and one unused formulation from
one study. `frozen_prompt.txt` contains exactly the application-visible inputs
before inference. Held-out answers, full paragraphs containing them, and the
public retrieval response are excluded from the prompt. Mean, nearest-recipe and loading-only linear-regression
baselines use the same two training labels. The linear comparator was added during
review of the first frozen result; the Astra response was not rerun or selected. There is no tuning, learned
preprocessing, uncertainty calibration or repeated best-result selection.

Cross-study evaluation and laboratory acceleration remain unfinished research
questions. Next acquisition priority: obtain the methods/supplements or permitted
manuscripts of the PP studies in `source_inventory.csv`, record electrolyte,
temperature, thickness, loading/binder basis and independent outcomes, then reserve
an entire new study/batch before expanding the model. Prospective validation must
freeze recommendations before measuring outcomes, compare against the partner's
actual selection workflow, and log experiment counts, time and cost.
