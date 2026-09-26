# Model Upgrade Research Notes (2026-04-20)

Purpose: record only the literature findings that are directly useful for the next `ai_for_bn` modeling wave.
This note is AI-facing and should guide implementation choices, not user-facing narrative.

## Why this research is needed now

The current repo has already fixed most reporting-side honesty gaps:
- grouped-by-formula robustness
- BN formula holdout
- BN family holdout
- BN vs non-BN stratified error
- uncertainty / abstention / decision-policy layers
- first-pass structure follow-up artifacts

The remaining bottleneck is no longer missing reporting. It is modeling capacity on the BN subdomain.
Current evidence says:
- overall 2D performance is decent
- BN-focused performance is still weak
- selected and formula-only screening models do not beat the global dummy on the strict BN diagnostics

Therefore, any next major wave should focus on **substantive model improvement**, not more cosmetic artifact layers.

## Hard constraint from this repo

`ai_for_bn` has two different prediction roles:

1. **Overall evaluation**
   - can use structure-aware features when present in the dataset
2. **Formula-only candidate screening**
   - must remain candidate-compatible
   - cannot assume crystal structures for hypothetical formulas

This means a single “modern model” is unlikely to solve both roles cleanly.
A realistic upgrade path should treat them separately:
- one modern **composition-only** model for formula-level screening
- one modern **structure-aware** model for overall evaluation and structure-conditioned follow-up

## Focused literature takeaways

### 1) Composition-only deep models are now a necessary baseline, not a luxury

**CrabNet** remains a serious reference point for composition-only band-gap/property prediction.
A quick external check of the Matbench `matbench_expt_gap` leaderboard still shows CrabNet-family models among the strongest composition-only baselines, with mean MAE around the low 0.3 range on that benchmark and clearly better than classic RF-style baselines.

Practical implication for this repo:
- `matminer_composition + hist_gradient_boosting` is no longer enough as the main formula-only screening model
- a modern composition-only neural baseline is now justified
- CrabNet is the cleanest first candidate because it matches the repo’s core screening constraint: **predict from composition alone**

Useful source checked:
- Matbench `matbench_expt_gap` leaderboard page
- CrabNet repository / original method description

### 2) Newer composition-only work is moving toward transfer/pretraining, not only deeper tabular regressors

A 2025 paper, **“Enhancing composition-based materials property prediction by cross-modal knowledge transfer”** (arXiv:2511.03371), argues that composition-based prediction can be strengthened by transfer from multimodal or structure-aware information.
Its core message is important for `ai_for_bn`:
- composition-only prediction is still crucial for exploring unknown chemical space
- but the best new results are increasingly obtained by **knowledge transfer**, not by staying with plain hand-crafted descriptors + shallow regressors

Practical implication for this repo:
- if a simple CrabNet-style upgrade helps but plateaus, the next serious step should be **pretraining or cross-modal transfer**, not endlessly tuning HGB/LR
- however, this is probably a **second** modeling wave, not the first one, because it is heavier and easier to over-engineer

### 3) Structure-aware SOTA has moved well beyond lightweight tabular structure summaries

Recent structure-aware materials papers are centered around crystal graphs, graph transformers, and hybrid transformer-graph models.
The practical message is straightforward:
- if `ai_for_bn` wants a modern structure-aware evaluation path, lightweight structure summary columns are only an intermediate baseline
- stronger candidates live in the ALIGNN / Matformer / crystal-graph-transformer family

A recent example checked during this scan:
- **“Accelerating materials property prediction via a hybrid Transformer Graph framework that leverages four body interactions”** (npj Computational Materials, 2025)

Practical implication for this repo:
- modern structure-aware modeling is justified for the **overall evaluation** branch
- but it does **not** solve formula-only screening for hypothetical formulas without structures
- therefore structure-aware model upgrades should not be mistaken for a direct fix to the screening bottleneck

### 4) Foundation-model thinking is relevant, but should not be the first implementation step here

Recent reviews on foundation models for chemistry/materials show that pretraining and large-scale atomistic models are becoming central.
This is strategically relevant, but for `ai_for_bn` the immediate question is not “can a foundation model exist?” It is:
- what can improve BN-focused diagnostics in a small, honest repo **without creating a large dependency and compute burden too early**?

Practical implication for this repo:
- foundation-model or pretrained-encoder routes are worth tracking
- but first implementation should remain a **targeted, benchmarkable model upgrade**, not an overly ambitious foundation-model integration

### 5) Evaluation honesty still matters when upgrading models

A 2024 paper on dataset redundancy control (**MD-HIT**) is a useful warning.
Its message is that conventional splits can overestimate model capability when redundant / nearby compositions leak across train and test.

This directly validates the repo’s current direction:
- grouped-by-formula split should be kept
- BN formula holdout should be kept
- BN family holdout should be kept
- BN vs non-BN stratified error should be kept

Practical implication for this repo:
- any new modern model must be judged under the **existing strict evaluation matrix**, not only by overall benchmark improvement
- otherwise the repo risks replacing one optimistic story with another

### 6) Simple/interpretable controls still matter for band gap

A 2025 paper (**“Predicting band gap from chemical composition: A simple learned model for a material property with atypical statistics”**, arXiv:2501.02932) is a reminder that band gap has unusual target statistics and that simple models can still be scientifically informative.

Practical implication for this repo:
- do not delete `linear_regression`, `dummy_mean`, or the BN-local baseline
- old models should remain as **controls** even after modern models are added
- the correct research move is **add modern models on top of the controls**, not replace the controls entirely

## Repo feasibility notes checked after the literature scan

Current repo reality:
- `requirements.txt` is still a lightweight scientific Python stack (`pandas`, `numpy`, `scikit-learn`, `matminer`, `pymatgen`, `streamlit`, etc.)
- there is **no deep-learning runtime** in the current dependency surface yet (`torch` is not part of the current requirements)
- `src/pipeline/features.py` currently wires model creation through a small sklearn-style factory (`LinearRegression`, `HistGradientBoostingRegressor`, `RandomForestRegressor`, `DummyRegressor`)

Practical implication:
- adding CrabNet or a modern structure-aware graph model is **not** a tiny config change
- it is a real modeling wave with new dependency, training, serialization, and test-surface consequences
- this further supports a phased upgrade rather than adding multiple new model families at once

## Decision guidance for the next coding wave

### What seems genuinely justified now

1. **Add one modern composition-only model first**
   - strongest candidate: CrabNet
   - reason: directly targets the current formula-only screening bottleneck

2. **Only after that, consider one modern structure-aware model**
   - candidates: ALIGNN / Matformer / comparable crystal-graph/transformer model
   - reason: improves overall evaluation realism, but does not by itself fix structureless candidate screening

3. **Keep all old baselines**
   - HGB, LR, Dummy, BN-local baseline remain necessary controls

4. **Judge success using the existing strict matrix**
   - overall benchmark
   - grouped robustness
   - BN formula holdout
   - BN family holdout
   - BN vs non-BN stratified error
   - candidate-compatible honesty tables

### What is probably not justified yet

- directly jumping to a very heavy foundation-model integration
- replacing the whole pipeline at once
- adding multiple new model families in a single uncontrolled wave
- using a general chat LLM directly as the main numeric band-gap regressor
- removing the old baselines just because they look old

## Recommended experimental order

### Phase A: low-risk, high-value
- integrate a modern composition-only model as an additional model family
- keep the rest of the pipeline unchanged
- compare against current formula-only HGB under the full strict matrix

### Phase B: only if Phase A is promising
- integrate one modern structure-aware model for overall evaluation
- compare against the current lightweight structure-summary + HGB route

### Phase C: only if A/B still leave a large gap
- explore transfer/pretraining or cross-modal knowledge transfer ideas
- treat this as a separate heavier wave with explicit justification

## Minimal success criteria for any new model

A new model is worth keeping only if it improves at least one of the following **without weakening honesty**:
- BN formula holdout MAE
- BN family holdout MAE
- BN vs non-BN MAE ratio
- formula-only screening quality under candidate-compatible constraints

Improving only the standard overall benchmark is **not enough** anymore.

## External references checked in this scan

1. CrabNet repository / method description
   - https://github.com/anthony-wang/CrabNet
2. Matbench `matbench_expt_gap` leaderboard
   - https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_gap/
3. Rubtsov et al., 2025, *Enhancing composition-based materials property prediction by cross-modal knowledge transfer*
   - https://arxiv.org/html/2511.03371
4. npj Computational Materials, 2025, *Accelerating materials property prediction via a hybrid Transformer Graph framework that leverages four body interactions*
   - https://www.nature.com/articles/s41524-024-01472-7
5. Nature Reviews Chemistry / related foundation-model review context
   - https://www.nature.com/articles/s41570-025-00793-5
6. MD-HIT, 2024, dataset redundancy control warning
   - https://www.nature.com/articles/s41524-024-01426-z
7. Ma et al., 2025, *Predicting band gap from chemical composition: A simple learned model for a material property with atypical statistics*
   - https://arxiv.org/html/2501.02932v1

## Implementation outcome from the first real modern-model wave

A first low-dependency modern-model wave has now been implemented in-repo.

What was actually added:
- a new candidate-compatible feature set: `fractional_composition_vector`
  - 118-dimensional periodic-table fraction vector
- new repo-local PyTorch models:
  - `torch_mlp`
  - `torch_mlp_ensemble`
  - both use sklearn-style `fit` / `predict` interfaces
  - `torch_mlp_ensemble` averages several member seeds and exposes member-level predictions to the uncertainty layer
  - this avoids the old external CrabNet package dependency problems under the current Python/macOS environment

What happened on the real verified run:
- overall best model **did not change**
  - still `matminer_composition_plus_structure_summary + hist_gradient_boosting`
- default formula-only screening model **did not change**
  - still `matminer_composition + hist_gradient_boosting`
- but the best BN candidate-compatible combo **did change**
  - first to `matminer_composition + torch_mlp`
  - then further to `matminer_composition + torch_mlp_ensemble`

Verified BN-facing gains from the current best candidate-compatible neural model:
- BN formula holdout MAE: `1.1257`
  - better than the earlier single-model `torch_mlp = 1.1942`
  - better than `dummy_mean = 1.3257`
  - better than old screening combo `matminer_composition + hist_gradient_boosting = 1.6383`
- BN family holdout MAE: `1.1382`
  - better than the earlier single-model `torch_mlp = 1.2032`
  - better than `dummy_mean = 1.3255`
  - better than old screening combo `1.4923`
- BN vs non-BN grouped error ratio: `1.8109`
  - essentially as good as the single-model result, but now from a more stable multi-seed ensemble view

Interpretation:
- the repo now has a **real candidate-compatible modern neural baseline that matters on strict BN diagnostics**
- this is stronger than the previous state, where no candidate-compatible combo beat the dummy baseline on BN views
- however, it is still **not** evidence that BN-centered generalization is solved
- it is evidence that the repo has finally produced one credible modern-model direction worth pushing further

## Implementation outcome from the attention-style follow-up pilots

After the initial `torch_mlp` / `torch_mlp_ensemble` wave, two further short pilot-only composition models were tested instead of being dropped blindly into the full pipeline.

### Dense fractional attention pilot

A pilot script was run on a reduced subset (`320` formulas / `455` rows) comparing:
- `torch_fractional_attention`
- `torch_mlp_ensemble`
- classic candidate-compatible controls

Observed BN-slice pilot result:
- `torch_mlp_ensemble`: `MAE ≈ 1.575`
- `torch_fractional_attention`: `MAE ≈ 1.689`
- `dummy_mean`: `MAE ≈ 1.398`

Interpretation:
- the dense fractional-attention variant did **not** justify broader rollout
- it should remain experimental only
- this was **not** a GPU-blocked success case; it was a weak-result case

### Sparse present-elements attention pilot

A second pilot implemented `torch_sparse_fractional_attention`, intended to be more CrabNet-like by attending only over present elements rather than all 118 positions.

Observed short pilot result (`106` formulas / `152` rows):
- `hist_gradient_boosting`: `MAE ≈ 1.299`
- `dummy_mean`: `MAE ≈ 1.306`
- `torch_mlp_ensemble`: `MAE ≈ 1.434`
- `torch_fractional_attention`: `MAE ≈ 1.442`
- `torch_sparse_fractional_attention`: `MAE ≈ 1.710`

Interpretation:
- the sparse attention variant also failed the short BN-slice pilot
- worse, validation on the tiny pilot selected it, but BN-slice evidence did not support it
- that suggests a selection-stability problem, not merely underpowered hardware
- therefore it should remain an experimental code path and should **not** enter the default candidate sweep

### Practical lesson from these two pilots

Short pilots are doing exactly what they are supposed to do in this repo:
- reject attractive but weak composition-model ideas early
- avoid wasting time on long `main.py` runs for models that have not yet earned that cost
- distinguish **needs GPU because promising** from **does not need GPU because not promising enough yet**

The current attention evidence supports the following conclusion:
- do **not** spend more time on transformer-style fractional-composition variants as the next main wave
- the better next composition-only direction is a **Roost-style present-elements stoichiometry model** with lightweight message passing / weighted soft-attention over only the elements present in the formula
- if that Roost-like path later shows clear promise but becomes compute-limited, that is the point where asking for the user's ready GPU machine would be justified

## Updated next-step recommendation

Given the short-pilot failures above, the best current composition-only modeling order is:

1. keep `torch_mlp_ensemble` as the current strongest verified candidate-compatible neural baseline
2. treat both dense and sparse fractional-attention variants as experimental only
3. try one compact **Roost-like** repo-local model next
4. only escalate to GPU-backed heavier work if that Roost-like pilot produces evidence stronger than the current ensemble baseline

## Bottom line

The literature scan supports a restrained conclusion:
- yes, this repo is now at the point where a **modern model upgrade is scientifically justified**
- but the first justified move is **not** “pick the flashiest model”
- the first justified move is to add a **composition-only modern model** that directly targets the formula-only screening role, while preserving strict BN diagnostics and old baselines as controls
- short pilot evidence now further narrows that recommendation: the next serious formula-only experiment should be **Roost-like present-elements message passing**, not another fractional-composition transformer variant

## Implementation outcome from the Roost-like follow-up pilots

A repo-local `torch_roost_like` path was added as an experimental present-element stoichiometry model over `fractional_composition_vector`.
It was evaluated only through short pilots, not promoted directly into the default model sweep.

### First Roost-like pilot

Pilot subset:
- `341` rows
- `240` formulas
- `10` BN formulas in the BN-slice leave-one-formula-out evaluation

Observed test benchmark on the pilot subset:
- `matminer_composition + hist_gradient_boosting`: `MAE ≈ 0.572`
- `fractional_composition_vector + torch_mlp_ensemble`: `MAE ≈ 0.825`
- `fractional_composition_vector + torch_roost_like`: `MAE ≈ 0.841`

Observed BN-slice result on the same pilot subset:
- `dummy_mean`: `MAE ≈ 1.344`
- `matminer_composition + hist_gradient_boosting`: `MAE ≈ 1.616`
- `fractional_composition_vector + torch_mlp_ensemble`: `MAE ≈ 1.477`
- `fractional_composition_vector + torch_roost_like`: `MAE ≈ 1.378`

Interpretation:
- the Roost-like model was **more promising than the earlier fractional attention variants**
- it also slightly improved over the same-pilot `torch_mlp_ensemble`
- but it still **did not beat the dummy baseline** on BN-slice
- therefore it did **not** earn default rollout

### Small Roost-like config sweep

A tiny configuration sweep tested three short-run settings:
- `roost_like_small`
- `roost_like_medium`
- `roost_like_wider`

Observed BN-slice MAE:
- `small`: `≈ 1.378`
- `medium`: `≈ 1.398`
- `wider`: `≈ 2.071`
- `dummy_mean`: `≈ 1.344`

Interpretation:
- increasing local capacity did **not** rescue the model
- the current evidence does **not** support the claim that Roost-like is merely “GPU-blocked but otherwise ready”
- the more honest conclusion is that this direction showed **some signal**, but still not enough signal

## Implementation outcome from a zero-code local-baseline pilot

A no-integration kNN pilot was run as a cheap sanity check before adding more code.

Best result found:
- `fractional_composition_vector + k=7 + distance`
- test `MAE ≈ 0.858`
- BN-slice `MAE ≈ 1.881`

Interpretation:
- local traditional-neighborhood regression is **not** the next meaningful upgrade path here
- this pilot usefully ruled out a low-cost alternative without complicating the codebase

## Implementation outcome from TabPFN feasibility checking

A more modern small-tabular route, **TabPFN**, was checked next because its design is far better aligned with the repo's current bottleneck than continuing to hand-design small composition architectures.

What was completed:
- `tabpfn==7.1.1` was installed successfully in the `quant` environment
- the Python API import succeeded

What blocked actual pilot execution:
- local model-weight download raised `TabPFNLicenseError`
- the issue was **not** local compute
- the issue was a one-time Prior Labs license acceptance plus missing `TABPFN_TOKEN`

Practical interpretation:
- TabPFN remains a plausible next short pilot
- but the repo cannot honestly claim to have evaluated it yet
- the next requirement is user-side license/token setup, not GPU escalation

## Updated decision guidance after the Roost-like / kNN / TabPFN checks

The modeling order should now be read as:

1. keep `matminer_composition + torch_mlp_ensemble` as the strongest currently verified candidate-compatible neural control
2. keep dense attention, sparse attention, and Roost-like as **experimental evidence-gathering paths**, not default-rollout winners
3. do **not** spend more time on kNN-style local baselines
4. if a next small-tabular modern baseline is desired, `TabPFN` is the cleanest currently identified candidate
5. if `TabPFN` later shows strong BN-slice signal but becomes compute-limited, *that* would be a justified moment to ask for a GPU machine
6. until then, the current blocker is license/token access, not hardware
