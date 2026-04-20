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
- a new repo-local PyTorch model: `torch_mlp`
  - sklearn-style `fit` / `predict` interface
  - avoids the old external CrabNet package dependency problems under the current Python/macOS environment

What happened on the real verified run:
- overall best model **did not change**
  - still `matminer_composition_plus_structure_summary + hist_gradient_boosting`
- default formula-only screening model **did not change**
  - still `matminer_composition + hist_gradient_boosting`
- but the best BN candidate-compatible combo **did change**
  - now `matminer_composition + torch_mlp`

Verified BN-facing gains from the new candidate-compatible neural model:
- BN formula holdout MAE: `1.1942`
  - better than `dummy_mean = 1.3257`
  - better than old screening combo `matminer_composition + hist_gradient_boosting = 1.6383`
- BN family holdout MAE: `1.2032`
  - better than `dummy_mean = 1.3255`
  - better than old screening combo `1.4923`
- BN vs non-BN grouped error ratio: `1.81`
  - better than old screening combo ratio `2.57`

Interpretation:
- the repo now has a **real candidate-compatible modern neural baseline that matters on strict BN diagnostics**
- this is stronger than the previous state, where no candidate-compatible combo beat the dummy baseline on BN views
- however, it is still **not** evidence that BN-centered generalization is solved
- it is evidence that the repo has finally produced one credible modern-model direction worth pushing further

## Bottom line

The literature scan supports a restrained conclusion:
- yes, this repo is now at the point where a **modern model upgrade is scientifically justified**
- but the first justified move is **not** “pick the flashiest model”
- the first justified move is to add a **composition-only modern model** that directly targets the formula-only screening role, while preserving strict BN diagnostics and old baselines as controls
