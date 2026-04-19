# PY_FILES_SUMMARY.md

AI-facing quick summary for the current Python surface of `ai_for_bn`.
This project is **not** maintained as a Python package. Default usage is from the repo root in the `quant` environment.

---

## main.py

### `main()`
Linear project entrypoint.

What it does, in order:
1. clears project cache
2. loads config
3. builds or reloads the normalized dataset
4. creates grouped split masks
5. builds all configured feature tables
6. selects `{feature_set} x {model_type}` on validation
7. retrains the selected overall-evaluation combo on `train + val`
8. benchmarks all configured combos on `test`
9. runs grouped-by-formula robustness benchmarking across configured combos
10. runs a dedicated BN-focused leave-one-BN-formula-out benchmark across configured combos plus baselines
11. builds formula-only candidate uncertainty stats
12. ranks candidates with the best candidate-compatible combo
13. writes metrics / plots / benchmark / robustness / BN-benchmark / ranking / shortlist artifacts

Important:
- overall evaluation combo and formula-only screening combo may differ
- `main.py` is intentionally kept linear and notebook-friendly

---

## src/core/io_utils.py

### `load_config(path)`
Loads the Python config module and returns its `CONFIG` dict.

### `ensure_runtime_dirs(cfg)`
Creates runtime directories such as the artifact folder if missing.

### `clear_project_cache(project_root_path='.')`
Delegates cache cleanup for this repo.
Use before tests or batch runs, per project skill requirements.

---

## src/pipeline/data.py

### `load_or_build_dataset(cfg)`
Builds the normalized dataframe from raw JARVIS / 2DMatPedia data or reloads the processed cache.
Returns:
- normalized dataframe
- manifest dict

Current behavior:
- prefers cached processed parquet when it already has the required normalized columns
- rebuilds stale processed cache from cached raw JSON when needed
- downloads from JARVIS only when cached raw JSON is absent
- writes lightweight structure-summary columns derived from cached `atoms` / lattice data

Important normalized columns include:
- `record_id`
- `source`
- `formula`
- `target`
- `energy_per_atom`
- `exfoliation_energy_per_atom`
- `total_magnetization`
- `abs_total_magnetization`
- `structure_n_sites`
- `structure_lattice_a`
- `structure_lattice_b`
- `structure_lattice_c`
- `structure_lattice_gamma`
- `structure_inplane_area`
- `structure_cell_height`
- `structure_thickness`
- `structure_vacuum`
- `structure_areal_number_density`
- `structure_thickness_fraction`

---

## src/pipeline/features.py

### `extract_elements(formula)`
Regex-based element-token extraction from a chemical formula string.

### `filter_bn(df, formula_col='formula')`
Returns the BN-themed slice, defined here as formulas containing both `B` and `N`.

### `generate_bn_candidates(cfg=None)`
Builds the current BN-anchored demo candidate space.
Default space is no longer the plain Group 13 / Group 15 cartesian product. It is now a 25-formula
BN-containing formula-family grid anchored by:
- BCN / h-BCN-style ternary motifs
- BC2N-style ternary motifs
- Si2BN-like motifs already observed in the dataset

The generator also writes source-space provenance fields such as:
- `candidate_generation_strategy`
- `candidate_space_name`
- `candidate_space_kind`
- `candidate_family`
- `candidate_template`
- `candidate_family_note`

And it adds lightweight formula-level chemical-plausibility annotations, including:
- `chemical_plausibility_pass`
- `chemical_plausibility_guess_count`
- `chemical_plausibility_primary_oxidation_state_guess`
- `chemical_plausibility_note`

### `get_candidate_feature_sets(cfg)`
Returns the ordered feature-set search space from config.
Current default search space:
- `basic_formula_composition`
- `matminer_composition`
- `matminer_composition_plus_structure_summary`

### `get_candidate_screening_feature_sets(cfg)`
Returns only the candidate-compatible feature sets.
Current default result:
- `basic_formula_composition`
- `matminer_composition`

### `get_candidate_model_types(cfg)`
Returns the ordered model-type search space from config.
Current default search space:
- `hist_gradient_boosting`
- `linear_regression`

### `build_feature_table(df, formula_col='formula', feature_set='basic_formula_composition')`
Builds one feature table for one configured feature representation.
Adds:
- feature columns
- `feature_set`
- `feature_generation_failed`
- `feature_generation_error`

Supported feature sets:
- `basic_formula_composition`
- `matminer_composition`
- `matminer_composition_plus_structure_summary`

Current feature counts:
- `basic_formula_composition`: 7
- `matminer_composition`: 19
- `matminer_composition_plus_structure_summary`: 30

### `build_feature_tables(df, cfg, formula_col='formula')`
Builds all configured feature tables at once and returns a `{feature_set: dataframe}` mapping.

### `make_split_masks(df, cfg)`
Builds split masks.
Current important mode:
- `group_by_formula`

Also stores split metadata including overlap counts.

### `summarize_feature_table(feature_df, feature_set=None)`
Returns metadata for one feature table, including:
- `feature_family`
- `candidate_compatible`
- `n_features`
- `status`
- whether the feature set is selection-eligible
- failed formula examples if featurization was incomplete

### `make_model(cfg, model_type=None)`
Factory for supported regressors.
Currently supports:
- `linear_regression`
- `hist_gradient_boosting`
- `random_forest`
- `dummy_mean`

### `train_baseline_model(df, split_masks, cfg, model_type=None, include_validation=False)`
Fits a regressor on the requested split scope and returns:
- trained model
- feature column list

### `evaluate_predictions(df, split_masks, model, feature_columns, split_name='test')`
Runs prediction on one split and returns:
- metrics dict (`mae`, `rmse`, `r2`)
- row-level prediction dataframe

Fails loudly if the requested feature set cannot evaluate every row in that split.

### `select_feature_model_combo(feature_tables, split_masks, cfg)`
Core validation-time selection routine.
Searches the configured `{feature_set} x {model_type}` space and returns a structured summary with:
- best overall evaluation feature set and model type
- best formula-only screening feature set and model type
- whether screening reuses the overall best combo
- per-feature-set status
- validation results for every candidate combo

Important:
- overall evaluation can select the structure-aware route
- formula-only screening is restricted to candidate-compatible feature sets

### `select_model_type(feature_tables, split_masks, cfg)`
Backward-compatible alias that forwards to `select_feature_model_combo(...)`.

### `benchmark_regressors(feature_tables, split_masks, cfg, selected_feature_set, selected_model_type)`
Evaluates the candidate feature/model combos plus dummy baseline on the test split and returns the benchmark dataframe.

Useful benchmark columns include:
- `feature_set`
- `feature_family`
- `candidate_compatible`
- `n_features`
- `model_type`
- `benchmark_role`
- `selected_by_validation`
- `benchmark_status`
- `mae`
- `rmse`
- `r2`

### `benchmark_grouped_robustness(feature_tables, cfg, selected_feature_set, selected_model_type)`
Runs grouped-by-formula cross-validation robustness benchmarking over the configured feature/model combos.
This is the new layer that checks whether the evaluation story survives more than one split.

Useful robustness columns include:
- `feature_set`
- `feature_family`
- `candidate_compatible`
- `model_type`
- `benchmark_role`
- `selected_by_validation`
- `robustness_method`
- `robustness_group_column`
- `requested_folds`
- `actual_folds`
- `completed_folds`
- `robustness_status`
- `mae_mean`
- `mae_std`
- `rmse_mean`
- `rmse_std`
- `r2_mean`
- `r2_std`

### `benchmark_bn_slice(dataset_df, feature_tables, cfg, selected_feature_set, selected_model_type, screening_feature_set, screening_model_type)`
Runs the dedicated BN-focused leave-one-BN-formula-out benchmark.
It returns two dataframes:
- a benchmark summary table across all configured feature/model combos plus `dummy_mean` and `bn_local_knn_mean`
- a per-held-out-formula prediction table for deeper inspection

Important:
- this exists because the standard grouped split can place all BN rows in train
- it is a small-sample BN-centered diagnostic, not a definitive BN-only benchmark
- the BN-slice best combo can differ from both the overall selected combo and the formula-only screening combo

Useful benchmark columns include:
- `feature_set`
- `feature_family`
- `candidate_compatible`
- `model_type`
- `benchmark_role`
- `selected_by_validation`
- `bn_slice_method`
- `bn_slice_train_scope`
- `bn_formula_count`
- `bn_row_count`
- `completed_holds`
- `benchmark_status`
- `mae`
- `rmse`
- `r2`
- `k_neighbors`

### `select_bn_centered_candidate_screening_combo(bn_slice_benchmark_df, cfg, fallback_feature_set=None, fallback_model_type=None)`
Selects the **best candidate-compatible BN-centered screening combo** from the BN-slice benchmark.
Current behavior:
- filters to `benchmark_status == 'ok'`
- keeps only candidate-compatible feature/model rows
- excludes the global dummy and BN-local reference baselines
- picks the lowest-MAE row
- returns metadata for the new BN-centered alternative ranking artifact

Important:
- this is not a new global selection rule for the whole project
- it exists to build a **BN-centered comparison view** alongside the default general ranking
- the chosen combo can differ from both the overall selected combo and the default formula-only screening combo

### `build_candidate_structure_generation_seeds(candidate_df, dataset_df, split_masks, cfg=None, bn_centered_candidate_df=None, formula_col='formula')`
Builds a **structure-generation bridge artifact** from the current screening outputs.
Current behavior:
- starts from the union of:
  - the general proposal shortlist
  - the formula-level extrapolation shortlist
  - the top-`n` BN-centered alternative ranking view
- parses nearby BN analog formulas already attached to each candidate
- maps each analog formula to a deterministic exemplar `train + val` BN reference record with observed structure summary columns
- writes a candidate-to-prototype seed table for downstream substitution / enumeration / relaxation work

Useful output fields include:
- `structure_generation_candidate_priority_reason`
- `bn_centered_ranking_rank`
- `seed_reference_formula`
- `seed_reference_record_id`
- `seed_reference_band_gap`
- `seed_formula_shared_elements`
- `seed_formula_candidate_only_elements`
- `seed_formula_seed_only_elements`
- `seed_formula_element_count_l1_distance`
- `seed_formula_edit_strategy`
- `seed_reference_energy_per_atom`
- `seed_reference_exfoliation_energy_per_atom`
- `seed_reference_structure_n_sites`
- `seed_reference_structure_inplane_area`
- `seed_reference_structure_thickness`

Important:
- this does **not** generate or validate new structures
- it is a handoff artifact that makes the pipeline less purely formula-only by surfacing concrete BN analog prototypes for follow-up work

### `build_candidate_prediction_ensemble(candidate_df, feature_tables, split_masks, cfg, candidate_feature_sets=None)`
Trains the tiny candidate-compatible feature/model pool on `train + val` and computes candidate-level ensemble prediction statistics:
- `ensemble_predicted_band_gap_mean`
- `ensemble_predicted_band_gap_std`
- `ensemble_member_count`

Important:
- this is a **small disagreement heuristic**, not calibrated physical uncertainty
- by default it only uses formula-only feature sets that can featurize candidates

### `build_candidate_grouped_robustness_predictions(candidate_df, feature_df, split_masks, cfg, feature_set, model_type, formula_col='formula')`
Builds candidate-level grouped-fold prediction statistics from the selected formula-only screening combo.
This layer is trained only on `train + val` reference formulas, so it does not leak test labels into candidate ranking.

Useful fields include:
- `grouped_robustness_prediction_enabled`
- `grouped_robustness_prediction_method`
- `grouped_robustness_prediction_fold_count`
- `grouped_robustness_predicted_band_gap_mean`
- `grouped_robustness_predicted_band_gap_std`

Interpretation:
- this is a **split-robustness heuristic**, not calibrated uncertainty
- large std means the candidate moves more across grouped-by-formula training folds

### `annotate_candidate_dataset_overlap(candidate_df, dataset_df, split_masks=None, formula_col='formula')`
Adds honesty-oriented candidate annotations such as:
- `seen_in_dataset`
- `dataset_formula_row_count`
- `seen_in_train_plus_val`
- `train_plus_val_formula_row_count`

Use this to distinguish demo-space rediscovery from true formula-level novelty.

### `annotate_candidate_novelty(candidate_df, formula_col='formula')`
Builds a simple novelty / rediscovery layer from the overlap fields.
This is still formula-level novelty inside the current demo candidate space, not validated discovery.
Adds:
- `candidate_is_seen_in_dataset`
- `candidate_is_seen_in_train_plus_val`
- `candidate_is_formula_level_extrapolation`
- `candidate_novelty_bucket`
- `candidate_novelty_priority`
- `candidate_novelty_note`

Current novelty buckets:
- `train_plus_val_rediscovery`
- `held_out_known_formula`
- `formula_level_extrapolation`

### `annotate_candidate_proposal_shortlist(ranked_candidate_df, cfg=None)`
Builds a separate advisor-facing shortlist from the raw ranking.
Current behavior:
- walks the raw ranking in order
- keeps chemical-plausibility priority
- caps repeats from the same `candidate_family`
- does not replace or mutate the full ranking artifact

Useful fields include:
- `proposal_shortlist_selected`
- `proposal_shortlist_rank`
- `proposal_shortlist_decision`
- `proposal_shortlist_family_count_before_selection`

### `annotate_candidate_extrapolation_shortlist(ranked_candidate_df, cfg=None)`
Builds a second advisor-facing shortlist that only considers `formula_level_extrapolation` candidates.
Current behavior:
- filters to the configured novelty bucket
- keeps chemical-plausibility priority
- applies an explicit `candidate_family` cap for diversity
- leaves the raw ranking artifact and general proposal shortlist untouched

Useful fields include:
- `extrapolation_shortlist_target_novelty_bucket`
- `extrapolation_shortlist_selected`
- `extrapolation_shortlist_rank`
- `extrapolation_shortlist_decision`
- `extrapolation_shortlist_family_count_before_selection`

### `annotate_candidate_domain_support(candidate_feature_df, reference_feature_df, split_masks, feature_columns, cfg=None, formula_col='formula')`
Annotates formula-only candidates with a lightweight train+val feature-space support signal.
Useful fields include:
- `domain_support_nearest_formula`
- `domain_support_nearest_distance`
- `domain_support_mean_k_distance`
- `domain_support_percentile`
- `domain_support_penalty`

### `annotate_candidate_bn_support(candidate_feature_df, reference_feature_df, split_masks, feature_columns, cfg=None, formula_col='formula')`
Annotates candidates against the **known BN slice only**.
This is the new BN-local support layer that tries to make BN part of the screening logic,
not just part of the report wording.
Useful fields include:
- `bn_support_nearest_formula`
- `bn_support_neighbor_formulas`
- `bn_support_nearest_distance`
- `bn_support_mean_k_distance`
- `bn_support_percentile`
- `bn_support_penalty`

### `annotate_candidate_bn_analog_evidence(candidate_df, dataset_df, split_masks, cfg=None, formula_col='formula')`
Adds observed-property evidence from nearby BN-containing train+val formulas.
This is meant to ground the ranking in **actual BN analog evidence**, not just feature-space distance.
Useful fields include:
- `bn_analog_nearest_formula`
- `bn_analog_neighbor_formulas`
- `bn_analog_nearest_band_gap`
- `bn_analog_nearest_energy_per_atom`
- `bn_analog_nearest_exfoliation_energy_per_atom`
- `bn_analog_neighbor_band_gap_mean`
- `bn_analog_neighbor_energy_per_atom_mean`
- `bn_analog_neighbor_exfoliation_energy_per_atom_mean`
- `bn_analog_exfoliation_support_label`
- `bn_analog_energy_support_label`
- `bn_analog_abs_total_magnetization_support_label`
- `bn_analog_support_vote_count`
- `bn_analog_support_available_metric_count`
- `bn_analog_validation_label`
- `bn_analog_validation_support_fraction`
- `bn_analog_validation_penalty`

### `screen_candidates(candidate_df, model, feature_columns, cfg, feature_set, model_type, best_overall_feature_set=None, best_overall_model_type=None, screening_selection_note=None, dataset_df=None, split_masks=None, ensemble_prediction_df=None, grouped_robustness_prediction_df=None, reference_feature_df=None)`
Builds the final demo ranking dataframe.
Current behavior:
- keeps the full candidate source pool in the artifact instead of silently dropping failed formulas
- sorts candidates by `chemical_plausibility_pass` first, then by ranking score
- marks the reported top-k explicitly with `screening_selected_for_top_k`
- records why a formula was or was not selected via `screening_selection_decision`
- predicts with the selected formula-only screening model
- merges small-pool disagreement statistics
- optionally merges grouped-fold candidate robustness predictions from the selected formula-only screening combo
- optionally merges dataset-overlap annotations
- optionally adds train+val feature-space domain-support annotations
- optionally adds BN-local support annotations against the known BN slice
- optionally adds BN analog-evidence annotations from observed BN reference properties
- optionally derives a lightweight BN analog-validation label from analog exfoliation / energy / magnetization alignment
- optionally applies a mild grouped candidate-robustness penalty in ranking space from fold-to-fold prediction spread
- optionally applies a mild BN analog-validation penalty in ranking space from the analog vote fraction
- adds novelty / rediscovery annotations from the overlap fields
- computes `ranking_score` and preserves `ranking_score_before_grouped_robustness_penalty`, `ranking_score_before_domain_support_penalty`, `ranking_score_before_bn_support_penalty`, and `ranking_score_before_bn_analog_validation_penalty`
- writes explicit honesty fields about whether screening matches the overall best evaluation combo
- annotates a family-aware proposal shortlist without replacing the raw ranking artifact
- annotates a second formula-level extrapolation shortlist for discovery-style follow-up inside the current demo space
- keeps the full ranking artifact and exposes novelty ranks instead of truncating the file to top-k only

Useful output columns include:
- `predicted_band_gap`
- `ensemble_predicted_band_gap_mean`
- `ensemble_predicted_band_gap_std`
- `ranking_score`
- `grouped_robustness_predicted_band_gap_std`
- `grouped_robustness_uncertainty_penalty`
- `domain_support_percentile`
- `bn_support_percentile`
- `bn_analog_neighbor_exfoliation_energy_per_atom_mean`
- `bn_analog_exfoliation_support_label`
- `bn_analog_validation_label`
- `bn_analog_validation_penalty`
- `candidate_novelty_bucket`
- `novelty_rank_within_bucket`
- `novel_formula_rank`
- `proposal_shortlist_selected`
- `proposal_shortlist_rank`
- `extrapolation_shortlist_selected`
- `extrapolation_shortlist_rank`
- `ranking_feature_set`
- `ranking_model_type`
- `best_overall_evaluation_feature_set`
- `best_overall_evaluation_model_type`
- `screening_matches_best_overall_evaluation`
- `screening_selection_note`

---

## src/pipeline/reporting.py

### `build_experiment_summary(dataset_df, bn_df, candidate_df, split_masks, selection_summary, cfg, robustness_df=None, bn_slice_benchmark_df=None, bn_centered_candidate_df=None, bn_centered_screening_selection=None, structure_generation_seed_df=None)`
Builds the structured experiment summary dict written to `artifacts/experiment_summary.json`.
Includes:
- dataset stats
- feature summary
- joint feature/model selection summary
- benchmark metadata
- grouped robustness metadata
- BN-slice benchmark metadata
- candidate ranking metadata
- BN-centered alternative ranking metadata and overlap diagnostics versus the default ranking

Important:
- preserves the distinction between best overall evaluation combo and formula-only screening combo
- now also records candidate-space provenance such as `candidate_generation_strategy` and `candidate_family_counts`
- now also summarizes BN-local support metadata so the screening story is not purely global-data driven
- now also summarizes BN analog-evidence metadata from observed BN reference properties
- now also summarizes analog-validation label counts derived from BN reference property alignment
- now also summarizes whether the BN analog-validation proxy is active in ranking and how many rows were penalized
- now also summarizes grouped-by-formula robustness results for the selected model, screening fallback, and dummy baseline
- now also summarizes the BN-slice benchmark, including standard-split BN row placement, selected/screening/baseline BN metrics, and the current BN-slice best configured combo
- now also summarizes the BN-centered alternative ranking view, including the chosen candidate-compatible combo, rank-shift statistics, and top-k overlap against the default ranking
- now also summarizes the structure-generation bridge artifact, including seeded candidate count, seed row count, unique BN reference prototype count, the JSON handoff artifact path, and the reference-record payload artifact path
- now also summarizes grouped candidate-robustness penalty settings, fold count, average spread, and penalized-row count
- now also summarizes the family-aware proposal shortlist and the formula-level extrapolation shortlist as separate advisor-facing outputs

### `save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, benchmark_df, robustness_df, bn_slice_benchmark_df, bn_slice_prediction_df, bn_centered_screened_df, structure_generation_seed_df, experiment_summary, manifest, cfg)`
Writes the main artifact files under `artifacts/`.
This now includes both shortlist CSVs, BN-slice benchmark artifacts, the BN-centered alternative ranking artifact, and the structure-generation bridge artifacts in addition to the full ranking artifact:
- `bn_slice_benchmark_results.csv`
- `bn_slice_predictions.csv`
- `demo_candidate_bn_centered_ranking.csv`
- `demo_candidate_structure_generation_seeds.csv`
- `demo_candidate_structure_generation_handoff.json`
- `demo_candidate_structure_generation_reference_records.json`
- `demo_candidate_proposal_shortlist.csv`
- `demo_candidate_extrapolation_shortlist.csv`

### `save_basic_plots(prediction_df, cfg)`
Writes the parity plot.

---

## apps/streamlit_app.py

No reusable exported functions.
It is a simple artifact viewer for:
- `metrics.json`
- `experiment_summary.json`
- `benchmark_results.csv`
- `robustness_results.csv`
- `bn_slice_benchmark_results.csv`
- `bn_slice_predictions.csv`
- `predictions.csv`
- `demo_candidate_ranking.csv`
- `demo_candidate_bn_centered_ranking.csv`
- `demo_candidate_structure_generation_seeds.csv`
- `demo_candidate_structure_generation_handoff.json`
- `demo_candidate_structure_generation_reference_records.json`
- `demo_candidate_proposal_shortlist.csv`
- `demo_candidate_extrapolation_shortlist.csv`

---

## Practical notes

- Keep `main.py` linear and notebook-friendly.
- Human-facing docs should track verified runtime behavior, not planned behavior.
- Before any commit or stage-worthy milestone, run:
  - cache clear via `clear_project_cache('.')`
  - `pytest -q`
  - `quant/bin/python main.py`
