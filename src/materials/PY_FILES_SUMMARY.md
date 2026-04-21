# materials module public surface

This file lists the documented public functions that other modules or top-level entrypoints may call from `materials`.
Anything underscore-prefixed or omitted here should be treated as internal implementation detail.

## data.py

- `load_cached_raw_record_lookup(cfg)`
  - Load the cached raw-record lookup used by downstream artifact writers.
- `load_or_build_dataset(cfg)`
  - Build or reload the normalized dataset and its manifest.

## candidate_space.py

- `extract_elements(formula)`
  - Parse element symbols from a chemical formula string.
- `filter_bn(df, formula_col='formula')`
  - Keep only BN-containing rows from a dataframe.
- `annotate_bn_families(df, formula_col='formula', grouping_method=...)`
  - Add BN-family labels for BN-local grouping logic.
- `generate_bn_candidates(cfg)`
  - Build the configured BN candidate space.
- `annotate_candidate_proposal_shortlist(ranked_candidate_df, cfg=None)`
  - Add the family-aware proposal-shortlist annotations.
- `annotate_candidate_extrapolation_shortlist(ranked_candidate_df, cfg=None)`
  - Add the formula-level extrapolation-shortlist annotations.
- `get_screening_ranking_metadata(cfg)`
  - Return ranking-metadata settings used by candidate screening.
- `annotate_candidate_chemical_plausibility(candidate_df, cfg=None, formula_col='formula')`
  - Add formula-level plausibility annotations.

## feature_building.py

- `get_candidate_feature_sets(cfg)`
  - Return the configured feature-set search space.
- `get_candidate_screening_feature_sets(cfg)`
  - Return candidate-compatible feature sets for formula-only screening.
- `get_candidate_model_types(cfg)`
  - Return the configured model-type search space.
- `compatible_model_types_for_feature_set(cfg, feature_set)`
  - Return model types compatible with a given feature set.
- `model_type_supports_feature_set(model_type, feature_set)`
  - Check feature/model compatibility.
- `incompatible_model_feature_note(model_type, feature_set)`
  - Return the human-readable incompatibility explanation.
- `get_feature_family(feature_set)`
  - Return the feature-family label for a feature set.
- `feature_set_supports_formula_only_screening(feature_set)`
  - Report whether a feature set is candidate-compatible.
- `get_feature_note(feature_set)`
  - Return the descriptive note for a feature set.
- `build_feature_table(df, feature_set, formula_col='formula')`
  - Build one feature table for a given feature set.
- `build_feature_tables(df, cfg, formula_col='formula')`
  - Build all configured feature tables.
- `make_split_masks(df, cfg)`
  - Create grouped train/val/test split masks.
- `summarize_feature_table(feature_df, feature_set=None)`
  - Summarize a feature table for selection and reporting.

## modeling.py

- `make_model(cfg, model_type=None)`
  - Instantiate the configured regression model.
- `train_baseline_model(df, split_masks, cfg, model_type=None, include_validation=False)`
  - Train one model and return `(model, feature_columns)`.
- `evaluate_predictions(df, split_masks, model, feature_columns, split_name='test')`
  - Evaluate a trained model on one split and return metrics plus predictions.

## selection.py

- `select_feature_model_combo(feature_tables, split_masks, cfg)`
  - Choose the overall-evaluation combo and formula-only screening combo.
- `select_model_type(feature_tables, split_masks, cfg)`
  - Compatibility wrapper for callers that need model-type-only selection output.

## benchmarking.py

- `benchmark_regressors(feature_tables, split_masks, cfg, ...)`
  - Run the standard benchmark sweep across configured combos.
- `benchmark_grouped_robustness(feature_tables, cfg, ...)`
  - Run grouped-by-formula robustness benchmarking.
- `benchmark_bn_slice(dataset_df, feature_tables, cfg, ...)`
  - Run the BN-focused leave-one-BN-formula-out benchmark.
- `benchmark_bn_family_holdout(dataset_df, feature_tables, cfg, ...)`
  - Run the BN-family holdout benchmark.
- `benchmark_bn_stratified_errors(feature_tables, cfg, ...)`
  - Run grouped BN-vs-non-BN stratified error benchmarking.
- `select_bn_centered_candidate_screening_combo(bn_slice_benchmark_df, cfg, ...)`
  - Pick the BN-centered alternative screening combo.

## screening.py

- `build_candidate_structure_generation_seeds(ranked_candidate_df, dataset_df, split_masks, cfg, ...)`
  - Build prototype-seed records for structure follow-up.
- `build_candidate_prediction_ensemble(candidate_df, feature_tables, split_masks, cfg, ...)`
  - Build ensemble candidate predictions.
- `build_candidate_prediction_members(candidate_df, feature_tables, split_masks, cfg, ...)`
  - Build per-member candidate predictions.
- `build_candidate_grouped_robustness_predictions(candidate_df, feature_df, split_masks, cfg, ...)`
  - Build grouped-fold candidate robustness summaries.
- `build_candidate_grouped_robustness_prediction_members(candidate_df, feature_df, split_masks, cfg, ...)`
  - Build grouped-fold candidate robustness member predictions.
- `annotate_candidate_dataset_overlap(candidate_df, dataset_df, split_masks=None, formula_col='formula')`
  - Add dataset-overlap annotations.
- `annotate_candidate_novelty(candidate_df, formula_col='formula')`
  - Add novelty/rediscovery annotations.
- `annotate_candidate_domain_support(candidate_feature_df, reference_feature_df, split_masks, feature_columns, cfg=None, formula_col='formula')`
  - Add train+val feature-space domain-support annotations.
- `annotate_candidate_bn_support(candidate_feature_df, reference_feature_df, split_masks, feature_columns, cfg=None, formula_col='formula')`
  - Add BN-local support annotations.
- `annotate_candidate_bn_analog_evidence(candidate_df, dataset_df, split_masks, cfg=None, formula_col='formula')`
  - Add BN analog-evidence annotations.
- `screen_candidates(candidate_df, model, feature_columns, cfg, ...)`
  - Build the final candidate ranking artifact.

## summary.py

- `build_experiment_summary(...)`
  - Build the structured experiment summary payload.

## artifacts.py

- `save_metrics_and_predictions(...)`
  - Write the main artifact bundle under the configured artifact directory.

## plots.py

- `save_basic_plots(prediction_df, cfg)`
  - Write the standard parity-plot artifact.

## structure_execution.py

- `build_structure_first_pass_execution_artifacts(...)`
  - Build the deterministic first-pass structure-execution artifacts.

## Internal-only files

These files currently expose no supported external call surface:
- `constants.py`
- `common.py`
- `ranking_tables.py`
- `structure_artifacts.py`
- `structure_helpers.py`
- `utils.py`
