# materials module public surface

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is user-owned contextual evidence, never runtime-owned state.

This file lists the documented public functions that other modules or top-level entrypoints may call from `materials`.
Anything underscore-prefixed or omitted here should be treated as internal implementation detail.

## aqueous_slurry.py

- `AQUEOUS_DATA_PATH`: pinned manifest for two public aqueous-slurry patent sources.
- `AqueousSlurryRecipe`: finite, bounded BN-volume request; locks solids, particle sizes, binder, co-filler, solvent and preparation settings to one matched series. Measurement temperature and shear rate are not invented.
- `load_aqueous_slurries(path=AQUEOUS_DATA_PATH)`: verify archive paths and hashes; extract 8 table cases and 4 paragraph cases, preserving missing labels, mass/volume bases and distinct cohorts.
- `predict_aqueous_viscosity(dataset, recipe)`: interpolate the six matched BaTiO3/BN/SBA points; return mPa·s, separate exact observations and neighbouring coating outcomes. Excludes the alumina control, PI recipes and unmeasured paste from the numeric fit.
- `evaluate_aqueous_viscosity(dataset)`: label-blind development leave-one-formula-out comparison on the four interior points; retain interpolation, nearest and linear estimates. This is not independent validation or measured experimental savings.

## separator_data.py

- `DATASET_PATH`: pinned public separator dataset location.
- `PRIMARY_COHORT`: PP/BNNT, LiTFSI/DOL/DME protocol identity.
- `SeparatorRecipe`: strict, finite, structured recipe request; rejects arbitrary prompts and extra fields.
- `CoatingRecipe`: bounded CA/BN coating-gap request with fixed source-matched chemistry and drying temperature.
- `predict_coating_thickness(dataset, recipe)`: interpolate the three matched coating settings; return estimated thickness in μm and a separate observed value for an exact match.
- `load_separator_dataset(path=DATASET_PATH)`: validate source hashes, original text, IDs, group splits and observations; derive readable section locations from source ancestry and return the public dataset.
- `build_separator_database(dataset, path)`: transactionally write the three public SQLite tables; return the database path.
- `search_separator_records(dataset, substrate=None, query='', limit=100)`: bounded literal search of public records.
- `assess_separator_recipe(dataset, recipe)`: gate protocol applicability, select training-only records, and compute simple comparison baselines. Numeric permission is exploratory, not a reliability claim.
- `separator_known_checks(dataset)`: reconstruct two published comparisons and an arithmetic discrepancy.

## separator_model.py

- `MODEL`: default requested model `gpt-6-astra`; trusted deployment configuration may select another model; no automatic fallback.
- `SeparatorModelResult`: bounded numeric hypothesis, allowed record citations and limitations.
- `make_separator_prompt(assessment, *, language='zh-TW')`: construct the exact answer-blind prompt from approved training examples, with a validated English/Simplified/Traditional Chinese response language; the historical benchmark defaults to Traditional Chinese.
- `run_separator_model(assessment, executable, runtime_dir, timeout_seconds=120, *, model_name=MODEL, language='zh-TW')`: one ephemeral read-only Codex CLI call with project/user configuration excluded, strict output/citation/event validation, finite process timeout and sanitized outward errors. Reject raw-BNNT results above the published 0.71 mS/cm peak; its input domain ends at 0.3 mg/cm². Public callers cannot supply commands, model IDs or free-form prompts. Private provider statistics are call evidence, not materials validation.
- `evaluate_separator_model(dataset, output_dir, executable=None)`: write frozen inputs before inference, compare one unused formulation against training mean and nearest formulation, and disclose the absence of independent-study/laboratory evaluation. Without an executable, mark the model not run.

## experiment_planning.py

- `ElectrolyteRecipe`: finite LiPF6/EC/DMC/EMC composition; salt molality 0.14–2, EC solvent mass fraction 0.3–0.5, DMC share of the remaining solvent 0–1; temperature restricted to the approximately 26–28°C study condition, not a learned temperature feature.
- `ExperimentPlan`: 2–38 already measured candidate IDs and finite conductivity values, with no arbitrary prompt.
- `SOURCE_DIR`, `PROTOCOL_PATH`, `FOREST_CONFIG`: public-source location, frozen evaluation protocol and fixed forest parameters.
- `load_electrolyte_records(source_dir=SOURCE_DIR)`: verify pinned source bytes, group all repeated measurements by exact composition, retain mean/std, temperature bounds and source CSV line locators.
- `suggest_experiments(candidates, observations, *, method='adaptive_forest', tie_order=None, limit=5)`: choose unseen recipes from composition-only candidates and explicitly supplied observed outcomes. Reject candidate labels, invalid identities/domains and invalid tie order. Reuse `materials.modeling.make_model`; acquisition is forest mean plus half the tree spread (a heuristic, not a calibrated uncertainty interval).
- `predict_electrolyte(records, recipe)`: fit the public grouped data and estimate an in-domain composition, with an optional separately labelled exact literature measurement.
- `evaluate_experiment_planner(records, protocol, output_dir)`: paired sequential replay and recipe-grouped leave-one-out MAE; write full query traces, all baselines/thresholds and conditional paired-start bootstrap intervals. Future outcomes remain in the evaluator; stopping counts every queried formulation, including early success during warm-up. No laboratory or calendar saving is fabricated.

## data.py

- `STRUCTURE_SUMMARY_COLUMNS`
  - Shared structure-derived column contract used by feature building, screening, benchmarking, and structure handoff.
- `REFERENCE_PROPERTY_COLUMNS`
  - Shared source-property column contract retained for provenance-aware BN records and reference evidence.
- `load_cached_raw_record_lookup(cfg)`
  - Load the cached raw-record lookup used by downstream artifact writers.
- `load_or_build_dataset(cfg)`
  - Build or reload the normalized dataset and its manifest.
  - Reuse a processed cache only when dataset name, source, required columns, and target column all match the request.
  - Preflight the concrete raw JSON, processed Parquet, manifest, JARVIS archive, and dependency-cache root before imports or writes; one validated metadata snapshot supplies a plain JSON archive name, URL, and canonical guarded raw directory directly to JARVIS.
  - Require a non-empty list of record objects before project cache writes; remove only a newly created unreadable, invalid, or partially streamed dependency archive—even on process interruption—so a later request can retry, while preserving pre-existing archive leaves and the original failure if cleanup itself fails.

## constants.py

No callable public surface. The following non-callable contracts are imported across `materials` files and tests for agent-visible v18 screening, novelty, support, and structure-boundary behavior:

- `FRACTIONAL_COMPOSITION_FEATURE_SET`
  - Formula-only fractional-composition feature-set identifier.
- `STRUCTURE_AWARE_FEATURE_SET`
  - Structure-dependent feature-set identifier that must stay out of formula-only screening.
- `NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY`
  - Novelty bucket for train/validation rediscovery candidates.
- `NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA`
  - Novelty bucket for known held-out formulas.
- `NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION`
  - Novelty bucket for formula-level extrapolation candidates.
- `DOMAIN_SUPPORT_RANKING_NOTE`
  - Ranking-note fragment for formula-space domain-support penalties.
- `BN_SUPPORT_RANKING_NOTE`
  - Ranking-note fragment for BN-local support penalties.
- `BN_BAND_GAP_ALIGNMENT_RANKING_NOTE`
  - Ranking-note fragment for BN-local band-gap alignment evidence.
- `BN_ANALOG_EVIDENCE_RANKING_NOTE`
  - Ranking-note fragment for BN analog-evidence context.
- `GROUPED_ROBUSTNESS_UNCERTAINTY_RANKING_NOTE`
  - Ranking-note fragment for grouped-fold candidate robustness uncertainty.
- `NOVELTY_ANNOTATION_RANKING_NOTE`
  - Ranking-note fragment for novelty annotation.

## candidate_space.py

- `extract_elements(formula)`
  - Parse element symbols from a chemical formula string.
- `filter_bn(df, formula_col='formula')`
  - Keep only BN-containing rows from a dataframe.
- `annotate_bn_families(df, *, formula_col='formula', grouping_method=...)`
  - Add BN-family labels for BN-local grouping logic.
- `generate_bn_candidates(cfg=None)`
  - Build the configured BN candidate space and apply the same configured chemical-plausibility policy to its returned annotations.
- `annotate_candidate_proposal_shortlist(ranked_candidate_df, cfg=None)`
  - Add the family-aware proposal-shortlist annotations.
- `annotate_candidate_extrapolation_shortlist(ranked_candidate_df, cfg=None)`
  - Add the formula-level extrapolation-shortlist annotations.
- `get_screening_ranking_metadata(cfg=None, ...)`
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
- `build_feature_table(df, formula_col='formula', feature_set=...)`
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
  - Run formula-grouped BN-vs-non-BN stratified error benchmarking; non-formula grouping is rejected to prevent duplicate-formula train/test leakage.
- `select_bn_centered_candidate_screening_combo(bn_slice_benchmark_df, cfg, ...)`
  - Pick the BN-centered alternative screening combo.

## screening.py

- `build_candidate_structure_generation_seeds(candidate_df, dataset_df, split_masks, cfg, ...)`
  - Build prototype-seed records for structure follow-up; preserve normalized null optional target/property evidence, while a disabled seed stage returns an empty schema and cannot enter downstream structure execution.
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
  - Reject structure-aware feature sets at this formula-only ranking boundary.
  - When decision policy is enabled, downstream artifacts hold candidates outside every configured application target window even when uncertainty/support/rank checks pass; disabling the policy emits neutral action fields instead.

## summary.py

- `build_experiment_summary(...)`
  - Build the structured experiment summary payload; tolerate data-insufficient BN diagnostics and advertise optional BN-prediction/structure outputs only when their rows will be emitted.

## artifacts.py

- `save_metrics_and_predictions(...)`
  - Write the main artifact bundle under the configured artifact directory.
  - Honor ranking-stability, decision-policy, shortlist, and structure-seed gates; remove stale optional outputs on a disabled second run, including case-equivalent CIF suffixes.
  - Preflight caller JSON and optional parity-plot inputs, replace CSV files atomically, invalidate any prior completion marker before the first bundle mutation, record only successfully emitted fixed/optional/configured/CIF/plot paths, and publish their v2 SHA-256 commitment in `artifact_provenance.json` as the final action; failed plotting or marker publication leaves no completion marker.
  - Reject wrong-shaped screening/structure-bridge containers and mismatched dynamic output declarations before artifact-directory creation, marker invalidation, or any bundle write; absent/null/empty containers and matching normalized/same-file roles remain valid.
  - Bind configured enabled/label/method/note metadata, structure-aware model availability, and any experiment-summary execution metadata/count/model fields to the canonical configuration and payload before mutation so structure output remains unrelaxed handoff evidence rather than a synthesis, stability, or discovery claim.
  - Reject nonempty structure-execution summary or variants tables that lack their canonical fields, use a candidate or variant status outside the builder-owned finite vocabularies, relabel the exact builder-selected seed record/formula, disagree with the cached raw record formula or source-atom composition, contradict the builder-produced relabel/vacancy plan or structure reconstructed from that plan, contradict atoms/formula/site-count/structure-summary/geometry/CIF/edit-count/relaxation evidence, or disagree across payload/summary/variants on membership, counts, statuses, geometry results, deterministic selected winner, or any of the ten selected-row projections, before mutation; free-form execution diagnostics remain descriptive and no-selection candidates keep every selected projection null except the canonical `not_executed` or invalid-reference error status.
  - Before publishing seed artifacts, bind every emitted record-level source, band-gap, property, has-summary, and structure-summary value to the selected cached raw record through the dataset normalizer; when the configured split groups by the active formula column, also reconstruct the train+val seed-formula row count and target mean from the complete BN frame before any write.
  - Keep each compact BN model-role comparison row bound to one feature/model identity across slice, family, and stratified diagnostics instead of splicing per-scope winners.
  - Preflight every fixed, configurable, dynamic, and stale-cleanup CIF leaf in its originally declared form before directory creation; contain structure-execution paths beneath their configured roots, reject kind/parent-chain/fixed-name/pairwise/cross-role-default/filesystem-alias collisions before mutation, and remove valid stale execution artifacts when the current run produces no execution payload.

## plots.py

- `save_basic_plots(prediction_df, cfg)`
  - Guard and canonicalize the Matplotlib cache before importing pyplot, preflight the standard parity-plot artifact, invalidate any older completion marker before plot mutation, close the figure on failure, and return the exact canonical published path.

## structure_execution.py

- `build_structure_first_pass_execution_artifacts(...)`
  - Build the deterministic first-pass structure-execution artifacts; builder and writer share the exact queue/follow-up seed-selection context, finite candidate/variant status authorities, one edit-plan projection contract, raw-source-formula-to-final-structure identity checks, atom/metadata/CIF identity checks, evidence-compatible relaxation/final-state derivation, canonical variant ordering, and the ten-field selected projection contract.

## Internal-only files

These files currently expose no supported external call surface:
- `common.py`
- `ranking_tables.py`
- `structure_artifacts.py`
- `structure_helpers.py`
- `utils.py`

## tests/

- Run `conda run -n quant python -m pytest -q src/materials/tests` from the repository root; prerequisites, emitted validation profiles and proof limits are in [TESTING.md](../../TESTING.md).
- `test_data.py` covers normalization, processed-cache identity, guarded JARVIS download fixtures and partial-archive cleanup; it is not a live provider check.
- `test_bn_filter.py` covers the B-and-N formula slice.
- `test_features_pipeline.py` covers features, grouped splits, model selection, candidate annotations/ranking and BN follow-up integration. Five independently reported neural-model cases share tiny CPU configuration and reuse fitted models for ensemble-member assertions.
- `test_diagnostic_edge_cases.py` locks disabled and insufficient-data status semantics for BN diagnostics and alternative screening selection.
- `test_reporting.py` locks artifact publication, provenance, finite candidate/variant structure-execution status vocabularies, exact builder-selected seed/raw-formula/source-atom and record-level normalized evidence identity, raw-source/edit-plan/final-structure identity, atom/metadata/CIF identity, evidence-compatible variant states, role-schema preflight, repeat-run cleanup, and failure-order behavior.
- `test_structure_execution_contracts.py` locks relabel, vacancy, unsupported edit, and structure-aware proxy execution behavior.
- `test_separator_prototype.py` covers BN source/database identity, training-only prompts/evaluation, recipe boundaries and the isolated provider wrapper. One local fake process exercises result/citation validation, private diagnostics and timeout/interruption cleanup without real inference.
- `test_aqueous_slurry.py` covers literal source anchors, cohort identity, bounded viscosity interpolation, incompatible recipes and held-out-label isolation.
- `test_experiment_planning.py` covers electrolyte measurement grouping, domain/coating estimates, unseen-outcome isolation, the published replay traces and a temporary reproduced comparison.

## separator_ingest.py

- `main()`: rebuild the explicitly authorized separator JSON and CSV from manually reviewed, literal source anchors in the four licensed XML documents. Fails on missing anchors. This command writes only `data/separators/`; it never downloads, interprets images, runs inference or changes the old band-gap dataset.
