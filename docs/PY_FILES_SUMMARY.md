# Python callable index

`HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task`; `human_docs/` is contextual evidence, never agent-owned Python or contract state.

Run from the repository root in `quant`. This flat `src/` tree is not an installed Python package. [COMMON_FUNCTIONS.md](../COMMON_FUNCTIONS.md) explains imports and shared helpers; module summaries list the complete supported public surfaces. Signatures below intentionally remain machine-checkable by `test_public_surfaces.py`; `...` means consult the implementation for remaining options.

Configuration is owned by [src/config.py](../src/config.py), not copied parameter lists. Test bootstrap, cache cleanup and non-vacuity rules are in [conftest.py](../conftest.py) and [TESTING.md](../TESTING.md). For detailed module contracts read [runtime](../src/runtime/PY_FILES_SUMMARY.md), [materials](../src/materials/PY_FILES_SUMMARY.md), [models](../src/torch_models/PY_FILES_SUMMARY.md), or [UI](../src/ui/PY_FILES_SUMMARY.md).

## main.py

### `main()`

Runs the linear pipeline: cache/config → dataset → grouped splits and feature tables → validation selection → train+val refit and held-out/grouped/BN diagnostics → formula-only ranking → unrelaxed structure follow-up → artifacts and plots. Returns `None`. Data-cache misses may download JARVIS data. Overall evaluation and screening can use different feature/model combinations.

### `run_dry_run()`

Returns and prints a compatibility report using three in-memory rows. Clears project caches, prepares runtime directories, generates candidates, builds features and constructs configured candidate/baseline models. It does not load the real dataset, train models or publish research outputs.

### `emit_agent_state(write_path=None, fail_on_error=False)`

Returns and prints JSON with Git state, manifest and validation results. `fail_on_error=True` makes blocking validation errors exit nonzero; `write_path` additionally writes guarded state. Control flags avoid importing the full scientific pipeline; dependency checks use isolated import probes.

### `emit_agent_commands()`

Returns and prints the command index: entrypoints, exact validation commands/profiles, dependencies, project skills, document ownership and research-plan alignment. Use it with [TESTING.md](../TESTING.md) to choose validation.

## src/runtime/io_utils.py

### `load_config(path)`

Trusted Python config path → `CONFIG` dict, compiled without bytecode. Rejects executable config under user-owned `human_docs/`.

### `validate_runtime_output_path(path, project_root_path=None, *, required_parent_path=None, reject_leaf_symlink=False, expected_output_kind=None)`

Path plus optional root/parent/kind/alias constraints → canonical guarded `Path`. Enforces human-document exclusion, configured containment, directory-only parent chains, leaf kind/symlink and hardlink rules before writes; alternate declared roots cannot weaken the canonical guard.

### `configure_matplotlib_cache()`

Guards/canonicalizes `MPLCONFIGDIR` (safe temporary default when blank/unset), returns the `Path` and exports it before Matplotlib/JARVIS imports; leaves directory creation to the dependency.

### `ensure_runtime_dirs(cfg, project_root_path='.')`

Config and project root → preflight every configured data/cache/artifact directory, then create them. Invalid leaves or parent chains fail before partial creation.

### `build_artifact_provenance(cfg, dataset_manifest=None, *, published_output_paths, project_root_path=None)`

Config, dataset manifest and successfully published paths → v2 identity dict containing source revision/dirty state, effective-config/dataset hashes, and artifact-relative output SHA-256 commitments. Root Python shadow modules participate in source identity.

### `assess_artifact_provenance(provenance, cfg, dataset_manifest=None, *, project_root_path=None)`

Stored provenance and current config/dataset/root → `{status, reason}` with `current`, `stale`, or `unverified`. Legacy/malformed markers, invalid manifests, unavailable/dirty source identity, or missing/changed committed bytes cannot assess current.

### `validate_json_payload(payload, ...)`

Payload and JSON serialization options → `None` on success, exception on invalid serialization; no filesystem mutation. Used before multi-output publication.

### `clear_project_cache(project_root_path='.')`

Existing project root → remove real cache directories within that root, preserving `human_docs/` and discovered cache symlinks. Rejects symlink components in the supplied root and escaping paths; concurrent already-removed caches are tolerated. Pytest and pipeline entrypoints already call it.

### `read_json_file`

Imported public helper: `read_json_file(path, ...)` → decoded JSON through the shared `myutils/file_utils/json_io.py` module.

### `write_json_file(payload, path, ...)`

Payload, guarded path and serialization options → shared JSON write. Canonicalizes the output and preflights serialization/encoding before creating parents.

### `make_json_safe`

Imported public helper: `make_json_safe(value)` → JSON-safe objects from NumPy/pandas/path-like values via `myutils`.

## src/runtime/agent_state.py

### `load_agent_manifest(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`

Project root and manifest path → checked-in manifest dict.

### `validate_agent_layout(project_root_path='.', manifest=None)`

Checks required agent-facing files; exact control and validation command/scope/capability records; profile reachability including mandatory dependency capabilities; repo-skill trigger frontmatter and repo-local `$skill` reference resolution; bidirectional normalized requirements/manifest specifier parity; source-derived external-import ownership (including production root-symbol parity against owner-symbol or exact immediate-descendant probes, exact static descendant target/symbol, literal-dynamic descendant target, and direct literal `getattr(importlib-import, symbol)` parity with fail-closed wildcards/ambiguous attributes, evaluation-scope and binding-position-aware `importlib`/`__import__` aliases, ambiguous late-bound owner sets, branch-compatible direct-call reaching owners, declared `global`/`nonlocal` targets, match-pattern shadows, precise comprehension shadows, fail-closed nonliteral dynamic names and delegated wrappers, relative-local exclusion, and identity-resolved direct literal module/attribute calls to the unique `main.py` `_bind_missing` loader); distribution/module identity; direct/backend and core/scientific/UI/test consumer constraints; local shared-owner records; timeout-bounded isolated dependency import probes with manifest-owned ordered preloads and active-consumer targets/symbols plus opaque import-environment/owner/target/symbol success-cache identity; exact retired-guidance and six-module records; local instruction paths; and the stable v18 alignment/scientific boundaries.
Returns:
- `status`
- `errors`
- `warnings`
- per-path `checks`

### `build_agent_state(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`

Project root/manifest → live JSON-ready project state used by inspection and contract verification.

### `build_agent_command_index(project_root_path='.', manifest_path='docs/AGENT_MANIFEST.json')`

Project root/manifest → JSON-ready command index, including module dependencies, ownership and v18 alignment.

### `agent_state_to_json(state)`

State dict → serialized JSON for stdout or logs.

### `write_agent_state(state, path)`

State dict and path → serialized state file after preflight. Rejects canonical, state-declared or filesystem-equivalent `human_docs/` targets and multi-hardlink leaves.

## src/runtime/schema.py

### `STRUCTURE_EXECUTION_OUTPUT_ROLES`

Three-role mapping of viewer key, summary field, configured path, suffix and default filename, shared by structure building, publication and persisted-state validation.

### `FIXED_REPORT_ARTIFACT_NAMES`

Fixed report filename set used by collision preflight and UI role validation; configured structure outputs cannot relabel fixed files.

## src/materials/data.py

### `load_or_build_dataset(cfg)`

Config → `(normalized_dataframe, manifest_dict)`. Reuses processed Parquet only when dataset/source/target/required-column identity matches; otherwise rebuilds from cached raw JSON, downloading through JARVIS only on a raw-cache miss. Preflights raw/processed/manifest/archive/cache paths before imports or writes, uses one validated metadata snapshot, and removes only newly created invalid/partial dependency archives on failure. Normalized source properties and structure columns are defined by `REFERENCE_PROPERTY_COLUMNS` and `STRUCTURE_SUMMARY_COLUMNS`.

## src/materials/candidate_space.py

### `extract_elements(formula)`

Formula string → regex-extracted element-symbol list (preserves token order and duplicates).

### `filter_bn(df, formula_col='formula')`

Dataframe and formula column → rows whose formulas contain both B and N.

### `generate_bn_candidates(cfg=None)`

Optional config → bounded BN formula-family grid with generation strategy/family/template provenance and configured chemical-plausibility annotations. Candidate definitions live in `candidate_space.py`; default settings come from `src/config.py`.

### `annotate_candidate_proposal_shortlist(ranked_candidate_df, cfg=None)`

Ranking/optional config → family-capped shortlist annotations in plausibility/ranking order, preserving the full ranking.

### `annotate_candidate_extrapolation_shortlist(ranked_candidate_df, cfg=None)`

Ranking/optional config → separate shortlist annotations restricted to the configured novelty bucket, with plausibility and family-diversity limits.

## src/materials/feature_building.py

### `get_candidate_feature_sets(cfg)`

Config → ordered feature-set search space. Current families are basic formula, matminer composition, fractional composition, and composition plus structure summaries.

### `get_candidate_screening_feature_sets(cfg)`

Config → candidate-compatible formula-only feature sets; excludes structure-dependent representations.

### `get_candidate_model_types(cfg)`

Config → ordered model search space. Default candidate types and baseline choices belong to `src/config.py`; attention/sparse-attention/Roost-like blocks remain experimental and outside the default sweep.

### `build_feature_table(df, formula_col='formula', feature_set='basic_formula_composition')`

Dataframe/formula column/feature-set name → feature dataframe including `feature_set`, `feature_generation_failed` and error information. The fractional representation has 118 element fractions; structure-summary features require actual structure evidence.

### `build_feature_tables(df, cfg, formula_col='formula')`

Dataframe/config/formula column → `{feature_set: dataframe}` mapping.

### `make_split_masks(df, cfg)`

Dataframe/config → split masks plus grouping/overlap metadata. The default `group_by_formula` keeps duplicate formulas in one split.

### `summarize_feature_table(feature_df, feature_set=None)`

Feature dataframe/name → family, candidate compatibility, feature count, status, selection eligibility and failed-formula examples.

## src/materials/modeling.py

### `make_model(cfg, model_type=None)`

Config/model type → unfitted regressor. Supports linear, HGB, random forest, dummy mean and local Torch families. Attention/sparse-attention/Roost-like models require `fractional_composition_vector`; compatibility helpers enforce that restriction.

### `train_baseline_model(df, split_masks, cfg, model_type=None, include_validation=False)`

Feature dataframe, masks and config → `(fitted_model, feature_columns)`; `include_validation` controls train versus train+val scope.

### `evaluate_predictions(df, split_masks, model, feature_columns, split_name='test')`

Feature dataframe, masks, model and columns → `(metrics_dict, prediction_dataframe)` for the chosen split. Metrics are MAE/RMSE/R²; incomplete feature coverage fails rather than dropping evaluation rows.

## src/torch_models/base.py

### `TorchMLPRegressor`

Numeric X/y → sklearn-style fit/predict regressor: standardized inputs/targets, LayerNorm/GELU MLP, AdamW, deterministic validation and early stopping. Auto device order is CUDA, MPS, CPU.

## src/torch_models/ensemble.py

### `TorchMLPEnsembleRegressor`

Multi-seed MLP ensemble; averages predictions and exposes `predict_members` for member disagreement.

## src/torch_models/attention.py

### `TorchFractionalAttentionRegressor`

Experimental attention over 118 element fractions with stoichiometric signals and weighted pooling. Auto device uses CUDA when available, otherwise CPU; explicit device settings are retained.

## src/torch_models/sparse_attention.py

### `TorchSparseFractionalAttentionRegressor`

Experimental attention using only present-element tokens, batch padding and fraction-weighted pooling; inherits the dense model's input/device contract.

## src/torch_models/roost_like.py

### `TorchRoostLikeRegressor`

Experimental present-element stoichiometric message-passing regressor with fraction-weighted pooling. These experimental families remain formula-compatible but do not establish a validated BN model.

## src/materials/selection.py

### `select_feature_model_combo(feature_tables, split_masks, cfg)`

Feature tables, masks and config → validation-selection summary: best overall and formula-only combinations, reuse decision, feature statuses and per-combination validation results.

## src/materials/benchmarking.py

### `benchmark_regressors(feature_tables, split_masks, cfg, selected_feature_set, selected_model_type)`

Feature tables/masks/config and selected identity → held-out test benchmark dataframe for configured combinations and baseline, including status, role and MAE/RMSE/R².

### `benchmark_grouped_robustness(feature_tables, cfg, selected_feature_set, selected_model_type)`

Feature tables/config and selected identity → formula-grouped cross-validation dataframe with requested/actual/completed folds, status and metric means/spreads.

### `benchmark_bn_slice(dataset_df, feature_tables, cfg, selected_feature_set, selected_model_type, screening_feature_set, screening_model_type)`

Dataset/features/config and model identities → `(summary_dataframe, held_out_prediction_dataframe)` for leave-one-BN-formula-out diagnostics, including global dummy and BN-local neighbor baselines. This small-sample view matters when ordinary splits place all BN rows in training; its best combination can differ from both main model roles.

### `benchmark_bn_family_holdout(dataset_df, feature_tables, cfg, selected_feature_set, selected_model_type, screening_feature_set, screening_model_type)`

Dataset/features/config and model identities → `(summary_dataframe, family_annotated_prediction_dataframe)` for reduced-BN-chemical-system family holdouts. This is still a small-sample diagnostic.

### `benchmark_bn_stratified_errors(feature_tables, cfg, selected_feature_set, selected_model_type, screening_feature_set, screening_model_type)`

Feature tables/config and model identities → dataframe of BN/non-BN MAE/RMSE/R² and their MAE ratio. Requires grouping by the active formula column; aggregates once per held-out formula to avoid duplicate-formula leakage.

### `select_bn_centered_candidate_screening_combo(bn_slice_benchmark_df, cfg, fallback_feature_set=None, fallback_model_type=None)`

BN-slice benchmark/config and optional fallback → alternative screening metadata. Selects the lowest-MAE successful candidate-compatible non-baseline row; this is a comparison view, not a replacement global selection rule.

## src/materials/screening.py

### `build_candidate_structure_generation_seeds(candidate_df, dataset_df, split_masks, cfg=None, bn_centered_candidate_df=None, formula_col='formula')`

Ranked candidates, dataset, masks and optional alternative ranking → candidate-to-reference seed dataframe. Uses the union of proposal/extrapolation shortlists and top BN-centered candidates, choosing deterministic train+val BN exemplars. Preserves record/source/property/structure evidence; edit-plan fields live in downstream job-plan JSON. A disabled stage returns an empty schema. This bridge itself does not generate or validate a structure.

### `build_candidate_prediction_ensemble(candidate_df, feature_tables, split_masks, cfg, candidate_feature_sets=None)`

Candidates/features/masks/config → candidate-level mean/std/member-count dataframe from small formula-compatible train+val model pools. Disagreement is a heuristic, not calibrated physical uncertainty.

### `build_candidate_grouped_robustness_predictions(candidate_df, feature_df, split_masks, cfg, feature_set, model_type, formula_col='formula')`

Candidates/reference features/masks/config and model identity → grouped-fold mean/std/count dataframe using train+val reference formulas only. Fold spread measures split sensitivity without test-label leakage.

### `annotate_candidate_dataset_overlap(candidate_df, dataset_df, split_masks=None, formula_col='formula')`

Candidates/dataset and optional masks → annotations distinguishing dataset presence from train+val presence, with formula row counts.

### `annotate_candidate_novelty(candidate_df, formula_col='formula')`

Candidates with overlap fields → formula-level novelty annotations: `train_plus_val_rediscovery`, `held_out_known_formula`, `formula_level_extrapolation`. Novelty is relative to this dataset/demo space.

### `annotate_candidate_domain_support(candidate_feature_df, reference_feature_df, split_masks, feature_columns, cfg=None, formula_col='formula')`

Candidate/reference features, masks and columns → train+val feature-space neighbor/distance/percentile/penalty annotations.

### `annotate_candidate_bn_support(candidate_feature_df, reference_feature_df, split_masks, feature_columns, cfg=None, formula_col='formula')`

Candidate/reference features, masks and columns → BN-local train+val support annotations, including neighbor formulas and penalties.

### `annotate_candidate_bn_analog_evidence(candidate_df, dataset_df, split_masks, cfg=None, formula_col='formula')`

Candidates/dataset/masks → observed BN train+val analog-property evidence (band gap, energy, exfoliation, magnetization), alignment labels and vote/penalty context. These are conservative analog proxies.

### `screen_candidates(candidate_df, model, feature_columns, cfg, feature_set, model_type, best_overall_feature_set=None, best_overall_model_type=None, screening_selection_note=None, dataset_df=None, split_masks=None, ensemble_prediction_df=None, grouped_robustness_prediction_df=None, reference_feature_df=None)`

Candidates, formula-compatible model/columns/config and optional reference evidence → full ranked dataframe. Rejects structure-aware features and any unfeaturizable candidate atomically. Sorts by plausibility then score, retains top-k decisions without truncating the full output, and records objective/signal/penalty decomposition, overlap, novelty, support and shortlist annotations. Overall/screening model identities stay explicit. Ranking-stability, abstention and application-track action columns are added downstream by artifact/summary writers, where multi-source predictions can be combined.

## src/materials/structure_execution.py

### `build_structure_first_pass_execution_artifacts(structure_generation_seed_df, *, cfg, formula_col='formula', structure_model=None, structure_feature_columns=None, structure_feature_set=None, structure_model_type=None)`

Seed dataframe, config and optional real structure-aware model → `(candidate_summary, variant_dataframe, JSON_payload)` including atoms/CIF text. Reuses reference cells or performs supported deterministic relabel/vacancy edits, geometry heuristics and optional band-gap proxies. No relaxation occurs. The canonical successful winner is selected by geometry, formula match, proxy, score and rank; ten selected fields and finite status vocabularies are shared with publication checks. Workflow-ready statuses are not physical validation.

## src/materials/summary.py

### `build_experiment_summary(dataset_df, bn_df, candidate_df, split_masks, selection_summary, cfg, robustness_df=None, bn_slice_benchmark_df=None, bn_family_benchmark_df=None, bn_stratified_error_df=None, bn_centered_candidate_df=None, bn_centered_screening_selection=None, structure_generation_seed_df=None, candidate_prediction_member_df=None, candidate_grouped_robustness_member_df=None, bn_centered_grouped_robustness_member_df=None, structure_first_pass_execution_summary_df=None, structure_first_pass_execution_payload=None, bn_slice_prediction_df=None, bn_family_prediction_df=None)`

Dataset, split, selection and optional diagnostic/ranking/structure evidence → JSON-ready experiment-summary dict. Records separate overall/screening roles, BN comparisons, ranking stability, uncertainty, decisions, shortlist and structure bridge/execution metadata. Data-insufficient comparisons are `null`; optional prediction paths are advertised only when corresponding rows will be emitted.

## src/materials/artifacts.py

### `save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, benchmark_df, robustness_df, bn_slice_benchmark_df, bn_slice_prediction_df, bn_centered_screened_df, structure_generation_seed_df, experiment_summary, manifest, cfg, ...)`

Metrics, prediction/benchmark/ranking/structure frames, summary, manifest and config → published artifact bundle. Preflights every output path, caller JSON, semantic role and structure/source/seed/edit-plan/selected-row relation before mutation. Honors optional gates and removes only preflighted stale outputs; uses atomic CSV replacement. Invalidates the old marker before writing and publishes the v2 commitment last; failures leave no completion marker. Detailed invariants are maintained once in [materials API](../src/materials/PY_FILES_SUMMARY.md#artifactspy).

## src/materials/plots.py

### `save_basic_plots(prediction_df, cfg)`

Predictions/config → guarded canonical parity-plot path after publication. Configures Matplotlib cache before pyplot import, invalidates an old marker before mutation and closes figures on failure.

## src/ui/streamlit_app.py

### `render_streamlit_app()`

No arguments → Streamlit artifact view (`None` return). Displays current committed outputs only after independent provenance, digest, role/path and shape checks. Missing/changed/legacy/uncommitted-known bundles suppress report tables; malformed JSON/CSV produces text warnings. Absent optional outputs stay absent. Rendering tests and optional startup commands are in [TESTING.md](../TESTING.md) and [SERVICES.md](../SERVICES.md).
