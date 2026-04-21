import argparse
import copy
import inspect
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime.io_utils import clear_project_cache, ensure_runtime_dirs, load_config
from materials.data import STRUCTURE_SUMMARY_COLUMNS, load_or_build_dataset
from materials.benchmarking import (
    benchmark_bn_family_holdout,
    benchmark_bn_slice,
    benchmark_bn_stratified_errors,
    benchmark_grouped_robustness,
    benchmark_regressors,
    select_bn_centered_candidate_screening_combo,
)
from materials.candidate_space import filter_bn, generate_bn_candidates
from materials.constants import STRUCTURE_AWARE_FEATURE_SET
from materials.feature_building import (
    build_feature_tables,
    compatible_model_types_for_feature_set,
    feature_set_supports_formula_only_screening,
    get_candidate_feature_sets,
    get_candidate_model_types,
    make_split_masks,
)
from materials.modeling import evaluate_predictions, make_model, train_baseline_model
from materials.screening import (
    build_candidate_grouped_robustness_prediction_members,
    build_candidate_grouped_robustness_predictions,
    build_candidate_prediction_ensemble,
    build_candidate_prediction_members,
    build_candidate_structure_generation_seeds,
    screen_candidates,
)
from materials.selection import select_feature_model_combo
from materials.artifacts import save_metrics_and_predictions
from materials.plots import save_basic_plots
from materials.summary import build_experiment_summary
from materials.structure_execution import build_structure_first_pass_execution_artifacts


def _build_dry_run_dataset(cfg: dict) -> pd.DataFrame:
    formula_col = str(cfg['data'].get('formula_column', 'formula'))
    target_col = str(cfg['data'].get('target_column', 'band_gap'))
    dry_run_rows = [
        ('BN', 5.2),
        ('AlN', 4.1),
        ('GaN', 3.4),
    ]
    records: list[dict[str, object]] = []
    for formula, target in dry_run_rows:
        row = {
            'formula': formula,
            formula_col: formula,
            'target': float(target),
            target_col: float(target),
        }
        for column in STRUCTURE_SUMMARY_COLUMNS:
            row[column] = 0.0
        records.append(row)
    return pd.DataFrame(records)


def run_dry_run() -> dict:
    clear_project_cache('.')

    config_path = Path('src/config.py')
    cfg = load_config(config_path)
    ensure_runtime_dirs(cfg)

    candidate_df = generate_bn_candidates(cfg)
    if candidate_df.empty:
        raise ValueError('Dry run failed because candidate generation returned zero rows')

    dry_run_dataset_df = _build_dry_run_dataset(cfg)
    feature_tables = build_feature_tables(
        dry_run_dataset_df,
        cfg,
        formula_col=cfg['data']['formula_column'],
    )
    configured_feature_sets = get_candidate_feature_sets(cfg)
    configured_model_types = get_candidate_model_types(cfg)
    available_feature_sets = [
        feature_set for feature_set in configured_feature_sets if feature_set in feature_tables
    ]
    if not available_feature_sets:
        raise ValueError('Dry run failed because no configured feature set produced a feature table')

    overall_compatible = {
        feature_set: compatible_model_types_for_feature_set(cfg, feature_set)
        for feature_set in available_feature_sets
    }
    screening_feature_sets = [
        feature_set
        for feature_set in available_feature_sets
        if feature_set_supports_formula_only_screening(feature_set)
    ]
    screening_compatible = {
        feature_set: compatible_model_types_for_feature_set(cfg, feature_set)
        for feature_set in screening_feature_sets
    }
    if not any(model_types for model_types in overall_compatible.values()):
        raise ValueError('Dry run failed because no configured feature/model combo is compatible')
    if not any(model_types for model_types in screening_compatible.values()):
        raise ValueError(
            'Dry run failed because no formula-only screening feature/model combo is compatible'
        )

    model_init_status = {}
    dry_run_model_types = list(
        dict.fromkeys(configured_model_types + list(cfg['model'].get('benchmark_baselines', [])))
    )
    for model_type in dry_run_model_types:
        try:
            make_model(cfg, model_type=model_type)
        except Exception as exc:  # pragma: no cover - exercised through failure path
            model_init_status[model_type] = f'{type(exc).__name__}: {exc}'
        else:
            model_init_status[model_type] = 'ok'
    failing_model_types = {
        model_type: status
        for model_type, status in model_init_status.items()
        if status != 'ok'
    }
    if failing_model_types:
        raise RuntimeError(
            'Dry run failed while instantiating configured models: '
            f'{failing_model_types}'
        )

    report = {
        'config_path': str(config_path),
        'candidate_row_count': int(len(candidate_df)),
        'dry_run_dataset_row_count': int(len(dry_run_dataset_df)),
        'configured_feature_sets': configured_feature_sets,
        'available_feature_sets': available_feature_sets,
        'screening_feature_sets': screening_feature_sets,
        'configured_model_types': configured_model_types,
        'benchmark_baselines': list(cfg['model'].get('benchmark_baselines', [])),
        'overall_compatible': overall_compatible,
        'screening_compatible': screening_compatible,
        'model_init_status': model_init_status,
    }

    print('=== BN AI PoC dry run completed ===')
    print(f"config path: {report['config_path']}")
    print(f"candidate rows: {report['candidate_row_count']}")
    print(f"dry-run dataset rows: {report['dry_run_dataset_row_count']}")
    print(f"configured feature sets: {report['configured_feature_sets']}")
    print(f"available feature sets: {report['available_feature_sets']}")
    print(f"screening feature sets: {report['screening_feature_sets']}")
    print(f"configured model types: {report['configured_model_types']}")
    print(f"benchmark baselines: {report['benchmark_baselines']}")
    print(f"overall compatible combos: {report['overall_compatible']}")
    print(f"screening compatible combos: {report['screening_compatible']}")
    print(f"model init status: {report['model_init_status']}")
    return report


def main() -> None:
    clear_project_cache('.')

    config_path = Path('src/config.py')
    cfg = load_config(config_path)

    ensure_runtime_dirs(cfg)

    dataset_df, manifest = load_or_build_dataset(cfg)
    bn_df = filter_bn(dataset_df, formula_col=cfg['data']['formula_column'])
    candidate_df = generate_bn_candidates(cfg)

    split_masks = make_split_masks(dataset_df, cfg)
    feature_tables = build_feature_tables(dataset_df, cfg, formula_col=cfg['data']['formula_column'])
    selection_summary = select_feature_model_combo(feature_tables, split_masks, cfg)
    selected_feature_set = selection_summary['selected_feature_set']
    selected_model_type = selection_summary['selected_model_type']
    ranking_feature_set = selection_summary['screening_selected_feature_set']
    ranking_model_type = selection_summary['screening_selected_model_type']
    feature_df = feature_tables[selected_feature_set]

    model, feature_columns = train_baseline_model(
        feature_df,
        split_masks,
        cfg,
        model_type=selected_model_type,
        include_validation=True,
    )
    metrics, prediction_df = evaluate_predictions(feature_df, split_masks, model, feature_columns)
    metrics = {
        **metrics,
        'selected_feature_set': selected_feature_set,
        'selected_model_type': selected_model_type,
        'selected_feature_family': selection_summary.get('selected_feature_family'),
        'screening_feature_set': ranking_feature_set,
        'screening_model_type': ranking_model_type,
        'screening_feature_family': selection_summary.get('screening_selected_feature_family'),
        'screening_matches_best_overall_evaluation': selection_summary.get(
            'screening_selection_matches_overall',
            True,
        ),
        'evaluation_split': 'test',
        'training_scope': 'train_plus_val',
        'split_method': split_masks['metadata']['method'],
        'feature_family': selection_summary.get('selected_feature_family'),
    }
    benchmark_df = benchmark_regressors(
        feature_tables,
        split_masks,
        cfg,
        selected_feature_set=selected_feature_set,
        selected_model_type=selected_model_type,
    )
    robustness_df = benchmark_grouped_robustness(
        feature_tables,
        cfg,
        selected_feature_set=selected_feature_set,
        selected_model_type=selected_model_type,
    )
    bn_slice_benchmark_df, bn_slice_prediction_df = benchmark_bn_slice(
        dataset_df,
        feature_tables,
        cfg,
        selected_feature_set=selected_feature_set,
        selected_model_type=selected_model_type,
        screening_feature_set=ranking_feature_set,
        screening_model_type=ranking_model_type,
    )
    bn_family_benchmark_df, bn_family_prediction_df = benchmark_bn_family_holdout(
        dataset_df,
        feature_tables,
        cfg,
        selected_feature_set=selected_feature_set,
        selected_model_type=selected_model_type,
        screening_feature_set=ranking_feature_set,
        screening_model_type=ranking_model_type,
    )
    bn_stratified_error_df = benchmark_bn_stratified_errors(
        feature_tables,
        cfg,
        selected_feature_set=selected_feature_set,
        selected_model_type=selected_model_type,
        screening_feature_set=ranking_feature_set,
        screening_model_type=ranking_model_type,
    )
    bn_centered_screening_selection = select_bn_centered_candidate_screening_combo(
        bn_slice_benchmark_df,
        cfg,
        fallback_feature_set=ranking_feature_set,
        fallback_model_type=ranking_model_type,
    )
    ranking_feature_df = feature_tables[ranking_feature_set]
    if ranking_feature_set == selected_feature_set and ranking_model_type == selected_model_type:
        ranking_model = model
        ranking_feature_columns = feature_columns
    else:
        ranking_model, ranking_feature_columns = train_baseline_model(
            ranking_feature_df,
            split_masks,
            cfg,
            model_type=ranking_model_type,
            include_validation=True,
        )
    candidate_prediction_member_df = build_candidate_prediction_members(
        candidate_df,
        feature_tables,
        split_masks,
        cfg,
        candidate_feature_sets=selection_summary.get('screening_candidate_feature_sets'),
    )
    candidate_ensemble_df = build_candidate_prediction_ensemble(
        candidate_df,
        feature_tables,
        split_masks,
        cfg,
        candidate_feature_sets=selection_summary.get('screening_candidate_feature_sets'),
    )
    candidate_grouped_robustness_member_df = build_candidate_grouped_robustness_prediction_members(
        candidate_df,
        ranking_feature_df,
        split_masks,
        cfg,
        feature_set=ranking_feature_set,
        model_type=ranking_model_type,
    )
    candidate_grouped_robustness_df = build_candidate_grouped_robustness_predictions(
        candidate_df,
        ranking_feature_df,
        split_masks,
        cfg,
        feature_set=ranking_feature_set,
        model_type=ranking_model_type,
    )

    ranked_candidate_df = screen_candidates(
        candidate_df,
        ranking_model,
        ranking_feature_columns,
        cfg,
        feature_set=ranking_feature_set,
        model_type=ranking_model_type,
        best_overall_feature_set=selected_feature_set,
        best_overall_model_type=selected_model_type,
        screening_selection_note=selection_summary.get('screening_selection_note'),
        dataset_df=dataset_df,
        split_masks=split_masks,
        ensemble_prediction_df=candidate_ensemble_df,
        grouped_robustness_prediction_df=candidate_grouped_robustness_df,
        reference_feature_df=ranking_feature_df,
    )

    bn_centered_ranked_candidate_df = None
    bn_centered_grouped_robustness_member_df = pd.DataFrame()
    if bool(bn_centered_screening_selection.get('enabled')):
        bn_centered_feature_set = str(bn_centered_screening_selection['feature_set'])
        bn_centered_model_type = str(bn_centered_screening_selection['model_type'])
        bn_centered_feature_df = feature_tables[bn_centered_feature_set]
        if bn_centered_feature_set == ranking_feature_set and bn_centered_model_type == ranking_model_type:
            bn_centered_model = ranking_model
            bn_centered_feature_columns = ranking_feature_columns
        elif bn_centered_feature_set == selected_feature_set and bn_centered_model_type == selected_model_type:
            bn_centered_model = model
            bn_centered_feature_columns = feature_columns
        else:
            bn_centered_model, bn_centered_feature_columns = train_baseline_model(
                bn_centered_feature_df,
                split_masks,
                cfg,
                model_type=bn_centered_model_type,
                include_validation=True,
            )
        bn_centered_grouped_robustness_member_df = (
            build_candidate_grouped_robustness_prediction_members(
                candidate_df,
                bn_centered_feature_df,
                split_masks,
                cfg,
                feature_set=bn_centered_feature_set,
                model_type=bn_centered_model_type,
            )
        )
        bn_centered_grouped_robustness_df = build_candidate_grouped_robustness_predictions(
            candidate_df,
            bn_centered_feature_df,
            split_masks,
            cfg,
            feature_set=bn_centered_feature_set,
            model_type=bn_centered_model_type,
        )
        bn_centered_cfg = copy.deepcopy(cfg)
        bn_centered_cfg.setdefault('screening', {})['use_model_disagreement'] = False
        bn_centered_ranked_candidate_df = screen_candidates(
            candidate_df,
            bn_centered_model,
            bn_centered_feature_columns,
            bn_centered_cfg,
            feature_set=bn_centered_feature_set,
            model_type=bn_centered_model_type,
            best_overall_feature_set=selected_feature_set,
            best_overall_model_type=selected_model_type,
            screening_selection_note=bn_centered_screening_selection.get('selection_note'),
            dataset_df=dataset_df,
            split_masks=split_masks,
            ensemble_prediction_df=None,
            grouped_robustness_prediction_df=bn_centered_grouped_robustness_df,
            reference_feature_df=bn_centered_feature_df,
        )
    structure_generation_seed_df = build_candidate_structure_generation_seeds(
        ranked_candidate_df,
        dataset_df,
        split_masks,
        cfg,
        bn_centered_candidate_df=bn_centered_ranked_candidate_df,
        formula_col=cfg['data']['formula_column'],
    )
    structure_first_pass_variant_df = pd.DataFrame()
    structure_first_pass_summary_df = pd.DataFrame()
    structure_first_pass_payload = {}
    if structure_generation_seed_df is not None and not structure_generation_seed_df.empty:
        structure_first_pass_variant_df, structure_first_pass_summary_df, structure_first_pass_payload = (
            build_structure_first_pass_execution_artifacts(
                structure_generation_seed_df,
                cfg=cfg,
                formula_col=cfg['data']['formula_column'],
                structure_model=(
                    model if selected_feature_set == STRUCTURE_AWARE_FEATURE_SET else None
                ),
                structure_feature_columns=(
                    feature_columns if selected_feature_set == STRUCTURE_AWARE_FEATURE_SET else None
                ),
                structure_feature_set=(
                    selected_feature_set if selected_feature_set == STRUCTURE_AWARE_FEATURE_SET else None
                ),
                structure_model_type=(
                    selected_model_type if selected_feature_set == STRUCTURE_AWARE_FEATURE_SET else None
                ),
            )
        )
    summary_kwargs = {
        'dataset_df': dataset_df,
        'bn_df': bn_df,
        'candidate_df': ranked_candidate_df,
        'split_masks': split_masks,
        'selection_summary': selection_summary,
        'robustness_df': robustness_df,
        'bn_slice_benchmark_df': bn_slice_benchmark_df,
        'bn_family_benchmark_df': bn_family_benchmark_df,
        'bn_stratified_error_df': bn_stratified_error_df,
        'bn_centered_candidate_df': bn_centered_ranked_candidate_df,
        'bn_centered_screening_selection': bn_centered_screening_selection,
        'structure_generation_seed_df': structure_generation_seed_df,
        'candidate_prediction_member_df': candidate_prediction_member_df,
        'candidate_grouped_robustness_member_df': candidate_grouped_robustness_member_df,
        'bn_centered_grouped_robustness_member_df': bn_centered_grouped_robustness_member_df,
        'structure_first_pass_execution_summary_df': structure_first_pass_summary_df,
        'structure_first_pass_execution_payload': structure_first_pass_payload,
        'cfg': cfg,
    }
    supported_summary_kwargs = {
        key: value
        for key, value in summary_kwargs.items()
        if key in inspect.signature(build_experiment_summary).parameters
    }
    experiment_summary = build_experiment_summary(**supported_summary_kwargs)

    save_kwargs = {
        'metrics': metrics,
        'prediction_df': prediction_df,
        'bn_df': bn_df,
        'screened_df': ranked_candidate_df,
        'benchmark_df': benchmark_df,
        'robustness_df': robustness_df,
        'bn_slice_benchmark_df': bn_slice_benchmark_df,
        'bn_slice_prediction_df': bn_slice_prediction_df,
        'bn_family_benchmark_df': bn_family_benchmark_df,
        'bn_family_prediction_df': bn_family_prediction_df,
        'bn_stratified_error_df': bn_stratified_error_df,
        'bn_centered_screened_df': bn_centered_ranked_candidate_df,
        'structure_generation_seed_df': structure_generation_seed_df,
        'experiment_summary': experiment_summary,
        'manifest': manifest,
        'cfg': cfg,
        'candidate_prediction_member_df': candidate_prediction_member_df,
        'candidate_grouped_robustness_member_df': candidate_grouped_robustness_member_df,
        'bn_centered_grouped_robustness_member_df': bn_centered_grouped_robustness_member_df,
        'structure_first_pass_execution_variant_df': structure_first_pass_variant_df,
        'structure_first_pass_execution_summary_df': structure_first_pass_summary_df,
        'structure_first_pass_execution_payload': structure_first_pass_payload,
    }
    supported_save_kwargs = {
        key: value
        for key, value in save_kwargs.items()
        if key in inspect.signature(save_metrics_and_predictions).parameters
    }
    save_metrics_and_predictions(**supported_save_kwargs)
    save_basic_plots(prediction_df, cfg)

    print('=== BN AI PoC pipeline completed ===')
    print(f"dataset rows: {len(dataset_df)}")
    print(f"bn rows: {len(bn_df)}")
    print(f"candidate rows: {len(ranked_candidate_df)}")
    if 'proposal_shortlist_selected' in ranked_candidate_df.columns:
        print(
            'proposal shortlist rows: '
            f"{int(ranked_candidate_df['proposal_shortlist_selected'].fillna(False).sum())}"
        )
    if 'extrapolation_shortlist_selected' in ranked_candidate_df.columns:
        print(
            'extrapolation shortlist rows: '
            f"{int(ranked_candidate_df['extrapolation_shortlist_selected'].fillna(False).sum())}"
        )
    if bn_slice_benchmark_df is not None and not bn_slice_benchmark_df.empty:
        print(f"bn slice benchmark rows: {len(bn_slice_benchmark_df)}")
    if bn_family_benchmark_df is not None and not bn_family_benchmark_df.empty:
        print(f"bn family benchmark rows: {len(bn_family_benchmark_df)}")
    if bn_stratified_error_df is not None and not bn_stratified_error_df.empty:
        print(f"bn stratified error rows: {len(bn_stratified_error_df)}")
    if bn_centered_ranked_candidate_df is not None:
        print(f"bn-centered alternative ranking rows: {len(bn_centered_ranked_candidate_df)}")
        print(
            'bn-centered alternative combo: '
            f"{bn_centered_screening_selection['feature_set']} + "
            f"{bn_centered_screening_selection['model_type']}"
        )
    if structure_generation_seed_df is not None and not structure_generation_seed_df.empty:
        print(f"structure-generation seed rows: {len(structure_generation_seed_df)}")
    print(f"split method: {split_masks['metadata']['method']}")
    print(f"selected feature set: {selected_feature_set}")
    print(f"selected model: {selected_model_type}")
    print(f"ranking feature set: {ranking_feature_set}")
    print(f"ranking model: {ranking_model_type}")
    print(f"metrics: {metrics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BN AI PoC pipeline entrypoint')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run a fast smoke check for config, feature generation, candidate generation, and model imports.',
    )
    args = parser.parse_args()
    if args.dry_run:
        run_dry_run()
    else:
        main()
