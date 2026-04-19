from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.io_utils import clear_project_cache, ensure_runtime_dirs, load_config
from pipeline.data import load_or_build_dataset
from pipeline.features import (
    benchmark_bn_slice,
    benchmark_grouped_robustness,
    benchmark_regressors,
    build_candidate_grouped_robustness_predictions,
    build_candidate_prediction_ensemble,
    build_feature_tables,
    evaluate_predictions,
    filter_bn,
    generate_bn_candidates,
    make_split_masks,
    select_feature_model_combo,
    screen_candidates,
    train_baseline_model,
)
from pipeline.reporting import (
    build_experiment_summary,
    save_basic_plots,
    save_metrics_and_predictions,
)


def main() -> None:
    clear_project_cache('.')

    config_path = Path('configs/default.py')
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
    candidate_ensemble_df = build_candidate_prediction_ensemble(
        candidate_df,
        feature_tables,
        split_masks,
        cfg,
        candidate_feature_sets=selection_summary.get('screening_candidate_feature_sets'),
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
    experiment_summary = build_experiment_summary(
        dataset_df=dataset_df,
        bn_df=bn_df,
        candidate_df=ranked_candidate_df,
        split_masks=split_masks,
        selection_summary=selection_summary,
        robustness_df=robustness_df,
        bn_slice_benchmark_df=bn_slice_benchmark_df,
        cfg=cfg,
    )

    save_metrics_and_predictions(
        metrics,
        prediction_df,
        bn_df,
        ranked_candidate_df,
        benchmark_df,
        robustness_df,
        bn_slice_benchmark_df,
        bn_slice_prediction_df,
        experiment_summary,
        manifest,
        cfg,
    )
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
    print(f"split method: {split_masks['metadata']['method']}")
    print(f"selected feature set: {selected_feature_set}")
    print(f"selected model: {selected_model_type}")
    print(f"ranking feature set: {ranking_feature_set}")
    print(f"ranking model: {ranking_model_type}")
    print(f"metrics: {metrics}")


if __name__ == '__main__':
    main()
