from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.io_utils import clear_project_cache, ensure_runtime_dirs, load_config
from pipeline.data import load_or_build_dataset
from pipeline.features import (
    benchmark_regressors,
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
    feature_df = feature_tables[selected_feature_set]

    model, feature_columns = train_baseline_model(
        feature_df,
        split_masks,
        cfg,
        model_type=selection_summary['selected_model_type'],
        include_validation=True,
    )
    metrics, prediction_df = evaluate_predictions(feature_df, split_masks, model, feature_columns)
    metrics = {
        **metrics,
        'selected_feature_set': selected_feature_set,
        'selected_model_type': selection_summary['selected_model_type'],
        'evaluation_split': 'test',
        'training_scope': 'train_plus_val',
        'split_method': split_masks['metadata']['method'],
        'feature_family': cfg['features'].get('feature_family', 'composition_only'),
    }
    benchmark_df = benchmark_regressors(
        feature_tables,
        split_masks,
        cfg,
        selected_feature_set=selected_feature_set,
        selected_model_type=selection_summary['selected_model_type'],
    )

    ranked_candidate_df = screen_candidates(
        candidate_df,
        model,
        feature_columns,
        cfg,
        feature_set=selected_feature_set,
        model_type=selection_summary['selected_model_type'],
    )
    experiment_summary = build_experiment_summary(
        dataset_df=dataset_df,
        bn_df=bn_df,
        candidate_df=candidate_df,
        split_masks=split_masks,
        selection_summary=selection_summary,
        cfg=cfg,
    )

    save_metrics_and_predictions(
        metrics,
        prediction_df,
        bn_df,
        ranked_candidate_df,
        benchmark_df,
        experiment_summary,
        manifest,
        cfg,
    )
    save_basic_plots(prediction_df, cfg)

    print('=== BN AI PoC pipeline completed ===')
    print(f"dataset rows: {len(dataset_df)}")
    print(f"bn rows: {len(bn_df)}")
    print(f"candidate rows: {len(ranked_candidate_df)}")
    print(f"split method: {split_masks['metadata']['method']}")
    print(f"selected feature set: {selected_feature_set}")
    print(f"selected model: {selection_summary['selected_model_type']}")
    print(f"metrics: {metrics}")


if __name__ == '__main__':
    main()
