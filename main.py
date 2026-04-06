from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.io_utils import clear_project_cache, ensure_runtime_dirs, load_config
from pipeline.data import load_or_build_dataset
from pipeline.features import (
    build_feature_table,
    evaluate_predictions,
    filter_bn,
    generate_bn_candidates,
    make_split_masks,
    screen_candidates,
    train_baseline_model,
)
from pipeline.reporting import save_basic_plots, save_metrics_and_predictions


def main() -> None:
    clear_project_cache('.')

    config_path = Path('configs/default.py')
    cfg = load_config(config_path)

    ensure_runtime_dirs(cfg)

    dataset_df, manifest = load_or_build_dataset(cfg)
    bn_df = filter_bn(dataset_df, formula_col=cfg['data']['formula_column'])
    candidate_df = generate_bn_candidates()

    feature_df = build_feature_table(dataset_df, formula_col=cfg['data']['formula_column'])
    split_masks = make_split_masks(feature_df, cfg)

    model, feature_columns = train_baseline_model(feature_df, split_masks, cfg)
    metrics, prediction_df = evaluate_predictions(feature_df, split_masks, model, feature_columns)

    screened_df = screen_candidates(candidate_df, model, feature_columns, cfg)

    save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, manifest, cfg)
    save_basic_plots(prediction_df, cfg)

    print('=== BN AI PoC pipeline completed ===')
    print(f"dataset rows: {len(dataset_df)}")
    print(f"bn rows: {len(bn_df)}")
    print(f"candidate rows: {len(screened_df)}")
    print(f"metrics: {metrics}")


if __name__ == '__main__':
    main()
