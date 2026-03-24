from pathlib import Path

from bnai.utils.io import delete_cache, ensure_runtime_dirs, load_yaml
from bnai.adapters.jarvis_twod_matpd import load_or_build_dataset
from bnai.preprocess.bn_filters import filter_bn, generate_bn_candidates
from bnai.preprocess.featurize import build_feature_table
from bnai.preprocess.splits import make_split_masks
from bnai.train.train import train_baseline_model
from bnai.train.evaluate import evaluate_predictions, save_metrics_and_predictions
from bnai.infer.predict import screen_candidates
from bnai.viz.plots import save_basic_plots


def main() -> None:
    delete_cache('.')

    config_path = Path('configs/default.yaml')
    cfg = load_yaml(config_path)

    ensure_runtime_dirs(cfg)

    dataset_df, manifest = load_or_build_dataset(cfg)
    bn_df = filter_bn(dataset_df, formula_col=cfg['data']['formula_column'])
    candidate_df = generate_bn_candidates()

    feature_df = build_feature_table(dataset_df, formula_col=cfg['data']['formula_column'])
    split_masks = make_split_masks(feature_df, cfg)

    model_bundle = train_baseline_model(feature_df, split_masks, cfg)
    metrics, prediction_df = evaluate_predictions(feature_df, split_masks, model_bundle, cfg)

    screened_df = screen_candidates(candidate_df, model_bundle, cfg)

    save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, manifest, cfg)
    save_basic_plots(prediction_df, cfg)

    print('=== BN AI PoC pipeline completed ===')
    print(f"dataset rows: {len(dataset_df)}")
    print(f"bn rows: {len(bn_df)}")
    print(f"candidate rows: {len(screened_df)}")
    print(f"metrics: {metrics}")


if __name__ == '__main__':
    main()
