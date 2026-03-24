from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from bnai.train.metrics import regression_metrics


def evaluate_predictions(df, split_masks, model_bundle, cfg):
    test_df = df.loc[split_masks['test']].copy()
    test_df = test_df[test_df['target'].notna()].copy()
    X = test_df[model_bundle.feature_columns]
    y = test_df['target']
    pred = model_bundle.model.predict(X)
    metrics = regression_metrics(y, pred)

    prediction_df = test_df[['formula', 'target']].copy()
    prediction_df['prediction'] = pred
    prediction_df['abs_error'] = (prediction_df['target'] - prediction_df['prediction']).abs()
    return metrics, prediction_df


def save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, manifest, cfg):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    prediction_df.to_csv(artifact_dir / 'predictions.csv', index=False)
    bn_df.to_csv(artifact_dir / 'bn_slice.csv', index=False)
    screened_df.to_csv(artifact_dir / 'screened_candidates.csv', index=False)
    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
