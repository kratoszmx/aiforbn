from __future__ import annotations

import json

import pandas as pd

from pipeline.reporting import save_basic_plots, save_metrics_and_predictions


def test_reporting_writes_expected_artifacts(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {'project': {'artifact_dir': str(artifact_dir)}}

    metrics = {'mae': 1.0, 'rmse': 2.0, 'r2': 0.5}
    prediction_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0], 'prediction': [4.8]})
    bn_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0]})
    screened_df = pd.DataFrame({'formula': ['BN'], 'prediction': [4.8]})
    manifest = {'name': 'twod_matpd'}

    save_metrics_and_predictions(metrics, prediction_df, bn_df, screened_df, manifest, cfg)
    save_basic_plots(prediction_df, cfg)

    assert json.loads((artifact_dir / 'metrics.json').read_text()) == metrics
    assert json.loads((artifact_dir / 'manifest.json').read_text()) == manifest
    assert (artifact_dir / 'predictions.csv').exists()
    assert (artifact_dir / 'bn_slice.csv').exists()
    assert (artifact_dir / 'screened_candidates.csv').exists()
    assert (artifact_dir / 'parity_plot.png').exists()
