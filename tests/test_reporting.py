from __future__ import annotations

import json

import pandas as pd

from pipeline.reporting import build_experiment_summary, save_basic_plots, save_metrics_and_predictions


def test_reporting_writes_expected_artifacts(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {
            'dataset': 'twod_matpd',
            'formula_column': 'formula',
            'target_column': 'band_gap',
        },
        'features': {
            'feature_set': 'basic_formula_composition',
            'candidate_sets': ['basic_formula_composition', 'matminer_composition'],
            'feature_family': 'composition_only',
        },
        'model': {
            'type': 'hist_gradient_boosting',
            'candidate_types': ['linear_regression', 'hist_gradient_boosting'],
            'benchmark_baselines': ['dummy_mean'],
        },
        'screening': {
            'candidate_space_name': 'toy_iii_v_demo_grid',
            'candidate_space_kind': 'toy_demo',
            'candidate_space_note': 'demo note',
            'top_k': 5,
        },
    }

    metrics = {
        'mae': 1.0,
        'rmse': 2.0,
        'r2': 0.5,
        'selected_model_type': 'linear_regression',
        'selected_feature_set': 'matminer_composition',
    }
    prediction_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0], 'prediction': [4.8]})
    bn_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0]})
    candidate_df = pd.DataFrame({'formula': ['BN'], 'candidate_space_name': ['toy_iii_v_demo_grid']})
    screened_df = pd.DataFrame({'formula': ['BN'], 'predicted_band_gap': [4.8]})
    benchmark_df = pd.DataFrame({
        'feature_set': ['matminer_composition', 'feature_agnostic_dummy'],
        'model_type': ['linear_regression', 'dummy_mean'],
        'mae': [1.0, 1.4],
    })
    split_masks = {
        'train': [True],
        'val': [False],
        'test': [False],
        'metadata': {'method': 'group_by_formula'},
    }
    selection_summary = {
        'selected_feature_set': 'matminer_composition',
        'selected_feature_count': 138,
        'selected_model_type': 'linear_regression',
        'used_validation_selection': True,
        'candidate_feature_sets': ['basic_formula_composition', 'matminer_composition'],
        'candidate_model_types': ['linear_regression', 'hist_gradient_boosting'],
        'feature_set_results': [
            {'feature_set': 'basic_formula_composition', 'status': 'ok'},
            {'feature_set': 'matminer_composition', 'status': 'ok'},
        ],
    }
    experiment_summary = build_experiment_summary(
        dataset_df=prediction_df,
        bn_df=bn_df,
        candidate_df=candidate_df,
        split_masks=split_masks,
        selection_summary=selection_summary,
        cfg=cfg,
    )
    manifest = {'name': 'twod_matpd'}

    save_metrics_and_predictions(
        metrics,
        prediction_df,
        bn_df,
        screened_df,
        benchmark_df,
        experiment_summary,
        manifest,
        cfg,
    )
    save_basic_plots(prediction_df, cfg)

    assert json.loads((artifact_dir / 'metrics.json').read_text()) == metrics
    assert json.loads((artifact_dir / 'manifest.json').read_text()) == manifest
    assert json.loads((artifact_dir / 'experiment_summary.json').read_text()) == experiment_summary
    assert experiment_summary['features']['selected_feature_set'] == 'matminer_composition'
    assert experiment_summary['feature_model_selection']['selected_model_type'] == 'linear_regression'
    assert (artifact_dir / 'predictions.csv').exists()
    assert (artifact_dir / 'bn_slice.csv').exists()
    assert (artifact_dir / 'demo_candidate_ranking.csv').exists()
    assert (artifact_dir / 'benchmark_results.csv').exists()
    assert (artifact_dir / 'parity_plot.png').exists()
