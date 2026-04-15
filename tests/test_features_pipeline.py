from __future__ import annotations

import pandas as pd
import pytest

from pipeline.features import (
    build_feature_table,
    evaluate_predictions,
    generate_bn_candidates,
    make_model,
    make_split_masks,
    screen_candidates,
    train_baseline_model,
)


CFG = {
    'project': {'random_seed': 42},
    'split': {'train_ratio': 0.6, 'val_ratio': 0.2, 'test_ratio': 0.2},
    'model': {
        'type': 'random_forest',
        'random_forest': {'n_estimators': 16, 'random_state': 42, 'n_jobs': 1},
        'hist_gradient_boosting': {
            'max_depth': 3,
            'learning_rate': 0.1,
            'max_iter': 20,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
    },
    'screening': {'top_k': 5},
}


def _sample_training_df() -> pd.DataFrame:
    formulas = ['BN', 'AlN', 'GaN', 'InN', 'BP', 'BAs', 'AlP', 'GaP', 'InP', 'TlN']
    targets = [5.0, 3.2, 3.1, 1.8, 2.0, 1.5, 2.3, 2.1, 1.2, 0.9]
    return pd.DataFrame({'formula': formulas, 'target': targets, 'source': 'twod_matpd'})


def test_feature_pipeline_can_train_evaluate_and_screen():
    candidates = generate_bn_candidates()
    feature_df = build_feature_table(_sample_training_df())
    split_masks = make_split_masks(feature_df, CFG)

    model, feature_columns = train_baseline_model(feature_df, split_masks, CFG)
    metrics, prediction_df = evaluate_predictions(feature_df, split_masks, model, feature_columns)
    screened_df = screen_candidates(candidates, model, feature_columns, CFG)

    assert len(candidates) == 25
    assert {'BN', 'TlBi'}.issubset(set(candidates['formula']))
    assert set(feature_columns) >= {'n_elements', 'sum_z', 'contains_B', 'contains_N'}
    assert set(metrics) == {'mae', 'rmse', 'r2'}
    assert not prediction_df.empty
    assert len(screened_df) == CFG['screening']['top_k']
    assert screened_df['prediction'].is_monotonic_decreasing


def test_make_model_rejects_unknown_model_type():
    bad_cfg = {
        **CFG,
        'model': {
            **CFG['model'],
            'type': 'does_not_exist',
        },
    }

    with pytest.raises(ValueError, match='Unsupported model type'):
        make_model(bad_cfg)
