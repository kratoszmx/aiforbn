from __future__ import annotations

import copy

import pandas as pd
import pytest

from pipeline.features import (
    benchmark_regressors,
    build_feature_table,
    build_feature_tables,
    evaluate_predictions,
    generate_bn_candidates,
    make_model,
    make_split_masks,
    screen_candidates,
    select_feature_model_combo,
    summarize_feature_table,
    train_baseline_model,
)


CFG = {
    'project': {'random_seed': 42},
    'split': {
        'method': 'group_by_formula',
        'group_column': 'formula',
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
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
        'use_validation_selection': True,
        'selection_metric': 'mae',
        'linear_regression': {},
        'random_forest': {'n_estimators': 16, 'random_state': 42, 'n_jobs': 1},
        'hist_gradient_boosting': {
            'max_depth': 3,
            'learning_rate': 0.1,
            'max_iter': 40,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'dummy_mean': {'strategy': 'mean'},
    },
    'screening': {
        'top_k': 5,
        'candidate_space_name': 'toy_iii_v_demo_grid',
        'candidate_space_kind': 'toy_demo',
        'candidate_space_note': 'demo note',
        'ranking_label': 'demo_candidate_ranking',
    },
}


def _sample_training_df() -> pd.DataFrame:
    formulas = [
        'BN', 'BN', 'AlN', 'AlN', 'GaN', 'GaN', 'InN', 'InN', 'BP', 'BP',
        'BAs', 'BAs', 'AlP', 'AlP', 'GaP', 'GaP', 'InP', 'InP', 'TlN', 'TlN',
    ]
    targets = [
        5.0, 5.1, 3.2, 3.1, 3.1, 3.0, 1.8, 1.7, 2.0, 2.1,
        1.5, 1.4, 2.3, 2.4, 2.1, 2.0, 1.2, 1.1, 0.9, 0.8,
    ]
    return pd.DataFrame({'formula': formulas, 'target': targets, 'source': 'twod_matpd'})


def _stoichiometry_signal_df() -> pd.DataFrame:
    base_df = pd.DataFrame({
        'formula': [
            'BN', 'BN2', 'B2N', 'B2N3', 'B3N2', 'B3N4',
            'AlN', 'AlN2', 'Al2N', 'Al2N3', 'Al3N2', 'Al3N4',
            'GaN', 'GaN2', 'Ga2N', 'Ga2N3', 'Ga3N2', 'Ga3N4',
        ],
        'source': 'twod_matpd',
    })
    matminer_df = build_feature_table(base_df, feature_set='matminer_composition')
    target = (
        matminer_df['matminer_2_norm'] * 6.0
        + matminer_df['matminer_magpiedata_mean_number'] * 0.02
    )
    return base_df.assign(target=target)


def test_make_split_masks_groups_duplicate_formulas_together():
    dataset_df = _sample_training_df()
    split_masks = make_split_masks(dataset_df, CFG)

    split_formulas = {
        name: set(dataset_df.loc[split_masks[name], 'formula'])
        for name in ('train', 'val', 'test')
    }

    assert split_masks['metadata']['method'] == 'group_by_formula'
    assert split_masks['metadata']['group_overlap_counts'] == {
        'train_val': 0,
        'train_test': 0,
        'val_test': 0,
    }

    for formula in dataset_df['formula'].unique():
        assigned_splits = [name for name, values in split_formulas.items() if formula in values]
        assert len(assigned_splits) == 1


def test_build_feature_table_with_matminer_marks_failed_formula_rows():
    feature_df = build_feature_table(
        pd.DataFrame({'formula': ['BN', '??'], 'target': [5.0, 0.0]}),
        feature_set='matminer_composition',
    )
    summary = summarize_feature_table(feature_df, feature_set='matminer_composition')

    assert bool(feature_df.loc[0, 'feature_generation_failed']) is False
    assert bool(feature_df.loc[1, 'feature_generation_failed']) is True
    assert isinstance(feature_df.loc[1, 'feature_generation_error'], str)
    assert summary['status'] == 'featurization_incomplete'
    assert summary['failed_formula_count'] == 1
    assert summary['failed_formula_examples'] == ['??']


def test_select_feature_model_combo_can_choose_matminer_representation():
    selection_cfg = copy.deepcopy(CFG)
    dataset_df = _stoichiometry_signal_df()
    feature_tables = build_feature_tables(dataset_df, selection_cfg)
    split_masks = make_split_masks(dataset_df, selection_cfg)
    summary = select_feature_model_combo(feature_tables, split_masks, selection_cfg)

    assert summary['used_validation_selection'] is True
    assert summary['selected_feature_set'] == 'matminer_composition'
    assert summary['selected_model_type'] in {'linear_regression', 'hist_gradient_boosting'}
    assert {
        (row['feature_set'], row['model_type'])
        for row in summary['validation_results']
        if row['status'] == 'ok'
    } == {
        ('basic_formula_composition', 'hist_gradient_boosting'),
        ('basic_formula_composition', 'linear_regression'),
        ('matminer_composition', 'hist_gradient_boosting'),
        ('matminer_composition', 'linear_regression'),
    }


def test_feature_pipeline_can_train_evaluate_benchmark_and_rank_demo_candidates():
    candidates = generate_bn_candidates(CFG)
    dataset_df = _sample_training_df()
    feature_tables = build_feature_tables(dataset_df, CFG)
    split_masks = make_split_masks(dataset_df, CFG)
    selection_summary = select_feature_model_combo(feature_tables, split_masks, CFG)
    selected_feature_df = feature_tables[selection_summary['selected_feature_set']]

    model, feature_columns = train_baseline_model(
        selected_feature_df,
        split_masks,
        CFG,
        model_type=selection_summary['selected_model_type'],
        include_validation=True,
    )
    metrics, prediction_df = evaluate_predictions(selected_feature_df, split_masks, model, feature_columns)
    benchmark_df = benchmark_regressors(
        feature_tables,
        split_masks,
        CFG,
        selected_feature_set=selection_summary['selected_feature_set'],
        selected_model_type=selection_summary['selected_model_type'],
    )
    screened_df = screen_candidates(
        candidates,
        model,
        feature_columns,
        CFG,
        feature_set=selection_summary['selected_feature_set'],
        model_type=selection_summary['selected_model_type'],
    )

    assert len(candidates) == 25
    assert {'BN', 'TlBi'}.issubset(set(candidates['formula']))
    assert candidates['candidate_space_kind'].eq('toy_demo').all()
    assert set(metrics) == {'mae', 'rmse', 'r2'}
    assert not prediction_df.empty
    assert len(benchmark_df) == 5
    assert set(benchmark_df['feature_set']) == {
        'basic_formula_composition',
        'matminer_composition',
        'feature_agnostic_dummy',
    }
    assert benchmark_df['selected_by_validation'].sum() == 1
    assert 'dummy_baseline' in set(benchmark_df['benchmark_role'])
    assert len(screened_df) == CFG['screening']['top_k']
    assert screened_df['predicted_band_gap'].is_monotonic_decreasing
    assert screened_df['ranking_label'].eq('demo_candidate_ranking').all()
    assert screened_df['ranking_basis'].eq('composition_only_predicted_band_gap').all()
    assert screened_df['ranking_feature_set'].eq(selection_summary['selected_feature_set']).all()
    assert screened_df['ranking_model_type'].eq(selection_summary['selected_model_type']).all()


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
