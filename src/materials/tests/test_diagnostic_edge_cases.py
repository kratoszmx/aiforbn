from __future__ import annotations

import copy

import pandas as pd
import pytest

from materials.benchmarking import (
    benchmark_bn_family_holdout,
    benchmark_bn_slice,
    benchmark_bn_stratified_errors,
    select_bn_centered_candidate_screening_combo,
)
from materials.constants import STRUCTURE_AWARE_FEATURE_SET


BASE_CFG = {
    'project': {'random_seed': 42},
    'data': {'formula_column': 'formula'},
    'features': {'candidate_sets': ['basic_formula_composition']},
    'model': {
        'type': 'linear_regression',
        'candidate_types': ['linear_regression'],
        'benchmark_baselines': ['dummy_mean'],
        'linear_regression': {},
        'dummy_mean': {'strategy': 'mean'},
    },
    'bn_slice_benchmark': {
        'enabled': True,
        'method': 'leave_one_bn_formula_out',
        'k_neighbors': 2,
        'note': 'edge-case test',
    },
    'bn_family_benchmark': {
        'enabled': True,
        'method': 'leave_one_bn_family_out',
        'grouping_method': 'reduced_bn_chemical_system',
        'k_neighbors': 2,
        'note': 'edge-case test',
    },
    'bn_stratified_error': {
        'enabled': True,
        'method': 'group_kfold_bn_vs_non_bn_formula_stratified_error',
        'group_column': 'formula',
        'n_splits': 3,
        'note': 'edge-case test',
    },
}


def _feature_table(formulas: list[str], targets: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        'formula': formulas,
        'target': targets,
        'feature_1': [float(index + 1) for index in range(len(formulas))],
        'feature_generation_failed': [False] * len(formulas),
        'feature_generation_error': [None] * len(formulas),
        'feature_set': ['basic_formula_composition'] * len(formulas),
    })


def test_disabled_bn_diagnostics_return_empty_artifacts():
    cfg = copy.deepcopy(BASE_CFG)
    cfg['bn_slice_benchmark']['enabled'] = False
    cfg['bn_family_benchmark']['enabled'] = False
    cfg['bn_stratified_error']['enabled'] = False
    dataset_df = pd.DataFrame({'formula': ['BN', 'AlN'], 'target': [5.0, 3.0]})
    feature_tables = {
        'basic_formula_composition': _feature_table(['BN', 'AlN'], [5.0, 3.0])
    }
    common_kwargs = {
        'selected_feature_set': 'basic_formula_composition',
        'selected_model_type': 'linear_regression',
        'screening_feature_set': 'basic_formula_composition',
        'screening_model_type': 'linear_regression',
    }

    slice_df, slice_predictions = benchmark_bn_slice(
        dataset_df, feature_tables, cfg, **common_kwargs
    )
    family_df, family_predictions = benchmark_bn_family_holdout(
        dataset_df, feature_tables, cfg, **common_kwargs
    )
    stratified_df = benchmark_bn_stratified_errors(
        feature_tables, cfg, **common_kwargs
    )

    assert slice_df.empty and slice_predictions.empty
    assert family_df.empty and family_predictions.empty
    assert stratified_df.empty


def test_bn_diagnostics_report_insufficient_single_formula_and_family():
    dataset_df = pd.DataFrame({
        'formula': ['BN', 'BN', 'AlN'],
        'target': [5.0, 5.1, 3.0],
    })
    feature_tables = {
        'basic_formula_composition': _feature_table(
            ['BN', 'BN', 'AlN'], [5.0, 5.1, 3.0]
        )
    }
    common_kwargs = {
        'selected_feature_set': 'basic_formula_composition',
        'selected_model_type': 'linear_regression',
        'screening_feature_set': 'basic_formula_composition',
        'screening_model_type': 'linear_regression',
    }

    slice_df, slice_predictions = benchmark_bn_slice(
        dataset_df, feature_tables, BASE_CFG, **common_kwargs
    )
    family_df, family_predictions = benchmark_bn_family_holdout(
        dataset_df, feature_tables, BASE_CFG, **common_kwargs
    )

    assert set(slice_df['benchmark_status']) == {'insufficient_bn_formulas'}
    assert slice_df['completed_holds'].eq(0).all()
    assert slice_df[['mae', 'rmse', 'r2']].isna().all().all()
    assert slice_predictions.empty
    assert set(family_df['benchmark_status']) == {'insufficient_bn_families'}
    assert family_df['completed_family_holds'].eq(0).all()
    assert family_df[['mae', 'rmse', 'r2']].isna().all().all()
    assert family_predictions.empty


def test_stratified_diagnostic_marks_missing_non_bn_stratum():
    formulas = ['BN', 'BCN', 'BC2N', 'Si2BN']
    feature_tables = {
        'basic_formula_composition': _feature_table(formulas, [5.0, 3.0, 2.5, 1.5])
    }

    result = benchmark_bn_stratified_errors(
        feature_tables,
        BASE_CFG,
        selected_feature_set='basic_formula_composition',
        selected_model_type='linear_regression',
        screening_feature_set='basic_formula_composition',
        screening_model_type='linear_regression',
    )

    assert set(result['benchmark_status']) == {'insufficient_stratified_formulas'}
    assert result['bn_formula_count'].ge(2).all()
    assert result['non_bn_formula_count'].eq(0).all()
    assert result['non_bn_mae'].isna().all()
    assert result['bn_to_non_bn_mae_ratio'].isna().all()


def test_stratified_diagnostic_rejects_non_formula_grouping_that_can_leak_duplicates():
    cfg = copy.deepcopy(BASE_CFG)
    cfg['bn_stratified_error']['group_column'] = 'record_id'
    feature_tables = {
        'basic_formula_composition': _feature_table(
            ['BN', 'BN', 'BCN', 'BCN', 'AlN', 'GaN'],
            [5.0, 5.1, 3.0, 3.1, 2.0, 2.1],
        ).assign(record_id=['1', '2', '3', '4', '5', '6']),
    }

    with pytest.raises(ValueError, match='must match data.formula_column'):
        benchmark_bn_stratified_errors(
            feature_tables,
            cfg,
            selected_feature_set='basic_formula_composition',
            selected_model_type='linear_regression',
            screening_feature_set='basic_formula_composition',
            screening_model_type='linear_regression',
        )


def test_bn_centered_selection_rechecks_formula_only_feature_metadata():
    benchmark_df = pd.DataFrame({
        'feature_set': [STRUCTURE_AWARE_FEATURE_SET, 'basic_formula_composition'],
        'model_type': ['linear_regression', 'linear_regression'],
        'benchmark_status': ['ok', 'ok'],
        'candidate_compatible': [True, True],
        'benchmark_role': ['candidate_model', 'candidate_model'],
        'mae': [0.1, 1.0],
        'rmse': [0.2, 1.1],
        'r2': [0.9, 0.1],
    })

    selection = select_bn_centered_candidate_screening_combo(
        benchmark_df,
        BASE_CFG,
        fallback_feature_set='basic_formula_composition',
        fallback_model_type='linear_regression',
    )

    assert selection['enabled'] is True
    assert selection['feature_set'] == 'basic_formula_composition'
    assert selection['mae'] == 1.0


def test_bn_centered_selection_reports_missing_contract_columns():
    selection = select_bn_centered_candidate_screening_combo(
        pd.DataFrame({'feature_set': ['basic_formula_composition']}),
        BASE_CFG,
    )

    assert selection['enabled'] is False
    assert 'missing required columns' in selection['selection_note']
