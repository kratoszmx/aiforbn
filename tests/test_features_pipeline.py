from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pipeline.data import STRUCTURE_SUMMARY_COLUMNS
from pipeline.features import (
    STRUCTURE_AWARE_FEATURE_SET,
    benchmark_regressors,
    build_candidate_prediction_ensemble,
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
        'candidate_sets': [
            'basic_formula_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
        ],
        'feature_family': 'mixed_formula_and_structure',
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
        'use_model_disagreement': True,
        'uncertainty_method': 'small_feature_model_disagreement',
        'uncertainty_penalty': 0.5,
        'domain_support': {
            'enabled': True,
            'method': 'train_plus_val_knn_feature_space_support',
            'distance_metric': 'z_scored_euclidean_rms',
            'k_neighbors': 5,
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.15,
            'penalize_below_percentile': 25.0,
            'note': 'demo domain-support note',
        },
        'chemical_plausibility': {
            'enabled': True,
            'method': 'pymatgen_common_oxidation_state_balance',
            'selection_policy': 'annotate_and_prioritize_passing_candidates',
            'note': 'demo plausibility note',
        },
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


def _structure_signal_df() -> pd.DataFrame:
    base_df = pd.DataFrame({
        'formula': [
            'BN', 'BN2', 'B2N', 'B2N3', 'B3N2', 'B3N4',
            'AlN', 'AlN2', 'Al2N', 'Al2N3', 'Al3N2', 'Al3N4',
            'GaN', 'GaN2', 'Ga2N', 'Ga2N3', 'Ga3N2', 'Ga3N4',
        ],
        'source': 'twod_matpd',
    })
    n_rows = len(base_df)
    base_df['structure_n_sites'] = [2 + (idx % 4) for idx in range(n_rows)]
    base_df['structure_lattice_a'] = [2.4 + 0.12 * idx for idx in range(n_rows)]
    base_df['structure_lattice_b'] = [2.9 + 0.07 * ((idx * 3) % 9) for idx in range(n_rows)]
    base_df['structure_lattice_c'] = [20.0] * n_rows
    base_df['structure_lattice_gamma'] = [120.0 if idx % 2 == 0 else 90.0 for idx in range(n_rows)]

    area_values = []
    thickness_values = []
    for idx in range(n_rows):
        gamma = np.deg2rad(base_df.loc[idx, 'structure_lattice_gamma'])
        area = (
            base_df.loc[idx, 'structure_lattice_a']
            * base_df.loc[idx, 'structure_lattice_b']
            * np.sin(gamma)
        )
        thickness = 1.2 + 0.35 * ((idx * 5) % 7)
        area_values.append(area)
        thickness_values.append(thickness)

    base_df['structure_inplane_area'] = area_values
    base_df['structure_cell_height'] = [20.0] * n_rows
    base_df['structure_thickness'] = thickness_values
    base_df['structure_vacuum'] = base_df['structure_cell_height'] - base_df['structure_thickness']
    base_df['structure_areal_number_density'] = (
        base_df['structure_n_sites'] / base_df['structure_inplane_area']
    )
    base_df['structure_thickness_fraction'] = (
        base_df['structure_thickness'] / base_df['structure_cell_height']
    )

    structure_feature_df = build_feature_table(
        base_df,
        feature_set=STRUCTURE_AWARE_FEATURE_SET,
    )
    target = (
        structure_feature_df['matminer_2_norm'] * 4.0
        + structure_feature_df['structure_thickness'] * 1.8
        + structure_feature_df['structure_areal_number_density'] * 6.0
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


def test_build_feature_table_with_structure_summary_features_requires_structure_columns():
    feature_df = build_feature_table(
        pd.DataFrame({'formula': ['BN'], 'target': [5.0], 'source': ['twod_matpd']}),
        feature_set=STRUCTURE_AWARE_FEATURE_SET,
    )
    summary = summarize_feature_table(feature_df, feature_set=STRUCTURE_AWARE_FEATURE_SET)

    assert set(STRUCTURE_SUMMARY_COLUMNS).issubset(feature_df.columns)
    assert bool(feature_df.loc[0, 'feature_generation_failed']) is True
    assert 'missing structure summary values' in feature_df.loc[0, 'feature_generation_error']
    assert summary['feature_family'] == 'structure_aware'
    assert summary['candidate_compatible'] is False


def test_generate_bn_candidates_adds_chemical_plausibility_annotations():
    candidate_df = generate_bn_candidates(CFG)

    assert len(candidate_df) == 25
    assert candidate_df['chemical_plausibility_enabled'].eq(True).all()
    assert candidate_df['chemical_plausibility_method'].eq(
        'pymatgen_common_oxidation_state_balance'
    ).all()
    assert candidate_df['chemical_plausibility_guess_count'].ge(0).all()
    assert candidate_df.loc[candidate_df['formula'] == 'BN', 'chemical_plausibility_pass'].iloc[0]
    assert (
        candidate_df.loc[candidate_df['formula'] == 'BN', 'chemical_plausibility_primary_oxidation_state_guess']
        .iloc[0]
        == 'B(+3), N(-3)'
    )
    assert bool(
        candidate_df.loc[candidate_df['formula'] == 'AlBi', 'chemical_plausibility_pass'].iloc[0]
    ) is False
    assert 'No charge-balanced common oxidation-state assignment' in (
        candidate_df.loc[candidate_df['formula'] == 'AlBi', 'chemical_plausibility_note'].iloc[0]
    )


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
    assert summary['screening_selected_feature_set'] in {
        'basic_formula_composition',
        'matminer_composition',
    }


def test_select_feature_model_combo_separates_structure_aware_evaluation_from_formula_only_screening():
    selection_cfg = copy.deepcopy(CFG)
    dataset_df = _structure_signal_df()
    feature_tables = build_feature_tables(dataset_df, selection_cfg)
    split_masks = make_split_masks(dataset_df, selection_cfg)
    summary = select_feature_model_combo(feature_tables, split_masks, selection_cfg)

    assert summary['selected_feature_set'] == STRUCTURE_AWARE_FEATURE_SET
    assert summary['selected_feature_family'] == 'structure_aware'
    assert summary['screening_selected_feature_set'] in {
        'basic_formula_composition',
        'matminer_composition',
    }
    assert summary['screening_selected_feature_set'] != summary['selected_feature_set']
    assert summary['screening_selection_matches_overall'] is False
    assert 'falls back to the best candidate-compatible validation combo' in summary['screening_selection_note']


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
    candidate_ensemble_df = build_candidate_prediction_ensemble(
        candidates,
        feature_tables,
        split_masks,
        CFG,
    )
    screened_df = screen_candidates(
        candidates,
        model,
        feature_columns,
        CFG,
        feature_set=selection_summary['selected_feature_set'],
        model_type=selection_summary['selected_model_type'],
        dataset_df=dataset_df,
        split_masks=split_masks,
        ensemble_prediction_df=candidate_ensemble_df,
    )

    assert len(candidates) == 25
    assert {'BN', 'TlBi'}.issubset(set(candidates['formula']))
    assert candidates['candidate_space_kind'].eq('toy_demo').all()
    assert candidates['chemical_plausibility_pass'].sum() == 21
    assert set(candidates.loc[~candidates['chemical_plausibility_pass'], 'formula']) == {
        'AlBi',
        'GaBi',
        'InBi',
        'TlBi',
    }
    assert set(metrics) == {'mae', 'rmse', 'r2'}
    assert not prediction_df.empty
    assert len(benchmark_df) == 7
    assert set(benchmark_df['feature_set']) == {
        'basic_formula_composition',
        'matminer_composition',
        STRUCTURE_AWARE_FEATURE_SET,
        'feature_agnostic_dummy',
    }
    structure_rows = benchmark_df[benchmark_df['feature_set'] == STRUCTURE_AWARE_FEATURE_SET]
    assert structure_rows['benchmark_status'].eq('skipped_featurization_failure').all()
    assert benchmark_df['selected_by_validation'].sum() == 1
    assert 'dummy_baseline' in set(benchmark_df['benchmark_role'])
    assert len(screened_df) == len(candidates)
    assert int(screened_df['screening_selected_for_top_k'].sum()) == CFG['screening']['top_k']
    assert screened_df.loc[screened_df['chemical_plausibility_pass'], 'ranking_score'].is_monotonic_decreasing
    assert screened_df['ranking_label'].eq('demo_candidate_ranking').all()
    assert screened_df['ranking_basis'].eq(
        'composition_only_mean_band_gap_minus_model_disagreement_and_low_support_penalties'
    ).all()
    assert screened_df['ranking_feature_set'].eq(selection_summary['selected_feature_set']).all()
    assert screened_df['ranking_model_type'].eq(selection_summary['selected_model_type']).all()
    assert screened_df['ranking_uncertainty_method'].eq('small_feature_model_disagreement').all()
    assert screened_df['ranking_feature_family'].eq('composition_only').all()
    assert screened_df['ensemble_member_count'].eq(4).all()
    assert screened_df['domain_support_enabled'].eq(True).all()
    assert screened_df['domain_support_method'].eq(
        'train_plus_val_knn_feature_space_support'
    ).all()
    assert screened_df['domain_support_distance_metric'].eq('z_scored_euclidean_rms').all()
    assert screened_df['domain_support_reference_split'].eq('train_plus_val_unique_formulas').all()
    expected_reference_formula_count = int(
        dataset_df.loc[
            np.asarray(split_masks['train']) | np.asarray(split_masks['val']),
            'formula',
        ]
        .astype(str)
        .nunique()
    )
    assert screened_df['domain_support_reference_formula_count'].eq(
        expected_reference_formula_count
    ).all()
    assert screened_df['domain_support_k_neighbors'].eq(5).all()
    assert (screened_df['ranking_score_before_domain_support_penalty'] >= screened_df['ranking_score']).all()
    assert screened_df['domain_support_penalty'].ge(0.0).all()
    assert (screened_df['domain_support_penalty'] > 0.0).any()
    assert screened_df['ranking_rank'].tolist() == list(range(1, len(screened_df) + 1))
    assert screened_df['chemical_plausibility_pass'].head(CFG['screening']['top_k']).all()
    assert screened_df.loc[
        ~screened_df['chemical_plausibility_pass'],
        'screening_selection_decision',
    ].eq('failed_chemical_plausibility').all()
    bn_row = screened_df.loc[screened_df['formula'] == 'BN'].iloc[0]
    bbi_row = screened_df.loc[screened_df['formula'] == 'BBi'].iloc[0]
    assert bn_row['domain_support_nearest_formula'] == 'BN'
    assert bn_row['domain_support_nearest_distance'] == pytest.approx(0.0)
    assert bn_row['domain_support_percentile'] == pytest.approx(100.0)
    assert bn_row['domain_support_penalty'] == pytest.approx(0.0)
    assert bbi_row['domain_support_nearest_distance'] > 0.0
    assert bn_row['domain_support_percentile'] >= bbi_row['domain_support_percentile']
    assert 'train+val feature-space domain-support layer' in screened_df['ranking_note'].iloc[0]
    assert 'Novelty is tracked only at the formula level' in screened_df['ranking_note'].iloc[0]
    assert bool(screened_df.loc[screened_df['formula'] == 'BN', 'seen_in_dataset'].iloc[0]) is True
    assert bool(screened_df.loc[screened_df['formula'] == 'BN', 'seen_in_train_plus_val'].iloc[0]) is True
    assert screened_df['dataset_formula_row_count'].ge(0).all()
    assert screened_df['train_plus_val_formula_row_count'].ge(0).all()
    assert screened_df['candidate_is_seen_in_dataset'].equals(screened_df['seen_in_dataset'])
    assert screened_df['candidate_is_seen_in_train_plus_val'].equals(screened_df['seen_in_train_plus_val'])
    assert (
        screened_df['candidate_is_formula_level_extrapolation']
        .eq(~screened_df['seen_in_dataset'])
        .all()
    )
    assert set(screened_df['candidate_novelty_bucket']) == {
        'train_plus_val_rediscovery',
        'held_out_known_formula',
        'formula_level_extrapolation',
    }
    assert set(screened_df['candidate_novelty_priority']) == {1, 2, 3}
    assert screened_df['novelty_rank_within_bucket'].ge(1).all()
    assert screened_df.loc[
        screened_df['candidate_is_formula_level_extrapolation'],
        'novel_formula_rank',
    ].notna().all()
    assert screened_df.loc[
        ~screened_df['candidate_is_formula_level_extrapolation'],
        'novel_formula_rank',
    ].isna().all()
    assert 'rediscovery / in-domain replay' in (
        screened_df.loc[
            screened_df['candidate_novelty_bucket'] == 'train_plus_val_rediscovery',
            'candidate_novelty_note',
        ]
        .iloc[0]
    )
    assert 'held-out-known formula' in (
        screened_df.loc[
            screened_df['candidate_novelty_bucket'] == 'held_out_known_formula',
            'candidate_novelty_note',
        ]
        .iloc[0]
    )
    assert 'formula-level extrapolation' in (
        screened_df.loc[
            screened_df['candidate_novelty_bucket'] == 'formula_level_extrapolation',
            'candidate_novelty_note',
        ]
        .iloc[0]
    )


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
