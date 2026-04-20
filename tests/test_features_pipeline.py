from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pipeline.data import STRUCTURE_SUMMARY_COLUMNS
from pipeline.features import (
    FRACTIONAL_COMPOSITION_FEATURE_SET,
    STRUCTURE_AWARE_FEATURE_SET,
    benchmark_bn_family_holdout,
    benchmark_bn_slice,
    benchmark_bn_stratified_errors,
    benchmark_grouped_robustness,
    benchmark_regressors,
    build_candidate_grouped_robustness_predictions,
    build_candidate_prediction_ensemble,
    build_candidate_prediction_members,
    build_candidate_structure_generation_seeds,
    build_feature_table,
    build_feature_tables,
    evaluate_predictions,
    generate_bn_candidates,
    make_model,
    make_split_masks,
    screen_candidates,
    select_bn_centered_candidate_screening_combo,
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
    'robustness': {
        'enabled': True,
        'method': 'group_kfold_by_formula',
        'group_column': 'formula',
        'n_splits': 5,
        'note': 'demo grouped robustness note',
    },
    'bn_slice_benchmark': {
        'enabled': True,
        'method': 'leave_one_bn_formula_out',
        'k_neighbors': 2,
        'note': 'demo bn slice benchmark note',
    },
    'bn_family_benchmark': {
        'enabled': True,
        'method': 'leave_one_bn_family_out',
        'grouping_method': 'reduced_bn_chemical_system',
        'k_neighbors': 2,
        'note': 'demo bn family benchmark note',
    },
    'bn_stratified_error': {
        'enabled': True,
        'method': 'group_kfold_bn_vs_non_bn_formula_stratified_error',
        'group_column': 'formula',
        'n_splits': 3,
        'note': 'demo bn stratified error note',
    },
    'screening': {
        'objective_name': 'bn_themed_formula_level_wide_gap_followup_prioritization',
        'objective_target_property': 'band_gap',
        'objective_target_direction': 'maximize',
        'objective_decision_unit': 'formula_level_candidate',
        'objective_decision_consequence': 'low_confidence_prioritization_for_structure_followup',
        'objective_note': 'Use the ranking as low-confidence formula-level prioritization for structure follow-up, not direct discovery.',
        'top_k': 5,
        'candidate_generation_strategy': 'bn_anchored_formula_family_grid',
        'candidate_space_name': 'bn_anchored_formula_family_grid',
        'candidate_space_kind': 'bn_family_demo',
        'candidate_space_note': 'bn-anchored demo note',
        'ranking_label': 'demo_candidate_ranking',
        'use_model_disagreement': True,
        'uncertainty_method': 'small_feature_model_disagreement',
        'uncertainty_penalty': 0.5,
        'grouped_robustness_uncertainty': {
            'enabled': True,
            'method': 'selected_formula_only_group_kfold_candidate_prediction_std',
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.15,
            'note': 'demo grouped candidate robustness note',
        },
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
        'bn_support': {
            'enabled': True,
            'method': 'train_plus_val_bn_knn_feature_space_support',
            'distance_metric': 'z_scored_euclidean_rms',
            'k_neighbors': 3,
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.1,
            'penalize_below_percentile': 25.0,
            'note': 'demo bn-support note',
        },
        'bn_analog_evidence': {
            'enabled': True,
            'aggregation': 'mean_over_k_nearest_bn_formulas',
            'reference_split': 'train_plus_val_bn_unique_formulas',
            'exfoliation_reference': 'train_plus_val_bn_formula_median',
            'note': 'demo bn-analog evidence note',
        },
        'bn_band_gap_alignment': {
            'enabled': True,
            'method': 'predicted_band_gap_vs_local_bn_analog_window',
            'reference_split': 'train_plus_val_bn_unique_formulas',
            'window_expansion_iqr_factor': 0.5,
            'minimum_neighbor_formula_count_for_penalty': 2,
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.08,
            'note': 'demo bn-local band-gap alignment note',
        },
        'bn_analog_validation': {
            'enabled': True,
            'method': 'bn_analog_alignment_vote_fraction',
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.12,
            'note': 'demo bn-analog validation note',
        },
        'chemical_plausibility': {
            'enabled': True,
            'method': 'pymatgen_common_oxidation_state_balance',
            'selection_policy': 'annotate_and_prioritize_passing_candidates',
            'note': 'demo plausibility note',
        },
        'proposal_shortlist': {
            'enabled': True,
            'label': 'family_aware_proposal_shortlist',
            'method': 'ranked_family_cap',
            'shortlist_size': 4,
            'max_per_candidate_family': 1,
            'chemical_plausibility_priority': True,
            'note': 'demo proposal shortlist note',
        },
        'extrapolation_shortlist': {
            'enabled': True,
            'label': 'formula_level_extrapolation_shortlist',
            'method': 'novelty_bucket_ranked_family_cap',
            'shortlist_size': 3,
            'max_per_candidate_family': 1,
            'required_novelty_bucket': 'formula_level_extrapolation',
            'chemical_plausibility_priority': True,
            'note': 'demo extrapolation shortlist note',
        },
        'structure_generation_seeds': {
            'enabled': True,
            'label': 'bn_structure_generation_seed_set',
            'method': 'bn_analog_reference_exemplar',
            'candidate_scope': 'proposal_shortlist_plus_extrapolation_shortlist_plus_bn_centered_top_n',
            'per_candidate_seed_limit': 2,
            'bn_centered_top_n': 2,
            'note': 'demo structure-generation seed note',
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
    energy_per_atom = [
        -8.30, -8.31, -6.20, -6.18, -5.95, -5.96, -5.40, -5.39, -6.85, -6.83,
        -6.10, -6.09, -5.90, -5.88, -5.70, -5.72, -5.35, -5.34, -5.15, -5.14,
    ]
    exfoliation_energy_per_atom = [
        0.060, 0.063, 0.120, 0.118, 0.110, 0.112, 0.150, 0.148, 0.090, 0.092,
        0.085, 0.084, 0.130, 0.128, 0.125, 0.127, 0.145, 0.144, 0.160, 0.158,
    ]
    total_magnetization = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.1, 0.1, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.3, 0.3,
    ]
    return pd.DataFrame({
        'formula': formulas,
        'target': targets,
        'source': 'twod_matpd',
        'energy_per_atom': energy_per_atom,
        'exfoliation_energy_per_atom': exfoliation_energy_per_atom,
        'total_magnetization': total_magnetization,
        'abs_total_magnetization': [abs(value) for value in total_magnetization],
    })


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


def _bn_alignment_reference_df() -> pd.DataFrame:
    formulas = ['BN', 'BC2N', 'Si2BN', 'BCN', 'AlN']
    targets = [5.2, 2.8, 1.6, 3.5, 3.2]
    energy_per_atom = [-8.30, -7.95, -7.60, -7.75, -6.20]
    exfoliation_energy_per_atom = [0.060, 0.082, 0.094, 0.086, 0.120]
    total_magnetization = [0.0, 0.0, 0.0, 0.0, 0.0]
    return pd.DataFrame({
        'formula': formulas,
        'target': targets,
        'source': 'twod_matpd',
        'energy_per_atom': energy_per_atom,
        'exfoliation_energy_per_atom': exfoliation_energy_per_atom,
        'total_magnetization': total_magnetization,
        'abs_total_magnetization': [abs(value) for value in total_magnetization],
    })


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


def test_generate_bn_candidates_adds_bn_anchored_family_metadata_and_chemical_plausibility_annotations():
    candidate_df = generate_bn_candidates(CFG)

    assert len(candidate_df) == 25
    assert candidate_df['candidate_space_name'].eq('bn_anchored_formula_family_grid').all()
    assert candidate_df['candidate_space_kind'].eq('bn_family_demo').all()
    assert candidate_df['candidate_generation_strategy'].eq(
        'bn_anchored_formula_family_grid'
    ).all()
    assert {'BN', 'BC2N', 'Si2BN', 'Ge2BN', 'AlBN2', 'Tl2BN'}.issubset(set(candidate_df['formula']))
    assert candidate_df['candidate_family'].nunique() >= 4
    assert candidate_df['candidate_template'].isin({'B1N1', 'B1X1N1', 'B1X2N1', 'B1X1N2', 'X1B1N1', 'X2B1N1', 'X1B1N2'}).all()
    assert candidate_df['candidate_family_note'].astype(str).str.len().gt(0).all()
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
        candidate_df.loc[candidate_df['formula'] == 'AlBN', 'chemical_plausibility_pass'].iloc[0]
    ) is False
    assert bool(
        candidate_df.loc[candidate_df['formula'] == 'TlBN', 'chemical_plausibility_pass'].iloc[0]
    ) is False
    assert 'No charge-balanced common oxidation-state assignment' in (
        candidate_df.loc[candidate_df['formula'] == 'AlBN', 'chemical_plausibility_note'].iloc[0]
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


def test_grouped_robustness_benchmark_summarizes_group_kfold_results():
    dataset_df = _stoichiometry_signal_df()
    feature_tables = build_feature_tables(dataset_df, CFG)
    split_masks = make_split_masks(dataset_df, CFG)
    selection_summary = select_feature_model_combo(feature_tables, split_masks, CFG)
    robustness_df = benchmark_grouped_robustness(
        feature_tables,
        CFG,
        selected_feature_set=selection_summary['selected_feature_set'],
        selected_model_type=selection_summary['selected_model_type'],
    )

    assert not robustness_df.empty
    assert robustness_df['robustness_method'].eq('group_kfold_by_formula').all()
    assert robustness_df['robustness_group_column'].eq('formula').all()
    ok_rows = robustness_df.loc[robustness_df['robustness_status'] == 'ok']
    assert ok_rows['requested_folds'].eq(5).all()
    assert ok_rows['actual_folds'].eq(5).all()
    assert ok_rows['completed_folds'].eq(5).all()
    assert ok_rows['mae_mean'].notna().all()
    assert ok_rows['rmse_mean'].notna().all()
    assert 'dummy_baseline' in set(robustness_df['benchmark_role'])
    structure_rows = robustness_df[robustness_df['feature_set'] == STRUCTURE_AWARE_FEATURE_SET]
    assert structure_rows['robustness_status'].eq('skipped_featurization_failure').all()
    assert robustness_df['selected_by_validation'].sum() == 1


def test_select_bn_centered_candidate_screening_combo_prefers_best_candidate_compatible_row():
    benchmark_df = pd.DataFrame({
        'feature_set': [
            'matminer_composition',
            'basic_formula_composition',
            'matminer_composition',
            'matminer_composition',
        ],
        'feature_family': [
            'composition_only',
            'composition_only',
            'composition_only',
            'composition_only',
        ],
        'model_type': [
            'linear_regression',
            'hist_gradient_boosting',
            'dummy_mean',
            'bn_local_knn_mean',
        ],
        'benchmark_role': [
            'candidate_model',
            'screening_model',
            'global_dummy_mean_baseline',
            'bn_local_reference_baseline',
        ],
        'candidate_compatible': [True, True, False, True],
        'benchmark_status': ['ok', 'ok', 'ok', 'ok'],
        'mae': [1.2, 1.4, 0.8, 2.0],
        'rmse': [1.5, 1.7, 1.0, 2.1],
        'r2': [0.1, -0.2, 0.0, -1.0],
    })

    summary = select_bn_centered_candidate_screening_combo(
        benchmark_df,
        CFG,
        fallback_feature_set='matminer_composition',
        fallback_model_type='hist_gradient_boosting',
    )

    assert summary['enabled'] is True
    assert summary['feature_set'] == 'matminer_composition'
    assert summary['model_type'] == 'linear_regression'
    assert summary['benchmark_role'] == 'candidate_model'
    assert summary['mae'] == pytest.approx(1.2)
    assert summary['matches_general_screening_combo'] is False
    assert summary['ranking_artifact'] == 'demo_candidate_bn_centered_ranking.csv'



def test_build_candidate_structure_generation_seeds_links_shortlisted_candidates_to_bn_reference_records():
    dataset_df = _structure_signal_df().copy()
    dataset_df['record_id'] = [f'jid-{idx}' for idx in range(len(dataset_df))]
    split_masks = {
        'train': [True] * 12 + [False] * 6,
        'val': [False] * 12 + [True] * 3 + [False] * 3,
        'test': [False] * 15 + [True] * 3,
        'metadata': {'method': 'group_by_formula'},
    }
    candidate_df = pd.DataFrame({
        'formula': ['BCN2', 'BCN', 'AlBN2'],
        'candidate_family': ['group14_bn_121_family', 'group14_bn_111_family', 'group13_bn_121_family'],
        'candidate_template': ['B1X1N2', 'B1X1N1', 'X1B1N2'],
        'candidate_novelty_bucket': ['formula_level_extrapolation'] * 3,
        'chemical_plausibility_pass': [True, True, True],
        'ranking_rank': [1, 2, 3],
        'proposal_shortlist_selected': [True, True, False],
        'proposal_shortlist_rank': [1, 2, pd.NA],
        'extrapolation_shortlist_selected': [True, False, True],
        'extrapolation_shortlist_rank': [1, pd.NA, 2],
        'bn_analog_neighbor_formulas': ['BN|B2N', 'BN|BN2', 'B2N|BN'],
        'bn_analog_nearest_formula': ['BN', 'BN', 'B2N'],
        'bn_support_neighbor_formulas': ['BN|B2N', 'BN|BN2', 'B2N|BN'],
    })
    bn_centered_candidate_df = pd.DataFrame({
        'formula': ['AlBN2', 'BCN2', 'BCN'],
        'ranking_rank': [1, 2, 3],
    })

    seed_df = build_candidate_structure_generation_seeds(
        candidate_df,
        dataset_df,
        split_masks,
        CFG,
        bn_centered_candidate_df=bn_centered_candidate_df,
    )

    assert not seed_df.empty
    assert set(seed_df['formula']) == {'BCN2', 'BCN', 'AlBN2'}
    assert seed_df['structure_generation_seed_status'].eq('ok').all()
    assert seed_df['structure_generation_seed_rank'].max() == 2
    assert seed_df['seed_reference_formula'].isin({'BN', 'B2N', 'BN2'}).all()
    assert seed_df['seed_reference_record_id'].notna().all()
    assert seed_df['seed_reference_has_structure_summary'].eq(True).all()
    assert seed_df['seed_formula_edit_strategy'].notna().all()
    assert seed_df['seed_formula_shared_elements'].str.contains('B').all()
    assert seed_df.loc[
        seed_df['formula'].eq('BCN2') & seed_df['seed_reference_formula'].eq('BN'),
        'seed_formula_candidate_only_elements',
    ].iloc[0] == 'C'
    assert seed_df.loc[
        seed_df['formula'].eq('BCN2') & seed_df['seed_reference_formula'].eq('BN'),
        'seed_formula_edit_strategy',
    ].iloc[0] == 'element_insertion_or_decoration'
    assert seed_df.loc[
        seed_df['formula'].eq('AlBN2'),
        'bn_centered_top_n_selected',
    ].iloc[0]
    assert seed_df.loc[
        seed_df['formula'].eq('BCN2'),
        'structure_generation_candidate_priority_reason',
    ].iloc[0] == 'proposal_shortlist|extrapolation_shortlist|bn_centered_top_2'



def test_bn_slice_benchmark_reports_bn_focused_holdout_metrics():
    benchmark_cfg = copy.deepcopy(CFG)
    benchmark_cfg['bn_slice_benchmark']['k_neighbors'] = 2
    dataset_df = pd.DataFrame({
        'formula': [
            'BN', 'BN', 'BCN', 'BCN', 'BC2N', 'BC2N', 'Si2BN', 'Si2BN', 'AlN', 'AlN', 'GaN', 'GaN'
        ],
        'target': [5.8, 5.7, 2.4, 2.5, 3.7, 3.6, 1.2, 1.3, 3.1, 3.0, 2.9, 2.8],
    })
    comp_df = pd.DataFrame({
        'formula': dataset_df['formula'],
        'target': dataset_df['target'],
        'feat_1': [0.10, 0.11, 0.30, 0.31, 0.40, 0.41, 0.60, 0.61, 0.20, 0.21, 0.25, 0.26],
        'feat_2': [0.90, 0.89, 0.70, 0.69, 0.55, 0.56, 0.20, 0.21, 0.80, 0.79, 0.74, 0.73],
        'feature_generation_failed': [False] * len(dataset_df),
        'feature_generation_error': [None] * len(dataset_df),
        'feature_set': ['matminer_composition'] * len(dataset_df),
    })
    structure_df = comp_df.copy()
    structure_df['structure_feat'] = [1.0, 1.02, 0.75, 0.76, 0.65, 0.66, 0.35, 0.34, 0.88, 0.87, 0.82, 0.81]
    structure_df['feature_set'] = STRUCTURE_AWARE_FEATURE_SET
    feature_tables = {
        'matminer_composition': comp_df,
        STRUCTURE_AWARE_FEATURE_SET: structure_df,
    }

    bn_slice_df, bn_slice_prediction_df = benchmark_bn_slice(
        dataset_df,
        feature_tables,
        benchmark_cfg,
        selected_feature_set=STRUCTURE_AWARE_FEATURE_SET,
        selected_model_type='linear_regression',
        screening_feature_set='matminer_composition',
        screening_model_type='linear_regression',
    )

    assert set(bn_slice_df['benchmark_role']) == {
        'selected_model',
        'screening_model',
        'candidate_model',
        'global_dummy_mean_baseline',
        'bn_local_reference_baseline',
    }
    assert bn_slice_df['benchmark_status'].eq('ok').all()
    assert bn_slice_df['bn_formula_count'].eq(4).all()
    assert bn_slice_df['bn_row_count'].eq(8).all()
    assert bn_slice_df['completed_holds'].eq(4).all()
    assert bn_slice_df['bn_slice_method'].eq('leave_one_bn_formula_out').all()
    assert bn_slice_df.loc[
        bn_slice_df['benchmark_role'].eq('bn_local_reference_baseline'),
        'k_neighbors',
    ].iloc[0] == 2
    assert bn_slice_df['mae'].notna().all()
    assert bn_slice_df['rmse'].notna().all()
    assert not bn_slice_prediction_df.empty
    assert set(bn_slice_prediction_df['benchmark_role']) == {
        'selected_model',
        'screening_model',
        'candidate_model',
        'global_dummy_mean_baseline',
        'bn_local_reference_baseline',
    }



def test_bn_family_benchmark_reports_family_holdout_metrics():
    benchmark_cfg = copy.deepcopy(CFG)
    benchmark_cfg['bn_family_benchmark']['k_neighbors'] = 2
    dataset_df = pd.DataFrame({
        'formula': [
            'BN', 'BN', 'BCN', 'BCN', 'BC2N', 'BC2N', 'Si2BN', 'Si2BN', 'AlN', 'AlN', 'GaN', 'GaN'
        ],
        'target': [5.8, 5.7, 2.4, 2.5, 3.7, 3.6, 1.2, 1.3, 3.1, 3.0, 2.9, 2.8],
    })
    comp_df = pd.DataFrame({
        'formula': dataset_df['formula'],
        'target': dataset_df['target'],
        'feat_1': [0.10, 0.11, 0.30, 0.31, 0.40, 0.41, 0.60, 0.61, 0.20, 0.21, 0.25, 0.26],
        'feat_2': [0.90, 0.89, 0.70, 0.69, 0.55, 0.56, 0.20, 0.21, 0.80, 0.79, 0.74, 0.73],
        'feature_generation_failed': [False] * len(dataset_df),
        'feature_generation_error': [None] * len(dataset_df),
        'feature_set': ['matminer_composition'] * len(dataset_df),
    })
    structure_df = comp_df.copy()
    structure_df['structure_feat'] = [1.0, 1.02, 0.75, 0.76, 0.65, 0.66, 0.35, 0.34, 0.88, 0.87, 0.82, 0.81]
    structure_df['feature_set'] = STRUCTURE_AWARE_FEATURE_SET
    feature_tables = {
        'matminer_composition': comp_df,
        STRUCTURE_AWARE_FEATURE_SET: structure_df,
    }

    bn_family_df, bn_family_prediction_df = benchmark_bn_family_holdout(
        dataset_df,
        feature_tables,
        benchmark_cfg,
        selected_feature_set=STRUCTURE_AWARE_FEATURE_SET,
        selected_model_type='linear_regression',
        screening_feature_set='matminer_composition',
        screening_model_type='linear_regression',
    )

    assert set(bn_family_df['benchmark_role']) == {
        'selected_model',
        'screening_model',
        'candidate_model',
        'global_dummy_mean_baseline',
        'bn_local_reference_baseline',
    }
    assert bn_family_df['benchmark_status'].eq('ok').all()
    assert bn_family_df['bn_formula_count'].eq(4).all()
    assert bn_family_df['bn_row_count'].eq(8).all()
    assert bn_family_df['bn_family_count'].eq(3).all()
    assert bn_family_df['completed_family_holds'].eq(3).all()
    assert bn_family_df['bn_family_benchmark_method'].eq('leave_one_bn_family_out').all()
    assert bn_family_df['bn_family_grouping_method'].eq('reduced_bn_chemical_system').all()
    assert bn_family_df.loc[
        bn_family_df['benchmark_role'].eq('bn_local_reference_baseline'),
        'k_neighbors',
    ].iloc[0] == 2
    assert bn_family_df['mae'].notna().all()
    assert bn_family_df['rmse'].notna().all()
    assert not bn_family_prediction_df.empty
    assert set(bn_family_prediction_df['benchmark_role']) == {
        'selected_model',
        'screening_model',
        'candidate_model',
        'global_dummy_mean_baseline',
        'bn_local_reference_baseline',
    }
    assert set(bn_family_prediction_df['bn_family']) == {'B-N', 'B-C-N', 'B-N-Si'}



def test_bn_stratified_errors_report_bn_vs_non_bn_metrics():
    benchmark_cfg = copy.deepcopy(CFG)
    benchmark_cfg['bn_stratified_error']['n_splits'] = 3
    dataset_df = pd.DataFrame({
        'formula': [
            'BN', 'BN', 'BCN', 'BCN', 'BC2N', 'BC2N', 'Si2BN', 'Si2BN', 'AlN', 'AlN', 'GaN', 'GaN'
        ],
        'target': [5.8, 5.7, 2.4, 2.5, 3.7, 3.6, 1.2, 1.3, 3.1, 3.0, 2.9, 2.8],
    })
    comp_df = pd.DataFrame({
        'formula': dataset_df['formula'],
        'target': dataset_df['target'],
        'feat_1': [0.10, 0.11, 0.30, 0.31, 0.40, 0.41, 0.60, 0.61, 0.20, 0.21, 0.25, 0.26],
        'feat_2': [0.90, 0.89, 0.70, 0.69, 0.55, 0.56, 0.20, 0.21, 0.80, 0.79, 0.74, 0.73],
        'feature_generation_failed': [False] * len(dataset_df),
        'feature_generation_error': [None] * len(dataset_df),
        'feature_set': ['matminer_composition'] * len(dataset_df),
    })
    structure_df = comp_df.copy()
    structure_df['structure_feat'] = [1.0, 1.02, 0.75, 0.76, 0.65, 0.66, 0.35, 0.34, 0.88, 0.87, 0.82, 0.81]
    structure_df['feature_set'] = STRUCTURE_AWARE_FEATURE_SET
    feature_tables = {
        'matminer_composition': comp_df,
        STRUCTURE_AWARE_FEATURE_SET: structure_df,
    }

    bn_stratified_df = benchmark_bn_stratified_errors(
        feature_tables,
        benchmark_cfg,
        selected_feature_set=STRUCTURE_AWARE_FEATURE_SET,
        selected_model_type='linear_regression',
        screening_feature_set='matminer_composition',
        screening_model_type='linear_regression',
    )

    assert set(bn_stratified_df['benchmark_role']) == {
        'selected_model',
        'screening_model',
        'candidate_model',
        'dummy_baseline',
    }
    assert bn_stratified_df['benchmark_status'].eq('ok').all()
    assert bn_stratified_df['bn_formula_count'].eq(4).all()
    assert bn_stratified_df['non_bn_formula_count'].eq(2).all()
    assert bn_stratified_df['requested_folds'].eq(3).all()
    assert bn_stratified_df['completed_folds'].eq(3).all()
    assert bn_stratified_df['bn_stratified_error_method'].eq(
        'group_kfold_bn_vs_non_bn_formula_stratified_error'
    ).all()
    assert bn_stratified_df['bn_mae'].notna().all()
    assert bn_stratified_df['non_bn_mae'].notna().all()
    assert bn_stratified_df['bn_to_non_bn_mae_ratio'].notna().all()



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
    candidate_grouped_robustness_df = build_candidate_grouped_robustness_predictions(
        candidates,
        selected_feature_df,
        split_masks,
        CFG,
        feature_set=selection_summary['selected_feature_set'],
        model_type=selection_summary['selected_model_type'],
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
        grouped_robustness_prediction_df=candidate_grouped_robustness_df,
    )

    assert len(candidates) == 25
    assert {'BN', 'BC2N', 'Si2BN', 'Tl2BN'}.issubset(set(candidates['formula']))
    assert candidates['candidate_space_kind'].eq('bn_family_demo').all()
    assert candidates['candidate_generation_strategy'].eq('bn_anchored_formula_family_grid').all()
    assert candidates['candidate_family'].nunique() >= 4
    assert candidates['chemical_plausibility_pass'].sum() == 23
    assert set(candidates.loc[~candidates['chemical_plausibility_pass'], 'formula']) == {
        'AlBN',
        'TlBN',
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
        'composition_only_mean_band_gap_minus_model_disagreement_low_support_and_bn_support_and_grouped_robustness_penalties'
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
    assert screened_df['grouped_robustness_prediction_enabled'].eq(True).all()
    assert screened_df['grouped_robustness_prediction_method'].eq(
        'selected_formula_only_group_kfold_candidate_prediction_std'
    ).all()
    assert screened_df['grouped_robustness_prediction_fold_count'].eq(5).all()
    assert screened_df['grouped_robustness_predicted_band_gap_std'].ge(0.0).all()
    assert (screened_df['grouped_robustness_uncertainty_penalty'] > 0.0).any()
    assert (
        screened_df['ranking_score_before_grouped_robustness_penalty']
        >= screened_df['ranking_score_before_domain_support_penalty']
    ).all()
    assert (
        screened_df['ranking_score_before_domain_support_penalty']
        >= screened_df['ranking_score_before_bn_support_penalty']
    ).all()
    assert (
        screened_df['ranking_score_before_bn_support_penalty']
        >= screened_df['ranking_score_before_bn_band_gap_alignment_penalty']
    ).all()
    assert (
        screened_df['ranking_score_before_bn_band_gap_alignment_penalty']
        >= screened_df['ranking_score_before_bn_analog_validation_penalty']
    ).all()
    assert (
        screened_df['ranking_score_before_bn_analog_validation_penalty']
        >= screened_df['ranking_score']
    ).all()
    assert screened_df['domain_support_penalty'].ge(0.0).all()
    assert (screened_df['domain_support_penalty'] > 0.0).any()
    assert screened_df['bn_support_enabled'].eq(True).all()
    assert screened_df['bn_support_method'].eq(
        'train_plus_val_bn_knn_feature_space_support'
    ).all()
    assert screened_df['bn_support_distance_metric'].eq('z_scored_euclidean_rms').all()
    assert screened_df['bn_support_reference_split'].eq('train_plus_val_bn_unique_formulas').all()
    expected_bn_reference_formula_count = int(
        dataset_df.loc[
            (np.asarray(split_masks['train']) | np.asarray(split_masks['val']))
            & dataset_df['formula'].astype(str).eq('BN'),
            'formula',
        ]
        .astype(str)
        .nunique()
    )
    assert screened_df['bn_support_reference_formula_count'].eq(
        expected_bn_reference_formula_count
    ).all()
    assert screened_df['bn_support_k_neighbors'].eq(3).all()
    assert screened_df['bn_support_penalty'].ge(0.0).all()
    assert (screened_df['bn_support_penalty'] > 0.0).any()
    assert screened_df['bn_support_neighbor_formulas'].astype(str).str.len().gt(0).all()
    assert screened_df['bn_support_neighbor_formula_count'].ge(1).all()
    assert screened_df['bn_analog_evidence_enabled'].eq(True).all()
    assert screened_df['bn_analog_evidence_aggregation'].eq(
        'mean_over_k_nearest_bn_formulas'
    ).all()
    assert screened_df['bn_band_gap_alignment_enabled'].eq(True).all()
    assert screened_df['bn_band_gap_alignment_method'].eq(
        'predicted_band_gap_vs_local_bn_analog_window'
    ).all()
    assert screened_df['bn_band_gap_alignment_reference_split'].eq(
        'train_plus_val_bn_unique_formulas'
    ).all()
    assert screened_df['bn_analog_validation_enabled'].eq(True).all()
    assert screened_df['bn_analog_validation_method'].eq(
        'bn_analog_alignment_vote_fraction'
    ).all()
    assert screened_df['bn_band_gap_alignment_neighbor_available_formula_count'].ge(1).all()
    assert screened_df['bn_band_gap_alignment_window_lower'].notna().all()
    assert screened_df['bn_band_gap_alignment_window_upper'].notna().all()
    assert screened_df['bn_band_gap_alignment_distance_to_window'].ge(0.0).all()
    assert screened_df['bn_band_gap_alignment_penalty'].eq(0.0).all()
    assert screened_df['bn_band_gap_alignment_penalty_eligible'].eq(False).all()
    assert screened_df['bn_analog_validation_penalty'].eq(0.0).all()
    assert screened_df['bn_analog_reference_formula_count'].eq(
        expected_bn_reference_formula_count
    ).all()
    assert np.allclose(
        screened_df['bn_analog_reference_band_gap_median'].to_numpy(dtype=float),
        5.05,
    )
    assert np.allclose(
        screened_df['bn_analog_reference_band_gap_iqr'].to_numpy(dtype=float),
        0.0,
    )
    assert np.allclose(
        screened_df['bn_analog_reference_exfoliation_energy_median'].to_numpy(dtype=float),
        0.0615,
    )
    assert np.allclose(
        screened_df['bn_analog_reference_energy_per_atom_median'].to_numpy(dtype=float),
        -8.305,
    )
    assert np.allclose(
        screened_df['bn_analog_reference_abs_total_magnetization_median'].to_numpy(dtype=float),
        0.0,
    )
    assert screened_df['ranking_rank'].tolist() == list(range(1, len(screened_df) + 1))
    assert screened_df['chemical_plausibility_pass'].head(CFG['screening']['top_k']).all()
    assert screened_df.loc[
        ~screened_df['chemical_plausibility_pass'],
        'screening_selection_decision',
    ].eq('failed_chemical_plausibility').all()
    assert screened_df['proposal_shortlist_enabled'].eq(True).all()
    assert screened_df['proposal_shortlist_label'].eq('family_aware_proposal_shortlist').all()
    assert screened_df['proposal_shortlist_method'].eq('ranked_family_cap').all()
    assert screened_df['proposal_shortlist_note'].eq('demo proposal shortlist note').all()
    assert screened_df['proposal_shortlist_size'].eq(4).all()
    assert screened_df['proposal_shortlist_family_cap'].eq(1).all()
    assert screened_df['proposal_shortlist_chemical_plausibility_priority'].eq(True).all()
    assert int(screened_df['proposal_shortlist_selected'].sum()) == 4
    proposal_shortlist_df = screened_df.loc[
        screened_df['proposal_shortlist_selected'].fillna(False).astype(bool)
    ]
    assert proposal_shortlist_df['proposal_shortlist_rank'].tolist() == [1, 2, 3, 4]
    assert proposal_shortlist_df['candidate_family'].nunique() == len(proposal_shortlist_df)
    assert proposal_shortlist_df['chemical_plausibility_pass'].all()
    assert proposal_shortlist_df['candidate_family'].value_counts().max() == 1
    assert proposal_shortlist_df['ranking_rank'].is_monotonic_increasing
    assert screened_df['proposal_shortlist_family_count_before_selection'].ge(0).all()
    assert screened_df.loc[
        screened_df['proposal_shortlist_selected'],
        'proposal_shortlist_decision',
    ].eq('selected_for_proposal_shortlist').all()
    assert (
        screened_df['proposal_shortlist_decision'].eq('not_selected_family_cap_reached').any()
        or screened_df['proposal_shortlist_decision'].eq('not_selected_shortlist_full').any()
    )
    assert screened_df.loc[
        ~screened_df['chemical_plausibility_pass'],
        'proposal_shortlist_decision',
    ].eq('not_selected_failed_chemical_plausibility').all()
    assert screened_df['extrapolation_shortlist_enabled'].eq(True).all()
    assert screened_df['extrapolation_shortlist_label'].eq(
        'formula_level_extrapolation_shortlist'
    ).all()
    assert screened_df['extrapolation_shortlist_method'].eq(
        'novelty_bucket_ranked_family_cap'
    ).all()
    assert screened_df['extrapolation_shortlist_note'].eq(
        'demo extrapolation shortlist note'
    ).all()
    assert screened_df['extrapolation_shortlist_size'].eq(3).all()
    assert screened_df['extrapolation_shortlist_family_cap'].eq(1).all()
    assert screened_df['extrapolation_shortlist_chemical_plausibility_priority'].eq(True).all()
    assert screened_df['extrapolation_shortlist_target_novelty_bucket'].eq(
        'formula_level_extrapolation'
    ).all()
    extrapolation_shortlist_df = screened_df.loc[
        screened_df['extrapolation_shortlist_selected'].fillna(False).astype(bool)
    ]
    assert int(extrapolation_shortlist_df.shape[0]) == 3
    assert extrapolation_shortlist_df['extrapolation_shortlist_rank'].tolist() == [1, 2, 3]
    assert extrapolation_shortlist_df['chemical_plausibility_pass'].all()
    assert extrapolation_shortlist_df['candidate_family'].nunique() == len(
        extrapolation_shortlist_df
    )
    assert extrapolation_shortlist_df['candidate_family'].value_counts().max() == 1
    assert extrapolation_shortlist_df['ranking_rank'].is_monotonic_increasing
    assert extrapolation_shortlist_df['candidate_novelty_bucket'].eq(
        'formula_level_extrapolation'
    ).all()
    assert screened_df['extrapolation_shortlist_family_count_before_selection'].ge(0).all()
    assert screened_df.loc[
        screened_df['extrapolation_shortlist_selected'],
        'extrapolation_shortlist_decision',
    ].eq('selected_for_extrapolation_shortlist').all()
    assert screened_df.loc[
        screened_df['candidate_novelty_bucket'] != 'formula_level_extrapolation',
        'extrapolation_shortlist_decision',
    ].eq('not_selected_novelty_bucket_mismatch').all()
    assert screened_df.loc[
        ~screened_df['chemical_plausibility_pass'],
        'extrapolation_shortlist_decision',
    ].isin(
        [
            'not_selected_failed_chemical_plausibility',
            'not_selected_novelty_bucket_mismatch',
        ]
    ).all()
    assert (
        screened_df['extrapolation_shortlist_decision'].eq('not_selected_family_cap_reached').any()
        or screened_df['extrapolation_shortlist_decision'].eq('not_selected_shortlist_full').any()
    )
    bn_row = screened_df.loc[screened_df['formula'] == 'BN'].iloc[0]
    ge2bn_row = screened_df.loc[screened_df['formula'] == 'Ge2BN'].iloc[0]
    assert bn_row['domain_support_nearest_formula'] == 'BN'
    assert bn_row['domain_support_nearest_distance'] == pytest.approx(0.0)
    assert bn_row['domain_support_percentile'] == pytest.approx(100.0)
    assert bn_row['domain_support_penalty'] == pytest.approx(0.0)
    assert bn_row['bn_support_nearest_formula'] == 'BN'
    assert bn_row['bn_support_neighbor_formulas'] == 'BN'
    assert bn_row['bn_support_neighbor_formula_count'] == 1
    assert bn_row['bn_support_nearest_distance'] == pytest.approx(0.0)
    assert bn_row['bn_support_percentile'] == pytest.approx(100.0)
    assert bn_row['bn_support_penalty'] == pytest.approx(0.0)
    assert bn_row['bn_analog_nearest_formula'] == 'BN'
    assert bn_row['bn_analog_neighbor_formulas'] == 'BN'
    assert bn_row['bn_analog_neighbor_formula_count'] == 1
    assert bn_row['bn_analog_reference_band_gap_median'] == pytest.approx(5.05)
    assert bn_row['bn_analog_reference_band_gap_iqr'] == pytest.approx(0.0)
    assert bn_row['bn_analog_nearest_band_gap'] == pytest.approx(5.05)
    assert bn_row['bn_analog_nearest_energy_per_atom'] == pytest.approx(-8.305)
    assert bn_row['bn_analog_nearest_exfoliation_energy_per_atom'] == pytest.approx(0.0615)
    assert bn_row['bn_analog_nearest_abs_total_magnetization'] == pytest.approx(0.0)
    assert bn_row['bn_analog_neighbor_band_gap_mean'] == pytest.approx(5.05)
    assert bn_row['bn_analog_neighbor_band_gap_min'] == pytest.approx(5.05)
    assert bn_row['bn_analog_neighbor_band_gap_max'] == pytest.approx(5.05)
    assert bn_row['bn_analog_neighbor_band_gap_std'] == pytest.approx(0.0)
    assert bn_row['bn_analog_neighbor_energy_per_atom_mean'] == pytest.approx(-8.305)
    assert bn_row['bn_analog_neighbor_exfoliation_energy_per_atom_mean'] == pytest.approx(0.0615)
    assert bn_row['bn_analog_neighbor_abs_total_magnetization_mean'] == pytest.approx(0.0)
    assert bn_row['bn_analog_neighbor_exfoliation_available_formula_count'] == 1
    assert bn_row['bn_band_gap_alignment_neighbor_available_formula_count'] == 1
    assert bn_row['bn_band_gap_alignment_window_lower'] == pytest.approx(5.05)
    assert bn_row['bn_band_gap_alignment_window_upper'] == pytest.approx(5.05)
    assert bool(bn_row['bn_band_gap_alignment_penalty_eligible']) is False
    assert bn_row['bn_band_gap_alignment_penalty'] == pytest.approx(0.0)
    assert bn_row['bn_band_gap_alignment_label'] in {
        'within_local_bn_analog_band_gap_window',
        'below_local_bn_analog_band_gap_window',
        'above_local_bn_analog_band_gap_window',
    }
    assert bn_row['bn_analog_exfoliation_support_label'] == 'lower_or_equal_bn_reference_median'
    assert bn_row['bn_analog_energy_support_label'] == 'lower_or_equal_bn_reference_median'
    assert bn_row['bn_analog_abs_total_magnetization_support_label'] == 'lower_or_equal_bn_reference_median'
    assert bn_row['bn_analog_support_vote_count'] == 3
    assert bn_row['bn_analog_support_available_metric_count'] == 3
    assert bn_row['bn_analog_validation_label'] == 'reference_like_on_available_metrics'
    assert bn_row['bn_analog_validation_support_fraction'] == pytest.approx(1.0)
    assert bn_row['bn_analog_validation_penalty'] == pytest.approx(0.0)
    assert ge2bn_row['domain_support_nearest_distance'] > 0.0
    assert ge2bn_row['bn_support_nearest_distance'] > 0.0
    assert bn_row['domain_support_percentile'] >= ge2bn_row['domain_support_percentile']
    assert bn_row['bn_support_percentile'] >= ge2bn_row['bn_support_percentile']
    assert ge2bn_row['bn_analog_validation_label'] in {
        'reference_like_on_available_metrics',
        'mixed_reference_alignment',
        'reference_divergent_on_available_metrics',
    }
    assert ge2bn_row['bn_band_gap_alignment_label'] in {
        'within_local_bn_analog_band_gap_window',
        'below_local_bn_analog_band_gap_window',
        'above_local_bn_analog_band_gap_window',
    }
    assert 0.0 <= ge2bn_row['bn_analog_validation_support_fraction'] <= 1.0
    assert ge2bn_row['bn_band_gap_alignment_distance_to_window'] >= 0.0
    assert ge2bn_row['bn_analog_validation_penalty'] >= 0.0
    assert 'grouped-fold candidate robustness penalty' in screened_df['ranking_note'].iloc[0]
    assert 'train+val feature-space domain-support layer' in screened_df['ranking_note'].iloc[0]
    assert 'known BN slice' in screened_df['ranking_note'].iloc[0]
    assert 'BN-local analog band-gap window' in screened_df['ranking_note'].iloc[0]
    assert 'observed-property evidence from nearby BN-containing train+val formulas' in screened_df['ranking_note'].iloc[0]
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
    assert {'train_plus_val_rediscovery', 'formula_level_extrapolation'}.issubset(
        set(screened_df['candidate_novelty_bucket'])
    )
    assert set(screened_df['candidate_novelty_priority']).issubset({1, 2, 3})
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
    held_out_known_notes = screened_df.loc[
        screened_df['candidate_novelty_bucket'] == 'held_out_known_formula',
        'candidate_novelty_note',
    ]
    if not held_out_known_notes.empty:
        assert 'held-out-known formula' in held_out_known_notes.iloc[0]
    assert 'formula-level extrapolation' in (
        screened_df.loc[
            screened_df['candidate_novelty_bucket'] == 'formula_level_extrapolation',
            'candidate_novelty_note',
        ]
        .iloc[0]
    )


def test_screen_candidates_can_apply_bn_local_band_gap_alignment_penalty():
    alignment_cfg = copy.deepcopy(CFG)
    alignment_cfg['screening']['top_k'] = 3
    alignment_cfg['screening']['use_model_disagreement'] = False
    alignment_cfg['screening']['domain_support']['ranking_penalty_enabled'] = False
    alignment_cfg['screening']['bn_support']['ranking_penalty_enabled'] = False
    alignment_cfg['screening']['grouped_robustness_uncertainty']['enabled'] = False
    alignment_cfg['screening']['bn_analog_validation']['ranking_penalty_enabled'] = False

    dataset_df = _bn_alignment_reference_df()
    feature_tables = build_feature_tables(dataset_df, alignment_cfg)
    feature_df = feature_tables['matminer_composition']
    split_masks = {
        'train': np.asarray([True, True, True, False, False], dtype=bool),
        'val': np.asarray([False, False, False, True, False], dtype=bool),
        'test': np.asarray([False, False, False, False, True], dtype=bool),
        'metadata': {'method': 'group_by_formula'},
    }
    _, feature_columns = train_baseline_model(
        feature_df,
        split_masks,
        alignment_cfg,
        model_type='linear_regression',
        include_validation=True,
    )

    class FixedPredictor:
        def __init__(self, predictions):
            self.predictions = np.asarray(predictions, dtype=float)

        def predict(self, x):
            assert len(x) == len(self.predictions)
            return self.predictions.copy()

    candidate_df = generate_bn_candidates(alignment_cfg)
    candidate_df = candidate_df.loc[
        candidate_df['formula'].isin(['BC2N', 'Si2BN', 'Ge2BN'])
    ].reset_index(drop=True)
    predicted_band_gaps = [7.0, 3.0, 0.2]
    candidate_ensemble_df = pd.DataFrame({
        'formula': candidate_df['formula'].astype(str),
        'ensemble_predicted_band_gap_mean': predicted_band_gaps,
        'ensemble_predicted_band_gap_std': [0.0, 0.0, 0.0],
        'ensemble_member_count': [1, 1, 1],
    })

    screened_df = screen_candidates(
        candidate_df,
        FixedPredictor(predicted_band_gaps),
        feature_columns,
        alignment_cfg,
        feature_set='matminer_composition',
        model_type='linear_regression',
        dataset_df=dataset_df,
        split_masks=split_masks,
        ensemble_prediction_df=candidate_ensemble_df,
        grouped_robustness_prediction_df=None,
        reference_feature_df=feature_df,
    )

    assert screened_df['ranking_basis'].eq(
        'composition_only_selected_model_band_gap_minus_bn_band_gap_alignment_penalty'
    ).all()
    assert screened_df['bn_band_gap_alignment_penalty_eligible'].eq(True).all()
    assert screened_df['bn_analog_reference_formula_count'].eq(4).all()
    assert np.allclose(
        screened_df['bn_analog_reference_band_gap_median'].to_numpy(dtype=float),
        3.15,
    )
    assert np.allclose(
        screened_df['bn_analog_reference_band_gap_iqr'].to_numpy(dtype=float),
        1.425,
    )

    bc2n_row = screened_df.loc[screened_df['formula'] == 'BC2N'].iloc[0]
    si2bn_row = screened_df.loc[screened_df['formula'] == 'Si2BN'].iloc[0]
    ge2bn_row = screened_df.loc[screened_df['formula'] == 'Ge2BN'].iloc[0]

    assert bc2n_row['bn_band_gap_alignment_label'] == 'above_local_bn_analog_band_gap_window'
    assert bc2n_row['bn_band_gap_alignment_distance_to_window'] > 0.0
    assert bc2n_row['bn_band_gap_alignment_relative_distance'] > 0.0
    assert bc2n_row['bn_band_gap_alignment_penalty'] > 0.0
    assert bc2n_row['ranking_score_before_bn_band_gap_alignment_penalty'] == pytest.approx(7.0)
    assert bc2n_row['ranking_score'] == pytest.approx(
        7.0 - bc2n_row['bn_band_gap_alignment_penalty']
    )

    assert si2bn_row['bn_band_gap_alignment_label'] == 'within_local_bn_analog_band_gap_window'
    assert si2bn_row['bn_band_gap_alignment_distance_to_window'] == pytest.approx(0.0)
    assert si2bn_row['bn_band_gap_alignment_penalty'] == pytest.approx(0.0)

    assert ge2bn_row['bn_band_gap_alignment_label'] == 'below_local_bn_analog_band_gap_window'
    assert ge2bn_row['bn_band_gap_alignment_distance_to_window'] > 0.0
    assert ge2bn_row['bn_band_gap_alignment_penalty'] > 0.0
    assert ge2bn_row['bn_band_gap_alignment_penalty'] <= 0.08 + 1e-12


def test_fractional_composition_feature_table_and_torch_models_fit_predict():
    pytest.importorskip('torch')

    cfg = copy.deepcopy(CFG)
    cfg['features']['candidate_sets'] = [FRACTIONAL_COMPOSITION_FEATURE_SET]
    cfg['model']['torch_mlp'] = {
        'hidden_dim': 32,
        'depth': 2,
        'dropout': 0.0,
        'learning_rate': 0.01,
        'weight_decay': 0.0,
        'max_epochs': 12,
        'patience': 3,
        'min_delta': 0.0,
        'val_fraction': 0.2,
        'device': 'cpu',
        'random_seed': 42,
    }
    cfg['model']['torch_mlp_ensemble'] = {
        'hidden_dim': 32,
        'depth': 2,
        'dropout': 0.0,
        'learning_rate': 0.01,
        'weight_decay': 0.0,
        'max_epochs': 10,
        'patience': 2,
        'min_delta': 0.0,
        'val_fraction': 0.2,
        'device': 'cpu',
        'random_seed': 42,
        'member_seeds': [42, 43],
    }

    dataset_df = _sample_training_df()
    split_masks = make_split_masks(dataset_df, cfg)
    feature_df = build_feature_table(dataset_df, feature_set=FRACTIONAL_COMPOSITION_FEATURE_SET)
    summary = summarize_feature_table(feature_df, feature_set=FRACTIONAL_COMPOSITION_FEATURE_SET)

    assert summary['candidate_compatible'] is True
    assert summary['status'] == 'ok'
    assert summary['n_features'] >= 100
    assert feature_df['frac_b'].sum() > 0.0
    assert feature_df['frac_n'].sum() > 0.0

    for model_type in ['torch_mlp', 'torch_mlp_ensemble']:
        model, feature_columns = train_baseline_model(
            feature_df,
            split_masks,
            cfg,
            model_type=model_type,
            include_validation=False,
        )
        metrics, prediction_df = evaluate_predictions(feature_df, split_masks, model, feature_columns)

        assert prediction_df['prediction'].notna().all()
        assert np.isfinite(prediction_df['prediction']).all()
        assert metrics['mae'] >= 0.0

    ensemble_model = make_model(cfg, model_type='torch_mlp_ensemble')
    feature_columns = [column for column in feature_df.columns if column.startswith('frac_')]
    ensemble_model.fit(feature_df[feature_columns], feature_df['target'])
    member_predictions = ensemble_model.predict_members(feature_df[feature_columns])
    assert member_predictions.shape[0] == 2
    assert member_predictions.shape[1] == len(feature_df)


def test_torch_mlp_ensemble_expands_candidate_uncertainty_sources():
    pytest.importorskip('torch')

    cfg = copy.deepcopy(CFG)
    cfg['features']['candidate_sets'] = [FRACTIONAL_COMPOSITION_FEATURE_SET]
    cfg['model']['type'] = 'torch_mlp_ensemble'
    cfg['model']['candidate_types'] = ['torch_mlp_ensemble']
    cfg['model']['torch_mlp_ensemble'] = {
        'hidden_dim': 24,
        'depth': 2,
        'dropout': 0.0,
        'learning_rate': 0.01,
        'weight_decay': 0.0,
        'max_epochs': 8,
        'patience': 2,
        'min_delta': 0.0,
        'val_fraction': 0.2,
        'device': 'cpu',
        'random_seed': 42,
        'member_seeds': [42, 43],
    }

    dataset_df = _sample_training_df()
    split_masks = make_split_masks(dataset_df, cfg)
    feature_tables = {
        FRACTIONAL_COMPOSITION_FEATURE_SET: build_feature_table(
            dataset_df,
            feature_set=FRACTIONAL_COMPOSITION_FEATURE_SET,
        )
    }
    candidate_df = pd.DataFrame({'formula': ['BN', 'BCN', 'B2N2']})

    prediction_df = build_candidate_prediction_members(
        candidate_df,
        feature_tables,
        split_masks,
        cfg,
        candidate_feature_sets=[FRACTIONAL_COMPOSITION_FEATURE_SET],
    )

    assert prediction_df['prediction_source'].nunique() == 2
    assert set(prediction_df['prediction_source_family']) == {'full_fit_candidate_model_member'}
    assert prediction_df.groupby('formula').size().eq(2).all()


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
