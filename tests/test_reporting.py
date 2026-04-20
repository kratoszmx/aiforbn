from __future__ import annotations

import json

import pandas as pd

from pipeline.reporting import build_experiment_summary, save_basic_plots, save_metrics_and_predictions
from pipeline.structure_execution import build_structure_first_pass_execution_artifacts


def test_reporting_writes_expected_artifacts(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    (raw_dir / 'twod_matpd.json').write_text(
        json.dumps(
            [
                {
                    'jid': 'jid-1',
                    'formula': 'BN',
                    'band_gap': 5.8,
                    'atoms': {
                        'elements': ['B', 'N'],
                        'coords': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
                        'lattice_mat': [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 20.0]],
                        'abc': [2.5, 2.5, 20.0],
                        'angles': [90.0, 90.0, 120.0],
                        'cartesian': False,
                    },
                },
                {
                    'jid': 'jid-2',
                    'formula': 'B2N',
                    'band_gap': 4.2,
                    'atoms': {
                        'elements': ['B', 'B', 'N'],
                        'coords': [[0.0, 0.0, 0.0], [0.33, 0.33, 0.0], [0.66, 0.66, 0.0]],
                        'lattice_mat': [[2.8, 0.0, 0.0], [0.0, 2.8, 0.0], [0.0, 0.0, 20.0]],
                        'abc': [2.8, 2.8, 20.0],
                        'angles': [90.0, 90.0, 120.0],
                        'cartesian': False,
                    },
                },
            ]
        ),
        encoding='utf-8',
    )
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'formula_column': 'formula',
            'target_column': 'band_gap',
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
        },
        'robustness': {
            'enabled': True,
            'method': 'group_kfold_by_formula',
            'group_column': 'formula',
            'n_splits': 4,
            'note': 'demo grouped robustness note',
        },
        'bn_slice_benchmark': {
            'enabled': True,
            'method': 'leave_one_bn_formula_out',
            'k_neighbors': 2,
            'note': 'demo bn slice benchmark note',
        },
        'screening': {
            'candidate_generation_strategy': 'bn_anchored_formula_family_grid',
            'candidate_space_name': 'bn_anchored_formula_family_grid',
            'candidate_space_kind': 'bn_family_demo',
            'candidate_space_note': 'bn-anchored demo note',
            'top_k': 5,
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
                'shortlist_size': 2,
                'max_per_candidate_family': 1,
                'chemical_plausibility_priority': True,
                'note': 'demo proposal shortlist note',
            },
            'extrapolation_shortlist': {
                'enabled': True,
                'label': 'formula_level_extrapolation_shortlist',
                'method': 'novelty_bucket_ranked_family_cap',
                'shortlist_size': 1,
                'max_per_candidate_family': 1,
                'required_novelty_bucket': 'formula_level_extrapolation',
                'chemical_plausibility_priority': True,
                'note': 'demo extrapolation shortlist note',
            },
            'structure_first_pass_execution': {
                'enabled': True,
                'max_candidates': 2,
                'max_variants_per_candidate': 2,
            },
        },
    }

    metrics = {
        'mae': 1.0,
        'rmse': 2.0,
        'r2': 0.5,
        'selected_model_type': 'linear_regression',
        'selected_feature_set': 'matminer_composition',
        'selected_feature_family': 'composition_only',
        'screening_feature_set': 'matminer_composition',
        'screening_model_type': 'linear_regression',
        'screening_feature_family': 'composition_only',
    }
    prediction_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0], 'prediction': [4.8]})
    bn_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0]})
    candidate_df = pd.DataFrame({
        'formula': ['BN', 'AlBN'],
        'candidate_space_name': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
        'candidate_space_kind': ['bn_family_demo', 'bn_family_demo'],
        'candidate_generation_strategy': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
        'candidate_family': ['bn_binary_anchor', 'group13_bn_111_family'],
        'candidate_template': ['B1N1', 'X1B1N1'],
        'candidate_family_note': ['BN anchor', 'Group-III BN ternary extension'],
        'ranking_rank': [1, 2],
        'ranking_score': [4.8, 1.2],
        'grouped_robustness_prediction_enabled': [True, True],
        'grouped_robustness_prediction_method': [
            'selected_formula_only_group_kfold_candidate_prediction_std',
            'selected_formula_only_group_kfold_candidate_prediction_std',
        ],
        'grouped_robustness_prediction_note': [
            'demo grouped candidate robustness note',
            'demo grouped candidate robustness note',
        ],
        'grouped_robustness_prediction_feature_set': ['matminer_composition', 'matminer_composition'],
        'grouped_robustness_prediction_model_type': ['linear_regression', 'linear_regression'],
        'grouped_robustness_prediction_fold_count': [4, 4],
        'grouped_robustness_predicted_band_gap_mean': [4.82, 1.24],
        'grouped_robustness_predicted_band_gap_std': [0.02, 0.30],
        'grouped_robustness_uncertainty_penalty': [0.003, 0.045],
        'ranking_score_before_grouped_robustness_penalty': [4.803, 1.245],
        'domain_support_reference_formula_count': [12, 12],
        'domain_support_k_neighbors': [5, 5],
        'domain_support_nearest_formula': ['BN', 'BN'],
        'domain_support_nearest_distance': [0.0, 0.8],
        'domain_support_mean_k_distance': [0.0, 1.1],
        'domain_support_percentile': [100.0, 10.0],
        'domain_support_penalty': [0.0, 0.09],
        'bn_support_reference_formula_count': [4, 4],
        'bn_support_k_neighbors': [3, 3],
        'bn_support_nearest_formula': ['BN', 'BN'],
        'bn_support_neighbor_formulas': ['BN', 'BN|Si2BN'],
        'bn_support_neighbor_formula_count': [1, 2],
        'bn_support_nearest_distance': [0.0, 0.4],
        'bn_support_mean_k_distance': [0.0, 0.6],
        'bn_support_percentile': [100.0, 0.0],
        'bn_support_penalty': [0.0, 0.1],
        'bn_analog_evidence_enabled': [True, True],
        'bn_analog_evidence_aggregation': ['mean_over_k_nearest_bn_formulas', 'mean_over_k_nearest_bn_formulas'],
        'bn_analog_reference_formula_count': [4, 4],
        'bn_analog_reference_band_gap_median': [3.6, 3.6],
        'bn_analog_reference_band_gap_iqr': [1.2, 1.2],
        'bn_analog_reference_exfoliation_energy_median': [0.07, 0.07],
        'bn_analog_reference_energy_per_atom_median': [-8.0, -8.0],
        'bn_analog_reference_abs_total_magnetization_median': [0.0, 0.0],
        'bn_analog_nearest_formula': ['BN', 'BN'],
        'bn_analog_neighbor_formulas': ['BN', 'BN|Si2BN'],
        'bn_analog_neighbor_formula_count': [1, 2],
        'bn_analog_nearest_band_gap': [4.8, 4.8],
        'bn_analog_nearest_energy_per_atom': [-8.3, -8.3],
        'bn_analog_nearest_exfoliation_energy_per_atom': [0.06, 0.06],
        'bn_analog_nearest_abs_total_magnetization': [0.0, 0.0],
        'bn_analog_neighbor_band_gap_mean': [4.8, 2.4],
        'bn_analog_neighbor_band_gap_min': [4.8, 0.0],
        'bn_analog_neighbor_band_gap_max': [4.8, 4.8],
        'bn_analog_neighbor_band_gap_std': [0.0, 2.4],
        'bn_analog_neighbor_energy_per_atom_mean': [-8.3, -7.3],
        'bn_analog_neighbor_exfoliation_energy_per_atom_mean': [0.06, 0.06],
        'bn_analog_neighbor_abs_total_magnetization_mean': [0.0, 0.0],
        'bn_analog_neighbor_exfoliation_available_formula_count': [1, 1],
        'bn_band_gap_alignment_enabled': [True, True],
        'bn_band_gap_alignment_method': [
            'predicted_band_gap_vs_local_bn_analog_window',
            'predicted_band_gap_vs_local_bn_analog_window',
        ],
        'bn_band_gap_alignment_reference_split': [
            'train_plus_val_bn_unique_formulas',
            'train_plus_val_bn_unique_formulas',
        ],
        'bn_band_gap_alignment_note': [
            'demo bn-local band-gap alignment note',
            'demo bn-local band-gap alignment note',
        ],
        'bn_band_gap_alignment_neighbor_available_formula_count': [1, 2],
        'bn_band_gap_alignment_window_lower': [4.2, -0.6],
        'bn_band_gap_alignment_window_upper': [5.4, 5.4],
        'bn_band_gap_alignment_distance_to_window': [0.0, 0.6],
        'bn_band_gap_alignment_relative_distance': [0.0, 0.5],
        'bn_band_gap_alignment_penalty_eligible': [False, True],
        'bn_band_gap_alignment_label': [
            'within_local_bn_analog_band_gap_window',
            'above_local_bn_analog_band_gap_window',
        ],
        'bn_band_gap_alignment_penalty': [0.0, 0.04],
        'bn_analog_exfoliation_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
        'bn_analog_energy_support_label': ['lower_or_equal_bn_reference_median', 'higher_than_bn_reference_median'],
        'bn_analog_abs_total_magnetization_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
        'bn_analog_support_vote_count': [3, 2],
        'bn_analog_support_available_metric_count': [3, 3],
        'bn_analog_validation_label': ['reference_like_on_available_metrics', 'mixed_reference_alignment'],
        'bn_analog_validation_support_fraction': [1.0, 2.0 / 3.0],
        'bn_analog_validation_penalty': [0.0, 0.04],
        'chemical_plausibility_pass': [True, False],
        'chemical_plausibility_guess_count': [1, 0],
        'chemical_plausibility_primary_oxidation_state_guess': ['B(+3), N(-3)', ''],
        'chemical_plausibility_note': ['pass', 'fail'],
        'seen_in_dataset': [True, False],
        'dataset_formula_row_count': [3, 0],
        'seen_in_train_plus_val': [True, False],
        'train_plus_val_formula_row_count': [2, 0],
        'candidate_is_seen_in_dataset': [True, False],
        'candidate_is_seen_in_train_plus_val': [True, False],
        'candidate_is_formula_level_extrapolation': [False, True],
        'candidate_novelty_bucket': ['train_plus_val_rediscovery', 'formula_level_extrapolation'],
        'candidate_novelty_priority': [1, 3],
        'candidate_novelty_note': ['rediscovery note', 'novel note'],
        'novelty_rank_within_bucket': [1, 1],
        'novel_formula_rank': [pd.NA, 1],
        'screening_selected_for_top_k': [True, False],
        'screening_selection_decision': ['selected_top_k', 'failed_chemical_plausibility'],
        'proposal_shortlist_enabled': [True, True],
        'proposal_shortlist_label': ['family_aware_proposal_shortlist', 'family_aware_proposal_shortlist'],
        'proposal_shortlist_method': ['ranked_family_cap', 'ranked_family_cap'],
        'proposal_shortlist_note': ['demo proposal shortlist note', 'demo proposal shortlist note'],
        'proposal_shortlist_size': [2, 2],
        'proposal_shortlist_family_cap': [1, 1],
        'proposal_shortlist_chemical_plausibility_priority': [True, True],
        'proposal_shortlist_family_count_before_selection': [0, 0],
        'proposal_shortlist_selected': [True, False],
        'proposal_shortlist_rank': [1, pd.NA],
        'proposal_shortlist_decision': [
            'selected_for_proposal_shortlist',
            'not_selected_failed_chemical_plausibility',
        ],
        'extrapolation_shortlist_enabled': [True, True],
        'extrapolation_shortlist_label': [
            'formula_level_extrapolation_shortlist',
            'formula_level_extrapolation_shortlist',
        ],
        'extrapolation_shortlist_method': [
            'novelty_bucket_ranked_family_cap',
            'novelty_bucket_ranked_family_cap',
        ],
        'extrapolation_shortlist_note': [
            'demo extrapolation shortlist note',
            'demo extrapolation shortlist note',
        ],
        'extrapolation_shortlist_size': [1, 1],
        'extrapolation_shortlist_family_cap': [1, 1],
        'extrapolation_shortlist_chemical_plausibility_priority': [True, True],
        'extrapolation_shortlist_target_novelty_bucket': [
            'formula_level_extrapolation',
            'formula_level_extrapolation',
        ],
        'extrapolation_shortlist_family_count_before_selection': [0, 0],
        'extrapolation_shortlist_selected': [False, False],
        'extrapolation_shortlist_rank': [pd.NA, pd.NA],
        'extrapolation_shortlist_decision': [
            'not_selected_novelty_bucket_mismatch',
            'not_selected_failed_chemical_plausibility',
        ],
    })
    screened_df = pd.DataFrame({
        'formula': ['BN', 'AlBN'],
        'predicted_band_gap': [4.8, 1.2],
        'screening_selected_for_top_k': [True, False],
        'screening_selection_decision': ['selected_top_k', 'failed_chemical_plausibility'],
        'proposal_shortlist_selected': [True, False],
        'proposal_shortlist_rank': [1, pd.NA],
        'proposal_shortlist_decision': [
            'selected_for_proposal_shortlist',
            'not_selected_failed_chemical_plausibility',
        ],
        'extrapolation_shortlist_selected': [False, False],
        'extrapolation_shortlist_rank': [pd.NA, pd.NA],
        'extrapolation_shortlist_decision': [
            'not_selected_novelty_bucket_mismatch',
            'not_selected_failed_chemical_plausibility',
        ],
    })
    benchmark_df = pd.DataFrame({
        'feature_set': ['matminer_composition', 'feature_agnostic_dummy'],
        'model_type': ['linear_regression', 'dummy_mean'],
        'mae': [1.0, 1.4],
    })
    robustness_df = pd.DataFrame({
        'feature_set': ['matminer_composition', 'basic_formula_composition', 'feature_agnostic_dummy'],
        'feature_family': ['composition_only', 'composition_only', 'baseline'],
        'candidate_compatible': [True, True, False],
        'n_features': [138, 5, 138],
        'model_type': ['linear_regression', 'hist_gradient_boosting', 'dummy_mean'],
        'benchmark_role': ['selected_model', 'candidate_model', 'dummy_baseline'],
        'selected_by_validation': [True, False, False],
        'robustness_method': ['group_kfold_by_formula'] * 3,
        'robustness_group_column': ['formula'] * 3,
        'requested_folds': [4, 4, 4],
        'actual_folds': [4, 4, 4],
        'completed_folds': [4, 4, 4],
        'robustness_status': ['ok', 'ok', 'ok'],
        'robustness_note': ['demo grouped robustness note', 'demo grouped robustness note', 'feature-agnostic dummy baseline'],
        'mae_mean': [1.1, 1.3, 1.6],
        'mae_std': [0.1, 0.2, 0.3],
        'rmse_mean': [1.4, 1.7, 2.0],
        'rmse_std': [0.1, 0.2, 0.3],
        'r2_mean': [0.6, 0.4, 0.1],
        'r2_std': [0.05, 0.08, 0.1],
    })
    bn_slice_benchmark_df = pd.DataFrame({
        'feature_set': [
            'matminer_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
            'matminer_composition',
            'matminer_composition',
        ],
        'feature_family': [
            'composition_only',
            'composition_only',
            'structure_aware',
            'composition_only',
            'composition_only',
        ],
        'model_type': [
            'linear_regression',
            'linear_regression',
            'hist_gradient_boosting',
            'dummy_mean',
            'bn_local_knn_mean',
        ],
        'benchmark_role': [
            'selected_model',
            'screening_model',
            'candidate_model',
            'global_dummy_mean_baseline',
            'bn_local_reference_baseline',
        ],
        'benchmark_status': ['ok', 'ok', 'ok', 'ok', 'ok'],
        'bn_slice_method': ['leave_one_bn_formula_out'] * 5,
        'bn_slice_train_scope': [
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'bn_only_reference_formulas',
        ],
        'bn_formula_count': [1, 1, 1, 1, 1],
        'bn_row_count': [1, 1, 1, 1, 1],
        'completed_holds': [3, 3, 3, 3, 3],
        'k_neighbors': [pd.NA, pd.NA, pd.NA, pd.NA, 2],
        'mae': [0.6, 0.6, 0.5, 0.9, 0.8],
        'rmse': [0.7, 0.7, 0.6, 1.0, 0.9],
        'r2': [0.5, 0.5, 0.6, 0.1, 0.2],
    })
    bn_slice_prediction_df = pd.DataFrame({
        'formula': ['BN', 'BN', 'BN', 'BN', 'BN'],
        'benchmark_role': [
            'selected_model',
            'screening_model',
            'candidate_model',
            'global_dummy_mean_baseline',
            'bn_local_reference_baseline',
        ],
        'feature_set': [
            'matminer_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
            'matminer_composition',
            'matminer_composition',
        ],
        'feature_family': [
            'composition_only',
            'composition_only',
            'structure_aware',
            'composition_only',
            'composition_only',
        ],
        'model_type': [
            'linear_regression',
            'linear_regression',
            'hist_gradient_boosting',
            'dummy_mean',
            'bn_local_knn_mean',
        ],
        'selected_by_validation': [True, False, False, False, False],
        'bn_slice_method': ['leave_one_bn_formula_out'] * 5,
        'bn_slice_train_scope': [
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'bn_only_reference_formulas',
        ],
        'target': [5.0, 5.0, 5.0, 5.0, 5.0],
        'prediction': [4.4, 4.3, 4.5, 4.1, 3.9],
        'absolute_error': [0.6, 0.7, 0.5, 0.9, 1.1],
    })
    bn_centered_candidate_df = pd.DataFrame({
        'formula': ['AlBN', 'BN'],
        'ranking_rank': [1, 2],
        'ranking_score': [1.5, 1.4],
        'ranking_basis': ['composition_only_selected_model_low_support_and_bn_support_and_grouped_robustness_and_bn_band_gap_alignment_and_bn_analog_validation_penalties'] * 2,
        'ranking_note': ['bn-centered alternative note'] * 2,
    })
    bn_centered_screening_selection = {
        'enabled': True,
        'selection_source_artifact': 'bn_slice_benchmark_results.csv',
        'selection_scope': 'bn_slice_candidate_compatible_best',
        'selection_note': 'bn-centered alternative ranking note',
        'ranking_artifact': 'demo_candidate_bn_centered_ranking.csv',
        'feature_set': 'matminer_composition',
        'feature_family': 'composition_only',
        'model_type': 'linear_regression',
        'benchmark_role': 'selected_model',
        'mae': 0.6,
        'rmse': 0.7,
        'r2': 0.5,
        'matches_general_screening_combo': True,
    }
    structure_generation_seed_df = pd.DataFrame({
        'formula': ['BN', 'AlBN'],
        'ranking_rank': [1, 2],
        'ranking_score': [4.8, 1.2],
        'bn_centered_ranking_rank': [2, 1],
        'candidate_family': ['bn_binary_anchor', 'group13_bn_111_family'],
        'candidate_novelty_bucket': ['train_plus_val_rediscovery', 'formula_level_extrapolation'],
        'chemical_plausibility_pass': [True, False],
        'proposal_shortlist_selected': [True, False],
        'proposal_shortlist_rank': [1, None],
        'extrapolation_shortlist_selected': [False, True],
        'extrapolation_shortlist_rank': [None, 1],
        'structure_generation_candidate_priority_reason': ['proposal_shortlist', 'extrapolation_shortlist'],
        'structure_generation_seed_rank': [1, 1],
        'structure_generation_seed_status': ['ok', 'ok'],
        'seed_reference_formula': ['BN', 'B2N'],
        'seed_reference_record_id': ['jid-1', 'jid-2'],
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
        'selected_feature_family': 'composition_only',
        'used_validation_selection': True,
        'candidate_feature_sets': [
            'basic_formula_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
        ],
        'candidate_model_types': ['linear_regression', 'hist_gradient_boosting'],
        'screening_selection_scope': 'candidate_compatible_formula_only',
        'screening_candidate_feature_sets': [
            'basic_formula_composition',
            'matminer_composition',
        ],
        'screening_selected_feature_set': 'matminer_composition',
        'screening_selected_feature_family': 'composition_only',
        'screening_selected_model_type': 'linear_regression',
        'screening_selected_feature_count': 138,
        'screening_selection_matches_overall': True,
        'screening_selection_note': 'Best overall validation combo is candidate-compatible, so screening reuses it.',
        'feature_set_results': [
            {'feature_set': 'basic_formula_composition', 'status': 'ok', 'candidate_compatible': True},
            {'feature_set': 'matminer_composition', 'status': 'ok', 'candidate_compatible': True},
            {
                'feature_set': 'matminer_composition_plus_structure_summary',
                'status': 'ok',
                'candidate_compatible': False,
            },
        ],
    }
    (
        structure_first_pass_variant_df,
        structure_first_pass_summary_df,
        structure_first_pass_payload,
    ) = build_structure_first_pass_execution_artifacts(
        structure_generation_seed_df,
        cfg=cfg,
        formula_col='formula',
    )
    experiment_summary = build_experiment_summary(
        dataset_df=prediction_df,
        bn_df=bn_df,
        candidate_df=candidate_df,
        split_masks=split_masks,
        selection_summary=selection_summary,
        cfg=cfg,
        robustness_df=robustness_df,
        bn_slice_benchmark_df=bn_slice_benchmark_df,
        bn_centered_candidate_df=bn_centered_candidate_df,
        bn_centered_screening_selection=bn_centered_screening_selection,
        structure_generation_seed_df=structure_generation_seed_df,
        structure_first_pass_execution_summary_df=structure_first_pass_summary_df,
        structure_first_pass_execution_payload=structure_first_pass_payload,
    )
    manifest = {'name': 'twod_matpd'}

    save_metrics_and_predictions(
        metrics,
        prediction_df,
        bn_df,
        screened_df,
        benchmark_df,
        robustness_df,
        bn_slice_benchmark_df,
        bn_slice_prediction_df,
        bn_centered_candidate_df,
        structure_generation_seed_df,
        experiment_summary,
        manifest,
        cfg,
        structure_first_pass_execution_variant_df=structure_first_pass_variant_df,
        structure_first_pass_execution_summary_df=structure_first_pass_summary_df,
        structure_first_pass_execution_payload=structure_first_pass_payload,
    )
    save_basic_plots(prediction_df, cfg)

    assert json.loads((artifact_dir / 'metrics.json').read_text()) == metrics
    assert json.loads((artifact_dir / 'manifest.json').read_text()) == manifest
    assert json.loads((artifact_dir / 'experiment_summary.json').read_text()) == experiment_summary
    assert experiment_summary['features']['selected_feature_set'] == 'matminer_composition'
    assert experiment_summary['features']['selected_feature_family'] == 'composition_only'
    assert experiment_summary['feature_model_selection']['selected_model_type'] == 'linear_regression'
    assert experiment_summary['robustness']['enabled'] is True
    assert experiment_summary['robustness']['robustness_artifact'] == 'robustness_results.csv'
    assert experiment_summary['robustness']['method'] == 'group_kfold_by_formula'
    assert experiment_summary['robustness']['group_column'] == 'formula'
    assert experiment_summary['robustness']['requested_folds'] == 4
    assert experiment_summary['robustness']['result_row_count'] == 3
    assert experiment_summary['robustness']['successful_result_rows'] == 3
    assert experiment_summary['robustness']['failed_result_rows'] == 0
    assert experiment_summary['robustness']['selected_model_metrics']['mae_mean'] == 1.1
    assert experiment_summary['robustness']['screening_model_metrics']['model_type'] == 'linear_regression'
    assert experiment_summary['robustness']['dummy_baseline_metrics']['model_type'] == 'dummy_mean'
    assert experiment_summary['bn_slice_benchmark']['enabled'] is True
    assert experiment_summary['bn_slice_benchmark']['benchmark_artifact'] == 'bn_slice_benchmark_results.csv'
    assert experiment_summary['bn_slice_benchmark']['prediction_artifact'] == 'bn_slice_predictions.csv'
    assert experiment_summary['bn_slice_benchmark']['method'] == 'leave_one_bn_formula_out'
    assert experiment_summary['bn_slice_benchmark']['k_neighbors'] == 2
    assert experiment_summary['bn_slice_benchmark']['standard_split_bn_train_rows'] == 1
    assert experiment_summary['bn_slice_benchmark']['standard_split_bn_val_rows'] == 0
    assert experiment_summary['bn_slice_benchmark']['standard_split_bn_test_rows'] == 0
    assert experiment_summary['bn_slice_benchmark']['standard_split_has_bn_eval_rows'] is False
    assert experiment_summary['bn_slice_benchmark']['selected_model_metrics']['mae'] == 0.6
    assert experiment_summary['bn_slice_benchmark']['screening_model_metrics']['benchmark_role'] == 'screening_model'
    assert experiment_summary['bn_slice_benchmark']['bn_local_reference_metrics']['model_type'] == 'bn_local_knn_mean'
    assert experiment_summary['bn_slice_benchmark']['global_dummy_baseline_metrics']['model_type'] == 'dummy_mean'
    assert experiment_summary['bn_slice_benchmark']['best_candidate_model_metrics']['benchmark_role'] == 'candidate_model'
    assert experiment_summary['bn_slice_benchmark']['best_candidate_model_metrics']['feature_set'] == (
        'matminer_composition_plus_structure_summary'
    )
    assert experiment_summary['bn_slice_benchmark']['selected_model_beats_global_dummy'] is True
    assert experiment_summary['bn_slice_benchmark']['screening_model_beats_global_dummy'] is True
    assert experiment_summary['bn_slice_benchmark']['best_candidate_model_beats_global_dummy'] is True
    assert experiment_summary['bn_slice_benchmark']['selected_model_matches_best_candidate'] is False
    assert experiment_summary['screening']['ranking_basis'] == (
        'composition_only_mean_band_gap_minus_model_disagreement_low_support_and_bn_support_and_grouped_robustness_and_bn_band_gap_alignment_and_bn_analog_validation_penalties'
    )
    assert experiment_summary['screening']['ranking_feature_family'] == 'composition_only'
    assert experiment_summary['screening']['bn_centered_alternative']['enabled'] is True
    assert (
        experiment_summary['screening']['bn_centered_alternative']['ranking_artifact']
        == 'demo_candidate_bn_centered_ranking.csv'
    )
    assert experiment_summary['screening']['bn_centered_alternative']['ranking_feature_set'] == 'matminer_composition'
    assert experiment_summary['screening']['bn_centered_alternative']['ranking_model_type'] == 'linear_regression'
    assert experiment_summary['screening']['bn_centered_alternative']['bn_slice_mae'] == 0.6
    assert experiment_summary['screening']['bn_centered_alternative']['top_k_overlap_count'] == 2
    assert experiment_summary['screening']['bn_centered_alternative']['top_k_overlap_formulas'] == ['BN', 'AlBN']
    assert experiment_summary['screening']['bn_centered_alternative']['general_top_k_formulas'] == ['BN', 'AlBN']
    assert experiment_summary['screening']['bn_centered_alternative']['bn_centered_top_k_formulas'] == ['AlBN', 'BN']
    assert experiment_summary['screening']['bn_centered_alternative']['mean_absolute_rank_shift'] == 1.0
    assert experiment_summary['screening']['bn_centered_alternative']['max_absolute_rank_shift'] == 1.0
    assert experiment_summary['screening']['bn_centered_alternative']['max_absolute_rank_shift_formula'] == 'BN'
    assert experiment_summary['screening']['structure_generation_bridge']['enabled'] is True
    assert (
        experiment_summary['screening']['structure_generation_bridge']['artifact']
        == 'demo_candidate_structure_generation_seeds.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['handoff_artifact']
        == 'demo_candidate_structure_generation_handoff.json'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['reference_record_payload_artifact']
        == 'demo_candidate_structure_generation_reference_records.json'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['job_plan_artifact']
        == 'demo_candidate_structure_generation_job_plan.json'
    )
    assert experiment_summary['screening']['structure_generation_bridge']['candidate_rows'] == 2
    assert experiment_summary['screening']['structure_generation_bridge']['seed_rows'] == 2
    assert experiment_summary['screening']['structure_generation_bridge']['job_count'] == 2
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_queue_artifact']
        == 'demo_candidate_structure_generation_first_pass_queue.json'
    )
    assert experiment_summary['screening']['structure_generation_bridge']['first_pass_queue_size'] == 2
    assert (
        experiment_summary['screening']['structure_generation_bridge']['direct_substitution_job_count']
        == 0
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['simple_relabeling_job_count']
        == 0
    )
    assert experiment_summary['screening']['structure_generation_bridge']['mean_edit_complexity_score'] == 1.5
    assert experiment_summary['screening']['structure_generation_bridge']['max_edit_complexity_score'] == 2.5
    assert experiment_summary['screening']['structure_generation_bridge']['job_action_counts'] == {
        'reference_reuse_control': 1,
        'element_insertion_enumeration': 1,
    }
    assert (
        experiment_summary['screening']['structure_generation_bridge']['followup_shortlist_artifact']
        == 'demo_candidate_structure_generation_followup_shortlist.csv'
    )
    assert experiment_summary['screening']['structure_generation_bridge']['followup_shortlist_size'] == 2
    assert experiment_summary['screening']['structure_generation_bridge']['followup_shortlist_formulas'] == ['BN', 'AlBN']
    assert experiment_summary['screening']['structure_generation_bridge']['followup_readiness_counts'] == {
        'reference_reuse_control_available': 1,
        'moderate_formula_edit_required': 1,
    }
    assert (
        experiment_summary['screening']['structure_generation_bridge'][
            'followup_extrapolation_shortlist_artifact'
        ]
        == 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['followup_extrapolation_shortlist_size']
        == 1
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['followup_extrapolation_shortlist_formulas']
        == ['AlBN']
    )
    assert experiment_summary['screening']['structure_generation_bridge']['unique_seed_reference_formulas'] == 2
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_artifact']
        == 'demo_candidate_structure_generation_first_pass_execution.json'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_summary_artifact']
        == 'demo_candidate_structure_generation_first_pass_execution_summary.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_variants_artifact']
        == 'demo_candidate_structure_generation_first_pass_execution_variants.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_structure_dir']
        == 'demo_candidate_structure_generation_first_pass_structures'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_method']
        == 'deterministic_unrelaxed_reference_reuse_species_edit'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_candidate_count']
        == 2
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_variant_count']
        == 3
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_successful_variant_count']
        == 3
    )
    assert experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_status_counts'] == {
        'executed': 2,
    }
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_executed_formulas']
        == ['BN', 'AlBN']
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_model_feature_set']
        is None
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_model_type']
        is None
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_model_available']
        is False
    )
    assert experiment_summary['screening']['ranking_matches_best_overall_evaluation'] is True
    assert experiment_summary['screening']['best_overall_evaluation_feature_set'] == 'matminer_composition'
    assert experiment_summary['screening']['ranking_uncertainty_method'] == 'small_feature_model_disagreement'
    assert experiment_summary['screening']['domain_support_enabled'] is True
    assert experiment_summary['screening']['domain_support_method'] == 'train_plus_val_knn_feature_space_support'
    assert experiment_summary['screening']['domain_support_reference_formula_count'] == 12
    assert experiment_summary['screening']['domain_support_penalty_enabled'] is True
    assert experiment_summary['screening']['domain_support_penalized_rows'] == 1
    assert experiment_summary['screening']['domain_support_low_support_rows'] == 1
    assert experiment_summary['screening']['bn_support_enabled'] is True
    assert experiment_summary['screening']['bn_support_method'] == 'train_plus_val_bn_knn_feature_space_support'
    assert experiment_summary['screening']['bn_support_reference_formula_count'] == 4
    assert experiment_summary['screening']['bn_support_penalty_enabled'] is True
    assert experiment_summary['screening']['bn_support_penalized_rows'] == 1
    assert experiment_summary['screening']['bn_support_low_support_rows'] == 1
    assert experiment_summary['screening']['grouped_robustness_uncertainty_enabled'] is True
    assert (
        experiment_summary['screening']['grouped_robustness_uncertainty_method']
        == 'selected_formula_only_group_kfold_candidate_prediction_std'
    )
    assert experiment_summary['screening']['grouped_robustness_penalty_enabled'] is True
    assert experiment_summary['screening']['grouped_robustness_penalty_active'] is True
    assert experiment_summary['screening']['grouped_robustness_penalty_weight'] == 0.15
    assert experiment_summary['screening']['grouped_robustness_prediction_fold_count'] == 4
    assert experiment_summary['screening']['grouped_robustness_prediction_std_mean'] == 0.16
    assert experiment_summary['screening']['grouped_robustness_penalized_rows'] == 2
    assert experiment_summary['screening']['bn_analog_evidence_enabled'] is True
    assert experiment_summary['screening']['bn_analog_reference_formula_count'] == 4
    assert experiment_summary['screening']['bn_analog_reference_band_gap_median'] == 3.6
    assert experiment_summary['screening']['bn_analog_reference_band_gap_iqr'] == 1.2
    assert experiment_summary['screening']['bn_analog_reference_exfoliation_energy_median'] == 0.07
    assert experiment_summary['screening']['bn_analog_reference_energy_per_atom_median'] == -8.0
    assert experiment_summary['screening']['bn_analog_reference_abs_total_magnetization_median'] == 0.0
    assert experiment_summary['screening']['bn_analog_exfoliation_available_rows'] == 2
    assert experiment_summary['screening']['bn_analog_lower_or_equal_reference_rows'] == 2
    assert experiment_summary['screening']['bn_analog_higher_reference_rows'] == 0
    assert experiment_summary['screening']['bn_band_gap_alignment_enabled'] is True
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_method']
        == 'predicted_band_gap_vs_local_bn_analog_window'
    )
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_reference_split']
        == 'train_plus_val_bn_unique_formulas'
    )
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_window_expansion_iqr_factor']
        == 0.5
    )
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_minimum_neighbor_formula_count_for_penalty']
        == 2
    )
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_enabled'] is True
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_active'] is True
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_weight'] == 0.08
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_eligible_rows'] == 1
    assert experiment_summary['screening']['bn_band_gap_alignment_within_window_rows'] == 1
    assert experiment_summary['screening']['bn_band_gap_alignment_below_window_rows'] == 0
    assert experiment_summary['screening']['bn_band_gap_alignment_above_window_rows'] == 1
    assert experiment_summary['screening']['bn_band_gap_alignment_penalized_rows'] == 1
    assert experiment_summary['screening']['bn_analog_reference_like_rows'] == 1
    assert experiment_summary['screening']['bn_analog_mixed_alignment_rows'] == 1
    assert experiment_summary['screening']['bn_analog_reference_divergent_rows'] == 0
    assert experiment_summary['screening']['bn_analog_validation_enabled'] is True
    assert experiment_summary['screening']['bn_analog_validation_method'] == 'bn_analog_alignment_vote_fraction'
    assert experiment_summary['screening']['bn_analog_validation_penalty_enabled'] is True
    assert experiment_summary['screening']['bn_analog_validation_penalty_active'] is True
    assert experiment_summary['screening']['bn_analog_validation_penalty_weight'] == 0.12
    assert experiment_summary['screening']['bn_analog_validation_penalized_rows'] == 1
    assert experiment_summary['screening']['chemical_plausibility_enabled'] is True
    assert experiment_summary['screening']['chemical_plausibility_passed_rows'] == 1
    assert experiment_summary['screening']['chemical_plausibility_failed_rows'] == 1
    assert experiment_summary['screening']['candidate_generation_strategy'] == 'bn_anchored_formula_family_grid'
    assert experiment_summary['screening']['candidate_family_counts'] == {
        'bn_binary_anchor': 1,
        'group13_bn_111_family': 1,
    }
    assert experiment_summary['screening']['proposal_shortlist_enabled'] is True
    assert experiment_summary['screening']['proposal_shortlist_artifact'] == (
        'demo_candidate_proposal_shortlist.csv'
    )
    assert experiment_summary['screening']['proposal_shortlist_label'] == (
        'family_aware_proposal_shortlist'
    )
    assert experiment_summary['screening']['proposal_shortlist_method'] == 'ranked_family_cap'
    assert experiment_summary['screening']['proposal_shortlist_note'] == 'demo proposal shortlist note'
    assert experiment_summary['screening']['proposal_shortlist_size'] == 2
    assert experiment_summary['screening']['proposal_shortlist_family_cap'] == 1
    assert experiment_summary['screening']['proposal_shortlist_selected_rows'] == 1
    assert experiment_summary['screening']['proposal_shortlist_selected_family_counts'] == {
        'bn_binary_anchor': 1,
    }
    assert experiment_summary['screening']['proposal_shortlist_novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 1,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 0,
    }
    assert experiment_summary['screening']['proposal_shortlist_formulas'] == [
        {
            'formula': 'BN',
            'proposal_shortlist_rank': 1,
            'ranking_rank': 1,
            'candidate_family': 'bn_binary_anchor',
            'ranking_score': 4.8,
        }
    ]
    assert experiment_summary['screening']['extrapolation_shortlist_enabled'] is True
    assert experiment_summary['screening']['extrapolation_shortlist_artifact'] == (
        'demo_candidate_extrapolation_shortlist.csv'
    )
    assert (
        experiment_summary['screening']['extrapolation_shortlist_label']
        == 'formula_level_extrapolation_shortlist'
    )
    assert (
        experiment_summary['screening']['extrapolation_shortlist_method']
        == 'novelty_bucket_ranked_family_cap'
    )
    assert (
        experiment_summary['screening']['extrapolation_shortlist_note']
        == 'demo extrapolation shortlist note'
    )
    assert experiment_summary['screening']['extrapolation_shortlist_size'] == 1
    assert experiment_summary['screening']['extrapolation_shortlist_family_cap'] == 1
    assert (
        experiment_summary['screening']['extrapolation_shortlist_target_novelty_bucket']
        == 'formula_level_extrapolation'
    )
    assert experiment_summary['screening']['extrapolation_shortlist_candidate_count'] == 1
    assert experiment_summary['screening']['extrapolation_shortlist_selected_rows'] == 0
    assert experiment_summary['screening']['extrapolation_shortlist_selected_family_counts'] == {}
    assert experiment_summary['screening']['extrapolation_shortlist_novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 0,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 0,
    }
    assert experiment_summary['screening']['extrapolation_shortlist_formulas'] == []
    assert experiment_summary['screening']['chemical_plausibility_failed_formulas'] == ['AlBN']
    assert experiment_summary['screening']['novelty_annotation_enabled'] is True
    assert experiment_summary['screening']['novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 1,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 1,
    }
    assert experiment_summary['screening']['standard_top_k_novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 1,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 0,
    }
    assert experiment_summary['screening']['formula_level_extrapolation_candidate_count'] == 1
    assert experiment_summary['screening']['formula_level_extrapolation_shortlist'] == [
        {
            'formula': 'AlBN',
            'ranking_rank': 2,
            'novel_formula_rank': 1,
            'ranking_score': 1.2,
            'chemical_plausibility_pass': False,
            'screening_selected_for_top_k': False,
            'screening_selection_decision': 'failed_chemical_plausibility',
            'extrapolation_shortlist_selected': False,
            'extrapolation_shortlist_decision': 'not_selected_failed_chemical_plausibility',
        }
    ]
    assert 'novelty should be interpreted separately' in (
        experiment_summary['screening']['novelty_interpretation_note']
    )
    assert 'Novelty is tracked only at the formula level' in experiment_summary['screening']['ranking_note']
    assert 'known BN slice' in experiment_summary['screening']['ranking_note']
    assert 'BN-local analog band-gap window' in experiment_summary['screening']['ranking_note']
    assert 'observed-property evidence from nearby BN-containing train+val formulas' in experiment_summary['screening']['ranking_note']
    assert 'BN analog-validation penalty' in experiment_summary['screening']['ranking_note']
    assert experiment_summary['bn_slice_benchmark']['candidate_compatible_evaluation_artifact'] == (
        'bn_candidate_compatible_evaluation.csv'
    )
    assert experiment_summary['bn_slice_benchmark']['candidate_compatible_result_row_count'] == 4
    assert experiment_summary['screening']['ranking_stability']['enabled'] is True
    assert experiment_summary['screening']['ranking_stability']['artifact'] == (
        'demo_candidate_ranking_uncertainty.csv'
    )
    assert experiment_summary['screening']['ranking_stability']['top_k_values'] == [3, 5, 10]
    assert experiment_summary['screening']['decision_policy']['enabled'] is True
    assert experiment_summary['screening']['decision_policy']['artifact'] == (
        'demo_candidate_ranking_uncertainty.csv'
    )
    assert experiment_summary['screening']['decision_policy']['abstained_candidate_count'] >= 0
    assert experiment_summary['screening']['candidate_annotations'] == [
        'candidate_family',
        'candidate_template',
        'candidate_family_note',
        'domain_support_reference_formula_count',
        'domain_support_k_neighbors',
        'domain_support_nearest_formula',
        'domain_support_nearest_distance',
        'domain_support_mean_k_distance',
        'domain_support_percentile',
        'domain_support_penalty',
        'bn_support_reference_formula_count',
        'bn_support_k_neighbors',
        'bn_support_nearest_formula',
        'bn_support_neighbor_formulas',
        'bn_support_neighbor_formula_count',
        'bn_support_nearest_distance',
        'bn_support_mean_k_distance',
        'bn_support_percentile',
        'bn_support_penalty',
        'bn_analog_nearest_formula',
        'bn_analog_neighbor_formulas',
        'bn_analog_neighbor_formula_count',
        'bn_analog_reference_band_gap_median',
        'bn_analog_reference_band_gap_iqr',
        'bn_analog_nearest_band_gap',
        'bn_analog_nearest_energy_per_atom',
        'bn_analog_nearest_exfoliation_energy_per_atom',
        'bn_analog_nearest_abs_total_magnetization',
        'bn_analog_neighbor_band_gap_mean',
        'bn_analog_neighbor_band_gap_min',
        'bn_analog_neighbor_band_gap_max',
        'bn_analog_neighbor_band_gap_std',
        'bn_analog_neighbor_energy_per_atom_mean',
        'bn_analog_neighbor_exfoliation_energy_per_atom_mean',
        'bn_analog_neighbor_abs_total_magnetization_mean',
        'bn_analog_neighbor_exfoliation_available_formula_count',
        'bn_band_gap_alignment_neighbor_available_formula_count',
        'bn_band_gap_alignment_window_lower',
        'bn_band_gap_alignment_window_upper',
        'bn_band_gap_alignment_distance_to_window',
        'bn_band_gap_alignment_relative_distance',
        'bn_band_gap_alignment_penalty_eligible',
        'bn_band_gap_alignment_label',
        'bn_band_gap_alignment_penalty',
        'bn_analog_exfoliation_support_label',
        'bn_analog_energy_support_label',
        'bn_analog_abs_total_magnetization_support_label',
        'bn_analog_support_vote_count',
        'bn_analog_support_available_metric_count',
        'bn_analog_validation_label',
        'bn_analog_validation_support_fraction',
        'bn_analog_validation_penalty',
        'chemical_plausibility_pass',
        'chemical_plausibility_guess_count',
        'chemical_plausibility_primary_oxidation_state_guess',
        'chemical_plausibility_note',
        'seen_in_dataset',
        'dataset_formula_row_count',
        'seen_in_train_plus_val',
        'train_plus_val_formula_row_count',
        'candidate_is_seen_in_dataset',
        'candidate_is_seen_in_train_plus_val',
        'candidate_is_formula_level_extrapolation',
        'candidate_novelty_bucket',
        'candidate_novelty_priority',
        'candidate_novelty_note',
        'novelty_rank_within_bucket',
        'novel_formula_rank',
        'screening_selected_for_top_k',
        'screening_selection_decision',
        'proposal_shortlist_family_count_before_selection',
        'proposal_shortlist_selected',
        'proposal_shortlist_rank',
        'proposal_shortlist_decision',
        'extrapolation_shortlist_target_novelty_bucket',
        'extrapolation_shortlist_family_count_before_selection',
        'extrapolation_shortlist_selected',
        'extrapolation_shortlist_rank',
        'extrapolation_shortlist_decision',
        'ranking_source_count',
        'predicted_band_gap_mean',
        'predicted_band_gap_std',
        'predicted_band_gap_interval_lower',
        'predicted_band_gap_interval_upper',
        'rank_mean',
        'rank_std',
        'rank_min',
        'rank_max',
        'top_3_selection_frequency',
        'top_5_selection_frequency',
        'top_10_selection_frequency',
        'bn_centered_ranking_rank',
        'structure_followup_priority_score',
        'structure_followup_best_queue_rank',
        'structure_followup_best_action_label',
        'structure_followup_readiness_label',
        'structure_followup_shortlist_selected',
        'structure_followup_shortlist_rank',
        'abstain_flag',
        'reason_for_abstention',
        'final_action_label',
    ]
    assert (artifact_dir / 'predictions.csv').exists()
    assert (artifact_dir / 'bn_slice.csv').exists()
    assert (artifact_dir / 'demo_candidate_ranking.csv').exists()
    assert (artifact_dir / 'demo_candidate_bn_centered_ranking.csv').exists()
    assert (artifact_dir / 'bn_candidate_compatible_evaluation.csv').exists()
    assert (artifact_dir / 'demo_candidate_ranking_uncertainty.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_seeds.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_handoff.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_reference_records.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_job_plan.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_summary.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_variants.csv').exists()
    handoff_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_handoff.json').read_text()
    )
    reference_record_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_reference_records.json').read_text()
    )
    job_plan_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_job_plan.json').read_text()
    )
    first_pass_queue_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json').read_text()
    )
    first_pass_execution_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution.json').read_text()
    )
    followup_shortlist_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv'
    )
    followup_extrapolation_shortlist_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'
    )
    first_pass_execution_summary_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_summary.csv'
    )
    first_pass_execution_variant_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_variants.csv'
    )
    assert handoff_payload['candidate_count'] == 2
    assert handoff_payload['seed_row_count'] == 2
    assert handoff_payload['candidates'][0]['formula'] == 'BN'
    assert handoff_payload['candidates'][0]['seeds'][0]['seed_reference_record_id'] == 'jid-1'
    assert handoff_payload['candidates'][0]['seeds'][0]['seed_formula_edit_strategy'] == (
        'same_reduced_formula_reference'
    )
    assert handoff_payload['candidates'][1]['seeds'][0]['seed_formula_candidate_only_elements'] == 'Al'
    assert reference_record_payload['record_count'] == 2
    assert reference_record_payload['reference_records'][0]['record_id'] == 'jid-1'
    assert reference_record_payload['reference_records'][0]['atoms']['elements'] == ['B', 'N']
    assert job_plan_payload['job_count'] == 2
    assert job_plan_payload['direct_substitution_job_count'] == 0
    assert job_plan_payload['simple_relabeling_job_count'] == 0
    assert job_plan_payload['job_action_counts'] == {
        'reference_reuse_control': 1,
        'element_insertion_enumeration': 1,
    }
    assert first_pass_execution_payload['candidate_count'] == 2
    assert first_pass_execution_payload['variant_count'] == 3
    assert first_pass_execution_payload['successful_variant_count'] == 3
    assert first_pass_execution_payload['status_counts'] == {'executed': 2}
    assert first_pass_execution_payload['executed_formulas'] == ['BN', 'AlBN']
    assert first_pass_execution_payload['model_available'] is False
    assert first_pass_execution_summary_df['first_pass_execution_status'].tolist() == [
        'executed',
        'executed',
    ]
    assert first_pass_execution_summary_df['first_pass_execution_selected_final_status'].tolist() == [
        'reference_control_ready',
        'geometry_sanity_failed',
    ]
    assert set(first_pass_execution_variant_df['formula']) == {'BN', 'AlBN'}
    assert first_pass_execution_variant_df['execution_status'].eq('ok').all()
    assert first_pass_execution_variant_df.groupby('formula')['geometry_sanity_pass'].agg(lambda values: list(values)).to_dict() == {
        'AlBN': [False, False],
        'BN': [True],
    }
    structure_dir = artifact_dir / 'demo_candidate_structure_generation_first_pass_structures'
    assert structure_dir.exists()
    assert len(list(structure_dir.glob('*.cif'))) == len(first_pass_execution_variant_df)
    assert job_plan_payload['candidates'][0]['jobs'][0]['job_action_label'] == 'reference_reuse_control'
    assert job_plan_payload['candidates'][0]['jobs'][0]['candidate_formula_element_counts'] == {
        'B': 1,
        'N': 1,
    }
    assert job_plan_payload['candidates'][1]['jobs'][0]['workflow_steps'][0] == 'load_reference_atoms'
    assert job_plan_payload['candidates'][1]['jobs'][0]['simple_element_relabeling_feasible'] is False
    assert job_plan_payload['candidates'][1]['jobs'][0]['element_count_deltas'] == {'Al': 1, 'B': -1}
    assert job_plan_payload['candidates'][1]['jobs'][0]['edit_operations'][0] == {
        'operation': 'increase_element_count',
        'element': 'Al',
        'delta': 1,
    }
    assert job_plan_payload['candidates'][1]['jobs'][0]['reference_record_payload_artifact'] == (
        'demo_candidate_structure_generation_reference_records.json'
    )
    assert first_pass_queue_payload['queue_entry_count'] == 2
    assert first_pass_queue_payload['simple_relabeling_job_count'] == 0
    assert first_pass_queue_payload['queue'][0]['job_id'] == 'bn__seed_1__jid_1'
    assert first_pass_queue_payload['queue'][1]['candidate_first_pass_rank'] == 1
    assert followup_shortlist_df['formula'].tolist() == ['BN', 'AlBN']
    assert followup_shortlist_df['structure_followup_shortlist_rank'].tolist() == [1, 2]
    assert followup_shortlist_df['structure_followup_best_action_label'].tolist() == [
        'reference_reuse_control',
        'element_insertion_enumeration',
    ]
    assert followup_shortlist_df['structure_followup_readiness_label'].tolist() == [
        'reference_reuse_control_available',
        'moderate_formula_edit_required',
    ]
    assert followup_extrapolation_shortlist_df['formula'].tolist() == ['AlBN']
    assert followup_extrapolation_shortlist_df[
        'structure_followup_extrapolation_shortlist_rank'
    ].tolist() == [1]
    assert (artifact_dir / 'demo_candidate_proposal_shortlist.csv').exists()
    assert (artifact_dir / 'demo_candidate_extrapolation_shortlist.csv').exists()
    assert (artifact_dir / 'benchmark_results.csv').exists()
    assert (artifact_dir / 'robustness_results.csv').exists()
    assert (artifact_dir / 'bn_slice_benchmark_results.csv').exists()
    assert (artifact_dir / 'bn_slice_predictions.csv').exists()
    assert (artifact_dir / 'parity_plot.png').exists()


def test_experiment_summary_explains_structure_aware_evaluation_vs_formula_only_screening():
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'formula_column': 'formula',
            'target_column': 'band_gap',
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
        },
        'screening': {
            'candidate_generation_strategy': 'bn_anchored_formula_family_grid',
            'candidate_space_name': 'bn_anchored_formula_family_grid',
            'candidate_space_kind': 'bn_family_demo',
            'candidate_space_note': 'bn-anchored demo note',
            'top_k': 5,
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
                'shortlist_size': 2,
                'max_per_candidate_family': 1,
                'chemical_plausibility_priority': True,
                'note': 'demo proposal shortlist note',
            },
            'extrapolation_shortlist': {
                'enabled': True,
                'label': 'formula_level_extrapolation_shortlist',
                'method': 'novelty_bucket_ranked_family_cap',
                'shortlist_size': 1,
                'max_per_candidate_family': 1,
                'required_novelty_bucket': 'formula_level_extrapolation',
                'chemical_plausibility_priority': True,
                'note': 'demo extrapolation shortlist note',
            },
        },
    }
    selection_summary = {
        'selected_feature_set': 'matminer_composition_plus_structure_summary',
        'selected_feature_family': 'structure_aware',
        'selected_feature_count': 30,
        'selected_model_type': 'hist_gradient_boosting',
        'screening_selection_scope': 'candidate_compatible_formula_only',
        'screening_candidate_feature_sets': ['basic_formula_composition', 'matminer_composition'],
        'screening_selected_feature_set': 'matminer_composition',
        'screening_selected_feature_family': 'composition_only',
        'screening_selected_feature_count': 19,
        'screening_selected_model_type': 'hist_gradient_boosting',
        'screening_selection_matches_overall': False,
        'screening_selection_note': (
            'Best overall validation combo requires structure-derived inputs, so formula-only '
            'candidate screening falls back to the best candidate-compatible validation combo.'
        ),
        'candidate_feature_sets': [
            'basic_formula_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
        ],
        'candidate_model_types': ['linear_regression', 'hist_gradient_boosting'],
        'feature_set_results': [],
    }

    summary = build_experiment_summary(
        dataset_df=pd.DataFrame({'formula': ['BN'], 'target': [5.0]}),
        bn_df=pd.DataFrame({'formula': ['BN'], 'target': [5.0]}),
        candidate_df=pd.DataFrame({
            'formula': ['BN', 'AlBN'],
            'candidate_space_name': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
            'candidate_space_kind': ['bn_family_demo', 'bn_family_demo'],
            'candidate_generation_strategy': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
            'candidate_family': ['bn_binary_anchor', 'group13_bn_111_family'],
            'candidate_template': ['B1N1', 'X1B1N1'],
            'candidate_family_note': ['BN anchor', 'Group-III BN ternary extension'],
            'grouped_robustness_prediction_enabled': [True, True],
            'grouped_robustness_prediction_method': [
                'selected_formula_only_group_kfold_candidate_prediction_std',
                'selected_formula_only_group_kfold_candidate_prediction_std',
            ],
            'grouped_robustness_prediction_note': [
                'demo grouped candidate robustness note',
                'demo grouped candidate robustness note',
            ],
            'grouped_robustness_prediction_feature_set': ['matminer_composition', 'matminer_composition'],
            'grouped_robustness_prediction_model_type': ['hist_gradient_boosting', 'hist_gradient_boosting'],
            'grouped_robustness_prediction_fold_count': [4, 4],
            'grouped_robustness_predicted_band_gap_mean': [4.82, 1.24],
            'grouped_robustness_predicted_band_gap_std': [0.02, 0.30],
            'grouped_robustness_uncertainty_penalty': [0.003, 0.045],
            'domain_support_reference_formula_count': [12, 12],
            'domain_support_k_neighbors': [5, 5],
            'domain_support_nearest_formula': ['BN', 'BN'],
            'domain_support_nearest_distance': [0.0, 0.8],
            'domain_support_mean_k_distance': [0.0, 1.1],
            'domain_support_percentile': [100.0, 10.0],
            'domain_support_penalty': [0.0, 0.09],
            'bn_support_reference_formula_count': [4, 4],
            'bn_support_k_neighbors': [3, 3],
            'bn_support_nearest_formula': ['BN', 'BN'],
            'bn_support_neighbor_formulas': ['BN', 'BN|Si2BN'],
            'bn_support_neighbor_formula_count': [1, 2],
            'bn_support_nearest_distance': [0.0, 0.4],
            'bn_support_mean_k_distance': [0.0, 0.6],
            'bn_support_percentile': [100.0, 0.0],
            'bn_support_penalty': [0.0, 0.1],
            'bn_analog_evidence_enabled': [True, True],
            'bn_analog_evidence_aggregation': ['mean_over_k_nearest_bn_formulas', 'mean_over_k_nearest_bn_formulas'],
            'bn_analog_reference_formula_count': [4, 4],
            'bn_analog_reference_band_gap_median': [3.6, 3.6],
            'bn_analog_reference_band_gap_iqr': [1.2, 1.2],
            'bn_analog_reference_exfoliation_energy_median': [0.07, 0.07],
            'bn_analog_reference_energy_per_atom_median': [-8.0, -8.0],
            'bn_analog_reference_abs_total_magnetization_median': [0.0, 0.0],
            'bn_analog_nearest_formula': ['BN', 'BN'],
            'bn_analog_neighbor_formulas': ['BN', 'BN|Si2BN'],
            'bn_analog_neighbor_formula_count': [1, 2],
            'bn_analog_nearest_band_gap': [4.8, 4.8],
            'bn_analog_nearest_energy_per_atom': [-8.3, -8.3],
            'bn_analog_nearest_exfoliation_energy_per_atom': [0.06, 0.06],
            'bn_analog_nearest_abs_total_magnetization': [0.0, 0.0],
            'bn_analog_neighbor_band_gap_mean': [4.8, 2.4],
            'bn_analog_neighbor_band_gap_min': [4.8, 0.0],
            'bn_analog_neighbor_band_gap_max': [4.8, 4.8],
            'bn_analog_neighbor_band_gap_std': [0.0, 2.4],
            'bn_analog_neighbor_energy_per_atom_mean': [-8.3, -7.3],
            'bn_analog_neighbor_exfoliation_energy_per_atom_mean': [0.06, 0.06],
            'bn_analog_neighbor_abs_total_magnetization_mean': [0.0, 0.0],
            'bn_analog_neighbor_exfoliation_available_formula_count': [1, 1],
            'bn_band_gap_alignment_enabled': [True, True],
            'bn_band_gap_alignment_method': [
                'predicted_band_gap_vs_local_bn_analog_window',
                'predicted_band_gap_vs_local_bn_analog_window',
            ],
            'bn_band_gap_alignment_reference_split': [
                'train_plus_val_bn_unique_formulas',
                'train_plus_val_bn_unique_formulas',
            ],
            'bn_band_gap_alignment_note': [
                'demo bn-local band-gap alignment note',
                'demo bn-local band-gap alignment note',
            ],
            'bn_band_gap_alignment_neighbor_available_formula_count': [1, 2],
            'bn_band_gap_alignment_window_lower': [4.2, -0.6],
            'bn_band_gap_alignment_window_upper': [5.4, 5.4],
            'bn_band_gap_alignment_distance_to_window': [0.0, 0.6],
            'bn_band_gap_alignment_relative_distance': [0.0, 0.5],
            'bn_band_gap_alignment_penalty_eligible': [False, True],
            'bn_band_gap_alignment_label': [
                'within_local_bn_analog_band_gap_window',
                'above_local_bn_analog_band_gap_window',
            ],
            'bn_band_gap_alignment_penalty': [0.0, 0.04],
            'bn_analog_exfoliation_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
            'bn_analog_energy_support_label': ['lower_or_equal_bn_reference_median', 'higher_than_bn_reference_median'],
            'bn_analog_abs_total_magnetization_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
            'bn_analog_support_vote_count': [3, 2],
            'bn_analog_support_available_metric_count': [3, 3],
            'bn_analog_validation_label': ['reference_like_on_available_metrics', 'mixed_reference_alignment'],
            'bn_analog_validation_support_fraction': [1.0, 2.0 / 3.0],
            'bn_analog_validation_penalty': [0.0, 0.04],
            'chemical_plausibility_pass': [True, False],
            'chemical_plausibility_guess_count': [1, 0],
            'chemical_plausibility_primary_oxidation_state_guess': ['B(+3), N(-3)', ''],
            'chemical_plausibility_note': ['pass', 'fail'],
            'proposal_shortlist_enabled': [True, True],
            'proposal_shortlist_label': ['family_aware_proposal_shortlist', 'family_aware_proposal_shortlist'],
            'proposal_shortlist_method': ['ranked_family_cap', 'ranked_family_cap'],
            'proposal_shortlist_note': ['demo proposal shortlist note', 'demo proposal shortlist note'],
            'proposal_shortlist_size': [2, 2],
            'proposal_shortlist_family_cap': [1, 1],
            'proposal_shortlist_chemical_plausibility_priority': [True, True],
            'proposal_shortlist_family_count_before_selection': [0, 0],
            'proposal_shortlist_selected': [True, False],
            'proposal_shortlist_rank': [1, pd.NA],
            'proposal_shortlist_decision': [
                'selected_for_proposal_shortlist',
                'not_selected_failed_chemical_plausibility',
            ],
            'extrapolation_shortlist_enabled': [True, True],
            'extrapolation_shortlist_label': [
                'formula_level_extrapolation_shortlist',
                'formula_level_extrapolation_shortlist',
            ],
            'extrapolation_shortlist_method': [
                'novelty_bucket_ranked_family_cap',
                'novelty_bucket_ranked_family_cap',
            ],
            'extrapolation_shortlist_note': [
                'demo extrapolation shortlist note',
                'demo extrapolation shortlist note',
            ],
            'extrapolation_shortlist_size': [1, 1],
            'extrapolation_shortlist_family_cap': [1, 1],
            'extrapolation_shortlist_chemical_plausibility_priority': [True, True],
            'extrapolation_shortlist_target_novelty_bucket': [
                'formula_level_extrapolation',
                'formula_level_extrapolation',
            ],
            'extrapolation_shortlist_family_count_before_selection': [0, 0],
            'extrapolation_shortlist_selected': [False, False],
            'extrapolation_shortlist_rank': [pd.NA, pd.NA],
            'extrapolation_shortlist_decision': [
                'not_selected_novelty_bucket_mismatch',
                'not_selected_failed_chemical_plausibility',
            ],
        }),
        split_masks={'metadata': {'method': 'group_by_formula'}},
        selection_summary=selection_summary,
        cfg=cfg,
    )

    assert summary['features']['selected_feature_family'] == 'structure_aware'
    assert summary['screening']['ranking_feature_set'] == 'matminer_composition'
    assert summary['screening']['ranking_feature_family'] == 'composition_only'
    assert summary['screening']['ranking_matches_best_overall_evaluation'] is False
    assert summary['screening']['best_overall_evaluation_feature_set'] == (
        'matminer_composition_plus_structure_summary'
    )
    assert 'falls back to the best candidate-compatible combo' in summary['screening']['ranking_note']
    assert 'grouped-fold candidate robustness penalty' in summary['screening']['ranking_note']
    assert 'train+val feature-space domain-support layer' in summary['screening']['ranking_note']
    assert 'known BN slice' in summary['screening']['ranking_note']
    assert 'BN-local analog band-gap window' in summary['screening']['ranking_note']
    assert 'observed-property evidence from nearby BN-containing train+val formulas' in summary['screening']['ranking_note']
    assert 'BN analog-validation penalty' in summary['screening']['ranking_note']
    assert 'lightweight pymatgen oxidation-state plausibility screen' in summary['screening']['ranking_note']
    assert summary['screening']['candidate_generation_strategy'] == 'bn_anchored_formula_family_grid'
    assert summary['screening']['candidate_family_counts'] == {
        'bn_binary_anchor': 1,
        'group13_bn_111_family': 1,
    }
    assert summary['screening']['proposal_shortlist_enabled'] is True
    assert summary['screening']['proposal_shortlist_label'] == 'family_aware_proposal_shortlist'
    assert summary['screening']['proposal_shortlist_method'] == 'ranked_family_cap'
    assert summary['screening']['proposal_shortlist_note'] == 'demo proposal shortlist note'
    assert summary['screening']['proposal_shortlist_size'] == 2
    assert summary['screening']['proposal_shortlist_family_cap'] == 1
    assert summary['screening']['proposal_shortlist_selected_rows'] == 1
    assert summary['screening']['proposal_shortlist_selected_family_counts'] == {
        'bn_binary_anchor': 1,
    }
    assert summary['screening']['proposal_shortlist_formulas'] == [
        {
            'formula': 'BN',
            'proposal_shortlist_rank': 1,
            'candidate_family': 'bn_binary_anchor',
        }
    ]
    assert summary['screening']['extrapolation_shortlist_enabled'] is True
    assert summary['screening']['extrapolation_shortlist_label'] == (
        'formula_level_extrapolation_shortlist'
    )
    assert summary['screening']['extrapolation_shortlist_method'] == (
        'novelty_bucket_ranked_family_cap'
    )
    assert summary['screening']['extrapolation_shortlist_note'] == (
        'demo extrapolation shortlist note'
    )
    assert summary['screening']['extrapolation_shortlist_size'] == 1
    assert summary['screening']['extrapolation_shortlist_family_cap'] == 1
    assert summary['screening']['extrapolation_shortlist_target_novelty_bucket'] == (
        'formula_level_extrapolation'
    )
    assert summary['screening']['extrapolation_shortlist_selected_rows'] == 0
    assert summary['screening']['extrapolation_shortlist_selected_family_counts'] == {}
    assert summary['screening']['extrapolation_shortlist_formulas'] == []
    assert summary['screening']['bn_support_reference_formula_count'] == 4
    assert summary['screening']['bn_support_penalized_rows'] == 1
    assert summary['screening']['grouped_robustness_uncertainty_enabled'] is True
    assert summary['screening']['grouped_robustness_penalized_rows'] == 2
    assert summary['screening']['bn_analog_evidence_enabled'] is True
    assert summary['screening']['bn_analog_reference_formula_count'] == 4
    assert summary['screening']['bn_analog_reference_band_gap_median'] == 3.6
    assert summary['screening']['bn_analog_reference_band_gap_iqr'] == 1.2
    assert summary['screening']['bn_analog_reference_energy_per_atom_median'] == -8.0
    assert summary['screening']['bn_analog_reference_abs_total_magnetization_median'] == 0.0
    assert summary['screening']['bn_analog_exfoliation_available_rows'] == 2
    assert summary['screening']['bn_band_gap_alignment_enabled'] is True
    assert summary['screening']['bn_band_gap_alignment_penalty_eligible_rows'] == 1
    assert summary['screening']['bn_band_gap_alignment_within_window_rows'] == 1
    assert summary['screening']['bn_band_gap_alignment_above_window_rows'] == 1
    assert summary['screening']['bn_band_gap_alignment_penalized_rows'] == 1
    assert summary['screening']['bn_analog_reference_like_rows'] == 1
    assert summary['screening']['bn_analog_mixed_alignment_rows'] == 1
    assert summary['screening']['bn_analog_reference_divergent_rows'] == 0
    assert summary['screening']['bn_analog_validation_enabled'] is True
    assert summary['screening']['bn_analog_validation_penalized_rows'] == 1
    assert summary['screening']['chemical_plausibility_failed_formulas'] == ['AlBN']
