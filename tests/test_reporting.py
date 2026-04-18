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
            'chemical_plausibility': {
                'enabled': True,
                'method': 'pymatgen_common_oxidation_state_balance',
                'selection_policy': 'annotate_and_prioritize_passing_candidates',
                'note': 'demo plausibility note',
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
        'bn_analog_neighbor_energy_per_atom_mean': [-8.3, -7.3],
        'bn_analog_neighbor_exfoliation_energy_per_atom_mean': [0.06, 0.06],
        'bn_analog_neighbor_abs_total_magnetization_mean': [0.0, 0.0],
        'bn_analog_neighbor_exfoliation_available_formula_count': [1, 1],
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
    })
    screened_df = pd.DataFrame({
        'formula': ['BN', 'AlBN'],
        'predicted_band_gap': [4.8, 1.2],
        'screening_selected_for_top_k': [True, False],
        'screening_selection_decision': ['selected_top_k', 'failed_chemical_plausibility'],
    })
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
    assert experiment_summary['features']['selected_feature_family'] == 'composition_only'
    assert experiment_summary['feature_model_selection']['selected_model_type'] == 'linear_regression'
    assert experiment_summary['screening']['ranking_basis'] == (
        'composition_only_mean_band_gap_minus_model_disagreement_low_support_and_bn_support_and_bn_analog_validation_penalties'
    )
    assert experiment_summary['screening']['ranking_feature_family'] == 'composition_only'
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
    assert experiment_summary['screening']['bn_analog_evidence_enabled'] is True
    assert experiment_summary['screening']['bn_analog_reference_formula_count'] == 4
    assert experiment_summary['screening']['bn_analog_reference_exfoliation_energy_median'] == 0.07
    assert experiment_summary['screening']['bn_analog_reference_energy_per_atom_median'] == -8.0
    assert experiment_summary['screening']['bn_analog_reference_abs_total_magnetization_median'] == 0.0
    assert experiment_summary['screening']['bn_analog_exfoliation_available_rows'] == 2
    assert experiment_summary['screening']['bn_analog_lower_or_equal_reference_rows'] == 2
    assert experiment_summary['screening']['bn_analog_higher_reference_rows'] == 0
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
        }
    ]
    assert 'novelty should be interpreted separately' in (
        experiment_summary['screening']['novelty_interpretation_note']
    )
    assert 'Novelty is tracked only at the formula level' in experiment_summary['screening']['ranking_note']
    assert 'known BN slice' in experiment_summary['screening']['ranking_note']
    assert 'observed-property evidence from nearby BN-containing train+val formulas' in experiment_summary['screening']['ranking_note']
    assert 'BN analog-validation penalty' in experiment_summary['screening']['ranking_note']
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
        'bn_analog_nearest_band_gap',
        'bn_analog_nearest_energy_per_atom',
        'bn_analog_nearest_exfoliation_energy_per_atom',
        'bn_analog_nearest_abs_total_magnetization',
        'bn_analog_neighbor_band_gap_mean',
        'bn_analog_neighbor_energy_per_atom_mean',
        'bn_analog_neighbor_exfoliation_energy_per_atom_mean',
        'bn_analog_neighbor_abs_total_magnetization_mean',
        'bn_analog_neighbor_exfoliation_available_formula_count',
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
    ]
    assert (artifact_dir / 'predictions.csv').exists()
    assert (artifact_dir / 'bn_slice.csv').exists()
    assert (artifact_dir / 'demo_candidate_ranking.csv').exists()
    assert (artifact_dir / 'benchmark_results.csv').exists()
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
            'chemical_plausibility': {
                'enabled': True,
                'method': 'pymatgen_common_oxidation_state_balance',
                'selection_policy': 'annotate_and_prioritize_passing_candidates',
                'note': 'demo plausibility note',
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
            'bn_analog_neighbor_energy_per_atom_mean': [-8.3, -7.3],
            'bn_analog_neighbor_exfoliation_energy_per_atom_mean': [0.06, 0.06],
            'bn_analog_neighbor_abs_total_magnetization_mean': [0.0, 0.0],
            'bn_analog_neighbor_exfoliation_available_formula_count': [1, 1],
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
    assert 'train+val feature-space domain-support layer' in summary['screening']['ranking_note']
    assert 'known BN slice' in summary['screening']['ranking_note']
    assert 'observed-property evidence from nearby BN-containing train+val formulas' in summary['screening']['ranking_note']
    assert 'BN analog-validation penalty' in summary['screening']['ranking_note']
    assert 'lightweight pymatgen oxidation-state plausibility screen' in summary['screening']['ranking_note']
    assert summary['screening']['candidate_generation_strategy'] == 'bn_anchored_formula_family_grid'
    assert summary['screening']['candidate_family_counts'] == {
        'bn_binary_anchor': 1,
        'group13_bn_111_family': 1,
    }
    assert summary['screening']['bn_support_reference_formula_count'] == 4
    assert summary['screening']['bn_support_penalized_rows'] == 1
    assert summary['screening']['bn_analog_evidence_enabled'] is True
    assert summary['screening']['bn_analog_reference_formula_count'] == 4
    assert summary['screening']['bn_analog_reference_energy_per_atom_median'] == -8.0
    assert summary['screening']['bn_analog_reference_abs_total_magnetization_median'] == 0.0
    assert summary['screening']['bn_analog_exfoliation_available_rows'] == 2
    assert summary['screening']['bn_analog_reference_like_rows'] == 1
    assert summary['screening']['bn_analog_mixed_alignment_rows'] == 1
    assert summary['screening']['bn_analog_reference_divergent_rows'] == 0
    assert summary['screening']['bn_analog_validation_enabled'] is True
    assert summary['screening']['bn_analog_validation_penalized_rows'] == 1
    assert summary['screening']['chemical_plausibility_failed_formulas'] == ['AlBN']
