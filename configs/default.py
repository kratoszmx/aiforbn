CONFIG = {
    'project': {
        'name': 'ai_for_bn_poc',
        'random_seed': 42,
        'artifact_dir': 'artifacts',
    },
    'data': {
        'dataset': 'twod_matpd',
        'raw_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'cache_raw_json': True,
        'formula_column': 'formula',
        'target_column': 'band_gap',
        'bn_required_elements': ['B', 'N'],
    },
    'split': {
        'method': 'group_by_formula',
        'group_column': 'formula',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
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
        'random_forest': {
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1,
        },
        'hist_gradient_boosting': {
            'max_depth': 6,
            'learning_rate': 0.05,
            'max_iter': 100,
            'min_samples_leaf': 20,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'random_state': 42,
        },
        'dummy_mean': {
            'strategy': 'mean',
        },
    },
    'robustness': {
        'enabled': True,
        'method': 'group_kfold_by_formula',
        'group_column': 'formula',
        'n_splits': 5,
        'note': (
            'Runs a grouped-by-formula cross-validation benchmark across configured feature/model '
            'combos so the advisor-facing story does not depend on a single lucky holdout split.'
        ),
    },
    'bn_slice_benchmark': {
        'enabled': True,
        'method': 'leave_one_bn_formula_out',
        'k_neighbors': 3,
        'note': (
            'Runs a dedicated BN-focused leave-one-formula-out benchmark because the standard '
            'grouped split can place zero BN formulas in validation/test. This is a small-sample '
            'diagnostic for BN-centered generalization, not a definitive BN-only benchmark.'
        ),
    },
    'screening': {
        'enabled': True,
        'top_k': 20,
        'candidate_generation_strategy': 'bn_anchored_formula_family_grid',
        'candidate_space_name': 'bn_anchored_formula_family_grid',
        'candidate_space_kind': 'bn_family_demo',
        'candidate_space_note': (
            'BN-containing formula-family grid anchored by BCN / BC2N-style ternary motifs from '
            'the literature and by Si2BN-like motifs already present in the dataset. This is still '
            'a formula-only candidate space and does not establish structure stability, synthesis '
            'feasibility, or real discovery.'
        ),
        'ranking_label': 'demo_candidate_ranking',
        'use_model_disagreement': True,
        'uncertainty_method': 'small_feature_model_disagreement',
        'uncertainty_penalty': 0.5,
        'grouped_robustness_uncertainty': {
            'enabled': True,
            'method': 'selected_formula_only_group_kfold_candidate_prediction_std',
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.15,
            'note': (
                'Uses grouped-by-formula fold-to-fold prediction spread from the selected '
                'formula-only screening combo as a candidate-ranking robustness penalty. '
                'This is a split-robustness heuristic, not calibrated uncertainty.'
            ),
        },
        'domain_support': {
            'enabled': True,
            'method': 'train_plus_val_knn_feature_space_support',
            'distance_metric': 'z_scored_euclidean_rms',
            'k_neighbors': 5,
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.15,
            'penalize_below_percentile': 25.0,
            'note': (
                'Measures support in the selected formula-only screening feature space using '
                'z-scored distances to unique train+val formulas. This is a lightweight '
                'transparency heuristic, not a calibrated confidence or stability estimate.'
            ),
        },
        'bn_support': {
            'enabled': True,
            'method': 'train_plus_val_bn_knn_feature_space_support',
            'distance_metric': 'z_scored_euclidean_rms',
            'k_neighbors': 3,
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.1,
            'penalize_below_percentile': 25.0,
            'note': (
                'Measures support relative to known BN-containing train+val formulas in the '
                'selected formula-only screening feature space. This is a BN-theme alignment '
                'heuristic, not a calibrated confidence or structure/stability estimate.'
            ),
        },
        'bn_analog_evidence': {
            'enabled': True,
            'aggregation': 'mean_over_k_nearest_bn_formulas',
            'reference_split': 'train_plus_val_bn_unique_formulas',
            'exfoliation_reference': 'train_plus_val_bn_formula_median',
            'note': (
                'Retrieves observed-property evidence from nearby BN-containing train+val formulas, '
                'including analog band gap, energy-per-atom, exfoliation-energy, and '
                'magnetization summaries. This is analog evidence, not predicted structure or '
                'stability validation for unseen candidates.'
            ),
        },
        'bn_band_gap_alignment': {
            'enabled': True,
            'method': 'predicted_band_gap_vs_local_bn_analog_window',
            'reference_split': 'train_plus_val_bn_unique_formulas',
            'window_expansion_iqr_factor': 0.5,
            'minimum_neighbor_formula_count_for_penalty': 2,
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.08,
            'note': (
                'Compares the predicted candidate band gap against a local window derived from '
                'nearby BN-containing train+val formulas. This is a heuristic local analog-'
                'consistency layer, not direct structure validation, stability proof, or '
                'discovery evidence.'
            ),
        },
        'bn_analog_validation': {
            'enabled': True,
            'method': 'bn_analog_alignment_vote_fraction',
            'ranking_penalty_enabled': True,
            'ranking_penalty_weight': 0.12,
            'note': (
                'Compresses BN analog exfoliation / energy / magnetization alignment into a '
                'lightweight vote-fraction validation proxy. This is still a heuristic ranking '
                'penalty, not direct structure, thermodynamic stability, or synthesis validation.'
            ),
        },
        'chemical_plausibility': {
            'enabled': True,
            'method': 'pymatgen_common_oxidation_state_balance',
            'selection_policy': 'annotate_and_prioritize_passing_candidates',
            'note': (
                'Uses pymatgen oxidation-state guesses as a lightweight formula-level credibility '
                'screen. This does not establish structure stability, synthesis feasibility, or '
                'experimental realizability.'
            ),
        },
        'proposal_shortlist': {
            'enabled': True,
            'label': 'family_aware_proposal_shortlist',
            'method': 'ranked_family_cap',
            'shortlist_size': 10,
            'max_per_candidate_family': 2,
            'chemical_plausibility_priority': True,
            'note': (
                'Creates a separate advisor-facing proposal shortlist by walking the raw ranking '
                'in order, keeping chemical-plausibility priority, and limiting over-concentration '
                'from any one candidate_family. This does not replace the full raw ranking artifact.'
            ),
        },
        'extrapolation_shortlist': {
            'enabled': True,
            'label': 'formula_level_extrapolation_shortlist',
            'method': 'novelty_bucket_ranked_family_cap',
            'shortlist_size': 5,
            'max_per_candidate_family': 1,
            'required_novelty_bucket': 'formula_level_extrapolation',
            'chemical_plausibility_priority': True,
            'note': (
                'Creates a separate advisor-facing novelty shortlist that only considers '
                'formula-level extrapolation candidates, keeps chemical-plausibility priority, '
                'and applies an explicit candidate_family cap for diversity. This does not '
                'replace the full raw ranking artifact or the general proposal shortlist.'
            ),
        },
        'structure_generation_seeds': {
            'enabled': True,
            'label': 'bn_structure_generation_seed_set',
            'method': 'bn_analog_reference_exemplar',
            'candidate_scope': 'proposal_shortlist_plus_extrapolation_shortlist_plus_bn_centered_top_n',
            'per_candidate_seed_limit': 3,
            'bn_centered_top_n': 10,
            'note': (
                'Builds a structure-generation handoff artifact by attaching shortlisted BN '
                'candidates to nearby BN-containing train+val reference formulas and exemplar '
                'records with observed structures. This does not generate, relax, or validate '
                'new candidate structures; it only surfaces prototype seeds for downstream '
                'substitution, enumeration, or relaxation workflows.'
            ),
        },
        'structure_followup_shortlist': {
            'enabled': True,
            'label': 'prototype_grounded_followup_shortlist',
            'method': 'first_pass_queue_candidate_aggregation',
            'shortlist_size': 5,
            'note': (
                'Aggregates the structure-generation first-pass queue back to the candidate '
                'level so advisor-facing follow-up can prioritize formulas that combine strong '
                'screening ranks with lower-complexity prototype-edit paths. This is still a '
                'planning artifact, not validated structure output.'
            ),
        },
    },
    'llm': {
        'enabled': False,
        'provider': 'openai',
        'model': 'gpt-5.4',
        'api_key_env': 'OPENAI_API_KEY',
        'use_for': ['explanation', 'summary'],
    },
    'api': {
        'enabled': False,
        'framework': 'fastapi',
        'host': '0.0.0.0',
        'port': 8000,
        'route_prefix': '/v1',
        'auth_enabled': False,
    },
    'ui': {
        'streamlit_enabled': True,
        'title': 'BN Explorer',
    },
}
