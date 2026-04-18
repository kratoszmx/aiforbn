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
