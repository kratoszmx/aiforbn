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
        'candidate_space_name': 'toy_iii_v_demo_grid',
        'candidate_space_kind': 'toy_demo',
        'candidate_space_note': (
            'Formula-only Group 13/15 demo grid without stability, structure, or synthesis constraints.'
        ),
        'ranking_label': 'demo_candidate_ranking',
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
