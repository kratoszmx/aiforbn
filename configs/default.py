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
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
    },
    'features': {
        'use_matminer': True,
        'fallback_basic_features': True,
    },
    'model': {
        'type': 'hist_gradient_boosting',
        'random_forest': {
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1,
        },
        'hist_gradient_boosting': {
            'max_depth': 6,
            'learning_rate': 0.05,
            'max_iter': 500,
            'min_samples_leaf': 20,
            'random_state': 42,
        },
    },
    'screening': {
        'enabled': True,
        'top_k': 20,
        'candidate_strategy': 'simple_bn_substitutions',
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
