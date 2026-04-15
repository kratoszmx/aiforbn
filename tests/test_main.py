from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_main_orchestrates_pipeline(monkeypatch, capsys):
    spec = spec_from_file_location('main_module_under_test', ROOT / 'main.py')
    main_module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(main_module)
    calls: list[str] = []

    dataset_df = pd.DataFrame({'formula': ['BN', 'AlN'], 'target': [5.0, 2.0]})
    bn_df = pd.DataFrame({'formula': ['BN']})
    candidate_df = pd.DataFrame({'formula': ['BN']})
    feature_df = dataset_df.assign(n_elements=[2, 2], sum_z=[12, 20], contains_B=[1, 0], contains_N=[1, 1])
    prediction_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0], 'prediction': [4.9]})
    screened_df = pd.DataFrame({'formula': ['BN'], 'prediction': [4.9]})
    metrics = {'mae': 0.1, 'rmse': 0.1, 'r2': 0.9}
    manifest = {'name': 'twod_matpd'}

    monkeypatch.setattr(main_module, 'clear_project_cache', lambda path: calls.append('clear_project_cache'))
    monkeypatch.setattr(main_module, 'load_config', lambda path: calls.append('load_config') or {'data': {'formula_column': 'formula'}})
    monkeypatch.setattr(main_module, 'ensure_runtime_dirs', lambda cfg: calls.append('ensure_runtime_dirs'))
    monkeypatch.setattr(main_module, 'load_or_build_dataset', lambda cfg: calls.append('load_or_build_dataset') or (dataset_df, manifest))
    monkeypatch.setattr(main_module, 'filter_bn', lambda df, formula_col='formula': calls.append('filter_bn') or bn_df)
    monkeypatch.setattr(main_module, 'generate_bn_candidates', lambda: calls.append('generate_bn_candidates') or candidate_df)
    monkeypatch.setattr(main_module, 'build_feature_table', lambda df, formula_col='formula': calls.append('build_feature_table') or feature_df)
    monkeypatch.setattr(main_module, 'make_split_masks', lambda df, cfg: calls.append('make_split_masks') or {'train': [True, False], 'val': [False, False], 'test': [False, True]})
    monkeypatch.setattr(main_module, 'train_baseline_model', lambda df, split_masks, cfg: calls.append('train_baseline_model') or ('model', ['n_elements']))
    monkeypatch.setattr(main_module, 'evaluate_predictions', lambda df, split_masks, model, feature_columns: calls.append('evaluate_predictions') or (metrics, prediction_df))
    monkeypatch.setattr(main_module, 'screen_candidates', lambda candidate_df, model, feature_columns, cfg: calls.append('screen_candidates') or screened_df)
    monkeypatch.setattr(main_module, 'save_metrics_and_predictions', lambda metrics, prediction_df, bn_df, screened_df, manifest, cfg: calls.append('save_metrics_and_predictions'))
    monkeypatch.setattr(main_module, 'save_basic_plots', lambda prediction_df, cfg: calls.append('save_basic_plots'))

    main_module.main()

    assert calls == [
        'clear_project_cache',
        'load_config',
        'ensure_runtime_dirs',
        'load_or_build_dataset',
        'filter_bn',
        'generate_bn_candidates',
        'build_feature_table',
        'make_split_masks',
        'train_baseline_model',
        'evaluate_predictions',
        'screen_candidates',
        'save_metrics_and_predictions',
        'save_basic_plots',
    ]

    out = capsys.readouterr().out
    assert 'BN AI PoC pipeline completed' in out
    assert "dataset rows: 2" in out
