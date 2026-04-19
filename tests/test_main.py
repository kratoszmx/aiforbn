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
    feature_df = pd.DataFrame({
        'formula': ['BN', 'AlN'],
        'target': [5.0, 2.0],
        'matminer_2_norm': [0.7, 0.7],
        'feature_generation_failed': [False, False],
        'feature_generation_error': [None, None],
        'feature_set': ['matminer_composition', 'matminer_composition'],
    })
    feature_tables = {'matminer_composition': feature_df}
    prediction_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0], 'prediction': [4.9]})
    candidate_ensemble_df = pd.DataFrame({
        'formula': ['BN'],
        'ensemble_predicted_band_gap_mean': [4.85],
        'ensemble_predicted_band_gap_std': [0.1],
        'ensemble_member_count': [4],
    })
    candidate_grouped_robustness_df = pd.DataFrame({
        'formula': ['BN'],
        'grouped_robustness_prediction_enabled': [True],
        'grouped_robustness_prediction_method': ['selected_formula_only_group_kfold_candidate_prediction_std'],
        'grouped_robustness_prediction_note': ['demo grouped candidate robustness note'],
        'grouped_robustness_prediction_feature_set': ['matminer_composition'],
        'grouped_robustness_prediction_model_type': ['linear_regression'],
        'grouped_robustness_prediction_fold_count': [5],
        'grouped_robustness_predicted_band_gap_mean': [4.87],
        'grouped_robustness_predicted_band_gap_std': [0.08],
    })
    ranked_candidate_df = pd.DataFrame({'formula': ['BN'], 'predicted_band_gap': [4.9], 'ranking_score': [4.8]})
    structure_generation_seed_df = pd.DataFrame({'formula': ['BN'], 'seed_reference_formula': ['BN']})
    benchmark_df = pd.DataFrame({
        'feature_set': ['matminer_composition'],
        'model_type': ['linear_regression'],
        'mae': [0.1],
    })
    robustness_df = pd.DataFrame({
        'feature_set': ['matminer_composition'],
        'model_type': ['linear_regression'],
        'mae_mean': [0.2],
    })
    bn_slice_benchmark_df = pd.DataFrame({
        'feature_set': ['matminer_composition'],
        'model_type': ['linear_regression'],
        'benchmark_role': ['selected_model'],
        'mae': [0.3],
    })
    bn_slice_prediction_df = pd.DataFrame({
        'formula': ['BN'],
        'benchmark_role': ['selected_model'],
        'prediction': [4.8],
    })
    metrics = {'mae': 0.1, 'rmse': 0.1, 'r2': 0.9}
    manifest = {'name': 'twod_matpd'}
    selection_summary = {
        'selected_feature_set': 'matminer_composition',
        'selected_model_type': 'linear_regression',
        'selected_feature_family': 'composition_only',
        'screening_selected_feature_set': 'matminer_composition',
        'screening_selected_model_type': 'linear_regression',
        'screening_selected_feature_family': 'composition_only',
        'screening_candidate_feature_sets': ['basic_formula_composition', 'matminer_composition'],
        'screening_selection_matches_overall': True,
        'screening_selection_note': 'Best overall validation combo is candidate-compatible, so screening reuses it.',
    }
    bn_centered_screening_selection = {'enabled': False, 'selection_note': 'disabled in test'}
    experiment_summary = {'dataset': {'rows': 2}}

    monkeypatch.setattr(main_module, 'clear_project_cache', lambda path: calls.append('clear_project_cache'))
    monkeypatch.setattr(
        main_module,
        'load_config',
        lambda path: calls.append('load_config') or {
            'data': {'formula_column': 'formula'},
            'features': {
                'feature_set': 'basic_formula_composition',
                'feature_family': 'composition_only',
            },
        },
    )
    monkeypatch.setattr(main_module, 'ensure_runtime_dirs', lambda cfg: calls.append('ensure_runtime_dirs'))
    monkeypatch.setattr(
        main_module,
        'load_or_build_dataset',
        lambda cfg: calls.append('load_or_build_dataset') or (dataset_df, manifest),
    )
    monkeypatch.setattr(main_module, 'filter_bn', lambda df, formula_col='formula': calls.append('filter_bn') or bn_df)
    monkeypatch.setattr(
        main_module,
        'generate_bn_candidates',
        lambda cfg: calls.append('generate_bn_candidates') or candidate_df,
    )
    monkeypatch.setattr(
        main_module,
        'make_split_masks',
        lambda df, cfg: calls.append('make_split_masks') or {
            'train': [True, False],
            'val': [False, False],
            'test': [False, True],
            'metadata': {'method': 'group_by_formula'},
        },
    )
    monkeypatch.setattr(
        main_module,
        'build_feature_tables',
        lambda df, cfg, formula_col='formula': calls.append('build_feature_tables') or feature_tables,
    )
    monkeypatch.setattr(
        main_module,
        'select_feature_model_combo',
        lambda feature_tables, split_masks, cfg: calls.append('select_feature_model_combo') or selection_summary,
    )
    monkeypatch.setattr(
        main_module,
        'train_baseline_model',
        lambda df, split_masks, cfg, model_type=None, include_validation=False: calls.append('train_baseline_model') or ('model', ['matminer_2_norm']),
    )
    monkeypatch.setattr(
        main_module,
        'evaluate_predictions',
        lambda df, split_masks, model, feature_columns: calls.append('evaluate_predictions') or (metrics, prediction_df),
    )
    monkeypatch.setattr(
        main_module,
        'benchmark_regressors',
        lambda feature_tables, split_masks, cfg, selected_feature_set, selected_model_type: calls.append('benchmark_regressors') or benchmark_df,
    )
    monkeypatch.setattr(
        main_module,
        'benchmark_grouped_robustness',
        lambda feature_tables, cfg, selected_feature_set, selected_model_type: calls.append('benchmark_grouped_robustness') or robustness_df,
    )
    monkeypatch.setattr(
        main_module,
        'benchmark_bn_slice',
        lambda dataset_df, feature_tables, cfg, selected_feature_set, selected_model_type, screening_feature_set, screening_model_type: calls.append('benchmark_bn_slice') or (bn_slice_benchmark_df, bn_slice_prediction_df),
    )
    monkeypatch.setattr(
        main_module,
        'select_bn_centered_candidate_screening_combo',
        lambda bn_slice_benchmark_df, cfg, fallback_feature_set=None, fallback_model_type=None: calls.append('select_bn_centered_candidate_screening_combo') or bn_centered_screening_selection,
    )
    monkeypatch.setattr(
        main_module,
        'build_candidate_prediction_ensemble',
        lambda candidate_df, feature_tables, split_masks, cfg, candidate_feature_sets=None: calls.append('build_candidate_prediction_ensemble') or candidate_ensemble_df,
    )
    monkeypatch.setattr(
        main_module,
        'build_candidate_grouped_robustness_predictions',
        lambda candidate_df, feature_df, split_masks, cfg, feature_set, model_type: calls.append('build_candidate_grouped_robustness_predictions') or candidate_grouped_robustness_df,
    )
    monkeypatch.setattr(
        main_module,
        'screen_candidates',
        lambda candidate_df, model, feature_columns, cfg, feature_set, model_type, best_overall_feature_set=None, best_overall_model_type=None, screening_selection_note=None, dataset_df=None, split_masks=None, ensemble_prediction_df=None, grouped_robustness_prediction_df=None, reference_feature_df=None: calls.append('screen_candidates') or ranked_candidate_df,
    )
    monkeypatch.setattr(
        main_module,
        'build_candidate_structure_generation_seeds',
        lambda candidate_df, dataset_df, split_masks, cfg, bn_centered_candidate_df=None, formula_col='formula': calls.append('build_candidate_structure_generation_seeds') or structure_generation_seed_df,
    )
    monkeypatch.setattr(
        main_module,
        'build_experiment_summary',
        lambda dataset_df, bn_df, candidate_df, split_masks, selection_summary, cfg, robustness_df=None, bn_slice_benchmark_df=None, bn_centered_candidate_df=None, bn_centered_screening_selection=None, structure_generation_seed_df=None: calls.append('build_experiment_summary') or experiment_summary,
    )
    monkeypatch.setattr(
        main_module,
        'save_metrics_and_predictions',
        lambda metrics, prediction_df, bn_df, screened_df, benchmark_df, robustness_df, bn_slice_benchmark_df, bn_slice_prediction_df, bn_centered_screened_df, structure_generation_seed_df, experiment_summary, manifest, cfg: calls.append('save_metrics_and_predictions'),
    )
    monkeypatch.setattr(main_module, 'save_basic_plots', lambda prediction_df, cfg: calls.append('save_basic_plots'))

    main_module.main()

    assert calls == [
        'clear_project_cache',
        'load_config',
        'ensure_runtime_dirs',
        'load_or_build_dataset',
        'filter_bn',
        'generate_bn_candidates',
        'make_split_masks',
        'build_feature_tables',
        'select_feature_model_combo',
        'train_baseline_model',
        'evaluate_predictions',
        'benchmark_regressors',
        'benchmark_grouped_robustness',
        'benchmark_bn_slice',
        'select_bn_centered_candidate_screening_combo',
        'build_candidate_prediction_ensemble',
        'build_candidate_grouped_robustness_predictions',
        'screen_candidates',
        'build_candidate_structure_generation_seeds',
        'build_experiment_summary',
        'save_metrics_and_predictions',
        'save_basic_plots',
    ]

    out = capsys.readouterr().out
    assert 'BN AI PoC pipeline completed' in out
    assert 'dataset rows: 2' in out
    assert 'bn slice benchmark rows: 1' in out
    assert 'structure-generation seed rows: 1' in out
    assert 'split method: group_by_formula' in out
    assert 'selected feature set: matminer_composition' in out
    assert 'selected model: linear_regression' in out
    assert 'ranking feature set: matminer_composition' in out
    assert 'ranking model: linear_regression' in out
