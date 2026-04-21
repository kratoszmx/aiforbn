from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from materials.data import REFERENCE_PROPERTY_COLUMNS, STRUCTURE_SUMMARY_COLUMNS

from materials.constants import *
from materials.candidate_space import *
from materials.candidate_space import (
    _bn_family_benchmark_config,
    _bn_slice_benchmark_config,
    _bn_stratified_error_config,
    _ordered_values,
    _robustness_config,
)
from materials.feature_building import *
from materials.feature_building import _feature_columns, _feature_valid_mask
from materials.modeling import *
from materials.selection import _ordered_model_types

def benchmark_regressors(
    feature_tables: dict[str, pd.DataFrame],
    split_masks,
    cfg: dict,
    selected_feature_set: str,
    selected_model_type: str,
) -> pd.DataFrame:
    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    baseline_types = _ordered_model_types(cfg['model'].get('benchmark_baselines', ['dummy_mean']))

    rows = []
    for feature_set in candidate_feature_sets:
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        for model_type in candidate_model_types:
            row = {
                'feature_set': feature_set,
                'feature_family': feature_info['feature_family'],
                'candidate_compatible': feature_info['candidate_compatible'],
                'n_features': feature_info['n_features'],
                'model_type': model_type,
                'benchmark_role': 'candidate_model',
                'selected_by_validation': bool(
                    feature_set == selected_feature_set and model_type == selected_model_type
                ),
                'training_scope': 'train_plus_val',
                'evaluation_split': 'test',
                'benchmark_status': 'ok',
                'benchmark_note': '',
                'mae': None,
                'rmse': None,
                'r2': None,
            }

            if row['selected_by_validation']:
                row['benchmark_role'] = 'selected_model'

            if not feature_info['selection_eligible']:
                row['benchmark_status'] = 'skipped_featurization_failure'
                row['benchmark_note'] = (
                    'Skipped because this feature set could not featurize every dataset formula.'
                )
                rows.append(row)
                continue
            if not model_type_supports_feature_set(model_type, feature_set):
                row['benchmark_status'] = 'skipped_model_feature_incompatible'
                row['benchmark_note'] = incompatible_model_feature_note(model_type, feature_set)
                rows.append(row)
                continue

            model, feature_columns = train_baseline_model(
                df=feature_df,
                split_masks=split_masks,
                cfg=cfg,
                model_type=model_type,
                include_validation=True,
            )
            metrics, _ = evaluate_predictions(
                df=feature_df,
                split_masks=split_masks,
                model=model,
                feature_columns=feature_columns,
                split_name='test',
            )
            row.update(metrics)
            rows.append(row)

    selected_feature_df = feature_tables[selected_feature_set]
    selected_feature_columns = _feature_columns(selected_feature_df)
    selected_feature_count = len(selected_feature_columns)
    for model_type in baseline_types:
        model, feature_columns = train_baseline_model(
            df=selected_feature_df,
            split_masks=split_masks,
            cfg=cfg,
            model_type=model_type,
            include_validation=True,
        )
        metrics, _ = evaluate_predictions(
            df=selected_feature_df,
            split_masks=split_masks,
            model=model,
            feature_columns=feature_columns,
            split_name='test',
        )
        rows.append({
            'feature_set': DUMMY_FEATURE_SET,
            'feature_family': get_feature_family(DUMMY_FEATURE_SET),
            'candidate_compatible': False,
            'n_features': int(selected_feature_count),
            'model_type': model_type,
            'benchmark_role': 'dummy_baseline',
            'selected_by_validation': False,
            'training_scope': 'train_plus_val',
            'evaluation_split': 'test',
            'benchmark_status': 'ok',
            'benchmark_note': get_feature_note(DUMMY_FEATURE_SET),
            **metrics,
        })

    benchmark_df = pd.DataFrame(rows)
    return benchmark_df.sort_values(
        ['selected_by_validation', 'benchmark_role', 'feature_set', 'model_type'],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _group_kfold_splits(
    df: pd.DataFrame,
    group_col: str,
    requested_splits: int,
) -> tuple[int, int, list[dict[str, np.ndarray]]]:
    if group_col not in df.columns:
        raise KeyError(f'Group split column not found for robustness evaluation: {group_col}')

    target_mask = df['target'].notna().to_numpy()
    eligible_indices = np.flatnonzero(target_mask)
    if len(eligible_indices) == 0:
        raise ValueError('Robustness evaluation requires at least one row with a target value')

    eligible_groups = df.loc[target_mask, group_col].astype(str).to_numpy()
    unique_group_count = int(pd.Series(eligible_groups).nunique())
    actual_splits = min(int(requested_splits), unique_group_count)
    if actual_splits < 2:
        raise ValueError(
            'Robustness evaluation requires at least two unique groups with target values'
        )

    splitter = GroupKFold(n_splits=actual_splits)
    split_payloads: list[dict[str, np.ndarray]] = []
    dummy_x = np.zeros(len(eligible_indices), dtype=float)
    for fold_index, (train_positions, test_positions) in enumerate(
        splitter.split(dummy_x, groups=eligible_groups),
        start=1,
    ):
        train_idx = eligible_indices[train_positions]
        test_idx = eligible_indices[test_positions]
        train_mask = np.zeros(len(df), dtype=bool)
        val_mask = np.zeros(len(df), dtype=bool)
        test_mask = np.zeros(len(df), dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        split_payloads.append({
            'fold_index': int(fold_index),
            'train': train_mask,
            'val': val_mask,
            'test': test_mask,
        })

    return int(requested_splits), int(actual_splits), split_payloads


def _aggregate_fold_metric_rows(metric_rows: list[dict[str, float]]) -> dict[str, float | int]:
    metric_df = pd.DataFrame(metric_rows)
    summary: dict[str, float | int] = {
        'completed_folds': int(len(metric_df)),
    }
    for metric_name in ('mae', 'rmse', 'r2'):
        values = metric_df[metric_name].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            summary[f'{metric_name}_mean'] = np.nan
            summary[f'{metric_name}_std'] = np.nan
            continue
        summary[f'{metric_name}_mean'] = float(np.mean(values))
        summary[f'{metric_name}_std'] = float(np.std(values, ddof=0))
    return summary


def benchmark_grouped_robustness(
    feature_tables: dict[str, pd.DataFrame],
    cfg: dict,
    selected_feature_set: str,
    selected_model_type: str,
) -> pd.DataFrame:
    robustness_cfg = _robustness_config(cfg)
    if not bool(robustness_cfg['enabled']):
        return pd.DataFrame()

    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    baseline_types = _ordered_model_types(cfg['model'].get('benchmark_baselines', ['dummy_mean']))
    requested_splits = int(robustness_cfg['n_splits'])
    group_col = str(robustness_cfg['group_column'])

    rows = []
    for feature_set in candidate_feature_sets:
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        for model_type in candidate_model_types:
            row = {
                'feature_set': feature_set,
                'feature_family': feature_info['feature_family'],
                'candidate_compatible': feature_info['candidate_compatible'],
                'n_features': feature_info['n_features'],
                'model_type': model_type,
                'benchmark_role': 'candidate_model',
                'selected_by_validation': bool(
                    feature_set == selected_feature_set and model_type == selected_model_type
                ),
                'robustness_method': robustness_cfg['method'],
                'robustness_group_column': group_col,
                'requested_folds': requested_splits,
                'actual_folds': None,
                'completed_folds': 0,
                'robustness_status': 'ok',
                'robustness_note': robustness_cfg['note'],
                'mae_mean': None,
                'mae_std': None,
                'rmse_mean': None,
                'rmse_std': None,
                'r2_mean': None,
                'r2_std': None,
            }
            if row['selected_by_validation']:
                row['benchmark_role'] = 'selected_model'

            if not feature_info['selection_eligible']:
                row['robustness_status'] = 'skipped_featurization_failure'
                row['robustness_note'] = (
                    'Skipped because this feature set could not featurize every dataset formula.'
                )
                rows.append(row)
                continue
            if not model_type_supports_feature_set(model_type, feature_set):
                row['robustness_status'] = 'skipped_model_feature_incompatible'
                row['robustness_note'] = incompatible_model_feature_note(model_type, feature_set)
                rows.append(row)
                continue

            try:
                requested_fold_count, actual_fold_count, split_payloads = _group_kfold_splits(
                    feature_df,
                    group_col=group_col,
                    requested_splits=requested_splits,
                )
                row['requested_folds'] = requested_fold_count
                row['actual_folds'] = actual_fold_count

                fold_metric_rows: list[dict[str, float]] = []
                for split_payload in split_payloads:
                    split_masks = {
                        'train': split_payload['train'],
                        'val': split_payload['val'],
                        'test': split_payload['test'],
                    }
                    model, feature_columns = train_baseline_model(
                        df=feature_df,
                        split_masks=split_masks,
                        cfg=cfg,
                        model_type=model_type,
                        include_validation=False,
                    )
                    metrics, _ = evaluate_predictions(
                        df=feature_df,
                        split_masks=split_masks,
                        model=model,
                        feature_columns=feature_columns,
                        split_name='test',
                    )
                    fold_metric_rows.append(metrics)

                row.update(_aggregate_fold_metric_rows(fold_metric_rows))
            except Exception as exc:
                row['robustness_status'] = 'evaluation_failed'
                row['robustness_note'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)

    selected_feature_df = feature_tables[selected_feature_set]
    selected_feature_columns = _feature_columns(selected_feature_df)
    selected_feature_count = len(selected_feature_columns)
    for model_type in baseline_types:
        row = {
            'feature_set': DUMMY_FEATURE_SET,
            'feature_family': get_feature_family(DUMMY_FEATURE_SET),
            'candidate_compatible': False,
            'n_features': int(selected_feature_count),
            'model_type': model_type,
            'benchmark_role': 'dummy_baseline',
            'selected_by_validation': False,
            'robustness_method': robustness_cfg['method'],
            'robustness_group_column': group_col,
            'robustness_status': 'ok',
            'robustness_note': get_feature_note(DUMMY_FEATURE_SET),
        }
        try:
            requested_fold_count, actual_fold_count, split_payloads = _group_kfold_splits(
                selected_feature_df,
                group_col=group_col,
                requested_splits=requested_splits,
            )
            row['requested_folds'] = requested_fold_count
            row['actual_folds'] = actual_fold_count

            fold_metric_rows: list[dict[str, float]] = []
            for split_payload in split_payloads:
                split_masks = {
                    'train': split_payload['train'],
                    'val': split_payload['val'],
                    'test': split_payload['test'],
                }
                model, feature_columns = train_baseline_model(
                    df=selected_feature_df,
                    split_masks=split_masks,
                    cfg=cfg,
                    model_type=model_type,
                    include_validation=False,
                )
                metrics, _ = evaluate_predictions(
                    df=selected_feature_df,
                    split_masks=split_masks,
                    model=model,
                    feature_columns=feature_columns,
                    split_name='test',
                )
                fold_metric_rows.append(metrics)

            row.update(_aggregate_fold_metric_rows(fold_metric_rows))
        except Exception as exc:
            row['robustness_status'] = 'evaluation_failed'
            row['robustness_note'] = f'{type(exc).__name__}: {exc}'

        rows.append(row)

    robustness_df = pd.DataFrame(rows)
    return robustness_df.sort_values(
        ['selected_by_validation', 'benchmark_role', 'feature_set', 'model_type'],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _formula_level_regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if len(y_true_array) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
    metrics = {
        'mae': float(mean_absolute_error(y_true_array, y_pred_array)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        'r2': np.nan,
    }
    if len(y_true_array) > 1:
        metrics['r2'] = float(r2_score(y_true_array, y_pred_array))
    return metrics


def benchmark_bn_slice(
    dataset_df: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    cfg: dict,
    selected_feature_set: str,
    selected_model_type: str,
    screening_feature_set: str,
    screening_model_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_cfg = _bn_slice_benchmark_config(cfg)
    if not bool(benchmark_cfg['enabled']):
        return pd.DataFrame(), pd.DataFrame()

    formula_col = (cfg.get('data', {}) or {}).get('formula_column', 'formula')
    bn_dataset_df = filter_bn(dataset_df, formula_col=formula_col)
    bn_dataset_df = bn_dataset_df.loc[bn_dataset_df['target'].notna()].copy()
    bn_formulas = bn_dataset_df[formula_col].astype(str).drop_duplicates().tolist()
    bn_row_count = int(len(bn_dataset_df))
    bn_formula_count = int(len(bn_formulas))

    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    benchmark_specs = []
    for feature_set in candidate_feature_sets:
        for model_type in candidate_model_types:
            benchmark_role = 'candidate_model'
            selected_by_validation = bool(
                feature_set == selected_feature_set and model_type == selected_model_type
            )
            if selected_by_validation:
                benchmark_role = 'selected_model'
            elif feature_set == screening_feature_set and model_type == screening_model_type:
                benchmark_role = 'screening_model'
            benchmark_specs.append(
                {
                    'feature_set': feature_set,
                    'model_type': model_type,
                    'benchmark_role': benchmark_role,
                    'selected_by_validation': selected_by_validation,
                    'train_scope': 'full_dataset_minus_held_out_bn_formula',
                }
            )
    benchmark_specs.append(
        {
            'feature_set': screening_feature_set,
            'model_type': 'dummy_mean',
            'benchmark_role': 'global_dummy_mean_baseline',
            'selected_by_validation': False,
            'train_scope': 'full_dataset_minus_held_out_bn_formula',
        }
    )

    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for spec in benchmark_specs:
        feature_set = str(spec['feature_set'])
        model_type = str(spec['model_type'])
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        row: dict[str, object] = {
            'feature_set': feature_set,
            'feature_family': feature_info['feature_family'],
            'candidate_compatible': feature_info['candidate_compatible'],
            'n_features': feature_info['n_features'],
            'model_type': model_type,
            'benchmark_role': spec['benchmark_role'],
            'selected_by_validation': bool(spec['selected_by_validation']),
            'bn_slice_method': benchmark_cfg['method'],
            'bn_slice_train_scope': spec['train_scope'],
            'bn_formula_count': bn_formula_count,
            'bn_row_count': bn_row_count,
            'completed_holds': 0,
            'benchmark_status': 'ok',
            'benchmark_note': benchmark_cfg['note'],
            'mae': None,
            'rmse': None,
            'r2': None,
            'k_neighbors': None,
        }
        if not feature_info['selection_eligible']:
            row['benchmark_status'] = 'skipped_featurization_failure'
            row['benchmark_note'] = (
                'Skipped because this feature set could not featurize every dataset formula.'
            )
            rows.append(row)
            continue
        if not model_type_supports_feature_set(model_type, feature_set):
            row['benchmark_status'] = 'skipped_model_feature_incompatible'
            row['benchmark_note'] = incompatible_model_feature_note(model_type, feature_set)
            rows.append(row)
            continue
        if bn_formula_count < 2:
            row['benchmark_status'] = 'insufficient_bn_formulas'
            row['benchmark_note'] = (
                'BN-focused benchmark requires at least two BN formulas with targets.'
            )
            rows.append(row)
            continue

        formula_values = feature_df[formula_col].astype(str).to_numpy()
        target_mask = feature_df['target'].notna().to_numpy()
        fold_true: list[float] = []
        fold_pred: list[float] = []
        for held_out_formula in bn_formulas:
            test_mask = target_mask & (formula_values == held_out_formula)
            if not bool(test_mask.any()):
                continue
            train_mask = target_mask & (formula_values != held_out_formula)
            split_masks = {
                'train': train_mask,
                'val': np.zeros(len(feature_df), dtype=bool),
                'test': test_mask,
            }
            model, feature_columns = train_baseline_model(
                df=feature_df,
                split_masks=split_masks,
                cfg=cfg,
                model_type=model_type,
                include_validation=False,
            )
            _, prediction_df = evaluate_predictions(
                df=feature_df,
                split_masks=split_masks,
                model=model,
                feature_columns=feature_columns,
                split_name='test',
            )
            if prediction_df.empty:
                continue
            formula_target = float(prediction_df['target'].mean())
            formula_prediction = float(prediction_df['prediction'].mean())
            fold_true.append(formula_target)
            fold_pred.append(formula_prediction)
            prediction_rows.append(
                {
                    'formula': held_out_formula,
                    'benchmark_role': spec['benchmark_role'],
                    'feature_set': feature_set,
                    'feature_family': feature_info['feature_family'],
                    'model_type': model_type,
                    'selected_by_validation': bool(spec['selected_by_validation']),
                    'bn_slice_method': benchmark_cfg['method'],
                    'bn_slice_train_scope': spec['train_scope'],
                    'target': formula_target,
                    'prediction': formula_prediction,
                    'absolute_error': abs(formula_target - formula_prediction),
                }
            )

        row['completed_holds'] = int(len(fold_true))
        if len(fold_true) < 2:
            row['benchmark_status'] = 'insufficient_bn_holds'
            row['benchmark_note'] = (
                'BN-focused benchmark could not assemble at least two held-out BN formulas.'
            )
            rows.append(row)
            continue

        row.update(_formula_level_regression_metrics(fold_true, fold_pred))
        rows.append(row)

    local_feature_df = feature_tables[screening_feature_set]
    local_feature_info = summarize_feature_table(local_feature_df, feature_set=screening_feature_set)
    baseline_row: dict[str, object] = {
        'feature_set': screening_feature_set,
        'feature_family': local_feature_info['feature_family'],
        'candidate_compatible': local_feature_info['candidate_compatible'],
        'n_features': local_feature_info['n_features'],
        'model_type': 'bn_local_knn_mean',
        'benchmark_role': 'bn_local_reference_baseline',
        'selected_by_validation': False,
        'bn_slice_method': benchmark_cfg['method'],
        'bn_slice_train_scope': 'bn_only_reference_formulas',
        'bn_formula_count': bn_formula_count,
        'bn_row_count': bn_row_count,
        'completed_holds': 0,
        'benchmark_status': 'ok',
        'benchmark_note': benchmark_cfg['note'],
        'mae': None,
        'rmse': None,
        'r2': None,
        'k_neighbors': int(benchmark_cfg['k_neighbors']),
    }
    if not local_feature_info['selection_eligible']:
        baseline_row['benchmark_status'] = 'skipped_featurization_failure'
        baseline_row['benchmark_note'] = (
            'Skipped because the screening feature set could not featurize every dataset formula.'
        )
        rows.append(baseline_row)
    elif bn_formula_count < 2:
        baseline_row['benchmark_status'] = 'insufficient_bn_formulas'
        baseline_row['benchmark_note'] = (
            'BN-focused benchmark requires at least two BN formulas with targets.'
        )
        rows.append(baseline_row)
    else:
        local_feature_columns = _feature_columns(local_feature_df)
        bn_local_feature_df = filter_bn(local_feature_df, formula_col=formula_col)
        bn_local_feature_df = bn_local_feature_df.loc[
            bn_local_feature_df['target'].notna()
            & _feature_valid_mask(bn_local_feature_df, local_feature_columns)
        ].copy()
        bn_local_feature_df[formula_col] = bn_local_feature_df[formula_col].astype(str)
        formula_feature_df = (
            bn_local_feature_df
            .groupby(formula_col, as_index=False)[['target', *local_feature_columns]]
            .mean()
        )

        fold_true: list[float] = []
        fold_pred: list[float] = []
        completed_holds = 0
        for held_out_formula in bn_formulas:
            held_out_df = formula_feature_df.loc[
                formula_feature_df[formula_col].eq(held_out_formula)
            ].copy()
            train_formula_df = formula_feature_df.loc[
                ~formula_feature_df[formula_col].eq(held_out_formula)
            ].copy()
            if held_out_df.empty or len(train_formula_df) == 0:
                continue

            train_matrix_raw = train_formula_df[local_feature_columns].to_numpy(dtype=float)
            held_out_matrix_raw = held_out_df[local_feature_columns].to_numpy(dtype=float)
            center = train_matrix_raw.mean(axis=0)
            spread = train_matrix_raw.std(axis=0)
            spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
            train_matrix = (train_matrix_raw - center) / spread
            held_out_matrix = (held_out_matrix_raw - center) / spread

            effective_k = min(int(benchmark_cfg['k_neighbors']), len(train_formula_df))
            neighbors = NearestNeighbors(metric='euclidean', n_neighbors=effective_k)
            neighbors.fit(train_matrix)
            _, neighbor_indices = neighbors.kneighbors(
                held_out_matrix,
                n_neighbors=effective_k,
            )
            neighbor_targets = train_formula_df.iloc[neighbor_indices[0]]['target'].to_numpy(dtype=float)
            formula_target = float(held_out_df['target'].iloc[0])
            formula_prediction = float(np.mean(neighbor_targets))
            fold_true.append(formula_target)
            fold_pred.append(formula_prediction)
            prediction_rows.append(
                {
                    'formula': held_out_formula,
                    'benchmark_role': 'bn_local_reference_baseline',
                    'feature_set': screening_feature_set,
                    'feature_family': local_feature_info['feature_family'],
                    'model_type': 'bn_local_knn_mean',
                    'selected_by_validation': False,
                    'bn_slice_method': benchmark_cfg['method'],
                    'bn_slice_train_scope': 'bn_only_reference_formulas',
                    'target': formula_target,
                    'prediction': formula_prediction,
                    'absolute_error': abs(formula_target - formula_prediction),
                }
            )
            completed_holds += 1

        baseline_row['completed_holds'] = int(completed_holds)
        if completed_holds < 2:
            baseline_row['benchmark_status'] = 'insufficient_bn_holds'
            baseline_row['benchmark_note'] = (
                'BN-local reference baseline could not assemble at least two held-out BN formulas.'
            )
        else:
            baseline_row.update(_formula_level_regression_metrics(fold_true, fold_pred))
        rows.append(baseline_row)

    benchmark_df = pd.DataFrame(rows)
    benchmark_df = benchmark_df.sort_values(
        ['selected_by_validation', 'benchmark_role', 'feature_set', 'model_type'],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows)
    if not prediction_df.empty:
        prediction_df = prediction_df.sort_values(
            ['selected_by_validation', 'benchmark_role', 'formula'],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return benchmark_df, prediction_df


def benchmark_bn_family_holdout(
    dataset_df: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    cfg: dict,
    selected_feature_set: str,
    selected_model_type: str,
    screening_feature_set: str,
    screening_model_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_cfg = _bn_family_benchmark_config(cfg)
    if not bool(benchmark_cfg['enabled']):
        return pd.DataFrame(), pd.DataFrame()

    formula_col = (cfg.get('data', {}) or {}).get('formula_column', 'formula')
    bn_dataset_df = filter_bn(dataset_df, formula_col=formula_col)
    bn_dataset_df = bn_dataset_df.loc[bn_dataset_df['target'].notna()].copy()
    bn_dataset_df = annotate_bn_families(
        bn_dataset_df,
        formula_col=formula_col,
        grouping_method=str(benchmark_cfg['grouping_method']),
    )
    bn_formulas = bn_dataset_df[formula_col].astype(str).drop_duplicates().tolist()
    bn_row_count = int(len(bn_dataset_df))
    bn_formula_count = int(len(bn_formulas))
    family_formula_df = (
        bn_dataset_df[[formula_col, 'bn_family']]
        .drop_duplicates()
        .groupby('bn_family')[formula_col]
        .apply(lambda values: sorted(set(values.astype(str))))
        .reset_index(name='holdout_formulas')
    )
    family_formula_df['holdout_formula_count'] = family_formula_df['holdout_formulas'].apply(len)
    holdout_groups = [
        (str(row['bn_family']), list(row['holdout_formulas']))
        for _, row in family_formula_df.iterrows()
    ]
    bn_family_count = int(len(holdout_groups))

    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    benchmark_specs = []
    for feature_set in candidate_feature_sets:
        for model_type in candidate_model_types:
            benchmark_role = 'candidate_model'
            selected_by_validation = bool(
                feature_set == selected_feature_set and model_type == selected_model_type
            )
            if selected_by_validation:
                benchmark_role = 'selected_model'
            elif feature_set == screening_feature_set and model_type == screening_model_type:
                benchmark_role = 'screening_model'
            benchmark_specs.append(
                {
                    'feature_set': feature_set,
                    'model_type': model_type,
                    'benchmark_role': benchmark_role,
                    'selected_by_validation': selected_by_validation,
                    'train_scope': 'full_dataset_minus_held_out_bn_family',
                }
            )
    benchmark_specs.append(
        {
            'feature_set': screening_feature_set,
            'model_type': 'dummy_mean',
            'benchmark_role': 'global_dummy_mean_baseline',
            'selected_by_validation': False,
            'train_scope': 'full_dataset_minus_held_out_bn_family',
        }
    )

    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for spec in benchmark_specs:
        feature_set = str(spec['feature_set'])
        model_type = str(spec['model_type'])
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        row: dict[str, object] = {
            'feature_set': feature_set,
            'feature_family': feature_info['feature_family'],
            'candidate_compatible': feature_info['candidate_compatible'],
            'n_features': feature_info['n_features'],
            'model_type': model_type,
            'benchmark_role': spec['benchmark_role'],
            'selected_by_validation': bool(spec['selected_by_validation']),
            'bn_family_benchmark_method': benchmark_cfg['method'],
            'bn_family_grouping_method': benchmark_cfg['grouping_method'],
            'bn_family_train_scope': spec['train_scope'],
            'bn_family_count': bn_family_count,
            'bn_formula_count': bn_formula_count,
            'bn_row_count': bn_row_count,
            'completed_family_holds': 0,
            'completed_formula_holds': 0,
            'benchmark_status': 'ok',
            'benchmark_note': benchmark_cfg['note'],
            'mae': None,
            'rmse': None,
            'r2': None,
            'k_neighbors': None,
        }
        if not feature_info['selection_eligible']:
            row['benchmark_status'] = 'skipped_featurization_failure'
            row['benchmark_note'] = (
                'Skipped because this feature set could not featurize every dataset formula.'
            )
            rows.append(row)
            continue
        if not model_type_supports_feature_set(model_type, feature_set):
            row['benchmark_status'] = 'skipped_model_feature_incompatible'
            row['benchmark_note'] = incompatible_model_feature_note(model_type, feature_set)
            rows.append(row)
            continue
        if bn_family_count < 2 or bn_formula_count < 2:
            row['benchmark_status'] = 'insufficient_bn_families'
            row['benchmark_note'] = (
                'BN-family benchmark requires at least two BN-local families and two BN formulas '
                'with targets.'
            )
            rows.append(row)
            continue

        feature_formula_col = formula_col if formula_col in feature_df.columns else 'formula'
        formula_values = feature_df[feature_formula_col].astype(str)
        target_mask = feature_df['target'].notna().to_numpy()
        fold_true: list[float] = []
        fold_pred: list[float] = []
        completed_formula_holds = 0
        completed_family_holds = 0
        for held_out_family, held_out_formulas in holdout_groups:
            held_out_formula_set = set(str(value) for value in held_out_formulas)
            test_mask = target_mask & formula_values.isin(held_out_formula_set).to_numpy()
            if not bool(test_mask.any()):
                continue
            train_mask = target_mask & ~formula_values.isin(held_out_formula_set).to_numpy()
            if not bool(train_mask.any()):
                continue
            split_masks = {
                'train': train_mask,
                'val': np.zeros(len(feature_df), dtype=bool),
                'test': test_mask,
            }
            model, feature_columns = train_baseline_model(
                df=feature_df,
                split_masks=split_masks,
                cfg=cfg,
                model_type=model_type,
                include_validation=False,
            )
            _, prediction_df = evaluate_predictions(
                df=feature_df,
                split_masks=split_masks,
                model=model,
                feature_columns=feature_columns,
                split_name='test',
            )
            if prediction_df.empty:
                continue
            prediction_formula_col = (
                feature_formula_col if feature_formula_col in prediction_df.columns else 'formula'
            )
            grouped_prediction_df = (
                prediction_df[[prediction_formula_col, 'target', 'prediction']]
                .copy()
                .groupby(prediction_formula_col, as_index=False)[['target', 'prediction']]
                .mean()
            )
            if grouped_prediction_df.empty:
                continue
            completed_family_holds += 1
            completed_formula_holds += int(len(grouped_prediction_df))
            for _, grouped_row in grouped_prediction_df.iterrows():
                formula_target = float(grouped_row['target'])
                formula_prediction = float(grouped_row['prediction'])
                formula_name = str(grouped_row[prediction_formula_col])
                fold_true.append(formula_target)
                fold_pred.append(formula_prediction)
                prediction_rows.append(
                    {
                        'formula': formula_name,
                        'bn_family': held_out_family,
                        'benchmark_role': spec['benchmark_role'],
                        'feature_set': feature_set,
                        'feature_family': feature_info['feature_family'],
                        'model_type': model_type,
                        'selected_by_validation': bool(spec['selected_by_validation']),
                        'bn_family_benchmark_method': benchmark_cfg['method'],
                        'bn_family_grouping_method': benchmark_cfg['grouping_method'],
                        'bn_family_train_scope': spec['train_scope'],
                        'target': formula_target,
                        'prediction': formula_prediction,
                        'absolute_error': abs(formula_target - formula_prediction),
                    }
                )

        row['completed_family_holds'] = int(completed_family_holds)
        row['completed_formula_holds'] = int(completed_formula_holds)
        if len(fold_true) < 2 or completed_family_holds < 2:
            row['benchmark_status'] = 'insufficient_bn_family_holds'
            row['benchmark_note'] = (
                'BN-family benchmark could not assemble at least two held-out BN families with '
                'formula-level predictions.'
            )
            rows.append(row)
            continue

        row.update(_formula_level_regression_metrics(fold_true, fold_pred))
        rows.append(row)

    local_feature_df = feature_tables[screening_feature_set]
    local_feature_info = summarize_feature_table(local_feature_df, feature_set=screening_feature_set)
    baseline_row: dict[str, object] = {
        'feature_set': screening_feature_set,
        'feature_family': local_feature_info['feature_family'],
        'candidate_compatible': local_feature_info['candidate_compatible'],
        'n_features': local_feature_info['n_features'],
        'model_type': 'bn_local_knn_mean',
        'benchmark_role': 'bn_local_reference_baseline',
        'selected_by_validation': False,
        'bn_family_benchmark_method': benchmark_cfg['method'],
        'bn_family_grouping_method': benchmark_cfg['grouping_method'],
        'bn_family_train_scope': 'bn_only_reference_formulas_minus_held_out_family',
        'bn_family_count': bn_family_count,
        'bn_formula_count': bn_formula_count,
        'bn_row_count': bn_row_count,
        'completed_family_holds': 0,
        'completed_formula_holds': 0,
        'benchmark_status': 'ok',
        'benchmark_note': benchmark_cfg['note'],
        'mae': None,
        'rmse': None,
        'r2': None,
        'k_neighbors': int(benchmark_cfg['k_neighbors']),
    }
    if not local_feature_info['selection_eligible']:
        baseline_row['benchmark_status'] = 'skipped_featurization_failure'
        baseline_row['benchmark_note'] = (
            'Skipped because the screening feature set could not featurize every dataset formula.'
        )
        rows.append(baseline_row)
    elif bn_family_count < 2 or bn_formula_count < 2:
        baseline_row['benchmark_status'] = 'insufficient_bn_families'
        baseline_row['benchmark_note'] = (
            'BN-family benchmark requires at least two BN-local families and two BN formulas '
            'with targets.'
        )
        rows.append(baseline_row)
    else:
        local_feature_columns = _feature_columns(local_feature_df)
        feature_formula_col = formula_col if formula_col in local_feature_df.columns else 'formula'
        bn_local_feature_df = filter_bn(local_feature_df, formula_col=feature_formula_col)
        bn_local_feature_df = bn_local_feature_df.loc[
            bn_local_feature_df['target'].notna()
            & _feature_valid_mask(bn_local_feature_df, local_feature_columns)
        ].copy()
        bn_local_feature_df = annotate_bn_families(
            bn_local_feature_df,
            formula_col=feature_formula_col,
            grouping_method=str(benchmark_cfg['grouping_method']),
        )
        bn_local_feature_df[feature_formula_col] = bn_local_feature_df[feature_formula_col].astype(str)
        formula_feature_df = (
            bn_local_feature_df
            .groupby([feature_formula_col, 'bn_family'], as_index=False)[['target', *local_feature_columns]]
            .mean()
        )

        fold_true: list[float] = []
        fold_pred: list[float] = []
        completed_family_holds = 0
        completed_formula_holds = 0
        for held_out_family, held_out_formulas in holdout_groups:
            held_out_formula_set = set(str(value) for value in held_out_formulas)
            held_out_df = formula_feature_df.loc[
                formula_feature_df[feature_formula_col].isin(held_out_formula_set)
            ].copy()
            train_formula_df = formula_feature_df.loc[
                ~formula_feature_df[feature_formula_col].isin(held_out_formula_set)
            ].copy()
            if held_out_df.empty or len(train_formula_df) == 0:
                continue

            train_matrix_raw = train_formula_df[local_feature_columns].to_numpy(dtype=float)
            center = train_matrix_raw.mean(axis=0)
            spread = train_matrix_raw.std(axis=0)
            spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
            train_matrix = (train_matrix_raw - center) / spread

            effective_k = min(int(benchmark_cfg['k_neighbors']), len(train_formula_df))
            neighbors = NearestNeighbors(metric='euclidean', n_neighbors=effective_k)
            neighbors.fit(train_matrix)

            family_completed = False
            for _, held_out_row in held_out_df.iterrows():
                held_out_matrix_raw = held_out_row[local_feature_columns].to_numpy(dtype=float).reshape(1, -1)
                held_out_matrix = (held_out_matrix_raw - center) / spread
                _, neighbor_indices = neighbors.kneighbors(
                    held_out_matrix,
                    n_neighbors=effective_k,
                )
                neighbor_targets = train_formula_df.iloc[neighbor_indices[0]]['target'].to_numpy(dtype=float)
                formula_target = float(held_out_row['target'])
                formula_prediction = float(np.mean(neighbor_targets))
                formula_name = str(held_out_row[feature_formula_col])
                fold_true.append(formula_target)
                fold_pred.append(formula_prediction)
                completed_formula_holds += 1
                family_completed = True
                prediction_rows.append(
                    {
                        'formula': formula_name,
                        'bn_family': held_out_family,
                        'benchmark_role': 'bn_local_reference_baseline',
                        'feature_set': screening_feature_set,
                        'feature_family': local_feature_info['feature_family'],
                        'model_type': 'bn_local_knn_mean',
                        'selected_by_validation': False,
                        'bn_family_benchmark_method': benchmark_cfg['method'],
                        'bn_family_grouping_method': benchmark_cfg['grouping_method'],
                        'bn_family_train_scope': 'bn_only_reference_formulas_minus_held_out_family',
                        'target': formula_target,
                        'prediction': formula_prediction,
                        'absolute_error': abs(formula_target - formula_prediction),
                    }
                )
            if family_completed:
                completed_family_holds += 1

        baseline_row['completed_family_holds'] = int(completed_family_holds)
        baseline_row['completed_formula_holds'] = int(completed_formula_holds)
        if len(fold_true) < 2 or completed_family_holds < 2:
            baseline_row['benchmark_status'] = 'insufficient_bn_family_holds'
            baseline_row['benchmark_note'] = (
                'BN-local family reference baseline could not assemble at least two held-out BN '
                'families with formula-level predictions.'
            )
        else:
            baseline_row.update(_formula_level_regression_metrics(fold_true, fold_pred))
        rows.append(baseline_row)

    benchmark_df = pd.DataFrame(rows)
    benchmark_df = benchmark_df.sort_values(
        ['selected_by_validation', 'benchmark_role', 'feature_set', 'model_type'],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows)
    if not prediction_df.empty:
        prediction_df = prediction_df.sort_values(
            ['selected_by_validation', 'benchmark_role', 'bn_family', 'formula'],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
    return benchmark_df, prediction_df


def benchmark_bn_stratified_errors(
    feature_tables: dict[str, pd.DataFrame],
    cfg: dict,
    selected_feature_set: str,
    selected_model_type: str,
    screening_feature_set: str,
    screening_model_type: str,
) -> pd.DataFrame:
    benchmark_cfg = _bn_stratified_error_config(cfg)
    if not bool(benchmark_cfg['enabled']):
        return pd.DataFrame()

    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    baseline_types = _ordered_model_types(cfg['model'].get('benchmark_baselines', ['dummy_mean']))
    requested_splits = int(benchmark_cfg['n_splits'])
    group_col = str(benchmark_cfg['group_column'])

    rows: list[dict[str, object]] = []
    for feature_set in candidate_feature_sets:
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        feature_formula_col = group_col if group_col in feature_df.columns else 'formula'
        for model_type in candidate_model_types:
            row: dict[str, object] = {
                'feature_set': feature_set,
                'feature_family': feature_info['feature_family'],
                'candidate_compatible': feature_info['candidate_compatible'],
                'n_features': feature_info['n_features'],
                'model_type': model_type,
                'benchmark_role': 'candidate_model',
                'selected_by_validation': bool(
                    feature_set == selected_feature_set and model_type == selected_model_type
                ),
                'bn_stratified_error_method': benchmark_cfg['method'],
                'bn_stratified_group_column': group_col,
                'requested_folds': requested_splits,
                'actual_folds': None,
                'completed_folds': 0,
                'benchmark_status': 'ok',
                'benchmark_note': benchmark_cfg['note'],
                'bn_formula_count': 0,
                'non_bn_formula_count': 0,
                'bn_mae': None,
                'bn_rmse': None,
                'bn_r2': None,
                'non_bn_mae': None,
                'non_bn_rmse': None,
                'non_bn_r2': None,
                'bn_to_non_bn_mae_ratio': None,
            }
            if row['selected_by_validation']:
                row['benchmark_role'] = 'selected_model'
            elif feature_set == screening_feature_set and model_type == screening_model_type:
                row['benchmark_role'] = 'screening_model'

            if not feature_info['selection_eligible']:
                row['benchmark_status'] = 'skipped_featurization_failure'
                row['benchmark_note'] = (
                    'Skipped because this feature set could not featurize every dataset formula.'
                )
                rows.append(row)
                continue
            if not model_type_supports_feature_set(model_type, feature_set):
                row['benchmark_status'] = 'skipped_model_feature_incompatible'
                row['benchmark_note'] = incompatible_model_feature_note(model_type, feature_set)
                rows.append(row)
                continue

            try:
                requested_fold_count, actual_fold_count, split_payloads = _group_kfold_splits(
                    feature_df,
                    group_col=group_col,
                    requested_splits=requested_splits,
                )
                row['requested_folds'] = requested_fold_count
                row['actual_folds'] = actual_fold_count

                formula_prediction_rows: list[pd.DataFrame] = []
                for split_payload in split_payloads:
                    split_masks = {
                        'train': split_payload['train'],
                        'val': split_payload['val'],
                        'test': split_payload['test'],
                    }
                    model, feature_columns = train_baseline_model(
                        df=feature_df,
                        split_masks=split_masks,
                        cfg=cfg,
                        model_type=model_type,
                        include_validation=False,
                    )
                    _, prediction_df = evaluate_predictions(
                        df=feature_df,
                        split_masks=split_masks,
                        model=model,
                        feature_columns=feature_columns,
                        split_name='test',
                    )
                    if prediction_df.empty:
                        continue
                    prediction_formula_col = (
                        feature_formula_col if feature_formula_col in prediction_df.columns else 'formula'
                    )
                    grouped_prediction_df = (
                        prediction_df[[prediction_formula_col, 'target', 'prediction']]
                        .copy()
                        .groupby(prediction_formula_col, as_index=False)[['target', 'prediction']]
                        .mean()
                    )
                    grouped_prediction_df = grouped_prediction_df.rename(
                        columns={prediction_formula_col: 'formula'}
                    )
                    grouped_prediction_df['is_bn'] = grouped_prediction_df['formula'].astype(str).apply(
                        lambda value: {'B', 'N'}.issubset(set(extract_elements(value)))
                    )
                    formula_prediction_rows.append(grouped_prediction_df)

                if not formula_prediction_rows:
                    row['benchmark_status'] = 'insufficient_predictions'
                    row['benchmark_note'] = (
                        'Grouped stratified benchmark could not collect any held-out formula '
                        'predictions.'
                    )
                    rows.append(row)
                    continue

                formula_prediction_df = pd.concat(formula_prediction_rows, ignore_index=True)
                bn_formula_df = formula_prediction_df.loc[formula_prediction_df['is_bn'].fillna(False)].copy()
                non_bn_formula_df = formula_prediction_df.loc[
                    ~formula_prediction_df['is_bn'].fillna(False)
                ].copy()
                row['completed_folds'] = int(len(split_payloads))
                row['bn_formula_count'] = int(bn_formula_df['formula'].astype(str).nunique())
                row['non_bn_formula_count'] = int(non_bn_formula_df['formula'].astype(str).nunique())
                if len(bn_formula_df) >= 2:
                    row.update({
                        'bn_mae': _formula_level_regression_metrics(
                            bn_formula_df['target'].tolist(),
                            bn_formula_df['prediction'].tolist(),
                        )['mae'],
                        'bn_rmse': _formula_level_regression_metrics(
                            bn_formula_df['target'].tolist(),
                            bn_formula_df['prediction'].tolist(),
                        )['rmse'],
                        'bn_r2': _formula_level_regression_metrics(
                            bn_formula_df['target'].tolist(),
                            bn_formula_df['prediction'].tolist(),
                        )['r2'],
                    })
                if len(non_bn_formula_df) >= 2:
                    row.update({
                        'non_bn_mae': _formula_level_regression_metrics(
                            non_bn_formula_df['target'].tolist(),
                            non_bn_formula_df['prediction'].tolist(),
                        )['mae'],
                        'non_bn_rmse': _formula_level_regression_metrics(
                            non_bn_formula_df['target'].tolist(),
                            non_bn_formula_df['prediction'].tolist(),
                        )['rmse'],
                        'non_bn_r2': _formula_level_regression_metrics(
                            non_bn_formula_df['target'].tolist(),
                            non_bn_formula_df['prediction'].tolist(),
                        )['r2'],
                    })
                if pd.notna(row['bn_mae']) and pd.notna(row['non_bn_mae']) and float(row['non_bn_mae']) > 0:
                    row['bn_to_non_bn_mae_ratio'] = float(row['bn_mae']) / float(row['non_bn_mae'])
            except Exception as exc:
                row['benchmark_status'] = 'evaluation_failed'
                row['benchmark_note'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)

    selected_feature_df = feature_tables[selected_feature_set]
    selected_feature_columns = _feature_columns(selected_feature_df)
    selected_feature_count = len(selected_feature_columns)
    feature_formula_col = group_col if group_col in selected_feature_df.columns else 'formula'
    for model_type in baseline_types:
        row = {
            'feature_set': DUMMY_FEATURE_SET,
            'feature_family': get_feature_family(DUMMY_FEATURE_SET),
            'candidate_compatible': False,
            'n_features': int(selected_feature_count),
            'model_type': model_type,
            'benchmark_role': 'dummy_baseline',
            'selected_by_validation': False,
            'bn_stratified_error_method': benchmark_cfg['method'],
            'bn_stratified_group_column': group_col,
            'requested_folds': requested_splits,
            'actual_folds': None,
            'completed_folds': 0,
            'benchmark_status': 'ok',
            'benchmark_note': get_feature_note(DUMMY_FEATURE_SET),
            'bn_formula_count': 0,
            'non_bn_formula_count': 0,
            'bn_mae': None,
            'bn_rmse': None,
            'bn_r2': None,
            'non_bn_mae': None,
            'non_bn_rmse': None,
            'non_bn_r2': None,
            'bn_to_non_bn_mae_ratio': None,
        }
        try:
            requested_fold_count, actual_fold_count, split_payloads = _group_kfold_splits(
                selected_feature_df,
                group_col=group_col,
                requested_splits=requested_splits,
            )
            row['requested_folds'] = requested_fold_count
            row['actual_folds'] = actual_fold_count

            formula_prediction_rows: list[pd.DataFrame] = []
            for split_payload in split_payloads:
                split_masks = {
                    'train': split_payload['train'],
                    'val': split_payload['val'],
                    'test': split_payload['test'],
                }
                model, feature_columns = train_baseline_model(
                    df=selected_feature_df,
                    split_masks=split_masks,
                    cfg=cfg,
                    model_type=model_type,
                    include_validation=False,
                )
                _, prediction_df = evaluate_predictions(
                    df=selected_feature_df,
                    split_masks=split_masks,
                    model=model,
                    feature_columns=feature_columns,
                    split_name='test',
                )
                if prediction_df.empty:
                    continue
                prediction_formula_col = (
                    feature_formula_col if feature_formula_col in prediction_df.columns else 'formula'
                )
                grouped_prediction_df = (
                    prediction_df[[prediction_formula_col, 'target', 'prediction']]
                    .copy()
                    .groupby(prediction_formula_col, as_index=False)[['target', 'prediction']]
                    .mean()
                )
                grouped_prediction_df = grouped_prediction_df.rename(
                    columns={prediction_formula_col: 'formula'}
                )
                grouped_prediction_df['is_bn'] = grouped_prediction_df['formula'].astype(str).apply(
                    lambda value: {'B', 'N'}.issubset(set(extract_elements(value)))
                )
                formula_prediction_rows.append(grouped_prediction_df)

            if formula_prediction_rows:
                formula_prediction_df = pd.concat(formula_prediction_rows, ignore_index=True)
                bn_formula_df = formula_prediction_df.loc[formula_prediction_df['is_bn'].fillna(False)].copy()
                non_bn_formula_df = formula_prediction_df.loc[
                    ~formula_prediction_df['is_bn'].fillna(False)
                ].copy()
                row['completed_folds'] = int(len(split_payloads))
                row['bn_formula_count'] = int(bn_formula_df['formula'].astype(str).nunique())
                row['non_bn_formula_count'] = int(non_bn_formula_df['formula'].astype(str).nunique())
                if len(bn_formula_df) >= 2:
                    bn_metrics = _formula_level_regression_metrics(
                        bn_formula_df['target'].tolist(),
                        bn_formula_df['prediction'].tolist(),
                    )
                    row['bn_mae'] = bn_metrics['mae']
                    row['bn_rmse'] = bn_metrics['rmse']
                    row['bn_r2'] = bn_metrics['r2']
                if len(non_bn_formula_df) >= 2:
                    non_bn_metrics = _formula_level_regression_metrics(
                        non_bn_formula_df['target'].tolist(),
                        non_bn_formula_df['prediction'].tolist(),
                    )
                    row['non_bn_mae'] = non_bn_metrics['mae']
                    row['non_bn_rmse'] = non_bn_metrics['rmse']
                    row['non_bn_r2'] = non_bn_metrics['r2']
                if pd.notna(row['bn_mae']) and pd.notna(row['non_bn_mae']) and float(row['non_bn_mae']) > 0:
                    row['bn_to_non_bn_mae_ratio'] = float(row['bn_mae']) / float(row['non_bn_mae'])
            else:
                row['benchmark_status'] = 'insufficient_predictions'
                row['benchmark_note'] = (
                    'Grouped stratified benchmark could not collect any held-out formula '
                    'predictions.'
                )
        except Exception as exc:
            row['benchmark_status'] = 'evaluation_failed'
            row['benchmark_note'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)

    stratified_df = pd.DataFrame(rows)
    return stratified_df.sort_values(
        ['selected_by_validation', 'benchmark_role', 'feature_set', 'model_type'],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def select_bn_centered_candidate_screening_combo(
    bn_slice_benchmark_df: pd.DataFrame,
    cfg: dict,
    fallback_feature_set: str | None = None,
    fallback_model_type: str | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        'enabled': False,
        'selection_scope': 'bn_slice_candidate_compatible_best',
        'selection_source_artifact': 'bn_slice_benchmark_results.csv',
        'ranking_artifact': 'demo_candidate_bn_centered_ranking.csv',
        'feature_set': None,
        'feature_family': None,
        'model_type': None,
        'benchmark_role': None,
        'mae': None,
        'rmse': None,
        'r2': None,
        'matches_general_screening_combo': None,
        'selection_note': (
            'No BN-centered alternative candidate-compatible combo was available from the '
            'BN-slice benchmark.'
        ),
    }
    benchmark_cfg = _bn_slice_benchmark_config(cfg)
    if not bool(benchmark_cfg['enabled']):
        summary['selection_note'] = (
            'BN-centered alternative ranking is disabled because the BN-slice benchmark is off.'
        )
        return summary
    if bn_slice_benchmark_df is None or bn_slice_benchmark_df.empty:
        return summary
    required_columns = {'feature_set', 'model_type', 'benchmark_status', 'mae'}
    if not required_columns.issubset(bn_slice_benchmark_df.columns):
        summary['selection_note'] = (
            'BN-centered alternative ranking is unavailable because the BN-slice benchmark '
            'artifact is missing required columns.'
        )
        return summary

    candidate_mask = bn_slice_benchmark_df['benchmark_status'].astype(str).eq('ok')
    if 'candidate_compatible' in bn_slice_benchmark_df.columns:
        candidate_mask &= bn_slice_benchmark_df['candidate_compatible'].fillna(False).astype(bool)
    else:
        candidate_mask &= bn_slice_benchmark_df['feature_set'].astype(str).map(
            feature_set_supports_formula_only_screening
        )
    if 'benchmark_role' in bn_slice_benchmark_df.columns:
        candidate_mask &= ~bn_slice_benchmark_df['benchmark_role'].astype(str).isin(
            ['global_dummy_mean_baseline', 'bn_local_reference_baseline']
        )

    candidate_result_df = bn_slice_benchmark_df.loc[candidate_mask].copy()
    if candidate_result_df.empty:
        return summary

    candidate_result_df['mae'] = candidate_result_df['mae'].astype(float)
    best_idx = candidate_result_df['mae'].idxmin()
    best_row = candidate_result_df.loc[best_idx]
    feature_set = str(best_row['feature_set'])
    model_type = str(best_row['model_type'])
    summary.update(
        {
            'enabled': True,
            'feature_set': feature_set,
            'feature_family': str(
                best_row.get('feature_family')
                if pd.notna(best_row.get('feature_family'))
                else get_feature_family(feature_set)
            ),
            'model_type': model_type,
            'benchmark_role': (
                str(best_row['benchmark_role'])
                if 'benchmark_role' in candidate_result_df.columns and pd.notna(best_row.get('benchmark_role'))
                else None
            ),
            'mae': float(best_row['mae']) if pd.notna(best_row.get('mae')) else None,
            'rmse': float(best_row['rmse']) if pd.notna(best_row.get('rmse')) else None,
            'r2': float(best_row['r2']) if pd.notna(best_row.get('r2')) else None,
            'matches_general_screening_combo': (
                feature_set == fallback_feature_set and model_type == fallback_model_type
                if fallback_feature_set is not None and fallback_model_type is not None
                else None
            ),
            'selection_note': (
                'BN-centered alternative ranking uses the lowest-MAE candidate-compatible '
                'formula-only combo under the leave-one-BN-formula-out benchmark and scores '
                'candidates with that single model rather than the general ensemble disagreement '
                'view. This is a comparison/control artifact, not a replacement for the default '
                'ranking.'
            ),
        }
    )
    return summary


def _split_pipe_delimited_values(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str):
        value = str(value)
    return _ordered_values([item.strip() for item in value.split('|') if item and item.strip()])

