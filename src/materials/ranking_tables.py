from __future__ import annotations

import os
from pathlib import Path
import re

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ai_for_bn_mplconfig')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runtime.io_utils import make_json_safe, write_json_file
from materials.data import load_cached_raw_record_lookup
from materials.constants import *
from materials.candidate_space import *
from materials.feature_building import *
from materials.benchmarking import *
from materials.common import *
from materials.common import _decision_policy_config, _ranking_stability_config

def _robustness_row_payload(robustness_df: pd.DataFrame, mask) -> dict | None:
    if robustness_df.empty:
        return None
    row_df = robustness_df.loc[mask, ROBUSTNESS_METRIC_COLUMNS]
    if row_df.empty:
        return None
    payload = row_df.iloc[0].to_dict()
    cleaned = {}
    for key, value in payload.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, (int, float, bool, str)):
            cleaned[key] = value
        else:
            cleaned[key] = value.item() if hasattr(value, 'item') else value
    return cleaned


def _prepare_candidate_ranking_df(candidate_df: pd.DataFrame, formula_col: str) -> pd.DataFrame:
    if candidate_df is None or candidate_df.empty or formula_col not in candidate_df.columns:
        return pd.DataFrame(columns=[formula_col, 'ranking_rank'])
    ranking_df = candidate_df.copy()
    ranking_df[formula_col] = ranking_df[formula_col].astype(str)
    if 'ranking_rank' in ranking_df.columns:
        ranking_df = ranking_df.sort_values('ranking_rank', ascending=True, kind='stable')
    else:
        ranking_df = ranking_df.reset_index(drop=True)
        ranking_df['ranking_rank'] = pd.RangeIndex(start=1, stop=len(ranking_df) + 1)
    return ranking_df[[formula_col, 'ranking_rank']].reset_index(drop=True)



def _candidate_ranking_comparison_payload(
    general_candidate_df: pd.DataFrame,
    alternative_candidate_df: pd.DataFrame,
    formula_col: str,
    top_k: int,
) -> dict[str, object]:
    general_ranking_df = _prepare_candidate_ranking_df(general_candidate_df, formula_col)
    alternative_ranking_df = _prepare_candidate_ranking_df(alternative_candidate_df, formula_col)
    if general_ranking_df.empty or alternative_ranking_df.empty:
        return {
            'top_k': int(top_k),
            'top_k_overlap_count': None,
            'top_k_overlap_formulas': [],
            'top_k_general_only_formulas': [],
            'top_k_bn_centered_only_formulas': [],
            'general_top_k_formulas': [],
            'bn_centered_top_k_formulas': [],
            'mean_absolute_rank_shift': None,
            'max_absolute_rank_shift': None,
            'max_absolute_rank_shift_formula': None,
            'spearman_rank_correlation': None,
            'kendall_tau': None,
        }

    top_k = max(int(top_k), 1)
    general_top_formulas = general_ranking_df.head(top_k)[formula_col].tolist()
    alternative_top_formulas = alternative_ranking_df.head(top_k)[formula_col].tolist()
    alternative_top_formula_set = set(alternative_top_formulas)
    general_top_formula_set = set(general_top_formulas)
    overlap_formulas = [
        formula for formula in general_top_formulas if formula in alternative_top_formula_set
    ]
    general_only_formulas = [
        formula for formula in general_top_formulas if formula not in alternative_top_formula_set
    ]
    alternative_only_formulas = [
        formula for formula in alternative_top_formulas if formula not in general_top_formula_set
    ]

    rank_comparison_df = general_ranking_df.rename(
        columns={'ranking_rank': 'general_ranking_rank'}
    ).merge(
        alternative_ranking_df.rename(columns={'ranking_rank': 'bn_centered_ranking_rank'}),
        on=formula_col,
        how='inner',
    )
    rank_comparison_df['absolute_rank_shift'] = (
        rank_comparison_df['general_ranking_rank'] - rank_comparison_df['bn_centered_ranking_rank']
    ).abs()
    mean_absolute_rank_shift = (
        float(rank_comparison_df['absolute_rank_shift'].mean())
        if not rank_comparison_df.empty
        else None
    )
    max_absolute_rank_shift = None
    max_absolute_rank_shift_formula = None
    spearman_rank_correlation = None
    kendall_tau = None
    if not rank_comparison_df.empty:
        max_idx = rank_comparison_df['absolute_rank_shift'].idxmax()
        max_absolute_rank_shift = float(rank_comparison_df.loc[max_idx, 'absolute_rank_shift'])
        max_absolute_rank_shift_formula = str(rank_comparison_df.loc[max_idx, formula_col])
        if len(rank_comparison_df) > 1:
            spearman_value = rank_comparison_df['general_ranking_rank'].corr(
                rank_comparison_df['bn_centered_ranking_rank'],
                method='spearman',
            )
            kendall_value = rank_comparison_df['general_ranking_rank'].corr(
                rank_comparison_df['bn_centered_ranking_rank'],
                method='kendall',
            )
            spearman_rank_correlation = (
                float(spearman_value) if pd.notna(spearman_value) else None
            )
            kendall_tau = float(kendall_value) if pd.notna(kendall_value) else None

    return {
        'top_k': int(top_k),
        'top_k_overlap_count': int(len(overlap_formulas)),
        'top_k_overlap_formulas': overlap_formulas,
        'top_k_general_only_formulas': general_only_formulas,
        'top_k_bn_centered_only_formulas': alternative_only_formulas,
        'general_top_k_formulas': general_top_formulas,
        'bn_centered_top_k_formulas': alternative_top_formulas,
        'mean_absolute_rank_shift': mean_absolute_rank_shift,
        'max_absolute_rank_shift': max_absolute_rank_shift,
        'max_absolute_rank_shift_formula': max_absolute_rank_shift_formula,
        'spearman_rank_correlation': spearman_rank_correlation,
        'kendall_tau': kendall_tau,
    }


def _build_bn_candidate_compatible_evaluation_table(
    bn_slice_benchmark_df: pd.DataFrame,
    bn_family_benchmark_df: pd.DataFrame | None = None,
    bn_stratified_error_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        'benchmark_role',
        'feature_set',
        'feature_family',
        'model_type',
        'candidate_compatible',
        'selected_by_validation',
        'bn_slice_train_scope',
        'bn_formula_count',
        'bn_row_count',
        'completed_holds',
        'mae',
        'rmse',
        'r2',
        'global_dummy_mae',
        'mae_minus_global_dummy',
        'beats_global_dummy',
        'family_holdout_mae',
        'family_holdout_rmse',
        'family_holdout_r2',
        'family_holdout_beats_global_dummy',
        'grouped_bn_mae',
        'grouped_non_bn_mae',
        'grouped_bn_to_non_bn_mae_ratio',
        'screening_eligible',
        'is_best_candidate_compatible',
        'is_best_candidate_compatible_family_holdout',
    ]
    if bn_slice_benchmark_df is None or bn_slice_benchmark_df.empty:
        return pd.DataFrame(columns=columns)

    evaluation_df = bn_slice_benchmark_df.copy()
    if 'candidate_compatible' not in evaluation_df.columns:
        evaluation_df['candidate_compatible'] = evaluation_df['feature_set'].astype(str).map(
            feature_set_supports_formula_only_screening
        )
    evaluation_df['screening_eligible'] = (
        evaluation_df['candidate_compatible'].fillna(False).astype(bool)
        | evaluation_df['benchmark_role'].astype(str).isin(
            ['global_dummy_mean_baseline', 'bn_local_reference_baseline']
        )
    )
    evaluation_df = evaluation_df.loc[
        evaluation_df['screening_eligible'].fillna(False).astype(bool)
    ].copy()
    if evaluation_df.empty:
        return pd.DataFrame(columns=columns)

    global_dummy_mae = None
    global_dummy_rows = evaluation_df.loc[
        evaluation_df['benchmark_role'].astype(str).eq('global_dummy_mean_baseline')
        & evaluation_df['mae'].notna()
    ]
    if not global_dummy_rows.empty:
        global_dummy_mae = float(global_dummy_rows.iloc[0]['mae'])
    evaluation_df['global_dummy_mae'] = global_dummy_mae
    evaluation_df['mae_minus_global_dummy'] = np.nan
    evaluation_df['beats_global_dummy'] = False
    if global_dummy_mae is not None:
        evaluation_df['mae_minus_global_dummy'] = (
            pd.to_numeric(evaluation_df['mae'], errors='coerce') - global_dummy_mae
        )
        evaluation_df['beats_global_dummy'] = (
            pd.to_numeric(evaluation_df['mae'], errors='coerce') < global_dummy_mae
        )

    evaluation_df['is_best_candidate_compatible'] = False
    candidate_rows = evaluation_df.loc[
        evaluation_df['candidate_compatible'].fillna(False).astype(bool)
        & ~evaluation_df['benchmark_role'].astype(str).isin(
            ['global_dummy_mean_baseline', 'bn_local_reference_baseline']
        )
        & evaluation_df['mae'].notna()
    ].copy()
    if not candidate_rows.empty:
        best_idx = candidate_rows['mae'].astype(float).idxmin()
        evaluation_df.loc[best_idx, 'is_best_candidate_compatible'] = True

    key_columns = ['benchmark_role', 'feature_set', 'feature_family', 'model_type']
    if bn_family_benchmark_df is not None and not bn_family_benchmark_df.empty:
        family_df = bn_family_benchmark_df.copy()
        family_keep = key_columns + [
            'mae',
            'rmse',
            'r2',
        ]
        available_family_keep = [column for column in family_keep if column in family_df.columns]
        family_df = family_df[available_family_keep].rename(
            columns={
                'mae': 'family_holdout_mae',
                'rmse': 'family_holdout_rmse',
                'r2': 'family_holdout_r2',
            }
        )
        evaluation_df = evaluation_df.merge(family_df, on=key_columns, how='left')

        family_dummy_mae = None
        family_dummy_rows = bn_family_benchmark_df.loc[
            bn_family_benchmark_df['benchmark_role'].astype(str).eq('global_dummy_mean_baseline')
            & pd.to_numeric(bn_family_benchmark_df['mae'], errors='coerce').notna()
        ]
        if not family_dummy_rows.empty:
            family_dummy_mae = float(pd.to_numeric(family_dummy_rows.iloc[0]['mae'], errors='coerce'))
        evaluation_df['family_holdout_beats_global_dummy'] = False
        if family_dummy_mae is not None:
            evaluation_df['family_holdout_beats_global_dummy'] = (
                pd.to_numeric(evaluation_df['family_holdout_mae'], errors='coerce') < family_dummy_mae
            )
    else:
        evaluation_df['family_holdout_mae'] = np.nan
        evaluation_df['family_holdout_rmse'] = np.nan
        evaluation_df['family_holdout_r2'] = np.nan
        evaluation_df['family_holdout_beats_global_dummy'] = pd.NA

    evaluation_df['is_best_candidate_compatible_family_holdout'] = False
    if 'family_holdout_mae' in evaluation_df.columns:
        family_candidate_rows = evaluation_df.loc[
            evaluation_df['candidate_compatible'].fillna(False).astype(bool)
            & ~evaluation_df['benchmark_role'].astype(str).isin(
                ['global_dummy_mean_baseline', 'bn_local_reference_baseline']
            )
            & pd.to_numeric(evaluation_df['family_holdout_mae'], errors='coerce').notna()
        ].copy()
        if not family_candidate_rows.empty:
            family_best_idx = pd.to_numeric(
                family_candidate_rows['family_holdout_mae'], errors='coerce'
            ).idxmin()
            evaluation_df.loc[family_best_idx, 'is_best_candidate_compatible_family_holdout'] = True

    if bn_stratified_error_df is not None and not bn_stratified_error_df.empty:
        stratified_df = bn_stratified_error_df.copy()
        stratified_keep = key_columns + [
            'bn_mae',
            'non_bn_mae',
            'bn_to_non_bn_mae_ratio',
        ]
        available_stratified_keep = [column for column in stratified_keep if column in stratified_df.columns]
        stratified_df = stratified_df[available_stratified_keep].rename(
            columns={
                'bn_mae': 'grouped_bn_mae',
                'non_bn_mae': 'grouped_non_bn_mae',
                'bn_to_non_bn_mae_ratio': 'grouped_bn_to_non_bn_mae_ratio',
            }
        )
        evaluation_df = evaluation_df.merge(stratified_df, on=key_columns, how='left')
    else:
        evaluation_df['grouped_bn_mae'] = np.nan
        evaluation_df['grouped_non_bn_mae'] = np.nan
        evaluation_df['grouped_bn_to_non_bn_mae_ratio'] = np.nan

    evaluation_df = evaluation_df.sort_values(
        [
            'screening_eligible',
            'candidate_compatible',
            'mae',
            'family_holdout_mae',
            'benchmark_role',
            'feature_set',
            'model_type',
        ],
        ascending=[False, False, True, True, True, True, True],
        kind='stable',
    ).reset_index(drop=True)
    for column in columns:
        if column not in evaluation_df.columns:
            evaluation_df[column] = pd.NA
    return evaluation_df[columns]



def _build_bn_evaluation_matrix_table(
    bn_slice_benchmark_df: pd.DataFrame,
    bn_family_benchmark_df: pd.DataFrame,
    bn_stratified_error_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        'benchmark_role',
        'feature_set',
        'feature_family',
        'model_type',
        'candidate_compatible',
        'selected_by_validation',
        'screening_eligible',
        'formula_holdout_mae',
        'formula_holdout_rmse',
        'formula_holdout_r2',
        'formula_holdout_beats_global_dummy',
        'family_holdout_mae',
        'family_holdout_rmse',
        'family_holdout_r2',
        'family_holdout_beats_global_dummy',
        'grouped_bn_mae',
        'grouped_non_bn_mae',
        'grouped_bn_to_non_bn_mae_ratio',
        'is_best_candidate_compatible_formula_holdout',
        'is_best_candidate_compatible_family_holdout',
    ]

    frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    key_columns = ['benchmark_role', 'feature_set', 'feature_family', 'model_type']
    metadata_columns = key_columns + ['candidate_compatible', 'selected_by_validation']
    if bn_slice_benchmark_df is not None and not bn_slice_benchmark_df.empty:
        formula_df = bn_slice_benchmark_df.copy()
        if 'candidate_compatible' not in formula_df.columns:
            formula_df['candidate_compatible'] = formula_df['feature_set'].astype(str).map(
                feature_set_supports_formula_only_screening
            )
        if 'selected_by_validation' not in formula_df.columns:
            formula_df['selected_by_validation'] = False
        formula_df = formula_df.rename(
            columns={
                'mae': 'formula_holdout_mae',
                'rmse': 'formula_holdout_rmse',
                'r2': 'formula_holdout_r2',
            }
        )
        metadata_frames.append(formula_df[metadata_columns].copy())
        frames.append(formula_df[key_columns + [
            'formula_holdout_mae',
            'formula_holdout_rmse',
            'formula_holdout_r2',
        ]].copy())
    if bn_family_benchmark_df is not None and not bn_family_benchmark_df.empty:
        family_df = bn_family_benchmark_df.copy()
        if 'candidate_compatible' not in family_df.columns:
            family_df['candidate_compatible'] = family_df['feature_set'].astype(str).map(
                feature_set_supports_formula_only_screening
            )
        if 'selected_by_validation' not in family_df.columns:
            family_df['selected_by_validation'] = False
        family_df = family_df.rename(
            columns={
                'mae': 'family_holdout_mae',
                'rmse': 'family_holdout_rmse',
                'r2': 'family_holdout_r2',
            }
        )
        metadata_frames.append(family_df[metadata_columns].copy())
        frames.append(family_df[key_columns + [
            'family_holdout_mae',
            'family_holdout_rmse',
            'family_holdout_r2',
        ]].copy())
    if bn_stratified_error_df is not None and not bn_stratified_error_df.empty:
        stratified_df = bn_stratified_error_df.copy()
        if 'candidate_compatible' not in stratified_df.columns:
            stratified_df['candidate_compatible'] = stratified_df['feature_set'].astype(str).map(
                feature_set_supports_formula_only_screening
            )
        if 'selected_by_validation' not in stratified_df.columns:
            stratified_df['selected_by_validation'] = False
        stratified_df = stratified_df.rename(
            columns={
                'bn_mae': 'grouped_bn_mae',
                'non_bn_mae': 'grouped_non_bn_mae',
                'bn_to_non_bn_mae_ratio': 'grouped_bn_to_non_bn_mae_ratio',
            }
        )
        metadata_frames.append(stratified_df[metadata_columns].copy())
        frames.append(stratified_df[key_columns + [
            'grouped_bn_mae',
            'grouped_non_bn_mae',
            'grouped_bn_to_non_bn_mae_ratio',
        ]].copy())

    if not frames:
        return pd.DataFrame(columns=columns)

    base_df = pd.concat(metadata_frames, ignore_index=True).drop_duplicates()
    if 'candidate_compatible' not in base_df.columns:
        base_df['candidate_compatible'] = base_df['feature_set'].astype(str).map(
            feature_set_supports_formula_only_screening
        )
    base_df['candidate_compatible'] = base_df['candidate_compatible'].fillna(False).astype(bool)
    if 'selected_by_validation' not in base_df.columns:
        base_df['selected_by_validation'] = False
    base_df['selected_by_validation'] = base_df['selected_by_validation'].fillna(False).astype(bool)
    base_df = base_df[key_columns + ['candidate_compatible', 'selected_by_validation']].drop_duplicates()
    for frame in frames:
        merge_columns = [column for column in frame.columns if column not in key_columns]
        base_df = base_df.merge(frame[key_columns + merge_columns], on=key_columns, how='left')
    base_df['screening_eligible'] = (
        base_df['candidate_compatible']
        | base_df['benchmark_role'].astype(str).isin(
            ['global_dummy_mean_baseline', 'bn_local_reference_baseline', 'dummy_baseline']
        )
    )

    formula_holdout_mae_series = pd.to_numeric(
        base_df['formula_holdout_mae'], errors='coerce'
    ) if 'formula_holdout_mae' in base_df.columns else pd.Series(np.nan, index=base_df.index, dtype=float)
    family_holdout_mae_series = pd.to_numeric(
        base_df['family_holdout_mae'], errors='coerce'
    ) if 'family_holdout_mae' in base_df.columns else pd.Series(np.nan, index=base_df.index, dtype=float)

    formula_dummy_mae = None
    formula_dummy_rows = base_df.loc[
        base_df['benchmark_role'].astype(str).eq('global_dummy_mean_baseline')
        & formula_holdout_mae_series.notna()
    ]
    if not formula_dummy_rows.empty:
        formula_dummy_mae = float(formula_dummy_rows.iloc[0]['formula_holdout_mae'])
    base_df['formula_holdout_beats_global_dummy'] = False
    if formula_dummy_mae is not None:
        base_df['formula_holdout_beats_global_dummy'] = formula_holdout_mae_series < formula_dummy_mae

    family_dummy_mae = None
    family_dummy_rows = base_df.loc[
        base_df['benchmark_role'].astype(str).eq('global_dummy_mean_baseline')
        & family_holdout_mae_series.notna()
    ]
    if not family_dummy_rows.empty:
        family_dummy_mae = float(family_dummy_rows.iloc[0]['family_holdout_mae'])
    base_df['family_holdout_beats_global_dummy'] = False
    if family_dummy_mae is not None:
        base_df['family_holdout_beats_global_dummy'] = family_holdout_mae_series < family_dummy_mae

    base_df['is_best_candidate_compatible_formula_holdout'] = False
    candidate_formula_df = base_df.loc[
        base_df['candidate_compatible']
        & ~base_df['benchmark_role'].astype(str).isin(
            ['global_dummy_mean_baseline', 'bn_local_reference_baseline', 'dummy_baseline']
        )
        & formula_holdout_mae_series.notna()
    ].copy()
    if not candidate_formula_df.empty:
        best_idx = pd.to_numeric(candidate_formula_df['formula_holdout_mae'], errors='coerce').idxmin()
        base_df.loc[best_idx, 'is_best_candidate_compatible_formula_holdout'] = True

    base_df['is_best_candidate_compatible_family_holdout'] = False
    candidate_family_df = base_df.loc[
        base_df['candidate_compatible']
        & ~base_df['benchmark_role'].astype(str).isin(
            ['global_dummy_mean_baseline', 'bn_local_reference_baseline', 'dummy_baseline']
        )
        & family_holdout_mae_series.notna()
    ].copy()
    if not candidate_family_df.empty:
        best_idx = pd.to_numeric(candidate_family_df['family_holdout_mae'], errors='coerce').idxmin()
        base_df.loc[best_idx, 'is_best_candidate_compatible_family_holdout'] = True

    for column in columns:
        if column not in base_df.columns:
            base_df[column] = pd.NA
    base_df = base_df.sort_values(
        [
            'screening_eligible',
            'candidate_compatible',
            'formula_holdout_mae',
            'family_holdout_mae',
            'benchmark_role',
            'feature_set',
            'model_type',
        ],
        ascending=[False, False, True, True, True, True, True],
        kind='stable',
    ).reset_index(drop=True)
    return base_df[columns]



def _quantile_from_group(values: pd.Series, quantile: float) -> float | None:
    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
    if numeric_values.empty:
        return None
    return float(numeric_values.quantile(quantile))



def _build_candidate_ranking_source_predictions(
    candidate_prediction_member_df: pd.DataFrame | None,
    candidate_grouped_robustness_member_df: pd.DataFrame | None,
    bn_centered_grouped_robustness_member_df: pd.DataFrame | None,
    *,
    formula_col: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_df, source_family in (
        (candidate_prediction_member_df, 'candidate_full_fit'),
        (candidate_grouped_robustness_member_df, 'default_group_kfold'),
        (bn_centered_grouped_robustness_member_df, 'bn_centered_group_kfold'),
    ):
        if source_df is None or source_df.empty:
            continue
        working_df = source_df.copy()
        if formula_col not in working_df.columns and 'formula' in working_df.columns:
            working_df[formula_col] = working_df['formula'].astype(str)
        working_df[formula_col] = working_df[formula_col].astype(str)
        if 'prediction_source' not in working_df.columns:
            working_df['prediction_source'] = source_family
        if 'prediction_source_family' not in working_df.columns:
            working_df['prediction_source_family'] = source_family
        frames.append(
            working_df[[
                formula_col,
                'prediction_source',
                'prediction_source_family',
                'feature_set',
                'model_type',
                'prediction',
            ]].copy()
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                formula_col,
                'prediction_source',
                'prediction_source_family',
                'feature_set',
                'model_type',
                'prediction',
                'source_rank',
            ]
        )
    prediction_df = pd.concat(frames, ignore_index=True)
    prediction_df = prediction_df.sort_values(
        ['prediction_source', 'prediction', formula_col],
        ascending=[True, False, True],
        kind='stable',
    ).reset_index(drop=True)
    prediction_df['source_rank'] = prediction_df.groupby('prediction_source').cumcount() + 1
    return prediction_df



def _candidate_ranking_uncertainty_table(
    candidate_df: pd.DataFrame,
    *,
    formula_col: str,
    cfg: dict,
    candidate_prediction_member_df: pd.DataFrame | None = None,
    candidate_grouped_robustness_member_df: pd.DataFrame | None = None,
    bn_centered_grouped_robustness_member_df: pd.DataFrame | None = None,
    bn_centered_candidate_df: pd.DataFrame | None = None,
    structure_followup_shortlist_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    stability_cfg = _ranking_stability_config(cfg)
    decision_cfg = _decision_policy_config(cfg)
    top_k_reference = int((cfg.get('screening') or {}).get('top_k', 10))
    requested_top_k_values = sorted({*stability_cfg['top_k_values'], top_k_reference})
    source_prediction_df = _build_candidate_ranking_source_predictions(
        candidate_prediction_member_df,
        candidate_grouped_robustness_member_df,
        bn_centered_grouped_robustness_member_df,
        formula_col=formula_col,
    )
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame(), {
            'source_count': 0,
            'top_k_reference': top_k_reference,
            'top_k_values': requested_top_k_values,
            'prediction_std_abstain_threshold': None,
            'rank_std_abstain_threshold': None,
            'abstained_candidate_count': 0,
            'final_action_counts': {},
        }

    summary_df = candidate_df.copy()
    summary_df[formula_col] = summary_df[formula_col].astype(str)
    selected_columns = [
        formula_col,
        'ranking_rank',
        'ranking_score',
        'predicted_band_gap',
        'ensemble_predicted_band_gap_mean',
        'ensemble_predicted_band_gap_std',
        'grouped_robustness_predicted_band_gap_mean',
        'grouped_robustness_predicted_band_gap_std',
        'domain_support_percentile',
        'domain_support_mean_k_distance',
        'bn_support_percentile',
        'bn_support_mean_k_distance',
        'chemical_plausibility_pass',
        'candidate_novelty_bucket',
        'proposal_shortlist_selected',
        'proposal_shortlist_rank',
        'extrapolation_shortlist_selected',
        'extrapolation_shortlist_rank',
    ]
    selected_columns = [column for column in selected_columns if column in summary_df.columns]
    summary_df = summary_df[selected_columns].copy()

    if not source_prediction_df.empty:
        aggregated_df = source_prediction_df.groupby(formula_col, as_index=False).agg(
            ranking_source_count=('prediction_source', 'nunique'),
            predicted_band_gap_mean=('prediction', 'mean'),
            predicted_band_gap_std=('prediction', 'std'),
            rank_mean=('source_rank', 'mean'),
            rank_std=('source_rank', 'std'),
            rank_min=('source_rank', 'min'),
            rank_max=('source_rank', 'max'),
        )
        aggregated_df['predicted_band_gap_std'] = (
            aggregated_df['predicted_band_gap_std'].fillna(0.0).astype(float)
        )
        aggregated_df['rank_std'] = aggregated_df['rank_std'].fillna(0.0).astype(float)
        lower_quantile = float(stability_cfg['prediction_interval_lower_quantile'])
        upper_quantile = float(stability_cfg['prediction_interval_upper_quantile'])
        lower_df = (
            source_prediction_df.groupby(formula_col)['prediction']
            .apply(lambda values: _quantile_from_group(values, lower_quantile))
            .reset_index(name='predicted_band_gap_interval_lower')
        )
        upper_df = (
            source_prediction_df.groupby(formula_col)['prediction']
            .apply(lambda values: _quantile_from_group(values, upper_quantile))
            .reset_index(name='predicted_band_gap_interval_upper')
        )
        aggregated_df = aggregated_df.merge(lower_df, on=formula_col, how='left').merge(
            upper_df,
            on=formula_col,
            how='left',
        )
        for top_k in requested_top_k_values:
            selection_df = (
                source_prediction_df.assign(
                    _selected=source_prediction_df['source_rank'].le(int(top_k)).astype(float)
                )
                .groupby(formula_col, as_index=False)['_selected']
                .mean()
                .rename(columns={'_selected': f'top_{int(top_k)}_selection_frequency'})
            )
            aggregated_df = aggregated_df.merge(selection_df, on=formula_col, how='left')
        summary_df = summary_df.merge(aggregated_df, on=formula_col, how='left')
    else:
        summary_df['ranking_source_count'] = 0
        predicted_mean_series = summary_df.get('ensemble_predicted_band_gap_mean')
        if predicted_mean_series is None:
            predicted_mean_series = summary_df.get(
                'predicted_band_gap',
                pd.Series(np.nan, index=summary_df.index, dtype=float),
            )
        predicted_mean_series = pd.to_numeric(predicted_mean_series, errors='coerce')
        predicted_std_series = summary_df.get('ensemble_predicted_band_gap_std')
        if predicted_std_series is None:
            predicted_std_series = pd.Series(0.0, index=summary_df.index, dtype=float)
        predicted_std_series = pd.to_numeric(predicted_std_series, errors='coerce').fillna(0.0)
        rank_series = pd.to_numeric(
            summary_df.get('ranking_rank', pd.Series(np.nan, index=summary_df.index, dtype=float)),
            errors='coerce',
        )
        summary_df['predicted_band_gap_mean'] = predicted_mean_series
        summary_df['predicted_band_gap_std'] = predicted_std_series
        summary_df['predicted_band_gap_interval_lower'] = summary_df['predicted_band_gap_mean']
        summary_df['predicted_band_gap_interval_upper'] = summary_df['predicted_band_gap_mean']
        summary_df['rank_mean'] = rank_series
        summary_df['rank_std'] = 0.0
        summary_df['rank_min'] = rank_series
        summary_df['rank_max'] = rank_series
        for top_k in requested_top_k_values:
            summary_df[f'top_{int(top_k)}_selection_frequency'] = rank_series.le(int(top_k)).astype(float)

    if bn_centered_candidate_df is not None and not bn_centered_candidate_df.empty:
        bn_centered_df = bn_centered_candidate_df[[formula_col, 'ranking_rank']].copy()
        bn_centered_df[formula_col] = bn_centered_df[formula_col].astype(str)
        bn_centered_df = bn_centered_df.rename(columns={'ranking_rank': 'bn_centered_ranking_rank'})
        summary_df = summary_df.merge(bn_centered_df, on=formula_col, how='left')
    else:
        summary_df['bn_centered_ranking_rank'] = pd.NA

    if structure_followup_shortlist_df is not None and not structure_followup_shortlist_df.empty:
        followup_columns = [
            formula_col,
            'structure_followup_priority_score',
            'structure_followup_best_queue_rank',
            'structure_followup_best_action_label',
            'structure_followup_readiness_label',
            'structure_followup_shortlist_selected',
            'structure_followup_shortlist_rank',
        ]
        available_columns = [column for column in followup_columns if column in structure_followup_shortlist_df.columns]
        followup_df = structure_followup_shortlist_df[available_columns].copy()
        followup_df[formula_col] = followup_df[formula_col].astype(str)
        summary_df = summary_df.merge(followup_df, on=formula_col, how='left')
    else:
        summary_df['structure_followup_priority_score'] = np.nan
        summary_df['structure_followup_best_queue_rank'] = np.nan
        summary_df['structure_followup_best_action_label'] = pd.NA
        summary_df['structure_followup_readiness_label'] = pd.NA
        summary_df['structure_followup_shortlist_selected'] = False
        summary_df['structure_followup_shortlist_rank'] = pd.NA

    top_10_frequency_column = 'top_10_selection_frequency'
    if top_10_frequency_column not in summary_df.columns:
        fallback_top_k = requested_top_k_values[-1]
        top_10_frequency_column = f'top_{int(fallback_top_k)}_selection_frequency'

    prediction_std_threshold = None
    rank_std_threshold = None
    numeric_prediction_std = pd.to_numeric(summary_df['predicted_band_gap_std'], errors='coerce').dropna()
    if not numeric_prediction_std.empty:
        prediction_std_threshold = float(
            numeric_prediction_std.quantile(float(decision_cfg['prediction_std_above_quantile']))
        )
    numeric_rank_std = pd.to_numeric(summary_df['rank_std'], errors='coerce').dropna()
    if not numeric_rank_std.empty:
        rank_std_threshold = float(
            numeric_rank_std.quantile(float(decision_cfg['rank_std_above_quantile']))
        )

    abstain_reasons = []
    abstain_flags = []
    final_action_labels = []
    for _, row in summary_df.iterrows():
        chemical_plausibility_pass = bool(row.get('chemical_plausibility_pass', True))
        candidate_novelty_bucket = str(row.get('candidate_novelty_bucket', ''))
        reasons: list[str] = []
        domain_support_percentile = pd.to_numeric(
            pd.Series([row.get('domain_support_percentile')]), errors='coerce'
        ).iloc[0]
        bn_support_percentile = pd.to_numeric(
            pd.Series([row.get('bn_support_percentile')]), errors='coerce'
        ).iloc[0]
        prediction_std_value = pd.to_numeric(
            pd.Series([row.get('predicted_band_gap_std')]), errors='coerce'
        ).iloc[0]
        rank_std_value = pd.to_numeric(pd.Series([row.get('rank_std')]), errors='coerce').iloc[0]
        top_10_frequency = pd.to_numeric(
            pd.Series([row.get(top_10_frequency_column)]), errors='coerce'
        ).iloc[0]
        if chemical_plausibility_pass:
            if pd.notna(domain_support_percentile) and (
                float(domain_support_percentile)
                < float(decision_cfg['global_support_abstain_below_percentile'])
            ):
                reasons.append('outside_global_support')
            if pd.notna(bn_support_percentile) and (
                float(bn_support_percentile)
                < float(decision_cfg['bn_support_abstain_below_percentile'])
            ):
                reasons.append('outside_bn_local_support')
            if prediction_std_threshold is not None and pd.notna(prediction_std_value) and (
                float(prediction_std_value) > prediction_std_threshold
            ):
                reasons.append('high_prediction_uncertainty')
            if rank_std_threshold is not None and pd.notna(rank_std_value) and (
                float(rank_std_value) > rank_std_threshold
            ):
                reasons.append('high_rank_instability')
            if pd.notna(top_10_frequency) and (
                float(top_10_frequency) < float(decision_cfg['minimum_top_10_selection_frequency'])
            ):
                reasons.append('low_top_10_selection_frequency')
        abstain_flag = bool(chemical_plausibility_pass and len(reasons) > 0)
        if not chemical_plausibility_pass:
            final_action_label = 'reject_formula_level'
        elif candidate_novelty_bucket == NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY:
            final_action_label = 'reuse_reference_control'
        elif abstain_flag:
            final_action_label = 'abstain_model_unreliable'
        elif bool(row.get('structure_followup_shortlist_selected', False)):
            final_action_label = 'low_risk_followup'
        else:
            final_action_label = 'uncertain_but_interesting'
        abstain_reasons.append('|'.join(reasons))
        abstain_flags.append(abstain_flag)
        final_action_labels.append(final_action_label)

    summary_df['abstain_flag'] = abstain_flags
    summary_df['reason_for_abstention'] = abstain_reasons
    summary_df['final_action_label'] = final_action_labels
    if 'ranking_rank' not in summary_df.columns:
        summary_df['ranking_rank'] = np.arange(1, len(summary_df) + 1, dtype=int)
    summary_df = summary_df.sort_values(
        ['ranking_rank', formula_col],
        ascending=[True, True],
        kind='stable',
    ).reset_index(drop=True)

    final_action_counts = summary_df['final_action_label'].value_counts().to_dict()
    final_action_counts = {str(key): int(value) for key, value in final_action_counts.items()}
    summary = {
        'source_count': int(source_prediction_df['prediction_source'].nunique()) if not source_prediction_df.empty else 0,
        'top_k_reference': top_k_reference,
        'top_k_values': [int(value) for value in requested_top_k_values],
        'prediction_interval_lower_quantile': float(stability_cfg['prediction_interval_lower_quantile']),
        'prediction_interval_upper_quantile': float(stability_cfg['prediction_interval_upper_quantile']),
        'prediction_std_abstain_threshold': prediction_std_threshold,
        'rank_std_abstain_threshold': rank_std_threshold,
        'abstained_candidate_count': int(summary_df['abstain_flag'].fillna(False).astype(bool).sum()),
        'final_action_counts': final_action_counts,
    }
    return summary_df, summary



def _bn_slice_benchmark_row_payload(bn_slice_benchmark_df: pd.DataFrame, mask) -> dict | None:
    if bn_slice_benchmark_df.empty:
        return None
    row_df = bn_slice_benchmark_df.loc[mask, BN_SLICE_BENCHMARK_COLUMNS]
    if row_df.empty:
        return None
    payload = row_df.iloc[0].to_dict()
    cleaned = {}
    for key, value in payload.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, (int, float, bool, str)):
            cleaned[key] = value
        else:
            cleaned[key] = value.item() if hasattr(value, 'item') else value
    return cleaned



def _bn_family_benchmark_row_payload(bn_family_benchmark_df: pd.DataFrame, mask) -> dict | None:
    if bn_family_benchmark_df.empty:
        return None
    row_df = bn_family_benchmark_df.loc[mask, BN_FAMILY_BENCHMARK_COLUMNS]
    if row_df.empty:
        return None
    payload = row_df.iloc[0].to_dict()
    cleaned = {}
    for key, value in payload.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, (int, float, bool, str)):
            cleaned[key] = value
        else:
            cleaned[key] = value.item() if hasattr(value, 'item') else value
    return cleaned



def _bn_stratified_error_row_payload(bn_stratified_error_df: pd.DataFrame, mask) -> dict | None:
    if bn_stratified_error_df.empty:
        return None
    row_df = bn_stratified_error_df.loc[mask, BN_STRATIFIED_ERROR_COLUMNS]
    if row_df.empty:
        return None
    payload = row_df.iloc[0].to_dict()
    cleaned = {}
    for key, value in payload.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, (int, float, bool, str)):
            cleaned[key] = value
        else:
            cleaned[key] = value.item() if hasattr(value, 'item') else value
    return cleaned

