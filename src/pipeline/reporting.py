from __future__ import annotations

import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ai_for_bn_mplconfig')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.data import load_cached_raw_record_lookup
from pipeline.features import (
    _bn_family_benchmark_config,
    _bn_slice_benchmark_config,
    _bn_stratified_error_config,
    _extrapolation_shortlist_config,
    _proposal_shortlist_config,
    _structure_generation_seed_config,
    _structure_seed_edit_metadata,
    _formula_amount_map,
    BN_ANALOG_EVIDENCE_RANKING_NOTE,
    BN_BAND_GAP_ALIGNMENT_RANKING_NOTE,
    BN_SUPPORT_RANKING_NOTE,
    DOMAIN_SUPPORT_RANKING_NOTE,
    GROUPED_ROBUSTNESS_UNCERTAINTY_RANKING_NOTE,
    NOVELTY_ANNOTATION_RANKING_NOTE,
    NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION,
    NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA,
    NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY,
    feature_set_supports_formula_only_screening,
    get_feature_family,
    get_screening_ranking_metadata,
)


ROBUSTNESS_METRIC_COLUMNS = [
    'feature_set',
    'model_type',
    'robustness_status',
    'requested_folds',
    'actual_folds',
    'completed_folds',
    'mae_mean',
    'mae_std',
    'rmse_mean',
    'rmse_std',
    'r2_mean',
    'r2_std',
]
BN_SLICE_BENCHMARK_COLUMNS = [
    'feature_set',
    'feature_family',
    'model_type',
    'benchmark_role',
    'benchmark_status',
    'bn_slice_method',
    'bn_slice_train_scope',
    'bn_formula_count',
    'bn_row_count',
    'completed_holds',
    'k_neighbors',
    'mae',
    'rmse',
    'r2',
]
BN_FAMILY_BENCHMARK_COLUMNS = [
    'feature_set',
    'feature_family',
    'model_type',
    'benchmark_role',
    'benchmark_status',
    'bn_family_benchmark_method',
    'bn_family_grouping_method',
    'bn_family_train_scope',
    'bn_family_count',
    'bn_formula_count',
    'bn_row_count',
    'completed_family_holds',
    'completed_formula_holds',
    'k_neighbors',
    'mae',
    'rmse',
    'r2',
]
BN_STRATIFIED_ERROR_COLUMNS = [
    'feature_set',
    'feature_family',
    'model_type',
    'benchmark_role',
    'benchmark_status',
    'bn_stratified_error_method',
    'bn_stratified_group_column',
    'requested_folds',
    'actual_folds',
    'completed_folds',
    'bn_formula_count',
    'non_bn_formula_count',
    'bn_mae',
    'bn_rmse',
    'bn_r2',
    'non_bn_mae',
    'non_bn_rmse',
    'non_bn_r2',
    'bn_to_non_bn_mae_ratio',
]

STRUCTURE_GENERATION_JOB_PLAN_LABEL = 'prototype_substitution_enumeration_job_plan'
STRUCTURE_GENERATION_JOB_PLAN_METHOD = 'candidate_seed_to_workflow_plan'
STRUCTURE_GENERATION_JOB_PLAN_NOTE = (
    'Converts each candidate/seed pairing into a deterministic downstream workflow plan '
    'for prototype substitution, enumeration, and relaxation. This is a planning artifact, '
    'not a generated-structure or validated-structure claim.'
)
STRUCTURE_GENERATION_FIRST_PASS_QUEUE_LABEL = 'prototype_edit_recipe_first_pass_queue'
STRUCTURE_GENERATION_FIRST_PASS_QUEUE_METHOD = 'job_plan_complexity_ranked_queue'
STRUCTURE_GENERATION_FIRST_PASS_QUEUE_NOTE = (
    'Builds a deterministic first-pass queue for downstream prototype work by combining '
    'candidate ranking context, seed rank, edit complexity, and reference readiness. '
    'This is still a planning artifact, not generated-structure output.'
)
STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_LABEL = 'prototype_grounded_followup_shortlist'
STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_METHOD = 'first_pass_queue_candidate_aggregation'
STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_NOTE = (
    'Aggregates the structure-generation first-pass queue back to the candidate level so '
    'advisor-facing follow-up can prioritize formulas that are not only high-ranked, but '
    'also better supported by prototype references and lower-complexity edit paths. This '
    'remains a planning artifact, not a validated structure result.'
)
STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_LABEL = (
    'novelty_aware_prototype_followup_shortlist'
)
STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_METHOD = (
    'structure_followup_filtered_by_formula_level_extrapolation'
)
STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_NOTE = (
    'Filters the prototype-grounded follow-up view down to formula-level extrapolation '
    'candidates so downstream structure work is not dominated by rediscovery or replay of '
    'already-observed BN formulas. This remains a planning artifact, not a discovery proof.'
)


def _structure_followup_shortlist_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    shortlist_cfg = screening_cfg.get('structure_followup_shortlist', {})
    out = {
        'enabled': bool(shortlist_cfg.get('enabled', True)),
        'label': str(shortlist_cfg.get('label', STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_LABEL)),
        'method': str(shortlist_cfg.get('method', STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_METHOD)),
        'shortlist_size': int(shortlist_cfg.get('shortlist_size', 5)),
        'note': str(shortlist_cfg.get('note', STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_NOTE)),
    }
    if out['shortlist_size'] <= 0:
        raise ValueError('structure_followup_shortlist.shortlist_size must be positive')
    return out


def _structure_followup_extrapolation_shortlist_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    shortlist_cfg = screening_cfg.get('structure_followup_extrapolation_shortlist', {})
    out = {
        'enabled': bool(shortlist_cfg.get('enabled', True)),
        'label': str(
            shortlist_cfg.get(
                'label', STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_LABEL
            )
        ),
        'method': str(
            shortlist_cfg.get(
                'method', STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_METHOD
            )
        ),
        'shortlist_size': int(shortlist_cfg.get('shortlist_size', 4)),
        'required_novelty_bucket': str(
            shortlist_cfg.get(
                'required_novelty_bucket', NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION
            )
        ),
        'note': str(
            shortlist_cfg.get(
                'note', STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_NOTE
            )
        ),
    }
    if out['shortlist_size'] <= 0:
        raise ValueError('structure_followup_extrapolation_shortlist.shortlist_size must be positive')
    return out


def _ranking_stability_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    stability_cfg = screening_cfg.get('ranking_stability', {})
    top_k_values = [int(value) for value in stability_cfg.get('top_k_values', [3, 5, 10])]
    top_k_values = sorted({value for value in top_k_values if value > 0})
    if not top_k_values:
        raise ValueError('ranking_stability.top_k_values must contain at least one positive integer')
    lower_quantile = float(stability_cfg.get('prediction_interval_lower_quantile', 0.1))
    upper_quantile = float(stability_cfg.get('prediction_interval_upper_quantile', 0.9))
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError('ranking_stability prediction interval quantiles must satisfy 0 <= lower < upper <= 1')
    return {
        'enabled': bool(stability_cfg.get('enabled', True)),
        'top_k_values': top_k_values,
        'prediction_interval_lower_quantile': lower_quantile,
        'prediction_interval_upper_quantile': upper_quantile,
        'note': str(
            stability_cfg.get(
                'note',
                'Summarizes prediction and rank stability across candidate-compatible full-fit '
                'models plus grouped-fold ranking views. This is a ranking-honesty layer, not a '
                'calibrated discovery-confidence estimate.'
            )
        ),
    }



def _decision_policy_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    policy_cfg = screening_cfg.get('decision_policy', {})
    return {
        'enabled': bool(policy_cfg.get('enabled', True)),
        'global_support_abstain_below_percentile': float(
            policy_cfg.get('global_support_abstain_below_percentile', 25.0)
        ),
        'bn_support_abstain_below_percentile': float(
            policy_cfg.get('bn_support_abstain_below_percentile', 25.0)
        ),
        'prediction_std_above_quantile': float(
            policy_cfg.get('prediction_std_above_quantile', 0.75)
        ),
        'rank_std_above_quantile': float(policy_cfg.get('rank_std_above_quantile', 0.75)),
        'minimum_top_10_selection_frequency': float(
            policy_cfg.get('minimum_top_10_selection_frequency', 0.5)
        ),
        'note': str(
            policy_cfg.get(
                'note',
                'Turns candidate ranking into a lightweight decision policy by combining '
                'chemical plausibility, domain support, BN-local support, prediction/rank '
                'instability, and prototype readiness into abstention flags plus action labels. '
                'This remains heuristic and should not be interpreted as validated discovery '
                'confidence.'
            )
        ),
    }


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


def _collect_structure_generation_seed_summary(
    structure_generation_seed_df: pd.DataFrame,
    *,
    formula_col: str,
    cfg_defaults: dict[str, object],
    artifact_name: str,
    handoff_artifact_name: str,
) -> dict[str, object]:
    summary = {
        'enabled': bool(cfg_defaults['enabled']),
        'artifact': artifact_name if bool(cfg_defaults['enabled']) else None,
        'handoff_artifact': handoff_artifact_name if bool(cfg_defaults['enabled']) else None,
        'label': str(cfg_defaults['label']),
        'method': str(cfg_defaults['method']),
        'candidate_scope': str(cfg_defaults['candidate_scope']),
        'per_candidate_seed_limit': int(cfg_defaults['per_candidate_seed_limit']),
        'bn_centered_top_n': int(cfg_defaults['bn_centered_top_n']),
        'note': str(cfg_defaults['note']),
        'candidate_rows': 0,
        'seed_rows': 0,
        'seeded_candidate_rows': 0,
        'candidates_without_seed_rows': 0,
        'unique_seed_reference_formulas': 0,
        'unique_seed_reference_records': 0,
        'candidate_formulas': [],
    }
    if structure_generation_seed_df is None or structure_generation_seed_df.empty:
        return summary

    seed_df = structure_generation_seed_df.copy()
    summary['candidate_rows'] = int(seed_df[formula_col].astype(str).nunique())
    summary['seed_rows'] = int(len(seed_df))
    if 'structure_generation_seed_status' in seed_df.columns:
        ok_mask = seed_df['structure_generation_seed_status'].astype(str).eq('ok')
        summary['seeded_candidate_rows'] = int(
            seed_df.loc[ok_mask, formula_col].astype(str).nunique()
        )
        summary['candidates_without_seed_rows'] = int(
            seed_df.loc[~ok_mask, formula_col].astype(str).nunique()
        )
    if 'seed_reference_formula' in seed_df.columns:
        summary['unique_seed_reference_formulas'] = int(
            seed_df['seed_reference_formula'].dropna().astype(str).nunique()
        )
    if 'seed_reference_record_id' in seed_df.columns:
        summary['unique_seed_reference_records'] = int(
            seed_df['seed_reference_record_id'].dropna().astype(str).nunique()
        )
    formula_rows = seed_df.drop_duplicates(subset=[formula_col], keep='first')
    summary['candidate_formulas'] = formula_rows[formula_col].astype(str).tolist()
    return summary


def _json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, 'item'):
        try:
            return _json_safe_value(value.item())
        except Exception:
            pass
    return value


def _split_pipe_delimited_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if not isinstance(value, str):
        value = str(value)
    return [item.strip() for item in value.split('|') if item and item.strip()]


def _slugify_structure_generation_job_component(value: object) -> str:
    text = str(value or '').strip()
    slug = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_').lower()
    return slug or 'unknown'


def _compact_formula_amount_map(formula: str | None) -> dict[str, object]:
    if not formula:
        return {}
    amount_map = _formula_amount_map(str(formula))
    compacted: dict[str, object] = {}
    for element, amount in sorted(amount_map.items()):
        numeric_amount = float(amount)
        compacted[element] = int(round(numeric_amount)) if numeric_amount.is_integer() else numeric_amount
    return compacted


def _formula_delta_map(candidate_formula: str | None, seed_formula: str | None) -> dict[str, object]:
    candidate_map = _compact_formula_amount_map(candidate_formula)
    seed_map = _compact_formula_amount_map(seed_formula)
    deltas: dict[str, object] = {}
    for element in sorted(set(candidate_map) | set(seed_map)):
        delta = float(candidate_map.get(element, 0)) - float(seed_map.get(element, 0))
        if abs(delta) < 1e-12:
            continue
        deltas[element] = int(round(delta)) if float(delta).is_integer() else float(delta)
    return deltas


def _structure_generation_edit_operations(
    *,
    action_label: str,
    substitution_pairs: list[dict[str, object]],
    element_count_deltas: dict[str, object],
) -> list[dict[str, object]]:
    if action_label == 'reference_reuse_control':
        return [{'operation': 'reuse_reference_control'}]
    if action_label == 'manual_reference_recovery':
        return [{'operation': 'recover_reference_record'}]

    operations: list[dict[str, object]] = []
    for pair in substitution_pairs:
        operations.append(
            {
                'operation': 'substitute_element',
                'from_element': pair['from_element'],
                'to_element': pair['to_element'],
            }
        )
    for element, delta in element_count_deltas.items():
        numeric_delta = float(delta)
        if numeric_delta > 0:
            operations.append(
                {
                    'operation': 'increase_element_count',
                    'element': element,
                    'delta': int(round(numeric_delta)) if numeric_delta.is_integer() else numeric_delta,
                }
            )
        elif numeric_delta < 0:
            positive_delta = abs(numeric_delta)
            operations.append(
                {
                    'operation': 'decrease_element_count',
                    'element': element,
                    'delta': int(round(positive_delta)) if positive_delta.is_integer() else positive_delta,
                }
            )
    return operations


def _structure_generation_job_workflow_steps(action_label: str) -> list[str]:
    workflow_map = {
        'reference_reuse_control': [
            'load_reference_atoms',
            'reuse_reference_as_control_seed',
            'relax_reference_control',
            'record_control_outcome',
        ],
        'stoichiometry_adjustment_enumeration': [
            'load_reference_atoms',
            'adjust_formula_stoichiometry',
            'enumerate_stoichiometric_variants',
            'relax_candidate_variants',
            'rank_relaxed_variants',
        ],
        'element_substitution_relabeling': [
            'load_reference_atoms',
            'apply_element_substitution',
            'relax_substituted_seed',
            'record_relabeled_prototype',
        ],
        'element_substitution_plus_stoichiometry_adjustment': [
            'load_reference_atoms',
            'apply_element_substitution',
            'adjust_formula_stoichiometry',
            'enumerate_substituted_variants',
            'relax_candidate_variants',
            'rank_relaxed_variants',
        ],
        'element_insertion_enumeration': [
            'load_reference_atoms',
            'insert_or_decorate_candidate_elements',
            'enumerate_insertion_sites',
            'relax_decorated_variants',
            'rank_relaxed_variants',
        ],
        'element_removal_enumeration': [
            'load_reference_atoms',
            'remove_or_vacancy_seed_elements',
            'enumerate_vacancy_variants',
            'relax_vacancy_variants',
            'rank_relaxed_variants',
        ],
        'mixed_formula_edit_enumeration': [
            'load_reference_atoms',
            'combine_substitution_and_stoichiometry_edits',
            'enumerate_candidate_variants',
            'relax_candidate_variants',
            'rank_relaxed_variants',
        ],
        'manual_reference_recovery': [
            'recover_reference_record',
            'rebuild_candidate_seed_link',
            'resume_prototype_enumeration',
        ],
    }
    return workflow_map.get(action_label, workflow_map['mixed_formula_edit_enumeration'])


def _structure_generation_job_payload(row: pd.Series, *, formula_col: str) -> dict[str, object]:
    candidate_formula = str(row[formula_col])
    seed_rank = _json_safe_value(row.get('structure_generation_seed_rank'))
    record_id = _json_safe_value(row.get('seed_reference_record_id'))
    seed_formula = _json_safe_value(row.get('seed_reference_formula'))
    seed_status = str(row.get('structure_generation_seed_status') or 'unknown')

    edit_payload = {}
    if seed_formula is not None:
        missing_edit_columns = [
            column
            for column in (
                'seed_formula_exact_element_match',
                'seed_formula_shared_elements',
                'seed_formula_candidate_only_elements',
                'seed_formula_seed_only_elements',
                'seed_formula_element_count_l1_distance',
                'seed_formula_edit_strategy',
            )
            if column not in row.index or pd.isna(row[column])
        ]
        if missing_edit_columns:
            edit_payload = _structure_seed_edit_metadata(candidate_formula, str(seed_formula))

    edit_strategy = _json_safe_value(row.get('seed_formula_edit_strategy'))
    if edit_strategy is None:
        edit_strategy = _json_safe_value(edit_payload.get('seed_formula_edit_strategy'))
    shared_elements = _split_pipe_delimited_values(
        row.get('seed_formula_shared_elements')
        if 'seed_formula_shared_elements' in row.index and pd.notna(row.get('seed_formula_shared_elements'))
        else edit_payload.get('seed_formula_shared_elements')
    )
    candidate_only_elements = _split_pipe_delimited_values(
        row.get('seed_formula_candidate_only_elements')
        if 'seed_formula_candidate_only_elements' in row.index and pd.notna(row.get('seed_formula_candidate_only_elements'))
        else edit_payload.get('seed_formula_candidate_only_elements')
    )
    seed_only_elements = _split_pipe_delimited_values(
        row.get('seed_formula_seed_only_elements')
        if 'seed_formula_seed_only_elements' in row.index and pd.notna(row.get('seed_formula_seed_only_elements'))
        else edit_payload.get('seed_formula_seed_only_elements')
    )
    exact_element_match_value = (
        row.get('seed_formula_exact_element_match')
        if 'seed_formula_exact_element_match' in row.index and pd.notna(row.get('seed_formula_exact_element_match'))
        else edit_payload.get('seed_formula_exact_element_match')
    )
    exact_element_match = bool(exact_element_match_value) if exact_element_match_value is not None else False
    element_count_l1_distance = _json_safe_value(
        row.get('seed_formula_element_count_l1_distance')
        if 'seed_formula_element_count_l1_distance' in row.index and pd.notna(row.get('seed_formula_element_count_l1_distance'))
        else edit_payload.get('seed_formula_element_count_l1_distance')
    )

    substitution_pairs = []
    if (
        edit_strategy == 'element_substitution_or_decoration'
        and len(candidate_only_elements) == len(seed_only_elements)
    ):
        substitution_pairs = [
            {'from_element': old_element, 'to_element': new_element}
            for old_element, new_element in zip(seed_only_elements, candidate_only_elements)
        ]

    requires_stoichiometry_adjustment = bool(
        element_count_l1_distance and float(element_count_l1_distance) > 0.0
    )

    if seed_status != 'ok':
        action_label = 'manual_reference_recovery'
    elif edit_strategy == 'same_reduced_formula_reference':
        action_label = 'reference_reuse_control'
    elif edit_strategy == 'same_elements_stoichiometry_adjustment':
        action_label = 'stoichiometry_adjustment_enumeration'
    elif substitution_pairs and requires_stoichiometry_adjustment:
        action_label = 'element_substitution_plus_stoichiometry_adjustment'
    elif substitution_pairs:
        action_label = 'element_substitution_relabeling'
    elif edit_strategy == 'element_insertion_or_decoration':
        action_label = 'element_insertion_enumeration'
    elif edit_strategy == 'element_removal_or_vacancy':
        action_label = 'element_removal_enumeration'
    else:
        action_label = 'mixed_formula_edit_enumeration'

    candidate_formula_element_counts = _compact_formula_amount_map(candidate_formula)
    seed_formula_element_counts = _compact_formula_amount_map(
        str(seed_formula) if seed_formula is not None else None
    )
    element_count_deltas = _formula_delta_map(
        candidate_formula,
        str(seed_formula) if seed_formula is not None else None,
    )
    edit_operations = _structure_generation_edit_operations(
        action_label=action_label,
        substitution_pairs=substitution_pairs,
        element_count_deltas=element_count_deltas,
    )

    workflow_steps = _structure_generation_job_workflow_steps(action_label)
    direct_substitution_feasible = bool(substitution_pairs)
    simple_element_relabeling_feasible = bool(substitution_pairs) and not requires_stoichiometry_adjustment
    requires_enumeration = action_label != 'reference_reuse_control'
    edit_complexity_score = float(
        len(substitution_pairs) * 1.5
        + sum(abs(float(delta)) for delta in element_count_deltas.values())
        + (0.5 if not bool(_json_safe_value(row.get('seed_reference_has_structure_summary'))) else 0.0)
        + (0.1 * max(int(seed_rank or 1) - 1, 0))
    )

    return {
        'job_id': '__'.join(
            [
                _slugify_structure_generation_job_component(candidate_formula),
                f"seed_{seed_rank or 'na'}",
                _slugify_structure_generation_job_component(record_id or seed_formula),
            ]
        ),
        'job_rank': seed_rank,
        'job_action_label': action_label,
        'job_status': seed_status,
        'candidate_formula': candidate_formula,
        'ranking_rank': _json_safe_value(row.get('ranking_rank')),
        'ranking_score': _json_safe_value(row.get('ranking_score')),
        'bn_centered_ranking_rank': _json_safe_value(row.get('bn_centered_ranking_rank')),
        'candidate_family': _json_safe_value(row.get('candidate_family')),
        'candidate_novelty_bucket': _json_safe_value(row.get('candidate_novelty_bucket')),
        'chemical_plausibility_pass': _json_safe_value(row.get('chemical_plausibility_pass')),
        'proposal_shortlist_selected': _json_safe_value(row.get('proposal_shortlist_selected')),
        'proposal_shortlist_rank': _json_safe_value(row.get('proposal_shortlist_rank')),
        'extrapolation_shortlist_selected': _json_safe_value(row.get('extrapolation_shortlist_selected')),
        'extrapolation_shortlist_rank': _json_safe_value(row.get('extrapolation_shortlist_rank')),
        'structure_generation_candidate_priority_reason': _json_safe_value(
            row.get('structure_generation_candidate_priority_reason')
        ),
        'seed_reference_formula': seed_formula,
        'seed_reference_record_id': record_id,
        'seed_reference_source': _json_safe_value(row.get('seed_reference_source')),
        'seed_reference_has_structure_summary': _json_safe_value(
            row.get('seed_reference_has_structure_summary')
        ),
        'seed_formula_edit_strategy': edit_strategy,
        'seed_formula_exact_element_match': exact_element_match,
        'seed_formula_shared_elements': shared_elements,
        'seed_formula_candidate_only_elements': candidate_only_elements,
        'seed_formula_seed_only_elements': seed_only_elements,
        'seed_formula_element_count_l1_distance': element_count_l1_distance,
        'candidate_formula_element_counts': candidate_formula_element_counts,
        'seed_formula_element_counts': seed_formula_element_counts,
        'element_count_deltas': element_count_deltas,
        'edit_operations': edit_operations,
        'edit_operation_count': len(edit_operations),
        'edit_complexity_score': edit_complexity_score,
        'suggested_element_substitutions': substitution_pairs,
        'direct_element_substitution_feasible': direct_substitution_feasible,
        'simple_element_relabeling_feasible': simple_element_relabeling_feasible,
        'requires_stoichiometry_adjustment': requires_stoichiometry_adjustment,
        'requires_enumeration': requires_enumeration,
        'requires_relaxation': True,
        'reference_record_payload_artifact': 'demo_candidate_structure_generation_reference_records.json',
        'workflow_steps': workflow_steps,
        'workflow_step_count': len(workflow_steps),
    }


def _build_structure_generation_job_plan_payload(
    structure_generation_seed_df: pd.DataFrame,
    *,
    formula_col: str,
    cfg_defaults: dict[str, object],
) -> dict[str, object]:
    payload = {
        'label': STRUCTURE_GENERATION_JOB_PLAN_LABEL,
        'method': STRUCTURE_GENERATION_JOB_PLAN_METHOD,
        'candidate_scope': str(cfg_defaults['candidate_scope']),
        'per_candidate_seed_limit': int(cfg_defaults['per_candidate_seed_limit']),
        'bn_centered_top_n': int(cfg_defaults['bn_centered_top_n']),
        'seed_bridge_label': str(cfg_defaults['label']),
        'seed_bridge_method': str(cfg_defaults['method']),
        'seed_bridge_note': str(cfg_defaults['note']),
        'note': STRUCTURE_GENERATION_JOB_PLAN_NOTE,
        'handoff_artifact': 'demo_candidate_structure_generation_handoff.json',
        'reference_record_payload_artifact': 'demo_candidate_structure_generation_reference_records.json',
        'candidate_count': 0,
        'job_count': 0,
        'direct_substitution_job_count': 0,
        'simple_relabeling_job_count': 0,
        'job_action_counts': {},
        'candidates': [],
    }
    if structure_generation_seed_df is None or structure_generation_seed_df.empty:
        return payload

    seed_df = structure_generation_seed_df.copy()
    if 'structure_generation_seed_rank' not in seed_df.columns:
        seed_df['structure_generation_seed_rank'] = seed_df.groupby(formula_col).cumcount() + 1
    payload['candidate_count'] = int(seed_df[formula_col].astype(str).nunique())
    payload['job_count'] = int(len(seed_df))

    sort_columns = [
        column
        for column in ('ranking_rank', 'structure_generation_seed_rank')
        if column in seed_df.columns
    ]
    if sort_columns:
        seed_df = seed_df.sort_values(sort_columns, ascending=True)

    action_counts: dict[str, int] = {}
    candidate_payloads: list[dict[str, object]] = []
    for formula_value, group_df in seed_df.groupby(formula_col, sort=False):
        group_df = group_df.reset_index(drop=True)
        jobs = [_structure_generation_job_payload(row, formula_col=formula_col) for _, row in group_df.iterrows()]
        for job in jobs:
            action_counts[job['job_action_label']] = action_counts.get(job['job_action_label'], 0) + 1
        candidate_payloads.append(
            {
                formula_col: str(formula_value),
                'ranking_rank': _json_safe_value(group_df.iloc[0].get('ranking_rank')),
                'bn_centered_ranking_rank': _json_safe_value(
                    group_df.iloc[0].get('bn_centered_ranking_rank')
                ),
                'structure_generation_candidate_priority_reason': _json_safe_value(
                    group_df.iloc[0].get('structure_generation_candidate_priority_reason')
                ),
                'job_count': len(jobs),
                'jobs': jobs,
            }
        )

    payload['job_action_counts'] = action_counts
    payload['direct_substitution_job_count'] = int(
        action_counts.get('element_substitution_relabeling', 0)
        + action_counts.get('element_substitution_plus_stoichiometry_adjustment', 0)
    )
    payload['simple_relabeling_job_count'] = int(
        action_counts.get('element_substitution_relabeling', 0)
    )
    payload['candidates'] = candidate_payloads
    return payload


def _build_structure_generation_first_pass_queue_payload(
    structure_generation_seed_df: pd.DataFrame,
    *,
    formula_col: str,
    cfg_defaults: dict[str, object],
) -> dict[str, object]:
    payload = {
        'label': STRUCTURE_GENERATION_FIRST_PASS_QUEUE_LABEL,
        'method': STRUCTURE_GENERATION_FIRST_PASS_QUEUE_METHOD,
        'candidate_scope': str(cfg_defaults['candidate_scope']),
        'per_candidate_seed_limit': int(cfg_defaults['per_candidate_seed_limit']),
        'bn_centered_top_n': int(cfg_defaults['bn_centered_top_n']),
        'note': STRUCTURE_GENERATION_FIRST_PASS_QUEUE_NOTE,
        'source_job_plan_artifact': 'demo_candidate_structure_generation_job_plan.json',
        'reference_record_payload_artifact': 'demo_candidate_structure_generation_reference_records.json',
        'candidate_count': 0,
        'queue_entry_count': 0,
        'direct_substitution_job_count': 0,
        'simple_relabeling_job_count': 0,
        'mean_edit_complexity_score': None,
        'max_edit_complexity_score': None,
        'queue': [],
    }
    if structure_generation_seed_df is None or structure_generation_seed_df.empty:
        return payload

    seed_df = structure_generation_seed_df.copy()
    if 'structure_generation_seed_rank' not in seed_df.columns:
        seed_df['structure_generation_seed_rank'] = seed_df.groupby(formula_col).cumcount() + 1
    payload['candidate_count'] = int(seed_df[formula_col].astype(str).nunique())

    queue_entries: list[dict[str, object]] = []
    for _, row in seed_df.iterrows():
        job = _structure_generation_job_payload(row, formula_col=formula_col)
        ranking_rank = int(job['ranking_rank']) if job['ranking_rank'] is not None else 999999
        job_rank = int(job['job_rank']) if job['job_rank'] is not None else 999999
        structure_bonus = 0.5 if job['seed_reference_has_structure_summary'] else 0.0
        plausibility_bonus = 0.25 if job['chemical_plausibility_pass'] else 0.0
        substitution_bonus = 0.25 if job['direct_element_substitution_feasible'] else 0.0
        first_pass_priority_score = float(
            5.0 / max(ranking_rank, 1)
            + 2.0 / max(job_rank, 1)
            + structure_bonus
            + plausibility_bonus
            + substitution_bonus
            - float(job['edit_complexity_score'])
        )
        queue_entry = {
            **job,
            'first_pass_priority_score': first_pass_priority_score,
        }
        queue_entries.append(queue_entry)

    queue_entries.sort(
        key=lambda item: (
            -float(item['first_pass_priority_score']),
            float(item['edit_complexity_score']),
            int(item['ranking_rank']) if item['ranking_rank'] is not None else 999999,
            int(item['job_rank']) if item['job_rank'] is not None else 999999,
            str(item['job_id']),
        )
    )

    candidate_rank_tracker: dict[str, int] = {}
    for queue_rank, entry in enumerate(queue_entries, start=1):
        candidate_formula = str(entry['candidate_formula'])
        candidate_rank_tracker[candidate_formula] = candidate_rank_tracker.get(candidate_formula, 0) + 1
        entry['queue_rank'] = queue_rank
        entry['candidate_first_pass_rank'] = candidate_rank_tracker[candidate_formula]

    complexity_scores = [float(entry['edit_complexity_score']) for entry in queue_entries]
    payload['queue_entry_count'] = len(queue_entries)
    payload['direct_substitution_job_count'] = int(
        sum(1 for entry in queue_entries if entry['direct_element_substitution_feasible'])
    )
    payload['simple_relabeling_job_count'] = int(
        sum(1 for entry in queue_entries if entry['simple_element_relabeling_feasible'])
    )
    payload['mean_edit_complexity_score'] = float(np.mean(complexity_scores)) if complexity_scores else None
    payload['max_edit_complexity_score'] = float(np.max(complexity_scores)) if complexity_scores else None
    payload['queue'] = queue_entries
    return payload


def _structure_followup_readiness_label(
    *,
    best_action_label: str,
    min_edit_complexity_score: float,
    direct_substitution_job_count: int,
    simple_relabeling_job_count: int,
    has_reference_reuse_control: bool,
) -> str:
    if has_reference_reuse_control:
        return 'reference_reuse_control_available'
    if simple_relabeling_job_count > 0:
        return 'simple_relabeling_available'
    if best_action_label == 'stoichiometry_adjustment_enumeration' and min_edit_complexity_score <= 2.5:
        return 'low_complexity_stoichiometry_adjustment'
    if direct_substitution_job_count > 0:
        return 'substitution_plus_adjustment_required'
    if min_edit_complexity_score <= 4.0:
        return 'moderate_formula_edit_required'
    return 'heavy_formula_edit_required'


def _build_structure_generation_followup_shortlist_df(
    structure_generation_first_pass_queue: dict[str, object] | None,
    *,
    formula_col: str,
    cfg_defaults: dict[str, object],
) -> pd.DataFrame:
    columns = [
        formula_col,
        'candidate_family',
        'candidate_novelty_bucket',
        'chemical_plausibility_pass',
        'ranking_rank',
        'ranking_score',
        'bn_centered_ranking_rank',
        'proposal_shortlist_selected',
        'proposal_shortlist_rank',
        'extrapolation_shortlist_selected',
        'extrapolation_shortlist_rank',
        'structure_generation_candidate_priority_reason',
        'structure_followup_priority_score',
        'structure_followup_best_queue_rank',
        'structure_followup_best_action_label',
        'structure_followup_best_seed_reference_formula',
        'structure_followup_best_seed_reference_record_id',
        'structure_followup_reference_formula_count',
        'structure_followup_reference_formulas',
        'structure_followup_job_action_labels',
        'structure_followup_min_edit_complexity_score',
        'structure_followup_mean_edit_complexity_score',
        'structure_followup_direct_substitution_job_count',
        'structure_followup_simple_relabeling_job_count',
        'structure_followup_readiness_label',
        'structure_followup_shortlist_selected',
        'structure_followup_shortlist_rank',
        'structure_followup_shortlist_decision',
    ]
    queue_rows = [] if structure_generation_first_pass_queue is None else structure_generation_first_pass_queue.get('queue', [])
    if not queue_rows:
        return pd.DataFrame(columns=columns)

    queue_df = pd.DataFrame(queue_rows).copy()
    queue_df[formula_col] = queue_df['candidate_formula'].astype(str)
    aggregated_rows: list[dict[str, object]] = []
    for candidate_formula, group_df in queue_df.groupby(formula_col, sort=False):
        candidate_group = group_df.sort_values(
            ['queue_rank', 'edit_complexity_score', 'job_id'],
            ascending=[True, True, True],
            kind='stable',
        ).reset_index(drop=True)
        best_row = candidate_group.iloc[0]
        reference_formulas = sorted(
            {str(value) for value in candidate_group['seed_reference_formula'].dropna().astype(str)}
        )
        action_labels = list(dict.fromkeys(candidate_group['job_action_label'].astype(str).tolist()))
        direct_substitution_job_count = int(
            candidate_group['direct_element_substitution_feasible'].fillna(False).astype(bool).sum()
        )
        simple_relabeling_job_count = int(
            candidate_group['simple_element_relabeling_feasible'].fillna(False).astype(bool).sum()
        )
        has_reference_reuse_control = bool(
            candidate_group['job_action_label'].astype(str).eq('reference_reuse_control').any()
        )
        min_edit_complexity_score = float(candidate_group['edit_complexity_score'].min())
        mean_edit_complexity_score = float(candidate_group['edit_complexity_score'].mean())
        aggregated_rows.append(
            {
                formula_col: str(candidate_formula),
                'candidate_family': _json_safe_value(best_row.get('candidate_family')),
                'candidate_novelty_bucket': _json_safe_value(best_row.get('candidate_novelty_bucket')),
                'chemical_plausibility_pass': _json_safe_value(best_row.get('chemical_plausibility_pass')),
                'ranking_rank': _json_safe_value(best_row.get('ranking_rank')),
                'ranking_score': _json_safe_value(best_row.get('ranking_score')),
                'bn_centered_ranking_rank': _json_safe_value(best_row.get('bn_centered_ranking_rank')),
                'proposal_shortlist_selected': _json_safe_value(best_row.get('proposal_shortlist_selected')),
                'proposal_shortlist_rank': _json_safe_value(best_row.get('proposal_shortlist_rank')),
                'extrapolation_shortlist_selected': _json_safe_value(
                    best_row.get('extrapolation_shortlist_selected')
                ),
                'extrapolation_shortlist_rank': _json_safe_value(
                    best_row.get('extrapolation_shortlist_rank')
                ),
                'structure_generation_candidate_priority_reason': _json_safe_value(
                    best_row.get('structure_generation_candidate_priority_reason')
                ),
                'structure_followup_priority_score': float(best_row['first_pass_priority_score']),
                'structure_followup_best_queue_rank': int(best_row['queue_rank']),
                'structure_followup_best_action_label': str(best_row['job_action_label']),
                'structure_followup_best_seed_reference_formula': _json_safe_value(
                    best_row.get('seed_reference_formula')
                ),
                'structure_followup_best_seed_reference_record_id': _json_safe_value(
                    best_row.get('seed_reference_record_id')
                ),
                'structure_followup_reference_formula_count': int(len(reference_formulas)),
                'structure_followup_reference_formulas': '|'.join(reference_formulas),
                'structure_followup_job_action_labels': '|'.join(action_labels),
                'structure_followup_min_edit_complexity_score': min_edit_complexity_score,
                'structure_followup_mean_edit_complexity_score': mean_edit_complexity_score,
                'structure_followup_direct_substitution_job_count': direct_substitution_job_count,
                'structure_followup_simple_relabeling_job_count': simple_relabeling_job_count,
                'structure_followup_readiness_label': _structure_followup_readiness_label(
                    best_action_label=str(best_row['job_action_label']),
                    min_edit_complexity_score=min_edit_complexity_score,
                    direct_substitution_job_count=direct_substitution_job_count,
                    simple_relabeling_job_count=simple_relabeling_job_count,
                    has_reference_reuse_control=has_reference_reuse_control,
                ),
            }
        )

    shortlist_df = pd.DataFrame(aggregated_rows)
    if shortlist_df.empty:
        return pd.DataFrame(columns=columns)

    shortlist_df = shortlist_df.sort_values(
        [
            'structure_followup_priority_score',
            'structure_followup_min_edit_complexity_score',
            'ranking_rank',
            'structure_followup_best_queue_rank',
            formula_col,
        ],
        ascending=[False, True, True, True, True],
        kind='stable',
    ).reset_index(drop=True)
    shortlist_df['structure_followup_shortlist_selected'] = False
    shortlist_df['structure_followup_shortlist_rank'] = pd.Series([None] * len(shortlist_df), dtype='object')
    shortlist_df['structure_followup_shortlist_decision'] = 'not_selected_for_structure_followup_shortlist'
    if bool(cfg_defaults['enabled']) and not shortlist_df.empty:
        selected_count = min(int(cfg_defaults['shortlist_size']), len(shortlist_df))
        selected_index = shortlist_df.index[:selected_count]
        shortlist_df.loc[selected_index, 'structure_followup_shortlist_selected'] = True
        shortlist_df.loc[selected_index, 'structure_followup_shortlist_rank'] = np.arange(
            1, selected_count + 1, dtype=int
        )
        shortlist_df.loc[
            selected_index, 'structure_followup_shortlist_decision'
        ] = 'selected_for_structure_followup_shortlist'
    return shortlist_df[columns]


def _build_structure_generation_followup_extrapolation_shortlist_df(
    structure_followup_shortlist_df: pd.DataFrame,
    *,
    formula_col: str,
    cfg_defaults: dict[str, object],
) -> pd.DataFrame:
    if structure_followup_shortlist_df is None or structure_followup_shortlist_df.empty:
        return pd.DataFrame(columns=list(structure_followup_shortlist_df.columns) if structure_followup_shortlist_df is not None else [formula_col])

    shortlist_df = structure_followup_shortlist_df.copy()
    required_novelty_bucket = str(cfg_defaults['required_novelty_bucket'])
    filtered_df = shortlist_df.loc[
        shortlist_df['candidate_novelty_bucket'].astype(str).eq(required_novelty_bucket)
    ].copy()
    if filtered_df.empty:
        filtered_df['structure_followup_extrapolation_shortlist_selected'] = pd.Series(dtype=bool)
        filtered_df['structure_followup_extrapolation_shortlist_rank'] = pd.Series(dtype='object')
        filtered_df['structure_followup_extrapolation_shortlist_decision'] = pd.Series(dtype='object')
        return filtered_df

    filtered_df = filtered_df.sort_values(
        [
            'structure_followup_priority_score',
            'structure_followup_min_edit_complexity_score',
            'ranking_rank',
            'structure_followup_best_queue_rank',
            formula_col,
        ],
        ascending=[False, True, True, True, True],
        kind='stable',
    ).reset_index(drop=True)
    filtered_df['structure_followup_extrapolation_shortlist_selected'] = False
    filtered_df['structure_followup_extrapolation_shortlist_rank'] = pd.Series(
        [None] * len(filtered_df), dtype='object'
    )
    filtered_df['structure_followup_extrapolation_shortlist_decision'] = (
        'not_selected_for_structure_followup_extrapolation_shortlist'
    )
    if bool(cfg_defaults['enabled']):
        selected_count = min(int(cfg_defaults['shortlist_size']), len(filtered_df))
        selected_index = filtered_df.index[:selected_count]
        filtered_df.loc[selected_index, 'structure_followup_extrapolation_shortlist_selected'] = True
        filtered_df.loc[selected_index, 'structure_followup_extrapolation_shortlist_rank'] = np.arange(
            1, selected_count + 1, dtype=int
        )
        filtered_df.loc[
            selected_index, 'structure_followup_extrapolation_shortlist_decision'
        ] = 'selected_for_structure_followup_extrapolation_shortlist'
    return filtered_df


def _build_structure_generation_handoff_payload(
    structure_generation_seed_df: pd.DataFrame,
    *,
    formula_col: str,
    cfg_defaults: dict[str, object],
) -> dict[str, object]:
    payload = {
        'label': str(cfg_defaults['label']),
        'method': str(cfg_defaults['method']),
        'candidate_scope': str(cfg_defaults['candidate_scope']),
        'per_candidate_seed_limit': int(cfg_defaults['per_candidate_seed_limit']),
        'bn_centered_top_n': int(cfg_defaults['bn_centered_top_n']),
        'note': str(cfg_defaults['note']),
        'candidate_count': 0,
        'seed_row_count': 0,
        'candidates': [],
    }
    if structure_generation_seed_df is None or structure_generation_seed_df.empty:
        return payload

    seed_df = structure_generation_seed_df.copy()
    if 'structure_generation_seed_rank' not in seed_df.columns:
        seed_df['structure_generation_seed_rank'] = seed_df.groupby(formula_col).cumcount() + 1
    payload['candidate_count'] = int(seed_df[formula_col].astype(str).nunique())
    payload['seed_row_count'] = int(len(seed_df))

    sort_columns = [
        column
        for column in ('ranking_rank', 'structure_generation_seed_rank')
        if column in seed_df.columns
    ]
    if sort_columns:
        seed_df = seed_df.sort_values(sort_columns, ascending=True)

    candidate_columns = [
        formula_col,
        'ranking_rank',
        'bn_centered_ranking_rank',
        'candidate_family',
        'candidate_template',
        'candidate_novelty_bucket',
        'chemical_plausibility_pass',
        'proposal_shortlist_selected',
        'proposal_shortlist_rank',
        'extrapolation_shortlist_selected',
        'extrapolation_shortlist_rank',
        'bn_centered_top_n_selected',
        'structure_generation_candidate_priority_reason',
    ]
    seed_columns = [
        'structure_generation_seed_rank',
        'structure_generation_seed_source_column',
        'structure_generation_seed_status',
        'seed_reference_formula',
        'seed_reference_record_id',
        'seed_reference_source',
        'seed_reference_formula_row_count',
        'seed_reference_formula_mean_band_gap',
        'seed_reference_band_gap',
        'seed_formula_exact_element_match',
        'seed_formula_shared_elements',
        'seed_formula_candidate_only_elements',
        'seed_formula_seed_only_elements',
        'seed_formula_element_count_l1_distance',
        'seed_formula_edit_strategy',
        'seed_reference_energy_per_atom',
        'seed_reference_exfoliation_energy_per_atom',
        'seed_reference_total_magnetization',
        'seed_reference_abs_total_magnetization',
        'seed_reference_has_structure_summary',
    ]
    seed_columns.extend(
        column
        for column in seed_df.columns
        if column.startswith('seed_reference_structure_')
    )

    edit_columns = [
        'seed_formula_exact_element_match',
        'seed_formula_shared_elements',
        'seed_formula_candidate_only_elements',
        'seed_formula_seed_only_elements',
        'seed_formula_element_count_l1_distance',
        'seed_formula_edit_strategy',
    ]
    candidate_payloads = []
    for formula_value, group_df in seed_df.groupby(formula_col, sort=False):
        group_df = group_df.reset_index(drop=True)
        candidate_row = {formula_col: str(formula_value)}
        for column in candidate_columns[1:]:
            if column in group_df.columns:
                candidate_row[column] = _json_safe_value(group_df.iloc[0][column])
        seeds = []
        for _, row in group_df.iterrows():
            edit_payload = {}
            if formula_col in row.index and 'seed_reference_formula' in row.index and pd.notna(row['seed_reference_formula']):
                missing_edit_columns = [column for column in edit_columns if column not in row.index or pd.isna(row[column])]
                if missing_edit_columns:
                    edit_payload = _structure_seed_edit_metadata(
                        str(row[formula_col]),
                        str(row['seed_reference_formula']),
                    )
            seed_row = {}
            for column in seed_columns:
                if column in row.index and pd.notna(row[column]):
                    seed_row[column] = _json_safe_value(row[column])
                elif column in edit_payload:
                    seed_row[column] = _json_safe_value(edit_payload[column])
            seeds.append(seed_row)
        candidate_row['seed_count'] = len(seeds)
        candidate_row['seeds'] = seeds
        candidate_payloads.append(candidate_row)

    payload['candidates'] = candidate_payloads
    return payload


def _build_structure_generation_reference_record_payload(
    structure_generation_seed_df: pd.DataFrame,
    *,
    cfg: dict,
) -> dict[str, object]:
    payload = {
        'dataset': ((cfg.get('data') or {}).get('dataset') or 'unknown'),
        'record_count': 0,
        'missing_record_ids': [],
        'reference_records': [],
    }
    if structure_generation_seed_df is None or structure_generation_seed_df.empty:
        return payload

    raw_lookup = load_cached_raw_record_lookup(cfg)
    if not raw_lookup:
        return payload

    if 'seed_reference_record_id' not in structure_generation_seed_df.columns:
        return payload

    record_rows = (
        structure_generation_seed_df[
            structure_generation_seed_df['seed_reference_record_id'].notna()
        ]
        .drop_duplicates(subset=['seed_reference_record_id'])
        .copy()
    )
    if record_rows.empty:
        return payload

    records = []
    missing_record_ids = []
    for _, row in record_rows.iterrows():
        record_id = str(row['seed_reference_record_id'])
        raw_entry = raw_lookup.get(record_id)
        if raw_entry is None:
            missing_record_ids.append(record_id)
            continue
        records.append({
            'record_id': record_id,
            'formula': _json_safe_value(row.get('seed_reference_formula')),
            'source': _json_safe_value(row.get('seed_reference_source')),
            'band_gap': _json_safe_value(row.get('seed_reference_band_gap')),
            'energy_per_atom': _json_safe_value(row.get('seed_reference_energy_per_atom')),
            'exfoliation_energy_per_atom': _json_safe_value(
                row.get('seed_reference_exfoliation_energy_per_atom')
            ),
            'total_magnetization': _json_safe_value(row.get('seed_reference_total_magnetization')),
            'abs_total_magnetization': _json_safe_value(
                row.get('seed_reference_abs_total_magnetization')
            ),
            'has_structure_summary': _json_safe_value(
                row.get('seed_reference_has_structure_summary')
            ),
            'atoms': raw_entry.get('atoms'),
        })

    payload['record_count'] = len(records)
    payload['missing_record_ids'] = missing_record_ids
    payload['reference_records'] = records
    return payload


def _collect_shortlist_summary(
    candidate_df: pd.DataFrame,
    *,
    formula_col: str,
    prefix: str,
    cfg_defaults: dict[str, object],
    artifact_name: str,
    novelty_annotation_enabled: bool,
    novelty_bucket_order: list[str] | None = None,
) -> dict[str, object]:
    summary = {
        f'{prefix}_enabled': bool(cfg_defaults['enabled']),
        f'{prefix}_artifact': artifact_name if bool(cfg_defaults['enabled']) else None,
        f'{prefix}_label': str(cfg_defaults['label']),
        f'{prefix}_method': str(cfg_defaults['method']),
        f'{prefix}_note': str(cfg_defaults['note']),
        f'{prefix}_size': int(cfg_defaults['shortlist_size']),
        f'{prefix}_family_cap': int(cfg_defaults['max_per_candidate_family']),
        f'{prefix}_selected_rows': None,
        f'{prefix}_selected_family_counts': None,
        f'{prefix}_novelty_bucket_counts': None,
        f'{prefix}_formulas': [],
    }
    if 'required_novelty_bucket' in cfg_defaults:
        summary[f'{prefix}_target_novelty_bucket'] = str(cfg_defaults['required_novelty_bucket'])
        summary[f'{prefix}_candidate_count'] = None

    if candidate_df.empty:
        return summary

    for column_name in (
        f'{prefix}_enabled',
        f'{prefix}_label',
        f'{prefix}_method',
        f'{prefix}_note',
        f'{prefix}_size',
        f'{prefix}_family_cap',
        f'{prefix}_target_novelty_bucket',
    ):
        if column_name not in candidate_df.columns:
            continue
        non_null_values = candidate_df[column_name].dropna()
        if non_null_values.empty:
            continue
        value = non_null_values.iloc[0]
        if column_name == f'{prefix}_enabled':
            summary[column_name] = bool(value)
            summary[f'{prefix}_artifact'] = artifact_name if bool(value) else None
        elif column_name in {f'{prefix}_size', f'{prefix}_family_cap'}:
            summary[column_name] = int(value)
        else:
            summary[column_name] = str(value)

    target_bucket_key = f'{prefix}_target_novelty_bucket'
    if (
        target_bucket_key in summary
        and summary[target_bucket_key]
        and 'candidate_novelty_bucket' in candidate_df.columns
    ):
        summary[f'{prefix}_candidate_count'] = int(
            candidate_df['candidate_novelty_bucket'].eq(summary[target_bucket_key]).sum()
        )

    selected_column = f'{prefix}_selected'
    if selected_column not in candidate_df.columns:
        return summary

    shortlist_df = candidate_df.loc[
        candidate_df[selected_column].fillna(False).astype(bool)
    ].copy()
    rank_column = f'{prefix}_rank'
    if rank_column in shortlist_df.columns:
        shortlist_df = shortlist_df.sort_values(rank_column, ascending=True)
    summary[f'{prefix}_selected_rows'] = int(len(shortlist_df))

    if 'candidate_family' in shortlist_df.columns:
        family_counts = shortlist_df['candidate_family'].dropna().astype(str).value_counts()
        summary[f'{prefix}_selected_family_counts'] = {
            family: int(count) for family, count in family_counts.items()
        }

    if novelty_annotation_enabled and novelty_bucket_order is not None:
        summary[f'{prefix}_novelty_bucket_counts'] = {
            bucket: int(shortlist_df['candidate_novelty_bucket'].eq(bucket).sum())
            for bucket in novelty_bucket_order
        }

    shortlist_rows: list[dict[str, object]] = []
    for _, row in shortlist_df.iterrows():
        shortlist_row = {'formula': str(row[formula_col])}
        if rank_column in row and pd.notna(row[rank_column]):
            shortlist_row[rank_column] = int(row[rank_column])
        if 'ranking_rank' in row and pd.notna(row['ranking_rank']):
            shortlist_row['ranking_rank'] = int(row['ranking_rank'])
        if 'novel_formula_rank' in row and pd.notna(row['novel_formula_rank']):
            shortlist_row['novel_formula_rank'] = int(row['novel_formula_rank'])
        if 'candidate_family' in row and pd.notna(row['candidate_family']):
            shortlist_row['candidate_family'] = str(row['candidate_family'])
        if 'ranking_score' in row and pd.notna(row['ranking_score']):
            shortlist_row['ranking_score'] = float(row['ranking_score'])
        shortlist_rows.append(shortlist_row)
    summary[f'{prefix}_formulas'] = shortlist_rows
    return summary


def build_experiment_summary(
    dataset_df,
    bn_df,
    candidate_df,
    split_masks,
    selection_summary,
    cfg,
    robustness_df=None,
    bn_slice_benchmark_df=None,
    bn_family_benchmark_df=None,
    bn_stratified_error_df=None,
    bn_centered_candidate_df=None,
    bn_centered_screening_selection=None,
    structure_generation_seed_df=None,
    candidate_prediction_member_df=None,
    candidate_grouped_robustness_member_df=None,
    bn_centered_grouped_robustness_member_df=None,
    structure_first_pass_execution_summary_df=None,
    structure_first_pass_execution_payload=None,
):
    formula_col = cfg['data']['formula_column']
    target_col = cfg['data']['target_column']
    split_metadata = split_masks.get('metadata', {})
    selected_feature_set = selection_summary.get('selected_feature_set', cfg['features']['feature_set'])
    selected_model_type = selection_summary.get('selected_model_type', cfg['model']['type'])
    screening_feature_set = selection_summary.get('screening_selected_feature_set', selected_feature_set)
    screening_model_type = selection_summary.get('screening_selected_model_type', selected_model_type)
    selected_feature_family = selection_summary.get(
        'selected_feature_family',
        get_feature_family(selected_feature_set),
    )
    screening_feature_family = selection_summary.get(
        'screening_selected_feature_family',
        get_feature_family(screening_feature_set),
    )
    screening_matches_overall = bool(
        selection_summary.get(
            'screening_selection_matches_overall',
            screening_feature_set == selected_feature_set and screening_model_type == selected_model_type,
        )
    )
    screening_selection_note = selection_summary.get(
        'screening_selection_note',
        'Formula-only screening reuses the overall selected combo.',
    )
    chemical_plausibility_cfg = cfg['screening'].get('chemical_plausibility', {})
    chemical_plausibility_enabled = bool(chemical_plausibility_cfg.get('enabled', True))
    proposal_shortlist_cfg = _proposal_shortlist_config(cfg)
    extrapolation_shortlist_cfg = _extrapolation_shortlist_config(cfg)
    structure_generation_seed_cfg = _structure_generation_seed_config(cfg)
    structure_followup_shortlist_cfg = _structure_followup_shortlist_config(cfg)
    structure_followup_extrapolation_shortlist_cfg = (
        _structure_followup_extrapolation_shortlist_config(cfg)
    )
    ranking_config_metadata = get_screening_ranking_metadata(cfg)
    robustness_cfg = cfg.get('robustness', {})
    robustness_enabled = bool(robustness_cfg.get('enabled', False))
    bn_slice_benchmark_cfg = _bn_slice_benchmark_config(cfg)
    bn_family_benchmark_cfg = _bn_family_benchmark_config(cfg)
    bn_stratified_error_cfg = _bn_stratified_error_config(cfg)
    robustness_df = pd.DataFrame() if robustness_df is None else robustness_df.copy()
    bn_slice_benchmark_df = (
        pd.DataFrame() if bn_slice_benchmark_df is None else bn_slice_benchmark_df.copy()
    )
    bn_family_benchmark_df = (
        pd.DataFrame() if bn_family_benchmark_df is None else bn_family_benchmark_df.copy()
    )
    bn_stratified_error_df = (
        pd.DataFrame() if bn_stratified_error_df is None else bn_stratified_error_df.copy()
    )
    bn_centered_candidate_df = (
        pd.DataFrame() if bn_centered_candidate_df is None else bn_centered_candidate_df.copy()
    )
    bn_centered_screening_selection = dict(bn_centered_screening_selection or {})
    structure_generation_seed_df = (
        pd.DataFrame() if structure_generation_seed_df is None else structure_generation_seed_df.copy()
    )
    candidate_prediction_member_df = (
        pd.DataFrame() if candidate_prediction_member_df is None else candidate_prediction_member_df.copy()
    )
    candidate_grouped_robustness_member_df = (
        pd.DataFrame()
        if candidate_grouped_robustness_member_df is None
        else candidate_grouped_robustness_member_df.copy()
    )
    bn_centered_grouped_robustness_member_df = (
        pd.DataFrame()
        if bn_centered_grouped_robustness_member_df is None
        else bn_centered_grouped_robustness_member_df.copy()
    )
    structure_first_pass_execution_summary_df = (
        pd.DataFrame()
        if structure_first_pass_execution_summary_df is None
        else structure_first_pass_execution_summary_df.copy()
    )
    structure_first_pass_execution_payload = dict(structure_first_pass_execution_payload or {})

    plausibility_pass_count = None
    plausibility_fail_count = None
    plausibility_failed_formulas = []
    if 'chemical_plausibility_pass' in candidate_df.columns:
        plausibility_pass_mask = candidate_df['chemical_plausibility_pass'].fillna(False).astype(bool)
        plausibility_pass_count = int(plausibility_pass_mask.sum())
        plausibility_fail_count = int((~plausibility_pass_mask).sum())
        plausibility_failed_formulas = (
            candidate_df.loc[~plausibility_pass_mask, formula_col].astype(str).tolist()
        )

    domain_support_reference_formula_count = None
    domain_support_penalized_rows = None
    domain_support_low_support_rows = None
    if 'domain_support_reference_formula_count' in candidate_df.columns and not candidate_df.empty:
        domain_support_reference_formula_count = int(
            candidate_df['domain_support_reference_formula_count'].fillna(0).iloc[0]
        )
    if 'domain_support_penalty' in candidate_df.columns:
        domain_support_penalties = candidate_df['domain_support_penalty'].fillna(0.0)
        domain_support_penalized_rows = int((domain_support_penalties > 0).sum())
    if 'domain_support_percentile' in candidate_df.columns:
        percentile_threshold = float(ranking_config_metadata['domain_support_penalize_below_percentile'])
        domain_support_low_support_rows = int(
            candidate_df['domain_support_percentile'].fillna(100.0).lt(percentile_threshold).sum()
        )

    bn_support_reference_formula_count = None
    bn_support_penalized_rows = None
    bn_support_low_support_rows = None
    if 'bn_support_reference_formula_count' in candidate_df.columns and not candidate_df.empty:
        bn_support_reference_formula_count = int(
            candidate_df['bn_support_reference_formula_count'].fillna(0).iloc[0]
        )
    if 'bn_support_penalty' in candidate_df.columns:
        bn_support_penalties = candidate_df['bn_support_penalty'].fillna(0.0)
        bn_support_penalized_rows = int((bn_support_penalties > 0).sum())
    if 'bn_support_percentile' in candidate_df.columns:
        percentile_threshold = float(ranking_config_metadata['bn_support_penalize_below_percentile'])
        bn_support_low_support_rows = int(
            candidate_df['bn_support_percentile'].fillna(100.0).lt(percentile_threshold).sum()
        )

    grouped_robustness_uncertainty_enabled = False
    grouped_robustness_prediction_fold_count = None
    grouped_robustness_penalized_rows = None
    grouped_robustness_prediction_std_mean = None
    bn_analog_evidence_enabled = False
    bn_analog_reference_formula_count = None
    bn_analog_reference_band_gap_median = None
    bn_analog_reference_band_gap_iqr = None
    bn_analog_reference_exfoliation_energy_median = None
    bn_analog_reference_energy_per_atom_median = None
    bn_analog_reference_abs_total_magnetization_median = None
    bn_analog_exfoliation_available_rows = None
    bn_analog_lower_or_equal_reference_rows = None
    bn_analog_higher_reference_rows = None
    bn_band_gap_alignment_penalty_eligible_rows = None
    bn_band_gap_alignment_within_window_rows = None
    bn_band_gap_alignment_below_window_rows = None
    bn_band_gap_alignment_above_window_rows = None
    bn_band_gap_alignment_penalized_rows = None
    bn_analog_reference_like_rows = None
    bn_analog_mixed_alignment_rows = None
    bn_analog_reference_divergent_rows = None
    bn_analog_validation_penalized_rows = None
    if 'grouped_robustness_prediction_enabled' in candidate_df.columns and not candidate_df.empty:
        grouped_robustness_uncertainty_enabled = bool(
            candidate_df['grouped_robustness_prediction_enabled'].fillna(False).iloc[0]
        )
    if 'grouped_robustness_prediction_fold_count' in candidate_df.columns and not candidate_df.empty:
        grouped_robustness_prediction_fold_count = int(
            candidate_df['grouped_robustness_prediction_fold_count'].fillna(0).iloc[0]
        )
    if 'grouped_robustness_predicted_band_gap_std' in candidate_df.columns:
        std_values = candidate_df['grouped_robustness_predicted_band_gap_std'].dropna()
        grouped_robustness_prediction_std_mean = (
            float(std_values.mean()) if not std_values.empty else None
        )
    if 'grouped_robustness_uncertainty_penalty' in candidate_df.columns:
        grouped_robustness_penalized_rows = int(
            candidate_df['grouped_robustness_uncertainty_penalty'].fillna(0.0).gt(0.0).sum()
        )
    if 'bn_analog_evidence_enabled' in candidate_df.columns and not candidate_df.empty:
        bn_analog_evidence_enabled = bool(candidate_df['bn_analog_evidence_enabled'].fillna(False).iloc[0])
    if 'bn_analog_reference_formula_count' in candidate_df.columns and not candidate_df.empty:
        bn_analog_reference_formula_count = int(
            candidate_df['bn_analog_reference_formula_count'].fillna(0).iloc[0]
        )
    if 'bn_analog_reference_band_gap_median' in candidate_df.columns and not candidate_df.empty:
        median_value = candidate_df['bn_analog_reference_band_gap_median'].iloc[0]
        bn_analog_reference_band_gap_median = (
            float(median_value) if pd.notna(median_value) else None
        )
    if 'bn_analog_reference_band_gap_iqr' in candidate_df.columns and not candidate_df.empty:
        iqr_value = candidate_df['bn_analog_reference_band_gap_iqr'].iloc[0]
        bn_analog_reference_band_gap_iqr = (
            float(iqr_value) if pd.notna(iqr_value) else None
        )
    if 'bn_analog_reference_exfoliation_energy_median' in candidate_df.columns and not candidate_df.empty:
        median_value = candidate_df['bn_analog_reference_exfoliation_energy_median'].iloc[0]
        bn_analog_reference_exfoliation_energy_median = (
            float(median_value) if pd.notna(median_value) else None
        )
    if 'bn_analog_reference_energy_per_atom_median' in candidate_df.columns and not candidate_df.empty:
        median_value = candidate_df['bn_analog_reference_energy_per_atom_median'].iloc[0]
        bn_analog_reference_energy_per_atom_median = (
            float(median_value) if pd.notna(median_value) else None
        )
    if 'bn_analog_reference_abs_total_magnetization_median' in candidate_df.columns and not candidate_df.empty:
        median_value = candidate_df['bn_analog_reference_abs_total_magnetization_median'].iloc[0]
        bn_analog_reference_abs_total_magnetization_median = (
            float(median_value) if pd.notna(median_value) else None
        )
    if 'bn_analog_neighbor_exfoliation_available_formula_count' in candidate_df.columns:
        bn_analog_exfoliation_available_rows = int(
            candidate_df['bn_analog_neighbor_exfoliation_available_formula_count'].fillna(0).gt(0).sum()
        )
    if 'bn_analog_exfoliation_support_label' in candidate_df.columns:
        bn_analog_lower_or_equal_reference_rows = int(
            candidate_df['bn_analog_exfoliation_support_label'].eq(
                'lower_or_equal_bn_reference_median'
            ).sum()
        )
        bn_analog_higher_reference_rows = int(
            candidate_df['bn_analog_exfoliation_support_label'].eq(
                'higher_than_bn_reference_median'
            ).sum()
        )
    if 'bn_band_gap_alignment_penalty_eligible' in candidate_df.columns:
        bn_band_gap_alignment_penalty_eligible_rows = int(
            candidate_df['bn_band_gap_alignment_penalty_eligible']
            .fillna(False)
            .astype(bool)
            .sum()
        )
    if 'bn_band_gap_alignment_label' in candidate_df.columns:
        bn_band_gap_alignment_within_window_rows = int(
            candidate_df['bn_band_gap_alignment_label'].eq(
                'within_local_bn_analog_band_gap_window'
            ).sum()
        )
        bn_band_gap_alignment_below_window_rows = int(
            candidate_df['bn_band_gap_alignment_label'].eq(
                'below_local_bn_analog_band_gap_window'
            ).sum()
        )
        bn_band_gap_alignment_above_window_rows = int(
            candidate_df['bn_band_gap_alignment_label'].eq(
                'above_local_bn_analog_band_gap_window'
            ).sum()
        )
    if 'bn_band_gap_alignment_penalty' in candidate_df.columns:
        bn_band_gap_alignment_penalized_rows = int(
            candidate_df['bn_band_gap_alignment_penalty'].fillna(0.0).gt(0.0).sum()
        )
    if 'bn_analog_validation_label' in candidate_df.columns:
        bn_analog_reference_like_rows = int(
            candidate_df['bn_analog_validation_label'].eq(
                'reference_like_on_available_metrics'
            ).sum()
        )
        bn_analog_mixed_alignment_rows = int(
            candidate_df['bn_analog_validation_label'].eq(
                'mixed_reference_alignment'
            ).sum()
        )
        bn_analog_reference_divergent_rows = int(
            candidate_df['bn_analog_validation_label'].eq(
                'reference_divergent_on_available_metrics'
            ).sum()
        )
    if 'bn_analog_validation_penalty' in candidate_df.columns:
        bn_analog_validation_penalized_rows = int(
            candidate_df['bn_analog_validation_penalty'].fillna(0.0).gt(0.0).sum()
        )

    ranking_metadata = get_screening_ranking_metadata(
        cfg,
        domain_support_penalty_applied=bool(domain_support_penalized_rows),
        bn_support_penalty_applied=bool(bn_support_penalized_rows),
        grouped_robustness_penalty_applied=bool(grouped_robustness_penalized_rows),
        bn_band_gap_alignment_penalty_applied=bool(bn_band_gap_alignment_penalized_rows),
        bn_analog_validation_penalty_applied=bool(bn_analog_validation_penalized_rows),
    )
    candidate_space_name = cfg['screening']['candidate_space_name']
    candidate_space_kind = cfg['screening']['candidate_space_kind']
    candidate_space_note = cfg['screening']['candidate_space_note']
    candidate_generation_strategy = cfg['screening'].get('candidate_generation_strategy')
    if not candidate_df.empty:
        for column_name, fallback_value in (
            ('candidate_space_name', candidate_space_name),
            ('candidate_space_kind', candidate_space_kind),
            ('candidate_space_note', candidate_space_note),
            ('candidate_generation_strategy', candidate_generation_strategy),
        ):
            if column_name in candidate_df.columns:
                non_null_values = candidate_df[column_name].dropna()
                if not non_null_values.empty:
                    if column_name == 'candidate_space_name':
                        candidate_space_name = str(non_null_values.iloc[0])
                    elif column_name == 'candidate_space_kind':
                        candidate_space_kind = str(non_null_values.iloc[0])
                    elif column_name == 'candidate_space_note':
                        candidate_space_note = str(non_null_values.iloc[0])
                    elif column_name == 'candidate_generation_strategy':
                        candidate_generation_strategy = str(non_null_values.iloc[0])

    candidate_family_counts = None
    if 'candidate_family' in candidate_df.columns:
        family_counts = candidate_df['candidate_family'].dropna().astype(str).value_counts()
        candidate_family_counts = {family: int(count) for family, count in family_counts.items()}

    ranking_basis = ranking_metadata['ranking_basis']
    ranking_note = ranking_metadata['ranking_note']
    if bool(ranking_metadata['domain_support_enabled']):
        ranking_note = f'{ranking_note} {DOMAIN_SUPPORT_RANKING_NOTE}'
    if bool(ranking_metadata['bn_support_enabled']):
        ranking_note = f'{ranking_note} {BN_SUPPORT_RANKING_NOTE}'
    if bool(ranking_metadata['bn_band_gap_alignment_enabled']):
        ranking_note = f'{ranking_note} {BN_BAND_GAP_ALIGNMENT_RANKING_NOTE}'
    if bool(ranking_metadata['grouped_robustness_penalty_active']):
        ranking_note = f'{ranking_note} {GROUPED_ROBUSTNESS_UNCERTAINTY_RANKING_NOTE}'
    if bn_analog_evidence_enabled:
        ranking_note = f'{ranking_note} {BN_ANALOG_EVIDENCE_RANKING_NOTE}'
    if not screening_matches_overall:
        ranking_note = (
            f'{ranking_note} The best overall validation combo is structure-aware, so formula-only '
            'candidate screening falls back to the best candidate-compatible combo.'
        )
    if chemical_plausibility_enabled:
        ranking_note = (
            f'{ranking_note} Candidates are also annotated with a lightweight pymatgen oxidation-state '
            'plausibility screen, and passing formulas are prioritized ahead of formulas that fail '
            'basic charge-balance checks.'
        )
    novelty_annotation_enabled = bool(
        {
            'candidate_novelty_bucket',
            'candidate_is_formula_level_extrapolation',
        }.issubset(candidate_df.columns)
    )
    novelty_bucket_order = [
        NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY,
        NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA,
        NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION,
    ]
    novelty_bucket_counts = None
    standard_top_k_novelty_bucket_counts = None
    formula_level_extrapolation_candidate_count = None
    formula_level_extrapolation_shortlist = []
    novelty_interpretation_note = None
    if novelty_annotation_enabled:
        novelty_bucket_counts = {
            bucket: int(
                candidate_df['candidate_novelty_bucket'].eq(bucket).sum()
            )
            for bucket in novelty_bucket_order
        }
        if 'screening_selected_for_top_k' in candidate_df.columns:
            top_k_candidate_df = candidate_df.loc[
                candidate_df['screening_selected_for_top_k'].fillna(False).astype(bool)
            ]
            standard_top_k_novelty_bucket_counts = {
                bucket: int(
                    top_k_candidate_df['candidate_novelty_bucket'].eq(bucket).sum()
                )
                for bucket in novelty_bucket_order
            }
        formula_level_extrapolation_candidate_count = int(
            candidate_df['candidate_is_formula_level_extrapolation'].fillna(False).astype(bool).sum()
        )
        novel_df = candidate_df.loc[
            candidate_df['candidate_novelty_bucket'].eq(NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION)
        ].copy()
        if 'ranking_rank' in novel_df.columns:
            novel_df = novel_df.sort_values('ranking_rank', ascending=True)
        for _, row in novel_df.head(5).iterrows():
            shortlist_row = {'formula': str(row[formula_col])}
            if 'ranking_rank' in row and pd.notna(row['ranking_rank']):
                shortlist_row['ranking_rank'] = int(row['ranking_rank'])
            if 'novel_formula_rank' in row and pd.notna(row['novel_formula_rank']):
                shortlist_row['novel_formula_rank'] = int(row['novel_formula_rank'])
            if 'ranking_score' in row and pd.notna(row['ranking_score']):
                shortlist_row['ranking_score'] = float(row['ranking_score'])
            if 'chemical_plausibility_pass' in row and pd.notna(row['chemical_plausibility_pass']):
                shortlist_row['chemical_plausibility_pass'] = bool(row['chemical_plausibility_pass'])
            if 'screening_selected_for_top_k' in row and pd.notna(row['screening_selected_for_top_k']):
                shortlist_row['screening_selected_for_top_k'] = bool(row['screening_selected_for_top_k'])
            if 'screening_selection_decision' in row and pd.notna(row['screening_selection_decision']):
                shortlist_row['screening_selection_decision'] = str(row['screening_selection_decision'])
            if (
                'extrapolation_shortlist_selected' in row
                and pd.notna(row['extrapolation_shortlist_selected'])
            ):
                shortlist_row['extrapolation_shortlist_selected'] = bool(
                    row['extrapolation_shortlist_selected']
                )
            if 'extrapolation_shortlist_rank' in row and pd.notna(row['extrapolation_shortlist_rank']):
                shortlist_row['extrapolation_shortlist_rank'] = int(
                    row['extrapolation_shortlist_rank']
                )
            if (
                'extrapolation_shortlist_decision' in row
                and pd.notna(row['extrapolation_shortlist_decision'])
            ):
                shortlist_row['extrapolation_shortlist_decision'] = str(
                    row['extrapolation_shortlist_decision']
                )
            formula_level_extrapolation_shortlist.append(shortlist_row)
        novelty_interpretation_note = (
            'Standard top-k remains the default ranking output, but novelty should be interpreted '
            'separately: train+val rediscovery is in-domain replay, held-out-known formulas are '
            'known elsewhere in the dataset, and formula-level extrapolation only means unseen '
            'formula compositions inside this demo candidate space.'
        )
        ranking_note = f'{ranking_note} {NOVELTY_ANNOTATION_RANKING_NOTE}'

    proposal_shortlist_summary = _collect_shortlist_summary(
        candidate_df,
        formula_col=formula_col,
        prefix='proposal_shortlist',
        cfg_defaults=proposal_shortlist_cfg,
        artifact_name='demo_candidate_proposal_shortlist.csv',
        novelty_annotation_enabled=novelty_annotation_enabled,
        novelty_bucket_order=novelty_bucket_order,
    )
    extrapolation_shortlist_summary = _collect_shortlist_summary(
        candidate_df,
        formula_col=formula_col,
        prefix='extrapolation_shortlist',
        cfg_defaults=extrapolation_shortlist_cfg,
        artifact_name='demo_candidate_extrapolation_shortlist.csv',
        novelty_annotation_enabled=novelty_annotation_enabled,
        novelty_bucket_order=novelty_bucket_order,
    )

    split_train_mask = pd.Series(split_masks.get('train', []), dtype=bool)
    split_val_mask = pd.Series(split_masks.get('val', []), dtype=bool)
    split_test_mask = pd.Series(split_masks.get('test', []), dtype=bool)
    bn_train_rows = int(split_train_mask.reindex(dataset_df.index, fill_value=False).loc[bn_df.index].sum()) if not bn_df.empty else 0
    bn_val_rows = int(split_val_mask.reindex(dataset_df.index, fill_value=False).loc[bn_df.index].sum()) if not bn_df.empty else 0
    bn_test_rows = int(split_test_mask.reindex(dataset_df.index, fill_value=False).loc[bn_df.index].sum()) if not bn_df.empty else 0

    bn_slice_selected_mask = pd.Series(False, index=bn_slice_benchmark_df.index)
    bn_slice_screening_mask = pd.Series(False, index=bn_slice_benchmark_df.index)
    bn_slice_bn_local_mask = pd.Series(False, index=bn_slice_benchmark_df.index)
    bn_slice_global_dummy_mask = pd.Series(False, index=bn_slice_benchmark_df.index)
    bn_slice_best_candidate_mask = pd.Series(False, index=bn_slice_benchmark_df.index)
    if 'benchmark_role' in bn_slice_benchmark_df.columns:
        bn_slice_selected_mask = bn_slice_benchmark_df['benchmark_role'].astype(str).eq('selected_model')
        bn_slice_screening_mask = bn_slice_benchmark_df['benchmark_role'].astype(str).eq('screening_model')
        bn_slice_bn_local_mask = bn_slice_benchmark_df['benchmark_role'].astype(str).eq('bn_local_reference_baseline')
        bn_slice_global_dummy_mask = bn_slice_benchmark_df['benchmark_role'].astype(str).eq('global_dummy_mean_baseline')
        if {'benchmark_status', 'mae'}.issubset(bn_slice_benchmark_df.columns):
            candidate_mask = bn_slice_benchmark_df['benchmark_role'].astype(str).isin(
                ['selected_model', 'screening_model', 'candidate_model']
            ) & bn_slice_benchmark_df['benchmark_status'].astype(str).eq('ok')
            bn_slice_candidate_result_df = bn_slice_benchmark_df.loc[candidate_mask].copy()
            if not bn_slice_candidate_result_df.empty:
                best_idx = bn_slice_candidate_result_df['mae'].astype(float).idxmin()
                bn_slice_best_candidate_mask = pd.Series(False, index=bn_slice_benchmark_df.index)
                bn_slice_best_candidate_mask.loc[best_idx] = True

    bn_slice_selected_metrics = _bn_slice_benchmark_row_payload(
        bn_slice_benchmark_df,
        bn_slice_selected_mask,
    ) if 'benchmark_role' in bn_slice_benchmark_df.columns else None
    bn_slice_screening_metrics = _bn_slice_benchmark_row_payload(
        bn_slice_benchmark_df,
        bn_slice_screening_mask,
    ) if 'benchmark_role' in bn_slice_benchmark_df.columns else None
    bn_slice_bn_local_metrics = _bn_slice_benchmark_row_payload(
        bn_slice_benchmark_df,
        bn_slice_bn_local_mask,
    ) if 'benchmark_role' in bn_slice_benchmark_df.columns else None
    bn_slice_global_dummy_metrics = _bn_slice_benchmark_row_payload(
        bn_slice_benchmark_df,
        bn_slice_global_dummy_mask,
    ) if 'benchmark_role' in bn_slice_benchmark_df.columns else None
    bn_slice_best_candidate_metrics = _bn_slice_benchmark_row_payload(
        bn_slice_benchmark_df,
        bn_slice_best_candidate_mask,
    ) if 'benchmark_role' in bn_slice_benchmark_df.columns else None
    bn_slice_selected_beats_global_dummy = None
    bn_slice_screening_beats_global_dummy = None
    bn_slice_best_candidate_beats_global_dummy = None
    bn_slice_selected_matches_best_candidate = None
    if bn_slice_selected_metrics and bn_slice_global_dummy_metrics:
        bn_slice_selected_beats_global_dummy = bool(
            bn_slice_selected_metrics['mae'] < bn_slice_global_dummy_metrics['mae']
        )
    if bn_slice_screening_metrics and bn_slice_global_dummy_metrics:
        bn_slice_screening_beats_global_dummy = bool(
            bn_slice_screening_metrics['mae'] < bn_slice_global_dummy_metrics['mae']
        )
    if bn_slice_best_candidate_metrics and bn_slice_global_dummy_metrics:
        bn_slice_best_candidate_beats_global_dummy = bool(
            bn_slice_best_candidate_metrics['mae'] < bn_slice_global_dummy_metrics['mae']
        )
    if bn_slice_best_candidate_metrics and bn_slice_selected_metrics:
        bn_slice_selected_matches_best_candidate = bool(
            bn_slice_best_candidate_metrics['feature_set'] == bn_slice_selected_metrics['feature_set']
            and bn_slice_best_candidate_metrics['model_type'] == bn_slice_selected_metrics['model_type']
        )

    bn_family_selected_mask = pd.Series(False, index=bn_family_benchmark_df.index)
    bn_family_screening_mask = pd.Series(False, index=bn_family_benchmark_df.index)
    bn_family_bn_local_mask = pd.Series(False, index=bn_family_benchmark_df.index)
    bn_family_global_dummy_mask = pd.Series(False, index=bn_family_benchmark_df.index)
    bn_family_best_candidate_mask = pd.Series(False, index=bn_family_benchmark_df.index)
    if 'benchmark_role' in bn_family_benchmark_df.columns:
        bn_family_selected_mask = bn_family_benchmark_df['benchmark_role'].astype(str).eq('selected_model')
        bn_family_screening_mask = bn_family_benchmark_df['benchmark_role'].astype(str).eq('screening_model')
        bn_family_bn_local_mask = bn_family_benchmark_df['benchmark_role'].astype(str).eq('bn_local_reference_baseline')
        bn_family_global_dummy_mask = bn_family_benchmark_df['benchmark_role'].astype(str).eq('global_dummy_mean_baseline')
        if {'benchmark_status', 'mae'}.issubset(bn_family_benchmark_df.columns):
            candidate_mask = bn_family_benchmark_df['benchmark_role'].astype(str).isin(
                ['selected_model', 'screening_model', 'candidate_model']
            ) & bn_family_benchmark_df['benchmark_status'].astype(str).eq('ok')
            bn_family_candidate_result_df = bn_family_benchmark_df.loc[candidate_mask].copy()
            if not bn_family_candidate_result_df.empty:
                best_idx = bn_family_candidate_result_df['mae'].astype(float).idxmin()
                bn_family_best_candidate_mask = pd.Series(False, index=bn_family_benchmark_df.index)
                bn_family_best_candidate_mask.loc[best_idx] = True

    bn_family_selected_metrics = _bn_family_benchmark_row_payload(
        bn_family_benchmark_df,
        bn_family_selected_mask,
    ) if 'benchmark_role' in bn_family_benchmark_df.columns else None
    bn_family_screening_metrics = _bn_family_benchmark_row_payload(
        bn_family_benchmark_df,
        bn_family_screening_mask,
    ) if 'benchmark_role' in bn_family_benchmark_df.columns else None
    bn_family_bn_local_metrics = _bn_family_benchmark_row_payload(
        bn_family_benchmark_df,
        bn_family_bn_local_mask,
    ) if 'benchmark_role' in bn_family_benchmark_df.columns else None
    bn_family_global_dummy_metrics = _bn_family_benchmark_row_payload(
        bn_family_benchmark_df,
        bn_family_global_dummy_mask,
    ) if 'benchmark_role' in bn_family_benchmark_df.columns else None
    bn_family_best_candidate_metrics = _bn_family_benchmark_row_payload(
        bn_family_benchmark_df,
        bn_family_best_candidate_mask,
    ) if 'benchmark_role' in bn_family_benchmark_df.columns else None
    bn_family_selected_beats_global_dummy = None
    bn_family_screening_beats_global_dummy = None
    bn_family_best_candidate_beats_global_dummy = None
    if bn_family_selected_metrics and bn_family_global_dummy_metrics:
        bn_family_selected_beats_global_dummy = bool(
            bn_family_selected_metrics['mae'] < bn_family_global_dummy_metrics['mae']
        )
    if bn_family_screening_metrics and bn_family_global_dummy_metrics:
        bn_family_screening_beats_global_dummy = bool(
            bn_family_screening_metrics['mae'] < bn_family_global_dummy_metrics['mae']
        )
    if bn_family_best_candidate_metrics and bn_family_global_dummy_metrics:
        bn_family_best_candidate_beats_global_dummy = bool(
            bn_family_best_candidate_metrics['mae'] < bn_family_global_dummy_metrics['mae']
        )

    bn_stratified_selected_mask = pd.Series(False, index=bn_stratified_error_df.index)
    bn_stratified_screening_mask = pd.Series(False, index=bn_stratified_error_df.index)
    bn_stratified_dummy_mask = pd.Series(False, index=bn_stratified_error_df.index)
    if 'benchmark_role' in bn_stratified_error_df.columns:
        bn_stratified_selected_mask = bn_stratified_error_df['benchmark_role'].astype(str).eq('selected_model')
        bn_stratified_screening_mask = bn_stratified_error_df['benchmark_role'].astype(str).eq('screening_model')
        bn_stratified_dummy_mask = bn_stratified_error_df['benchmark_role'].astype(str).eq('dummy_baseline')

    bn_stratified_selected_metrics = _bn_stratified_error_row_payload(
        bn_stratified_error_df,
        bn_stratified_selected_mask,
    ) if 'benchmark_role' in bn_stratified_error_df.columns else None
    bn_stratified_screening_metrics = _bn_stratified_error_row_payload(
        bn_stratified_error_df,
        bn_stratified_screening_mask,
    ) if 'benchmark_role' in bn_stratified_error_df.columns else None
    bn_stratified_dummy_metrics = _bn_stratified_error_row_payload(
        bn_stratified_error_df,
        bn_stratified_dummy_mask,
    ) if 'benchmark_role' in bn_stratified_error_df.columns else None

    bn_centered_summary = {
        'enabled': bool(bn_centered_screening_selection.get('enabled', False)),
        'selection_source_artifact': bn_centered_screening_selection.get(
            'selection_source_artifact',
            'bn_slice_benchmark_results.csv',
        ),
        'selection_scope': bn_centered_screening_selection.get(
            'selection_scope',
            'bn_slice_candidate_compatible_best',
        ),
        'selection_note': bn_centered_screening_selection.get('selection_note'),
        'ranking_artifact': (
            bn_centered_screening_selection.get('ranking_artifact', 'demo_candidate_bn_centered_ranking.csv')
            if bool(bn_centered_screening_selection.get('enabled', False))
            else None
        ),
        'ranking_feature_set': bn_centered_screening_selection.get('feature_set'),
        'ranking_feature_family': bn_centered_screening_selection.get('feature_family'),
        'ranking_model_type': bn_centered_screening_selection.get('model_type'),
        'benchmark_role': bn_centered_screening_selection.get('benchmark_role'),
        'bn_slice_mae': bn_centered_screening_selection.get('mae'),
        'bn_slice_rmse': bn_centered_screening_selection.get('rmse'),
        'bn_slice_r2': bn_centered_screening_selection.get('r2'),
        'matches_general_screening_combo': bn_centered_screening_selection.get(
            'matches_general_screening_combo'
        ),
        'ranking_basis': None,
        'ranking_note': None,
        'candidate_rows': int(len(bn_centered_candidate_df)),
        'top_k': int(cfg['screening']['top_k']),
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
        'comparison_top_k_views': {},
    }
    if not bn_centered_candidate_df.empty:
        if 'ranking_basis' in bn_centered_candidate_df.columns:
            non_null_values = bn_centered_candidate_df['ranking_basis'].dropna()
            if not non_null_values.empty:
                bn_centered_summary['ranking_basis'] = str(non_null_values.iloc[0])
        if 'ranking_note' in bn_centered_candidate_df.columns:
            non_null_values = bn_centered_candidate_df['ranking_note'].dropna()
            if not non_null_values.empty:
                bn_centered_summary['ranking_note'] = str(non_null_values.iloc[0])
        bn_centered_summary.update(
            _candidate_ranking_comparison_payload(
                candidate_df,
                bn_centered_candidate_df,
                formula_col=formula_col,
                top_k=int(cfg['screening']['top_k']),
            )
        )

    structure_generation_seed_summary = _collect_structure_generation_seed_summary(
        structure_generation_seed_df,
        formula_col=formula_col,
        cfg_defaults=structure_generation_seed_cfg,
        artifact_name='demo_candidate_structure_generation_seeds.csv',
        handoff_artifact_name='demo_candidate_structure_generation_handoff.json',
    )
    if structure_generation_seed_summary.get('enabled'):
        structure_generation_seed_summary['reference_record_payload_artifact'] = (
            'demo_candidate_structure_generation_reference_records.json'
        )
        structure_generation_job_plan = _build_structure_generation_job_plan_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        structure_generation_seed_summary['job_plan_artifact'] = (
            'demo_candidate_structure_generation_job_plan.json'
        )
        structure_generation_seed_summary['job_count'] = int(
            structure_generation_job_plan['job_count']
        )
        structure_generation_seed_summary['job_action_counts'] = dict(
            structure_generation_job_plan['job_action_counts']
        )
        structure_generation_seed_summary['direct_substitution_job_count'] = int(
            structure_generation_job_plan['direct_substitution_job_count']
        )
        structure_generation_seed_summary['simple_relabeling_job_count'] = int(
            structure_generation_job_plan['simple_relabeling_job_count']
        )
        structure_generation_first_pass_queue = _build_structure_generation_first_pass_queue_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        structure_generation_seed_summary['first_pass_queue_artifact'] = (
            'demo_candidate_structure_generation_first_pass_queue.json'
        )
        structure_generation_seed_summary['first_pass_queue_size'] = int(
            structure_generation_first_pass_queue['queue_entry_count']
        )
        structure_generation_seed_summary['mean_edit_complexity_score'] = (
            structure_generation_first_pass_queue['mean_edit_complexity_score']
        )
        structure_generation_seed_summary['max_edit_complexity_score'] = (
            structure_generation_first_pass_queue['max_edit_complexity_score']
        )
        structure_followup_shortlist_df = _build_structure_generation_followup_shortlist_df(
            structure_generation_first_pass_queue,
            formula_col=formula_col,
            cfg_defaults=structure_followup_shortlist_cfg,
        )
        if bool(structure_followup_shortlist_cfg['enabled']):
            structure_generation_seed_summary['followup_shortlist_artifact'] = (
                'demo_candidate_structure_generation_followup_shortlist.csv'
            )
        selected_followup_df = structure_followup_shortlist_df.loc[
            structure_followup_shortlist_df['structure_followup_shortlist_selected'].fillna(False).astype(bool)
        ].copy() if not structure_followup_shortlist_df.empty else pd.DataFrame()
        structure_generation_seed_summary['followup_shortlist_size'] = int(len(selected_followup_df))
        structure_generation_seed_summary['followup_shortlist_formulas'] = (
            selected_followup_df.sort_values('structure_followup_shortlist_rank', ascending=True)[
                formula_col
            ].astype(str).tolist()
            if not selected_followup_df.empty
            else []
        )
        structure_generation_seed_summary['followup_readiness_counts'] = (
            selected_followup_df['structure_followup_readiness_label'].astype(str).value_counts().to_dict()
            if not selected_followup_df.empty
            else {}
        )
        structure_followup_extrapolation_shortlist_df = (
            _build_structure_generation_followup_extrapolation_shortlist_df(
                structure_followup_shortlist_df,
                formula_col=formula_col,
                cfg_defaults=structure_followup_extrapolation_shortlist_cfg,
            )
        )
        selected_followup_extrapolation_df = structure_followup_extrapolation_shortlist_df.loc[
            structure_followup_extrapolation_shortlist_df[
                'structure_followup_extrapolation_shortlist_selected'
            ].fillna(False).astype(bool)
        ].copy() if not structure_followup_extrapolation_shortlist_df.empty else pd.DataFrame()
        if bool(structure_followup_extrapolation_shortlist_cfg['enabled']):
            structure_generation_seed_summary['followup_extrapolation_shortlist_artifact'] = (
                'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'
            )
        structure_generation_seed_summary['followup_extrapolation_shortlist_size'] = int(
            len(selected_followup_extrapolation_df)
        )
        structure_generation_seed_summary['followup_extrapolation_shortlist_formulas'] = (
            selected_followup_extrapolation_df.sort_values(
                'structure_followup_extrapolation_shortlist_rank', ascending=True
            )[formula_col].astype(str).tolist()
            if not selected_followup_extrapolation_df.empty
            else []
        )
        if structure_first_pass_execution_payload:
            structure_generation_seed_summary['first_pass_execution_artifact'] = (
                structure_first_pass_execution_payload.get('artifact')
            )
            structure_generation_seed_summary['first_pass_execution_summary_artifact'] = (
                structure_first_pass_execution_payload.get('summary_artifact')
            )
            structure_generation_seed_summary['first_pass_execution_variants_artifact'] = (
                structure_first_pass_execution_payload.get('variants_artifact')
            )
            structure_generation_seed_summary['first_pass_execution_structure_dir'] = (
                structure_first_pass_execution_payload.get('structure_dir')
            )
            structure_generation_seed_summary['first_pass_execution_method'] = (
                structure_first_pass_execution_payload.get('method')
            )
            structure_generation_seed_summary['first_pass_execution_note'] = (
                structure_first_pass_execution_payload.get('note')
            )
            structure_generation_seed_summary['first_pass_execution_candidate_count'] = int(
                structure_first_pass_execution_payload.get(
                    'candidate_count',
                    len(structure_first_pass_execution_summary_df),
                )
                or 0
            )
            structure_generation_seed_summary['first_pass_execution_variant_count'] = int(
                structure_first_pass_execution_payload.get('variant_count', 0) or 0
            )
            structure_generation_seed_summary['first_pass_execution_successful_variant_count'] = int(
                structure_first_pass_execution_payload.get('successful_variant_count', 0) or 0
            )
            structure_generation_seed_summary['first_pass_execution_status_counts'] = {
                str(key): int(value)
                for key, value in (structure_first_pass_execution_payload.get('status_counts') or {}).items()
            }
            structure_generation_seed_summary['first_pass_execution_executed_formulas'] = [
                str(value)
                for value in (structure_first_pass_execution_payload.get('executed_formulas') or [])
            ]
            structure_generation_seed_summary['first_pass_execution_model_feature_set'] = (
                structure_first_pass_execution_payload.get('model_feature_set')
            )
            structure_generation_seed_summary['first_pass_execution_model_type'] = (
                structure_first_pass_execution_payload.get('model_type')
            )
            structure_generation_seed_summary['first_pass_execution_model_available'] = bool(
                structure_first_pass_execution_payload.get('model_available', False)
            )

    bn_candidate_compatible_evaluation_df = _build_bn_candidate_compatible_evaluation_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df=bn_family_benchmark_df,
        bn_stratified_error_df=bn_stratified_error_df,
    )
    bn_evaluation_matrix_df = _build_bn_evaluation_matrix_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df,
        bn_stratified_error_df,
    )
    candidate_ranking_uncertainty_df, candidate_ranking_uncertainty_summary = (
        _candidate_ranking_uncertainty_table(
            candidate_df,
            formula_col=formula_col,
            cfg=cfg,
            candidate_prediction_member_df=candidate_prediction_member_df,
            candidate_grouped_robustness_member_df=candidate_grouped_robustness_member_df,
            bn_centered_grouped_robustness_member_df=bn_centered_grouped_robustness_member_df,
            bn_centered_candidate_df=bn_centered_candidate_df,
            structure_followup_shortlist_df=(
                selected_followup_df if 'selected_followup_df' in locals() else pd.DataFrame()
            ),
        )
    )
    ranking_stability_cfg = _ranking_stability_config(cfg)
    decision_policy_cfg = _decision_policy_config(cfg)
    bn_centered_top_k_views = {}
    for top_k_value in ranking_stability_cfg['top_k_values']:
        bn_centered_top_k_views[f'top_{int(top_k_value)}'] = _candidate_ranking_comparison_payload(
            candidate_df,
            bn_centered_candidate_df,
            formula_col=formula_col,
            top_k=int(top_k_value),
        )
    bn_centered_summary['comparison_top_k_views'] = bn_centered_top_k_views

    return {
        'dataset': {
            'dataset_name': cfg['data']['dataset'],
            'target_column': target_col,
            'rows': int(len(dataset_df)),
            'target_non_null_rows': int(dataset_df['target'].notna().sum()),
            'unique_formulas': int(dataset_df[formula_col].astype(str).nunique()),
            'bn_rows': int(len(bn_df)),
            'bn_unique_formulas': int(bn_df[formula_col].astype(str).nunique()) if not bn_df.empty else 0,
        },
        'features': {
            'feature_space_family': cfg['features'].get('feature_family', 'mixed_formula_and_structure'),
            'configured_default_feature_set': cfg['features']['feature_set'],
            'candidate_feature_sets': selection_summary.get(
                'candidate_feature_sets',
                cfg['features'].get('candidate_sets', [cfg['features']['feature_set']]),
            ),
            'selected_feature_set': selected_feature_set,
            'selected_feature_family': selected_feature_family,
            'selected_feature_count': selection_summary.get('selected_feature_count'),
            'feature_set_results': selection_summary.get('feature_set_results', []),
        },
        'split': split_metadata,
        'feature_model_selection': selection_summary,
        'benchmarking': {
            'benchmark_artifact': 'benchmark_results.csv',
            'candidate_model_types': selection_summary.get(
                'candidate_model_types',
                cfg['model'].get('candidate_types', [cfg['model']['type']]),
            ),
            'dummy_baselines': cfg['model'].get('benchmark_baselines', ['dummy_mean']),
        },
        'robustness': {
            'enabled': robustness_enabled,
            'robustness_artifact': 'robustness_results.csv' if robustness_enabled else None,
            'method': robustness_cfg.get('method', 'group_kfold_by_formula'),
            'group_column': robustness_cfg.get(
                'group_column',
                cfg.get('split', {}).get('group_column', formula_col),
            ),
            'requested_folds': int(robustness_cfg.get('n_splits', 5)),
            'note': robustness_cfg.get(
                'note',
                (
                    'Runs grouped-by-formula cross-validation across configured feature/model '
                    'combos so the evaluation story does not depend on a single holdout split.'
                ),
            ),
            'result_row_count': int(len(robustness_df)),
            'successful_result_rows': int(
                robustness_df['robustness_status'].eq('ok').sum()
            ) if 'robustness_status' in robustness_df.columns else 0,
            'failed_result_rows': int(
                robustness_df['robustness_status'].eq('evaluation_failed').sum()
            ) if 'robustness_status' in robustness_df.columns else 0,
            'selected_feature_set': selected_feature_set,
            'selected_model_type': selected_model_type,
            'screening_feature_set': screening_feature_set,
            'screening_model_type': screening_model_type,
            'selected_model_metrics': _robustness_row_payload(
                robustness_df,
                robustness_df['selected_by_validation'].fillna(False).astype(bool),
            ) if 'selected_by_validation' in robustness_df.columns else None,
            'screening_model_metrics': _robustness_row_payload(
                robustness_df,
                robustness_df['feature_set'].astype(str).eq(screening_feature_set)
                & robustness_df['model_type'].astype(str).eq(screening_model_type),
            ) if {'feature_set', 'model_type'}.issubset(robustness_df.columns) else None,
            'dummy_baseline_metrics': _robustness_row_payload(
                robustness_df,
                robustness_df['benchmark_role'].astype(str).eq('dummy_baseline'),
            ) if 'benchmark_role' in robustness_df.columns else None,
        },
        'bn_slice_benchmark': {
            'enabled': bool(bn_slice_benchmark_cfg['enabled']),
            'benchmark_artifact': (
                'bn_slice_benchmark_results.csv' if bool(bn_slice_benchmark_cfg['enabled']) else None
            ),
            'prediction_artifact': (
                'bn_slice_predictions.csv' if bool(bn_slice_benchmark_cfg['enabled']) else None
            ),
            'method': bn_slice_benchmark_cfg['method'],
            'k_neighbors': int(bn_slice_benchmark_cfg['k_neighbors']),
            'note': bn_slice_benchmark_cfg['note'],
            'bn_rows': int(len(bn_df)),
            'bn_unique_formulas': int(bn_df[formula_col].astype(str).nunique()) if not bn_df.empty else 0,
            'standard_split_bn_train_rows': bn_train_rows,
            'standard_split_bn_val_rows': bn_val_rows,
            'standard_split_bn_test_rows': bn_test_rows,
            'standard_split_has_bn_eval_rows': bool((bn_val_rows + bn_test_rows) > 0),
            'result_row_count': int(len(bn_slice_benchmark_df)),
            'successful_result_rows': int(
                bn_slice_benchmark_df['benchmark_status'].eq('ok').sum()
            ) if 'benchmark_status' in bn_slice_benchmark_df.columns else 0,
            'non_ok_result_rows': int(
                (~bn_slice_benchmark_df['benchmark_status'].eq('ok')).sum()
            ) if 'benchmark_status' in bn_slice_benchmark_df.columns else 0,
            'selected_model_metrics': bn_slice_selected_metrics,
            'screening_model_metrics': bn_slice_screening_metrics,
            'bn_local_reference_metrics': bn_slice_bn_local_metrics,
            'global_dummy_baseline_metrics': bn_slice_global_dummy_metrics,
            'best_candidate_model_metrics': bn_slice_best_candidate_metrics,
            'selected_model_beats_global_dummy': bn_slice_selected_beats_global_dummy,
            'screening_model_beats_global_dummy': bn_slice_screening_beats_global_dummy,
            'best_candidate_model_beats_global_dummy': bn_slice_best_candidate_beats_global_dummy,
            'selected_model_matches_best_candidate': bn_slice_selected_matches_best_candidate,
            'candidate_compatible_evaluation_artifact': (
                'bn_candidate_compatible_evaluation.csv'
                if not bn_candidate_compatible_evaluation_df.empty
                else None
            ),
            'candidate_compatible_result_row_count': int(len(bn_candidate_compatible_evaluation_df)),
            'family_benchmark_artifact': (
                'bn_family_benchmark_results.csv' if bool(bn_family_benchmark_cfg['enabled']) else None
            ),
            'family_prediction_artifact': (
                'bn_family_predictions.csv' if bool(bn_family_benchmark_cfg['enabled']) else None
            ),
            'family_benchmark_method': bn_family_benchmark_cfg['method'],
            'family_grouping_method': bn_family_benchmark_cfg['grouping_method'],
            'family_k_neighbors': int(bn_family_benchmark_cfg['k_neighbors']),
            'family_note': bn_family_benchmark_cfg['note'],
            'family_result_row_count': int(len(bn_family_benchmark_df)),
            'family_successful_result_rows': int(
                bn_family_benchmark_df['benchmark_status'].eq('ok').sum()
            ) if 'benchmark_status' in bn_family_benchmark_df.columns else 0,
            'family_selected_model_metrics': bn_family_selected_metrics,
            'family_screening_model_metrics': bn_family_screening_metrics,
            'family_bn_local_reference_metrics': bn_family_bn_local_metrics,
            'family_global_dummy_baseline_metrics': bn_family_global_dummy_metrics,
            'family_best_candidate_model_metrics': bn_family_best_candidate_metrics,
            'family_selected_model_beats_global_dummy': bn_family_selected_beats_global_dummy,
            'family_screening_model_beats_global_dummy': bn_family_screening_beats_global_dummy,
            'family_best_candidate_model_beats_global_dummy': bn_family_best_candidate_beats_global_dummy,
            'stratified_error_artifact': (
                'bn_stratified_error_results.csv'
                if bool(bn_stratified_error_cfg['enabled'])
                else None
            ),
            'stratified_error_method': bn_stratified_error_cfg['method'],
            'stratified_error_group_column': bn_stratified_error_cfg['group_column'],
            'stratified_error_requested_folds': int(bn_stratified_error_cfg['n_splits']),
            'stratified_error_note': bn_stratified_error_cfg['note'],
            'stratified_error_result_row_count': int(len(bn_stratified_error_df)),
            'stratified_selected_model_metrics': bn_stratified_selected_metrics,
            'stratified_screening_model_metrics': bn_stratified_screening_metrics,
            'stratified_dummy_baseline_metrics': bn_stratified_dummy_metrics,
            'evaluation_matrix_artifact': (
                'bn_evaluation_matrix.csv' if not bn_evaluation_matrix_df.empty else None
            ),
            'evaluation_matrix_row_count': int(len(bn_evaluation_matrix_df)),
        },
        'screening': {
            'candidate_space_name': candidate_space_name,
            'candidate_space_kind': candidate_space_kind,
            'candidate_space_note': candidate_space_note,
            'candidate_generation_strategy': candidate_generation_strategy,
            'objective': {
                'name': cfg['screening'].get(
                    'objective_name',
                    'bn_themed_formula_level_wide_gap_followup_prioritization',
                ),
                'target_property': cfg['screening'].get('objective_target_property', target_col),
                'target_direction': cfg['screening'].get('objective_target_direction', 'maximize'),
                'decision_unit': cfg['screening'].get(
                    'objective_decision_unit',
                    'formula_level_candidate',
                ),
                'decision_consequence': cfg['screening'].get(
                    'objective_decision_consequence',
                    'low_confidence_prioritization_for_structure_followup',
                ),
                'note': cfg['screening'].get(
                    'objective_note',
                    'The screening objective is low-confidence formula-level candidate '
                    'prioritization for downstream structure follow-up, not direct discovery.',
                ),
            },
            'candidate_family_counts': candidate_family_counts,
            'candidate_rows': int(len(candidate_df)),
            'candidate_formulas_have_structures': False,
            'top_k': int(cfg['screening']['top_k']),
            'ranking_artifact': 'demo_candidate_ranking.csv',
            'bn_centered_alternative': bn_centered_summary,
            'structure_generation_bridge': structure_generation_seed_summary,
            'ranking_stability': {
                'enabled': bool(ranking_stability_cfg['enabled']),
                'artifact': (
                    'demo_candidate_ranking_uncertainty.csv'
                    if bool(ranking_stability_cfg['enabled'])
                    else None
                ),
                'note': ranking_stability_cfg['note'],
                'source_count': int(candidate_ranking_uncertainty_summary['source_count']),
                'top_k_reference': int(candidate_ranking_uncertainty_summary['top_k_reference']),
                'top_k_values': candidate_ranking_uncertainty_summary['top_k_values'],
                'prediction_interval_lower_quantile': float(
                    candidate_ranking_uncertainty_summary['prediction_interval_lower_quantile']
                ),
                'prediction_interval_upper_quantile': float(
                    candidate_ranking_uncertainty_summary['prediction_interval_upper_quantile']
                ),
                'prediction_std_abstain_threshold': candidate_ranking_uncertainty_summary[
                    'prediction_std_abstain_threshold'
                ],
                'rank_std_abstain_threshold': candidate_ranking_uncertainty_summary[
                    'rank_std_abstain_threshold'
                ],
                'comparison_top_k_views': bn_centered_top_k_views,
            },
            'decision_policy': {
                'enabled': bool(decision_policy_cfg['enabled']),
                'artifact': (
                    'demo_candidate_ranking_uncertainty.csv'
                    if bool(decision_policy_cfg['enabled'])
                    else None
                ),
                'note': decision_policy_cfg['note'],
                'global_support_abstain_below_percentile': float(
                    decision_policy_cfg['global_support_abstain_below_percentile']
                ),
                'bn_support_abstain_below_percentile': float(
                    decision_policy_cfg['bn_support_abstain_below_percentile']
                ),
                'prediction_std_above_quantile': float(
                    decision_policy_cfg['prediction_std_above_quantile']
                ),
                'rank_std_above_quantile': float(decision_policy_cfg['rank_std_above_quantile']),
                'minimum_top_10_selection_frequency': float(
                    decision_policy_cfg['minimum_top_10_selection_frequency']
                ),
                'abstained_candidate_count': int(
                    candidate_ranking_uncertainty_summary['abstained_candidate_count']
                ),
                'final_action_counts': candidate_ranking_uncertainty_summary['final_action_counts'],
            },
            **proposal_shortlist_summary,
            **extrapolation_shortlist_summary,
            'ranking_basis': ranking_basis,
            'ranking_note': ranking_note,
            'ranking_feature_set': screening_feature_set,
            'ranking_feature_family': screening_feature_family,
            'ranking_model_type': screening_model_type,
            'domain_support_enabled': bool(ranking_metadata['domain_support_enabled']),
            'domain_support_method': ranking_metadata['domain_support_method'],
            'domain_support_distance_metric': ranking_metadata['domain_support_distance_metric'],
            'domain_support_reference_split': ranking_metadata['domain_support_reference_split'],
            'domain_support_reference_formula_count': domain_support_reference_formula_count,
            'domain_support_k_neighbors': int(ranking_metadata['domain_support_k_neighbors']),
            'domain_support_note': ranking_metadata['domain_support_note'],
            'domain_support_penalty_enabled': bool(ranking_metadata['domain_support_penalty_enabled']),
            'domain_support_penalty_active': bool(ranking_metadata['domain_support_penalty_active']),
            'domain_support_penalty_weight': float(ranking_metadata['domain_support_penalty_weight']),
            'domain_support_penalize_below_percentile': float(
                ranking_metadata['domain_support_penalize_below_percentile']
            ),
            'domain_support_penalized_rows': domain_support_penalized_rows,
            'domain_support_low_support_rows': domain_support_low_support_rows,
            'bn_support_enabled': bool(ranking_metadata['bn_support_enabled']),
            'bn_support_method': ranking_metadata['bn_support_method'],
            'bn_support_distance_metric': ranking_metadata['bn_support_distance_metric'],
            'bn_support_reference_split': ranking_metadata['bn_support_reference_split'],
            'bn_support_reference_formula_count': bn_support_reference_formula_count,
            'bn_support_k_neighbors': int(ranking_metadata['bn_support_k_neighbors']),
            'bn_support_note': ranking_metadata['bn_support_note'],
            'bn_support_penalty_enabled': bool(ranking_metadata['bn_support_penalty_enabled']),
            'bn_support_penalty_active': bool(ranking_metadata['bn_support_penalty_active']),
            'bn_support_penalty_weight': float(ranking_metadata['bn_support_penalty_weight']),
            'bn_support_penalize_below_percentile': float(
                ranking_metadata['bn_support_penalize_below_percentile']
            ),
            'bn_support_penalized_rows': bn_support_penalized_rows,
            'bn_support_low_support_rows': bn_support_low_support_rows,
            'grouped_robustness_uncertainty_enabled': bool(
                ranking_metadata['grouped_robustness_uncertainty_enabled']
            ),
            'grouped_robustness_uncertainty_method': ranking_metadata[
                'grouped_robustness_uncertainty_method'
            ],
            'grouped_robustness_uncertainty_note': ranking_metadata[
                'grouped_robustness_uncertainty_note'
            ],
            'grouped_robustness_penalty_enabled': bool(
                ranking_metadata['grouped_robustness_penalty_enabled']
            ),
            'grouped_robustness_penalty_active': bool(
                ranking_metadata['grouped_robustness_penalty_active']
            ),
            'grouped_robustness_penalty_weight': float(
                ranking_metadata['grouped_robustness_penalty_weight']
            ),
            'grouped_robustness_prediction_fold_count': grouped_robustness_prediction_fold_count,
            'grouped_robustness_prediction_std_mean': grouped_robustness_prediction_std_mean,
            'grouped_robustness_penalized_rows': grouped_robustness_penalized_rows,
            'bn_analog_evidence_enabled': bn_analog_evidence_enabled,
            'bn_analog_reference_formula_count': bn_analog_reference_formula_count,
            'bn_analog_reference_band_gap_median': bn_analog_reference_band_gap_median,
            'bn_analog_reference_band_gap_iqr': bn_analog_reference_band_gap_iqr,
            'bn_analog_reference_exfoliation_energy_median': bn_analog_reference_exfoliation_energy_median,
            'bn_analog_reference_energy_per_atom_median': bn_analog_reference_energy_per_atom_median,
            'bn_analog_reference_abs_total_magnetization_median': bn_analog_reference_abs_total_magnetization_median,
            'bn_analog_exfoliation_available_rows': bn_analog_exfoliation_available_rows,
            'bn_analog_lower_or_equal_reference_rows': bn_analog_lower_or_equal_reference_rows,
            'bn_analog_higher_reference_rows': bn_analog_higher_reference_rows,
            'bn_band_gap_alignment_enabled': bool(
                ranking_metadata['bn_band_gap_alignment_enabled']
            ),
            'bn_band_gap_alignment_method': ranking_metadata['bn_band_gap_alignment_method'],
            'bn_band_gap_alignment_reference_split': ranking_metadata[
                'bn_band_gap_alignment_reference_split'
            ],
            'bn_band_gap_alignment_window_expansion_iqr_factor': float(
                ranking_metadata['bn_band_gap_alignment_window_expansion_iqr_factor']
            ),
            'bn_band_gap_alignment_minimum_neighbor_formula_count_for_penalty': int(
                ranking_metadata[
                    'bn_band_gap_alignment_minimum_neighbor_formula_count_for_penalty'
                ]
            ),
            'bn_band_gap_alignment_note': ranking_metadata['bn_band_gap_alignment_note'],
            'bn_band_gap_alignment_penalty_enabled': bool(
                ranking_metadata['bn_band_gap_alignment_penalty_enabled']
            ),
            'bn_band_gap_alignment_penalty_active': bool(
                ranking_metadata['bn_band_gap_alignment_penalty_active']
            ),
            'bn_band_gap_alignment_penalty_weight': float(
                ranking_metadata['bn_band_gap_alignment_penalty_weight']
            ),
            'bn_band_gap_alignment_penalty_eligible_rows': (
                bn_band_gap_alignment_penalty_eligible_rows
            ),
            'bn_band_gap_alignment_within_window_rows': bn_band_gap_alignment_within_window_rows,
            'bn_band_gap_alignment_below_window_rows': bn_band_gap_alignment_below_window_rows,
            'bn_band_gap_alignment_above_window_rows': bn_band_gap_alignment_above_window_rows,
            'bn_band_gap_alignment_penalized_rows': bn_band_gap_alignment_penalized_rows,
            'bn_analog_reference_like_rows': bn_analog_reference_like_rows,
            'bn_analog_mixed_alignment_rows': bn_analog_mixed_alignment_rows,
            'bn_analog_reference_divergent_rows': bn_analog_reference_divergent_rows,
            'bn_analog_validation_enabled': bool(ranking_metadata['bn_analog_validation_enabled']),
            'bn_analog_validation_method': ranking_metadata['bn_analog_validation_method'],
            'bn_analog_validation_note': ranking_metadata['bn_analog_validation_note'],
            'bn_analog_validation_penalty_enabled': bool(
                ranking_metadata['bn_analog_validation_penalty_enabled']
            ),
            'bn_analog_validation_penalty_active': bool(
                ranking_metadata['bn_analog_validation_penalty_active']
            ),
            'bn_analog_validation_penalty_weight': float(
                ranking_metadata['bn_analog_validation_penalty_weight']
            ),
            'bn_analog_validation_penalized_rows': bn_analog_validation_penalized_rows,
            'chemical_plausibility_enabled': chemical_plausibility_enabled,
            'chemical_plausibility_method': chemical_plausibility_cfg.get(
                'method',
                'pymatgen_common_oxidation_state_balance',
            ),
            'chemical_plausibility_selection_policy': chemical_plausibility_cfg.get(
                'selection_policy',
                'annotate_and_prioritize_passing_candidates',
            ),
            'chemical_plausibility_note': chemical_plausibility_cfg.get(
                'note',
                (
                    'Formula-level plausibility annotation using pymatgen oxidation-state guesses. '
                    'This is not a structure, thermodynamic stability, phonon stability, or '
                    'synthesis feasibility filter.'
                ),
            ),
            'chemical_plausibility_passed_rows': plausibility_pass_count,
            'chemical_plausibility_failed_rows': plausibility_fail_count,
            'chemical_plausibility_failed_formulas': plausibility_failed_formulas,
            'novelty_annotation_enabled': novelty_annotation_enabled,
            'novelty_bucket_counts': novelty_bucket_counts,
            'standard_top_k_novelty_bucket_counts': standard_top_k_novelty_bucket_counts,
            'formula_level_extrapolation_candidate_count': formula_level_extrapolation_candidate_count,
            'formula_level_extrapolation_shortlist': formula_level_extrapolation_shortlist,
            'novelty_interpretation_note': novelty_interpretation_note,
            'screening_selection_scope': selection_summary.get(
                'screening_selection_scope',
                'candidate_compatible_formula_only',
            ),
            'screening_candidate_feature_sets': selection_summary.get(
                'screening_candidate_feature_sets',
                [],
            ),
            'screening_selection_note': screening_selection_note,
            'ranking_matches_best_overall_evaluation': screening_matches_overall,
            'best_overall_evaluation_feature_set': selected_feature_set,
            'best_overall_evaluation_feature_family': selected_feature_family,
            'best_overall_evaluation_model_type': selected_model_type,
            'best_overall_evaluation_candidate_compatible': feature_set_supports_formula_only_screening(
                selected_feature_set
            ),
            'ranking_uncertainty_method': ranking_metadata['ranking_uncertainty_method'],
            'ranking_uncertainty_penalty': float(ranking_metadata['ranking_uncertainty_penalty']),
            'candidate_annotations': [
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
            ],
        },
    }


def save_metrics_and_predictions(
    metrics,
    prediction_df,
    bn_df,
    screened_df,
    benchmark_df,
    robustness_df,
    bn_slice_benchmark_df,
    bn_slice_prediction_df,
    bn_centered_screened_df,
    structure_generation_seed_df,
    experiment_summary,
    manifest,
    cfg,
    candidate_prediction_member_df=None,
    candidate_grouped_robustness_member_df=None,
    bn_centered_grouped_robustness_member_df=None,
    structure_first_pass_execution_variant_df=None,
    structure_first_pass_execution_summary_df=None,
    structure_first_pass_execution_payload=None,
    bn_family_benchmark_df=None,
    bn_family_prediction_df=None,
    bn_stratified_error_df=None,
):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bn_family_benchmark_df = (
        pd.DataFrame() if bn_family_benchmark_df is None else bn_family_benchmark_df.copy()
    )
    bn_family_prediction_df = (
        pd.DataFrame() if bn_family_prediction_df is None else bn_family_prediction_df.copy()
    )
    bn_stratified_error_df = (
        pd.DataFrame() if bn_stratified_error_df is None else bn_stratified_error_df.copy()
    )
    structure_first_pass_execution_variant_df = (
        pd.DataFrame()
        if structure_first_pass_execution_variant_df is None
        else structure_first_pass_execution_variant_df.copy()
    )
    structure_first_pass_execution_summary_df = (
        pd.DataFrame()
        if structure_first_pass_execution_summary_df is None
        else structure_first_pass_execution_summary_df.copy()
    )
    structure_first_pass_execution_payload = dict(structure_first_pass_execution_payload or {})
    (artifact_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    prediction_df.to_csv(artifact_dir / 'predictions.csv', index=False)
    bn_df.to_csv(artifact_dir / 'bn_slice.csv', index=False)
    screened_df.to_csv(artifact_dir / 'demo_candidate_ranking.csv', index=False)
    candidate_uncertainty_path = artifact_dir / 'demo_candidate_ranking_uncertainty.csv'
    bn_candidate_compatible_evaluation_path = artifact_dir / 'bn_candidate_compatible_evaluation.csv'
    bn_family_benchmark_path = artifact_dir / 'bn_family_benchmark_results.csv'
    bn_family_prediction_path = artifact_dir / 'bn_family_predictions.csv'
    bn_stratified_error_path = artifact_dir / 'bn_stratified_error_results.csv'
    bn_evaluation_matrix_path = artifact_dir / 'bn_evaluation_matrix.csv'
    bn_centered_ranking_path = artifact_dir / 'demo_candidate_bn_centered_ranking.csv'
    if bn_centered_screened_df is not None and not bn_centered_screened_df.empty:
        bn_centered_screened_df.to_csv(bn_centered_ranking_path, index=False)
    elif bn_centered_ranking_path.exists():
        bn_centered_ranking_path.unlink()
    structure_generation_seed_path = artifact_dir / 'demo_candidate_structure_generation_seeds.csv'
    structure_generation_handoff_path = artifact_dir / 'demo_candidate_structure_generation_handoff.json'
    structure_generation_reference_records_path = (
        artifact_dir / 'demo_candidate_structure_generation_reference_records.json'
    )
    structure_generation_job_plan_path = (
        artifact_dir / 'demo_candidate_structure_generation_job_plan.json'
    )
    structure_generation_first_pass_queue_path = (
        artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json'
    )
    structure_generation_followup_shortlist_path = (
        artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv'
    )
    structure_generation_followup_extrapolation_shortlist_path = (
        artifact_dir / 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'
    )
    structure_first_pass_execution_path = None
    structure_first_pass_execution_summary_path = None
    structure_first_pass_execution_variants_path = None
    structure_first_pass_execution_structure_dir = None
    if structure_first_pass_execution_payload:
        structure_first_pass_execution_path = artifact_dir / str(
            structure_first_pass_execution_payload.get('artifact')
        )
        structure_first_pass_execution_summary_path = artifact_dir / str(
            structure_first_pass_execution_payload.get('summary_artifact')
        )
        structure_first_pass_execution_variants_path = artifact_dir / str(
            structure_first_pass_execution_payload.get('variants_artifact')
        )
        structure_first_pass_execution_structure_dir = artifact_dir / str(
            structure_first_pass_execution_payload.get('structure_dir')
        )
    selected_followup_df = pd.DataFrame()
    if structure_generation_seed_df is not None and not structure_generation_seed_df.empty:
        structure_generation_seed_df.to_csv(structure_generation_seed_path, index=False)
        structure_generation_seed_cfg = _structure_generation_seed_config(cfg)
        formula_col = ((cfg.get('data') or {}).get('formula_column') or 'formula')
        structure_generation_handoff = _build_structure_generation_handoff_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        structure_generation_handoff_path.write_text(
            json.dumps(structure_generation_handoff, indent=2, ensure_ascii=False)
        )
        structure_generation_reference_records = _build_structure_generation_reference_record_payload(
            structure_generation_seed_df,
            cfg=cfg,
        )
        structure_generation_reference_records_path.write_text(
            json.dumps(structure_generation_reference_records, indent=2, ensure_ascii=False)
        )
        structure_generation_job_plan = _build_structure_generation_job_plan_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        structure_generation_job_plan_path.write_text(
            json.dumps(structure_generation_job_plan, indent=2, ensure_ascii=False)
        )
        structure_generation_first_pass_queue = _build_structure_generation_first_pass_queue_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        structure_generation_first_pass_queue_path.write_text(
            json.dumps(structure_generation_first_pass_queue, indent=2, ensure_ascii=False)
        )
        structure_followup_shortlist_cfg = _structure_followup_shortlist_config(cfg)
        structure_followup_shortlist_df = _build_structure_generation_followup_shortlist_df(
            structure_generation_first_pass_queue,
            formula_col=formula_col,
            cfg_defaults=structure_followup_shortlist_cfg,
        )
        selected_followup_df = (
            structure_followup_shortlist_df.loc[
                structure_followup_shortlist_df['structure_followup_shortlist_selected']
                .fillna(False)
                .astype(bool)
            ].copy()
            if not structure_followup_shortlist_df.empty
            else pd.DataFrame()
        )
        if not selected_followup_df.empty:
            if 'structure_followup_shortlist_rank' in selected_followup_df.columns:
                selected_followup_df = selected_followup_df.sort_values(
                    'structure_followup_shortlist_rank', ascending=True
                )
            selected_followup_df.to_csv(structure_generation_followup_shortlist_path, index=False)
        elif structure_generation_followup_shortlist_path.exists():
            structure_generation_followup_shortlist_path.unlink()
        structure_followup_extrapolation_shortlist_cfg = (
            _structure_followup_extrapolation_shortlist_config(cfg)
        )
        structure_followup_extrapolation_shortlist_df = (
            _build_structure_generation_followup_extrapolation_shortlist_df(
                structure_followup_shortlist_df,
                formula_col=formula_col,
                cfg_defaults=structure_followup_extrapolation_shortlist_cfg,
            )
        )
        selected_followup_extrapolation_df = (
            structure_followup_extrapolation_shortlist_df.loc[
                structure_followup_extrapolation_shortlist_df[
                    'structure_followup_extrapolation_shortlist_selected'
                ].fillna(False).astype(bool)
            ].copy()
            if not structure_followup_extrapolation_shortlist_df.empty
            else pd.DataFrame()
        )
        if not selected_followup_extrapolation_df.empty:
            if 'structure_followup_extrapolation_shortlist_rank' in selected_followup_extrapolation_df.columns:
                selected_followup_extrapolation_df = selected_followup_extrapolation_df.sort_values(
                    'structure_followup_extrapolation_shortlist_rank', ascending=True
                )
            selected_followup_extrapolation_df.to_csv(
                structure_generation_followup_extrapolation_shortlist_path, index=False
            )
        elif structure_generation_followup_extrapolation_shortlist_path.exists():
            structure_generation_followup_extrapolation_shortlist_path.unlink()
    else:
        if structure_generation_seed_path.exists():
            structure_generation_seed_path.unlink()
        if structure_generation_handoff_path.exists():
            structure_generation_handoff_path.unlink()
        if structure_generation_reference_records_path.exists():
            structure_generation_reference_records_path.unlink()
        if structure_generation_job_plan_path.exists():
            structure_generation_job_plan_path.unlink()
        if structure_generation_first_pass_queue_path.exists():
            structure_generation_first_pass_queue_path.unlink()
        if structure_generation_followup_shortlist_path.exists():
            structure_generation_followup_shortlist_path.unlink()
        if structure_generation_followup_extrapolation_shortlist_path.exists():
            structure_generation_followup_extrapolation_shortlist_path.unlink()
    if (
        structure_first_pass_execution_payload
        and structure_first_pass_execution_summary_path is not None
        and structure_first_pass_execution_variants_path is not None
        and structure_first_pass_execution_path is not None
        and not structure_first_pass_execution_summary_df.empty
    ):
        structure_first_pass_execution_summary_df.to_csv(
            structure_first_pass_execution_summary_path,
            index=False,
        )
        structure_first_pass_execution_variant_df.to_csv(
            structure_first_pass_execution_variants_path,
            index=False,
        )
        if structure_first_pass_execution_structure_dir is not None:
            structure_first_pass_execution_structure_dir.mkdir(parents=True, exist_ok=True)
            for existing_cif_path in structure_first_pass_execution_structure_dir.glob('*.cif'):
                existing_cif_path.unlink()
        sanitized_candidates = []
        for candidate_payload in structure_first_pass_execution_payload.get('candidates', []):
            sanitized_candidate = {
                key: value
                for key, value in candidate_payload.items()
                if key != 'variants'
            }
            sanitized_variants = []
            for variant_payload in candidate_payload.get('variants', []):
                cif_text = variant_payload.get('_cif_text')
                cif_relative_path = variant_payload.get('generated_structure_cif_path')
                if (
                    cif_text
                    and cif_relative_path
                    and structure_first_pass_execution_structure_dir is not None
                ):
                    cif_output_path = artifact_dir / str(cif_relative_path)
                    cif_output_path.parent.mkdir(parents=True, exist_ok=True)
                    cif_output_path.write_text(str(cif_text), encoding='utf-8')
                sanitized_variants.append(
                    {
                        key: value
                        for key, value in variant_payload.items()
                        if key != '_cif_text'
                    }
                )
            sanitized_candidate['variants'] = sanitized_variants
            sanitized_candidates.append(sanitized_candidate)
        sanitized_payload = {
            **structure_first_pass_execution_payload,
            'candidates': sanitized_candidates,
        }
        structure_first_pass_execution_path.write_text(
            json.dumps(sanitized_payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
    else:
        for cleanup_path in (
            structure_first_pass_execution_summary_path,
            structure_first_pass_execution_variants_path,
            structure_first_pass_execution_path,
        ):
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink()
        if structure_first_pass_execution_structure_dir is not None and structure_first_pass_execution_structure_dir.exists():
            for existing_cif_path in structure_first_pass_execution_structure_dir.glob('*.cif'):
                existing_cif_path.unlink()
    for selected_column, rank_column, artifact_name in (
        (
            'proposal_shortlist_selected',
            'proposal_shortlist_rank',
            'demo_candidate_proposal_shortlist.csv',
        ),
        (
            'extrapolation_shortlist_selected',
            'extrapolation_shortlist_rank',
            'demo_candidate_extrapolation_shortlist.csv',
        ),
    ):
        shortlist_path = artifact_dir / artifact_name
        if selected_column in screened_df.columns:
            shortlist_df = screened_df.loc[
                screened_df[selected_column].fillna(False).astype(bool)
            ].copy()
            if rank_column in shortlist_df.columns:
                shortlist_df = shortlist_df.sort_values(rank_column, ascending=True)
            shortlist_df.to_csv(shortlist_path, index=False)
        elif shortlist_path.exists():
            shortlist_path.unlink()
    bn_candidate_compatible_evaluation_df = _build_bn_candidate_compatible_evaluation_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df=bn_family_benchmark_df,
        bn_stratified_error_df=bn_stratified_error_df,
    )
    if not bn_candidate_compatible_evaluation_df.empty:
        bn_candidate_compatible_evaluation_df.to_csv(
            bn_candidate_compatible_evaluation_path,
            index=False,
        )
    elif bn_candidate_compatible_evaluation_path.exists():
        bn_candidate_compatible_evaluation_path.unlink()

    if bn_family_benchmark_df is not None and not bn_family_benchmark_df.empty:
        bn_family_benchmark_df.to_csv(bn_family_benchmark_path, index=False)
    elif bn_family_benchmark_path.exists():
        bn_family_benchmark_path.unlink()
    if bn_family_prediction_df is not None and not bn_family_prediction_df.empty:
        bn_family_prediction_df.to_csv(bn_family_prediction_path, index=False)
    elif bn_family_prediction_path.exists():
        bn_family_prediction_path.unlink()
    if bn_stratified_error_df is not None and not bn_stratified_error_df.empty:
        bn_stratified_error_df.to_csv(bn_stratified_error_path, index=False)
    elif bn_stratified_error_path.exists():
        bn_stratified_error_path.unlink()

    bn_evaluation_matrix_df = _build_bn_evaluation_matrix_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df,
        bn_stratified_error_df,
    )
    if not bn_evaluation_matrix_df.empty:
        bn_evaluation_matrix_df.to_csv(bn_evaluation_matrix_path, index=False)
    elif bn_evaluation_matrix_path.exists():
        bn_evaluation_matrix_path.unlink()

    candidate_ranking_uncertainty_df, _ = _candidate_ranking_uncertainty_table(
        screened_df,
        formula_col=((cfg.get('data') or {}).get('formula_column') or 'formula'),
        cfg=cfg,
        candidate_prediction_member_df=candidate_prediction_member_df,
        candidate_grouped_robustness_member_df=candidate_grouped_robustness_member_df,
        bn_centered_grouped_robustness_member_df=bn_centered_grouped_robustness_member_df,
        bn_centered_candidate_df=bn_centered_screened_df,
        structure_followup_shortlist_df=selected_followup_df,
    )
    if not candidate_ranking_uncertainty_df.empty:
        candidate_ranking_uncertainty_df.to_csv(candidate_uncertainty_path, index=False)
    elif candidate_uncertainty_path.exists():
        candidate_uncertainty_path.unlink()

    benchmark_df.to_csv(artifact_dir / 'benchmark_results.csv', index=False)
    robustness_path = artifact_dir / 'robustness_results.csv'
    if robustness_df is not None and not robustness_df.empty:
        robustness_df.to_csv(robustness_path, index=False)
    elif robustness_path.exists():
        robustness_path.unlink()
    bn_slice_benchmark_path = artifact_dir / 'bn_slice_benchmark_results.csv'
    if bn_slice_benchmark_df is not None and not bn_slice_benchmark_df.empty:
        bn_slice_benchmark_df.to_csv(bn_slice_benchmark_path, index=False)
    elif bn_slice_benchmark_path.exists():
        bn_slice_benchmark_path.unlink()
    bn_slice_prediction_path = artifact_dir / 'bn_slice_predictions.csv'
    if bn_slice_prediction_df is not None and not bn_slice_prediction_df.empty:
        bn_slice_prediction_df.to_csv(bn_slice_prediction_path, index=False)
    elif bn_slice_prediction_path.exists():
        bn_slice_prediction_path.unlink()
    (artifact_dir / 'experiment_summary.json').write_text(
        json.dumps(experiment_summary, indent=2, ensure_ascii=False)
    )
    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    legacy_screen_path = artifact_dir / 'screened_candidates.csv'
    if legacy_screen_path.exists():
        legacy_screen_path.unlink()


def save_basic_plots(prediction_df, cfg):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(prediction_df['target'], prediction_df['prediction'], alpha=0.7)
    ax.set_xlabel('True target')
    ax.set_ylabel('Predicted target')
    ax.set_title('Parity plot')
    fig.tight_layout()
    fig.savefig(artifact_dir / 'parity_plot.png', dpi=160)
    plt.close(fig)
