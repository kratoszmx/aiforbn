from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ai_for_bn_mplconfig')
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.features import (
    _bn_slice_benchmark_config,
    _extrapolation_shortlist_config,
    _proposal_shortlist_config,
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
    ranking_config_metadata = get_screening_ranking_metadata(cfg)
    robustness_cfg = cfg.get('robustness', {})
    robustness_enabled = bool(robustness_cfg.get('enabled', False))
    bn_slice_benchmark_cfg = _bn_slice_benchmark_config(cfg)
    robustness_df = pd.DataFrame() if robustness_df is None else robustness_df.copy()
    bn_slice_benchmark_df = (
        pd.DataFrame() if bn_slice_benchmark_df is None else bn_slice_benchmark_df.copy()
    )

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
            candidate_df = bn_slice_benchmark_df.loc[candidate_mask].copy()
            if not candidate_df.empty:
                best_idx = candidate_df['mae'].astype(float).idxmin()
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
        },
        'screening': {
            'candidate_space_name': candidate_space_name,
            'candidate_space_kind': candidate_space_kind,
            'candidate_space_note': candidate_space_note,
            'candidate_generation_strategy': candidate_generation_strategy,
            'candidate_family_counts': candidate_family_counts,
            'candidate_rows': int(len(candidate_df)),
            'candidate_formulas_have_structures': False,
            'top_k': int(cfg['screening']['top_k']),
            'ranking_artifact': 'demo_candidate_ranking.csv',
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
    experiment_summary,
    manifest,
    cfg,
):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    prediction_df.to_csv(artifact_dir / 'predictions.csv', index=False)
    bn_df.to_csv(artifact_dir / 'bn_slice.csv', index=False)
    screened_df.to_csv(artifact_dir / 'demo_candidate_ranking.csv', index=False)
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
    (artifact_dir / 'experiment_summary.json').write_text(json.dumps(experiment_summary, indent=2))
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
