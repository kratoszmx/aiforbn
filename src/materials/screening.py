from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from materials.data import REFERENCE_PROPERTY_COLUMNS, STRUCTURE_SUMMARY_COLUMNS

from materials.constants import *
from materials.candidate_space import *
from materials.candidate_space import (
    _bn_analog_evidence_config,
    _bn_analog_validation_config,
    _bn_band_gap_alignment_config,
    _grouped_robustness_uncertainty_config,
    _ordered_values,
    _robustness_config,
    _structure_generation_seed_config,
    _structure_seed_edit_metadata,
)
from materials.feature_building import *
from materials.feature_building import _feature_columns, _feature_valid_mask
from materials.modeling import *
from materials.benchmarking import *
from materials.benchmarking import _group_kfold_splits, _split_pipe_delimited_values
from materials.selection import (
    _ranking_active_penalty_terms,
    _ranking_decision_summary,
    _ranking_main_penalty_driver,
    _screening_selection_note,
)

def build_candidate_structure_generation_seeds(
    candidate_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    split_masks,
    cfg: dict | None = None,
    bn_centered_candidate_df: pd.DataFrame | None = None,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    seed_cfg = _structure_generation_seed_config(cfg)
    out_columns = [
        formula_col,
        'candidate_family',
        'candidate_template',
        'candidate_novelty_bucket',
        'chemical_plausibility_pass',
        'ranking_rank',
        'proposal_shortlist_selected',
        'proposal_shortlist_rank',
        'extrapolation_shortlist_selected',
        'extrapolation_shortlist_rank',
        'bn_centered_ranking_rank',
        'bn_centered_top_n_selected',
        'rank_shift_general_minus_bn_centered',
        'absolute_rank_shift_vs_bn_centered',
        'structure_generation_seed_enabled',
        'structure_generation_seed_label',
        'structure_generation_seed_method',
        'structure_generation_seed_candidate_scope',
        'structure_generation_seed_note',
        'structure_generation_candidate_selected',
        'structure_generation_candidate_priority_reason',
        'structure_generation_seed_limit',
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
        *[f'seed_reference_{column}' for column in REFERENCE_PROPERTY_COLUMNS],
        'seed_reference_has_structure_summary',
        *[f'seed_reference_{column}' for column in STRUCTURE_SUMMARY_COLUMNS],
    ]
    if candidate_df is None or candidate_df.empty or formula_col not in candidate_df.columns:
        return pd.DataFrame(columns=out_columns)

    candidate_ranking_df = candidate_df.copy()
    candidate_ranking_df[formula_col] = candidate_ranking_df[formula_col].astype(str)
    if 'ranking_rank' in candidate_ranking_df.columns:
        candidate_ranking_df = candidate_ranking_df.sort_values(
            'ranking_rank', ascending=True, kind='stable'
        ).reset_index(drop=True)
    else:
        candidate_ranking_df = candidate_ranking_df.reset_index(drop=True)
        candidate_ranking_df['ranking_rank'] = np.arange(1, len(candidate_ranking_df) + 1, dtype=int)

    for column_name, default_value in (
        ('proposal_shortlist_selected', False),
        ('proposal_shortlist_rank', pd.NA),
        ('extrapolation_shortlist_selected', False),
        ('extrapolation_shortlist_rank', pd.NA),
        ('candidate_novelty_bucket', pd.NA),
        ('chemical_plausibility_pass', pd.NA),
        ('candidate_family', pd.NA),
        ('candidate_template', pd.NA),
    ):
        if column_name not in candidate_ranking_df.columns:
            candidate_ranking_df[column_name] = default_value

    bn_centered_top_formulas: set[str] = set()
    bn_centered_rank_map: dict[str, int] = {}
    if (
        bn_centered_candidate_df is not None
        and not bn_centered_candidate_df.empty
        and formula_col in bn_centered_candidate_df.columns
    ):
        alternative_df = bn_centered_candidate_df.copy()
        alternative_df[formula_col] = alternative_df[formula_col].astype(str)
        if 'ranking_rank' in alternative_df.columns:
            alternative_df = alternative_df.sort_values(
                'ranking_rank', ascending=True, kind='stable'
            ).reset_index(drop=True)
        else:
            alternative_df = alternative_df.reset_index(drop=True)
            alternative_df['ranking_rank'] = np.arange(1, len(alternative_df) + 1, dtype=int)
        bn_centered_rank_map = {
            str(row[formula_col]): int(row['ranking_rank'])
            for _, row in alternative_df[[formula_col, 'ranking_rank']].iterrows()
        }
        bn_centered_top_formulas = set(
            alternative_df.head(int(seed_cfg['bn_centered_top_n']))[formula_col].astype(str)
        )

    candidate_ranking_df['bn_centered_ranking_rank'] = candidate_ranking_df[formula_col].map(
        bn_centered_rank_map
    )
    candidate_ranking_df['bn_centered_ranking_rank'] = candidate_ranking_df[
        'bn_centered_ranking_rank'
    ].astype('Int64')
    candidate_ranking_df['bn_centered_top_n_selected'] = candidate_ranking_df[formula_col].isin(
        bn_centered_top_formulas
    )
    candidate_ranking_df['rank_shift_general_minus_bn_centered'] = pd.Series(
        pd.NA, index=candidate_ranking_df.index, dtype='Float64'
    )
    comparable_rank_mask = candidate_ranking_df['bn_centered_ranking_rank'].notna()
    candidate_ranking_df.loc[comparable_rank_mask, 'rank_shift_general_minus_bn_centered'] = (
        candidate_ranking_df.loc[comparable_rank_mask, 'ranking_rank'].astype(float)
        - candidate_ranking_df.loc[comparable_rank_mask, 'bn_centered_ranking_rank'].astype(float)
    )
    candidate_ranking_df['absolute_rank_shift_vs_bn_centered'] = (
        candidate_ranking_df['rank_shift_general_minus_bn_centered'].abs().astype('Float64')
    )

    proposal_mask = candidate_ranking_df['proposal_shortlist_selected'].fillna(False).astype(bool)
    extrapolation_mask = candidate_ranking_df['extrapolation_shortlist_selected'].fillna(False).astype(bool)
    bn_top_mask = candidate_ranking_df['bn_centered_top_n_selected'].fillna(False).astype(bool)
    candidate_ranking_df['structure_generation_candidate_selected'] = (
        proposal_mask | extrapolation_mask | bn_top_mask
    )
    if not candidate_ranking_df['structure_generation_candidate_selected'].any():
        fallback_n = min(int(seed_cfg['bn_centered_top_n']), len(candidate_ranking_df))
        candidate_ranking_df['structure_generation_candidate_selected'] = candidate_ranking_df[
            'ranking_rank'
        ].le(fallback_n)
    else:
        fallback_n = None

    priority_reasons: list[str] = []
    for _, row in candidate_ranking_df.iterrows():
        reasons: list[str] = []
        if bool(row['proposal_shortlist_selected']):
            reasons.append('proposal_shortlist')
        if bool(row['extrapolation_shortlist_selected']):
            reasons.append('extrapolation_shortlist')
        if bool(row['bn_centered_top_n_selected']):
            reasons.append(f"bn_centered_top_{int(seed_cfg['bn_centered_top_n'])}")
        if not reasons and bool(row['structure_generation_candidate_selected']) and fallback_n is not None:
            reasons.append(f'general_top_{fallback_n}_fallback')
        priority_reasons.append('|'.join(reasons) if reasons else 'not_selected')
    candidate_ranking_df['structure_generation_candidate_priority_reason'] = priority_reasons

    candidate_ranking_df['structure_generation_seed_enabled'] = bool(seed_cfg['enabled'])
    candidate_ranking_df['structure_generation_seed_label'] = str(seed_cfg['label'])
    candidate_ranking_df['structure_generation_seed_method'] = str(seed_cfg['method'])
    candidate_ranking_df['structure_generation_seed_candidate_scope'] = str(
        seed_cfg['candidate_scope']
    )
    candidate_ranking_df['structure_generation_seed_note'] = str(seed_cfg['note'])
    candidate_ranking_df['structure_generation_seed_limit'] = int(seed_cfg['per_candidate_seed_limit'])

    if not bool(seed_cfg['enabled']):
        return candidate_ranking_df.loc[
            candidate_ranking_df['structure_generation_candidate_selected'].fillna(False).astype(bool),
            [column for column in out_columns if column in candidate_ranking_df.columns],
        ].copy()

    target_col = ((cfg or {}).get('data') or {}).get('target_column', 'target')
    train_mask = pd.Series(split_masks.get('train', []), index=dataset_df.index).fillna(False).astype(bool)
    val_mask = pd.Series(split_masks.get('val', []), index=dataset_df.index).fillna(False).astype(bool)
    reference_df = dataset_df.loc[train_mask | val_mask].copy()
    reference_df = filter_bn(reference_df, formula_col=formula_col)
    if reference_df.empty:
        return pd.DataFrame(columns=out_columns)

    if target_col not in reference_df.columns:
        if 'target' in reference_df.columns:
            target_col = 'target'
        else:
            raise ValueError(
                'Structure-generation seed construction requires the target column to be present '
                'in the reference dataset.'
            )

    structure_columns = [column for column in STRUCTURE_SUMMARY_COLUMNS if column in reference_df.columns]
    reference_df['reference_has_structure_summary'] = (
        reference_df[structure_columns].notna().any(axis=1)
        if structure_columns
        else False
    )
    reference_df['reference_formula_row_count'] = reference_df.groupby(formula_col)[formula_col].transform('size')
    reference_df['reference_formula_mean_band_gap'] = reference_df.groupby(formula_col)[target_col].transform('mean')
    reference_df['reference_formula_mean_absolute_deviation'] = (
        reference_df[target_col] - reference_df['reference_formula_mean_band_gap']
    ).abs()
    reference_df['_seed_sort_record_id'] = (
        reference_df['record_id'].astype(str) if 'record_id' in reference_df.columns else ''
    )
    reference_df = reference_df.sort_values(
        [
            formula_col,
            'reference_has_structure_summary',
            'reference_formula_mean_absolute_deviation',
            '_seed_sort_record_id',
        ],
        ascending=[True, False, True, True],
        kind='stable',
    )
    exemplar_df = reference_df.drop_duplicates(subset=[formula_col], keep='first').copy()
    exemplar_lookup: dict[str, dict[str, object]] = {}
    for _, row in exemplar_df.iterrows():
        payload: dict[str, object] = {
            'seed_reference_formula': str(row[formula_col]),
            'seed_reference_record_id': row['record_id'] if 'record_id' in row else None,
            'seed_reference_source': row['source'] if 'source' in row else None,
            'seed_reference_formula_row_count': int(row['reference_formula_row_count']),
            'seed_reference_formula_mean_band_gap': float(row['reference_formula_mean_band_gap']),
            'seed_reference_band_gap': float(row[target_col]),
            'seed_reference_has_structure_summary': bool(row['reference_has_structure_summary']),
        }
        for column_name in REFERENCE_PROPERTY_COLUMNS:
            payload[f'seed_reference_{column_name}'] = (
                float(row[column_name]) if column_name in row and pd.notna(row[column_name]) else None
            )
        for column_name in STRUCTURE_SUMMARY_COLUMNS:
            payload[f'seed_reference_{column_name}'] = (
                float(row[column_name]) if column_name in row and pd.notna(row[column_name]) else None
            )
        exemplar_lookup[str(row[formula_col])] = payload

    selected_candidate_df = candidate_ranking_df.loc[
        candidate_ranking_df['structure_generation_candidate_selected'].fillna(False).astype(bool)
    ].copy()
    selected_candidate_df = selected_candidate_df.sort_values(
        ['proposal_shortlist_rank', 'extrapolation_shortlist_rank', 'ranking_rank'],
        ascending=[True, True, True],
        na_position='last',
        kind='stable',
    ).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _, row in selected_candidate_df.iterrows():
        candidate_payload = {
            formula_col: str(row[formula_col]),
            'candidate_family': row.get('candidate_family'),
            'candidate_template': row.get('candidate_template'),
            'candidate_novelty_bucket': row.get('candidate_novelty_bucket'),
            'chemical_plausibility_pass': (
                bool(row['chemical_plausibility_pass'])
                if pd.notna(row.get('chemical_plausibility_pass'))
                else None
            ),
            'ranking_rank': int(row['ranking_rank']),
            'proposal_shortlist_selected': bool(row['proposal_shortlist_selected']),
            'proposal_shortlist_rank': (
                int(row['proposal_shortlist_rank'])
                if pd.notna(row.get('proposal_shortlist_rank'))
                else None
            ),
            'extrapolation_shortlist_selected': bool(row['extrapolation_shortlist_selected']),
            'extrapolation_shortlist_rank': (
                int(row['extrapolation_shortlist_rank'])
                if pd.notna(row.get('extrapolation_shortlist_rank'))
                else None
            ),
            'bn_centered_ranking_rank': (
                int(row['bn_centered_ranking_rank'])
                if pd.notna(row.get('bn_centered_ranking_rank'))
                else None
            ),
            'bn_centered_top_n_selected': bool(row['bn_centered_top_n_selected']),
            'rank_shift_general_minus_bn_centered': (
                float(row['rank_shift_general_minus_bn_centered'])
                if pd.notna(row.get('rank_shift_general_minus_bn_centered'))
                else None
            ),
            'absolute_rank_shift_vs_bn_centered': (
                float(row['absolute_rank_shift_vs_bn_centered'])
                if pd.notna(row.get('absolute_rank_shift_vs_bn_centered'))
                else None
            ),
            'structure_generation_seed_enabled': bool(seed_cfg['enabled']),
            'structure_generation_seed_label': str(seed_cfg['label']),
            'structure_generation_seed_method': str(seed_cfg['method']),
            'structure_generation_seed_candidate_scope': str(seed_cfg['candidate_scope']),
            'structure_generation_seed_note': str(seed_cfg['note']),
            'structure_generation_candidate_selected': True,
            'structure_generation_candidate_priority_reason': str(
                row['structure_generation_candidate_priority_reason']
            ),
            'structure_generation_seed_limit': int(seed_cfg['per_candidate_seed_limit']),
        }

        seed_source_column = None
        seed_formulas = _split_pipe_delimited_values(row.get('bn_analog_neighbor_formulas'))
        if seed_formulas:
            seed_source_column = 'bn_analog_neighbor_formulas'
        if not seed_formulas:
            seed_formulas = _split_pipe_delimited_values(row.get('bn_support_neighbor_formulas'))
            if seed_formulas:
                seed_source_column = 'bn_support_neighbor_formulas'
        if not seed_formulas:
            nearest_formula = row.get('bn_analog_nearest_formula')
            if pd.notna(nearest_formula):
                seed_formulas = [str(nearest_formula)]
                seed_source_column = 'bn_analog_nearest_formula'
        if not seed_formulas:
            nearest_formula = row.get('bn_support_nearest_formula')
            if pd.notna(nearest_formula):
                seed_formulas = [str(nearest_formula)]
                seed_source_column = 'bn_support_nearest_formula'
        seed_formulas = seed_formulas[: int(seed_cfg['per_candidate_seed_limit'])]

        if not seed_formulas:
            rows.append(
                {
                    **candidate_payload,
                    'structure_generation_seed_rank': None,
                    'structure_generation_seed_source_column': None,
                    'structure_generation_seed_status': 'no_reference_formula_seed_available',
                }
            )
            continue

        for seed_rank, seed_formula in enumerate(seed_formulas, start=1):
            edit_payload = _structure_seed_edit_metadata(str(row[formula_col]), str(seed_formula))
            seed_payload = exemplar_lookup.get(str(seed_formula))
            if seed_payload is None:
                rows.append(
                    {
                        **candidate_payload,
                        'structure_generation_seed_rank': int(seed_rank),
                        'structure_generation_seed_source_column': seed_source_column,
                        'structure_generation_seed_status': 'reference_formula_missing_from_train_plus_val_bn_slice',
                        'seed_reference_formula': str(seed_formula),
                        **edit_payload,
                    }
                )
                continue
            rows.append(
                {
                    **candidate_payload,
                    'structure_generation_seed_rank': int(seed_rank),
                    'structure_generation_seed_source_column': seed_source_column,
                    'structure_generation_seed_status': 'ok',
                    **edit_payload,
                    **seed_payload,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=out_columns)
    for column_name in out_columns:
        if column_name not in out.columns:
            out[column_name] = pd.NA
    return out[out_columns].reset_index(drop=True)


def build_candidate_prediction_ensemble(
    candidate_df: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    split_masks,
    cfg: dict,
    candidate_feature_sets: list[str] | None = None,
) -> pd.DataFrame:
    prediction_df = build_candidate_prediction_members(
        candidate_df,
        feature_tables,
        split_masks,
        cfg,
        candidate_feature_sets=candidate_feature_sets,
    )
    aggregated = (
        prediction_df
        .groupby('formula', as_index=False)
        .agg(
            ensemble_predicted_band_gap_mean=('prediction', 'mean'),
            ensemble_predicted_band_gap_std=('prediction', 'std'),
            ensemble_member_count=('prediction', 'size'),
        )
    )
    aggregated['ensemble_predicted_band_gap_std'] = (
        aggregated['ensemble_predicted_band_gap_std'].fillna(0.0)
    )
    aggregated['ensemble_member_count'] = aggregated['ensemble_member_count'].astype(int)
    return aggregated


def build_candidate_prediction_members(
    candidate_df: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    split_masks,
    cfg: dict,
    candidate_feature_sets: list[str] | None = None,
) -> pd.DataFrame:
    candidate_feature_sets = candidate_feature_sets or [
        value for value in get_candidate_screening_feature_sets(cfg) if value in feature_tables
    ]
    candidate_feature_sets = [value for value in candidate_feature_sets if value in feature_tables]
    candidate_feature_tables = {
        feature_set: build_feature_table(candidate_df, formula_col='formula', feature_set=feature_set)
        for feature_set in candidate_feature_sets
    }
    candidate_model_types = get_candidate_model_types(cfg)

    prediction_frames = []
    for feature_set in candidate_feature_sets:
        train_feature_df = feature_tables[feature_set]
        candidate_feature_df = candidate_feature_tables[feature_set]
        train_feature_info = summarize_feature_table(train_feature_df, feature_set=feature_set)
        candidate_feature_info = summarize_feature_table(candidate_feature_df, feature_set=feature_set)

        if not train_feature_info['selection_eligible']:
            continue
        if not candidate_feature_info['selection_eligible']:
            raise ValueError(
                'Candidate uncertainty estimation aborted because the feature set could not '
                f'featurize candidate formulas: {candidate_feature_info["failed_formula_examples"]}'
            )

        feature_columns = _feature_columns(train_feature_df)
        compatible_model_types = [
            model_type
            for model_type in candidate_model_types
            if model_type_supports_feature_set(model_type, feature_set)
        ]
        for model_type in compatible_model_types:
            model, _ = train_baseline_model(
                df=train_feature_df,
                split_masks=split_masks,
                cfg=cfg,
                model_type=model_type,
                include_validation=True,
            )
            candidate_matrix = candidate_feature_df[feature_columns]
            member_predictions = None
            if hasattr(model, 'predict_members'):
                try:
                    member_predictions = getattr(model, 'predict_members')(candidate_matrix)
                except Exception:
                    member_predictions = None
            if member_predictions is not None:
                member_predictions = np.asarray(member_predictions, dtype=float)
            if member_predictions is not None and member_predictions.ndim == 2 and member_predictions.shape[0] > 1:
                for member_idx, member_prediction in enumerate(member_predictions, start=1):
                    prediction_frames.append(pd.DataFrame({
                        'formula': candidate_feature_df['formula'].astype(str),
                        'prediction_source': (
                            f'full_fit__{feature_set}__{model_type}__member_{member_idx}'
                        ),
                        'prediction_source_family': 'full_fit_candidate_model_member',
                        'feature_set': feature_set,
                        'model_type': model_type,
                        'prediction': np.asarray(member_prediction, dtype=float),
                    }))
            else:
                prediction_frames.append(pd.DataFrame({
                    'formula': candidate_feature_df['formula'].astype(str),
                    'prediction_source': f'full_fit__{feature_set}__{model_type}',
                    'prediction_source_family': 'full_fit_candidate_model',
                    'feature_set': feature_set,
                    'model_type': model_type,
                    'prediction': model.predict(candidate_matrix),
                }))

    if not prediction_frames:
        raise ValueError('No candidate feature/model combination was available for uncertainty estimation')

    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    return prediction_df.sort_values(
        ['prediction_source_family', 'feature_set', 'model_type', 'formula'],
        ascending=[True, True, True, True],
        kind='stable',
    ).reset_index(drop=True)


def build_candidate_grouped_robustness_predictions(
    candidate_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    split_masks,
    cfg: dict,
    feature_set: str,
    model_type: str,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    robustness_cfg = _robustness_config(cfg)
    grouped_uncertainty_cfg = _grouped_robustness_uncertainty_config(cfg)
    out = pd.DataFrame({
        formula_col: candidate_df[formula_col].astype(str).reset_index(drop=True),
    })
    out['grouped_robustness_prediction_enabled'] = bool(
        robustness_cfg['enabled'] and grouped_uncertainty_cfg['enabled']
    )
    out['grouped_robustness_prediction_method'] = grouped_uncertainty_cfg['method']
    out['grouped_robustness_prediction_note'] = grouped_uncertainty_cfg['note']
    out['grouped_robustness_prediction_feature_set'] = feature_set
    out['grouped_robustness_prediction_model_type'] = model_type
    out['grouped_robustness_prediction_fold_count'] = 0
    out['grouped_robustness_predicted_band_gap_mean'] = np.nan
    out['grouped_robustness_predicted_band_gap_std'] = 0.0

    if not bool(robustness_cfg['enabled'] and grouped_uncertainty_cfg['enabled']):
        return out
    if not model_type_supports_feature_set(model_type, feature_set):
        out['grouped_robustness_prediction_enabled'] = False
        out['grouped_robustness_prediction_note'] = incompatible_model_feature_note(
            model_type,
            feature_set,
        )
        return out

    prediction_df = build_candidate_grouped_robustness_prediction_members(
        candidate_df,
        feature_df,
        split_masks,
        cfg,
        feature_set=feature_set,
        model_type=model_type,
        formula_col=formula_col,
    )

    aggregated = (
        prediction_df.groupby(formula_col, as_index=False)
        .agg(
            grouped_robustness_prediction_fold_count=('prediction', 'size'),
            grouped_robustness_predicted_band_gap_mean=('prediction', 'mean'),
            grouped_robustness_predicted_band_gap_std=('prediction', 'std'),
        )
        .sort_values(formula_col)
        .reset_index(drop=True)
    )
    aggregated['grouped_robustness_predicted_band_gap_std'] = (
        aggregated['grouped_robustness_predicted_band_gap_std'].fillna(0.0).astype(float)
    )
    aggregated['grouped_robustness_prediction_fold_count'] = (
        aggregated['grouped_robustness_prediction_fold_count'].astype(int)
    )
    return out.merge(aggregated, on=formula_col, how='left', suffixes=('', '_agg')).assign(
        grouped_robustness_prediction_fold_count=lambda df: (
            df['grouped_robustness_prediction_fold_count_agg']
            .fillna(df['grouped_robustness_prediction_fold_count'])
            .astype(int)
        ),
        grouped_robustness_predicted_band_gap_mean=lambda df: (
            df['grouped_robustness_predicted_band_gap_mean_agg']
            .fillna(df['grouped_robustness_predicted_band_gap_mean'])
            .astype(float)
        ),
        grouped_robustness_predicted_band_gap_std=lambda df: (
            df['grouped_robustness_predicted_band_gap_std_agg']
            .fillna(df['grouped_robustness_predicted_band_gap_std'])
            .astype(float)
        ),
    ).drop(
        columns=[
            'grouped_robustness_prediction_fold_count_agg',
            'grouped_robustness_predicted_band_gap_mean_agg',
            'grouped_robustness_predicted_band_gap_std_agg',
        ]
    )


def build_candidate_grouped_robustness_prediction_members(
    candidate_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    split_masks,
    cfg: dict,
    feature_set: str,
    model_type: str,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    robustness_cfg = _robustness_config(cfg)
    grouped_uncertainty_cfg = _grouped_robustness_uncertainty_config(cfg)
    empty_prediction_df = pd.DataFrame(
        columns=[
            formula_col,
            'prediction_source',
            'prediction_source_family',
            'feature_set',
            'model_type',
            'fold_index',
            'prediction',
        ]
    )
    if not bool(robustness_cfg['enabled'] and grouped_uncertainty_cfg['enabled']):
        return empty_prediction_df
    if not model_type_supports_feature_set(model_type, feature_set):
        return empty_prediction_df

    feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
    if not feature_info['selection_eligible']:
        raise ValueError(
            'Grouped candidate robustness prediction aborted because the selected feature set '
            f'was not selection-eligible: {feature_info["failed_formula_examples"]}'
        )
    feature_columns = _feature_columns(feature_df)

    train_plus_val_mask = np.asarray(split_masks['train']) | np.asarray(split_masks['val'])
    if len(train_plus_val_mask) != len(feature_df):
        raise ValueError('Feature table length does not match split masks for grouped candidate robustness')
    reference_feature_df = feature_df.loc[train_plus_val_mask].reset_index(drop=True)
    _, _, split_payloads = _group_kfold_splits(
        reference_feature_df,
        formula_col,
        int(robustness_cfg['n_splits']),
    )
    candidate_feature_df = build_feature_table(candidate_df, formula_col=formula_col, feature_set=feature_set)
    candidate_valid_mask = _feature_valid_mask(candidate_feature_df, feature_columns)
    if not bool(candidate_valid_mask.all()):
        failed_formulas = (
            candidate_feature_df.loc[~candidate_valid_mask, formula_col]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(
            'Grouped candidate robustness prediction aborted because candidate features were '
            f'invalid for formulas: {failed_formulas}'
        )

    candidate_matrix = candidate_feature_df[feature_columns].to_numpy(dtype=float)
    fold_predictions: list[pd.DataFrame] = []
    for fold_payload in split_payloads:
        fold_idx = int(fold_payload['fold_index'])
        train_df = reference_feature_df.loc[fold_payload['train']].reset_index(drop=True)
        model = make_model(cfg, model_type)
        model.fit(
            train_df[feature_columns],
            train_df['target'],
        )
        fold_predictions.append(pd.DataFrame({
            formula_col: candidate_feature_df[formula_col].astype(str),
            'prediction_source': f'group_kfold__{feature_set}__{model_type}__fold_{fold_idx + 1}',
            'prediction_source_family': 'group_kfold_candidate_model',
            'feature_set': feature_set,
            'model_type': model_type,
            'fold_index': fold_idx,
            'prediction': model.predict(candidate_matrix).astype(float),
        }))

    if not fold_predictions:
        return empty_prediction_df
    prediction_df = pd.concat(fold_predictions, ignore_index=True)
    return prediction_df.sort_values(
        ['prediction_source_family', 'fold_index', formula_col],
        ascending=[True, True, True],
        kind='stable',
    ).reset_index(drop=True)


def annotate_candidate_dataset_overlap(
    candidate_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    split_masks=None,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    dataset_formula_counts = dataset_df[formula_col].astype(str).value_counts()
    out = pd.DataFrame({'formula': candidate_df['formula'].astype(str)})
    out['seen_in_dataset'] = out['formula'].map(dataset_formula_counts).fillna(0).astype(int) > 0
    out['dataset_formula_row_count'] = out['formula'].map(dataset_formula_counts).fillna(0).astype(int)

    if split_masks is not None:
        train_plus_val_mask = np.asarray(split_masks['train']) | np.asarray(split_masks['val'])
        train_plus_val_formula_counts = (
            dataset_df.loc[train_plus_val_mask, formula_col].astype(str).value_counts()
        )
        out['seen_in_train_plus_val'] = (
            out['formula'].map(train_plus_val_formula_counts).fillna(0).astype(int) > 0
        )
        out['train_plus_val_formula_row_count'] = (
            out['formula'].map(train_plus_val_formula_counts).fillna(0).astype(int)
        )
    return out


def annotate_candidate_novelty(
    candidate_df: pd.DataFrame,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    required_columns = {formula_col, 'seen_in_dataset', 'seen_in_train_plus_val'}
    missing_columns = sorted(required_columns.difference(candidate_df.columns))
    if missing_columns:
        raise KeyError(f'Candidate novelty annotation requires columns: {missing_columns}')

    out = pd.DataFrame({formula_col: candidate_df[formula_col].astype(str).reset_index(drop=True)})
    out['candidate_is_seen_in_dataset'] = (
        candidate_df['seen_in_dataset'].fillna(False).astype(bool).reset_index(drop=True)
    )
    out['candidate_is_seen_in_train_plus_val'] = (
        candidate_df['seen_in_train_plus_val'].fillna(False).astype(bool).reset_index(drop=True)
    )
    out['candidate_is_formula_level_extrapolation'] = ~out['candidate_is_seen_in_dataset']

    novelty_bucket = np.select(
        [
            out['candidate_is_seen_in_train_plus_val'],
            out['candidate_is_seen_in_dataset'],
        ],
        [
            NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY,
            NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA,
        ],
        default=NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION,
    )
    novelty_bucket_series = pd.Series(novelty_bucket, index=out.index, dtype='object')
    out['candidate_novelty_bucket'] = novelty_bucket_series
    out['candidate_novelty_priority'] = novelty_bucket_series.map(NOVELTY_BUCKET_PRIORITY).astype(int)
    out['candidate_novelty_note'] = novelty_bucket_series.map(NOVELTY_BUCKET_NOTE)
    return out


def annotate_candidate_domain_support(
    candidate_feature_df: pd.DataFrame,
    reference_feature_df: pd.DataFrame,
    split_masks,
    feature_columns: list[str],
    cfg: dict | None = None,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    if formula_col not in candidate_feature_df.columns:
        raise KeyError(f'Formula column not found in candidate features: {formula_col}')
    if formula_col not in reference_feature_df.columns:
        raise KeyError(f'Formula column not found in reference features: {formula_col}')

    support_metadata = get_screening_ranking_metadata(cfg)
    out = pd.DataFrame({
        formula_col: candidate_feature_df[formula_col].astype(str).reset_index(drop=True),
    })
    out['domain_support_enabled'] = bool(support_metadata['domain_support_enabled'])
    out['domain_support_method'] = support_metadata['domain_support_method']
    out['domain_support_distance_metric'] = support_metadata['domain_support_distance_metric']
    out['domain_support_reference_split'] = support_metadata['domain_support_reference_split']
    out['domain_support_reference_formula_count'] = 0
    out['domain_support_k_neighbors'] = int(support_metadata['domain_support_k_neighbors'])
    out['domain_support_nearest_formula'] = ''
    out['domain_support_nearest_distance'] = np.nan
    out['domain_support_mean_k_distance'] = np.nan
    out['domain_support_percentile'] = np.nan
    out['domain_support_penalty'] = 0.0

    if not bool(support_metadata['domain_support_enabled']):
        return out
    if split_masks is None:
        raise ValueError('Domain-support annotation requires split masks for the reference dataset')

    train_plus_val_mask = np.asarray(split_masks['train']) | np.asarray(split_masks['val'])
    if len(train_plus_val_mask) != len(reference_feature_df):
        raise ValueError('Reference feature table length does not match split masks')

    reference_df = reference_feature_df.loc[train_plus_val_mask].copy()
    reference_df = reference_df.loc[_feature_valid_mask(reference_df, feature_columns)].copy()
    reference_df[formula_col] = reference_df[formula_col].astype(str)
    reference_df = reference_df.drop_duplicates(subset=formula_col, keep='first').reset_index(drop=True)
    out['domain_support_reference_formula_count'] = int(len(reference_df))

    candidate_valid_mask = _feature_valid_mask(candidate_feature_df, feature_columns)
    if not bool(candidate_valid_mask.all()):
        failed_formulas = (
            candidate_feature_df.loc[~candidate_valid_mask, formula_col]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(
            'Domain-support annotation aborted because candidate features were invalid for formulas: '
            f'{failed_formulas}'
        )
    if reference_df.empty:
        return out

    reference_matrix_raw = reference_df[feature_columns].to_numpy(dtype=float)
    candidate_matrix_raw = candidate_feature_df[feature_columns].to_numpy(dtype=float)
    center = reference_matrix_raw.mean(axis=0)
    spread = reference_matrix_raw.std(axis=0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)

    reference_matrix = (reference_matrix_raw - center) / spread
    candidate_matrix = (candidate_matrix_raw - center) / spread
    distance_scale = float(np.sqrt(max(len(feature_columns), 1)))
    effective_k = min(int(support_metadata['domain_support_k_neighbors']), len(reference_df))

    reference_neighbors = NearestNeighbors(metric='euclidean', n_neighbors=effective_k)
    reference_neighbors.fit(reference_matrix)
    candidate_distances, candidate_indices = reference_neighbors.kneighbors(
        candidate_matrix,
        n_neighbors=effective_k,
    )
    candidate_nearest_formulas = (
        reference_df.iloc[candidate_indices[:, 0]][formula_col]
        .astype(str)
        .reset_index(drop=True)
    )
    candidate_nearest_distances = candidate_distances[:, 0] / distance_scale
    candidate_mean_k_distances = candidate_distances.mean(axis=1) / distance_scale

    candidate_percentiles = np.full(len(out), np.nan, dtype=float)
    if len(reference_df) > 1:
        reference_effective_k = min(int(support_metadata['domain_support_k_neighbors']), len(reference_df) - 1)
        reference_distances, _ = reference_neighbors.kneighbors(
            reference_matrix,
            n_neighbors=reference_effective_k + 1,
        )
        reference_mean_k_distances = reference_distances[:, 1:].mean(axis=1) / distance_scale
        candidate_percentiles = np.asarray([
            100.0 * float((reference_mean_k_distances >= value).mean())
            for value in candidate_mean_k_distances
        ])

    out['domain_support_nearest_formula'] = candidate_nearest_formulas
    out['domain_support_nearest_distance'] = candidate_nearest_distances.astype(float)
    out['domain_support_mean_k_distance'] = candidate_mean_k_distances.astype(float)
    out['domain_support_percentile'] = candidate_percentiles

    if bool(support_metadata['domain_support_penalty_enabled']):
        percentile_threshold = float(support_metadata['domain_support_penalize_below_percentile'])
        safe_threshold = max(percentile_threshold, 1e-12)
        low_support_gap = np.clip(
            (percentile_threshold - np.nan_to_num(candidate_percentiles, nan=0.0)) / safe_threshold,
            a_min=0.0,
            a_max=None,
        )
        out['domain_support_penalty'] = (
            float(support_metadata['domain_support_penalty_weight']) * low_support_gap
        ).astype(float)

    return out


def annotate_candidate_bn_support(
    candidate_feature_df: pd.DataFrame,
    reference_feature_df: pd.DataFrame,
    split_masks,
    feature_columns: list[str],
    cfg: dict | None = None,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    if formula_col not in candidate_feature_df.columns:
        raise KeyError(f'Formula column not found in candidate features: {formula_col}')
    if formula_col not in reference_feature_df.columns:
        raise KeyError(f'Formula column not found in reference features: {formula_col}')

    support_metadata = get_screening_ranking_metadata(cfg)
    out = pd.DataFrame({
        formula_col: candidate_feature_df[formula_col].astype(str).reset_index(drop=True),
    })
    out['bn_support_enabled'] = bool(support_metadata['bn_support_enabled'])
    out['bn_support_method'] = support_metadata['bn_support_method']
    out['bn_support_distance_metric'] = support_metadata['bn_support_distance_metric']
    out['bn_support_reference_split'] = support_metadata['bn_support_reference_split']
    out['bn_support_reference_formula_count'] = 0
    out['bn_support_k_neighbors'] = int(support_metadata['bn_support_k_neighbors'])
    out['bn_support_nearest_formula'] = ''
    out['bn_support_neighbor_formulas'] = ''
    out['bn_support_neighbor_formula_count'] = 0
    out['bn_support_nearest_distance'] = np.nan
    out['bn_support_mean_k_distance'] = np.nan
    out['bn_support_percentile'] = np.nan
    out['bn_support_penalty'] = 0.0

    if not bool(support_metadata['bn_support_enabled']):
        return out
    if split_masks is None:
        raise ValueError('BN-support annotation requires split masks for the reference dataset')

    train_plus_val_mask = np.asarray(split_masks['train']) | np.asarray(split_masks['val'])
    if len(train_plus_val_mask) != len(reference_feature_df):
        raise ValueError('Reference feature table length does not match split masks')

    reference_df = reference_feature_df.loc[train_plus_val_mask].copy()
    reference_df = reference_df.loc[_feature_valid_mask(reference_df, feature_columns)].copy()
    reference_df[formula_col] = reference_df[formula_col].astype(str)
    reference_df = reference_df.loc[
        reference_df[formula_col].apply(
            lambda value: {'B', 'N'}.issubset(set(extract_elements(value)))
        )
    ].copy()
    reference_df = reference_df.drop_duplicates(subset=formula_col, keep='first').reset_index(drop=True)
    out['bn_support_reference_formula_count'] = int(len(reference_df))

    candidate_valid_mask = _feature_valid_mask(candidate_feature_df, feature_columns)
    if not bool(candidate_valid_mask.all()):
        failed_formulas = (
            candidate_feature_df.loc[~candidate_valid_mask, formula_col]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(
            'BN-support annotation aborted because candidate features were invalid for formulas: '
            f'{failed_formulas}'
        )
    if reference_df.empty:
        return out

    reference_matrix_raw = reference_df[feature_columns].to_numpy(dtype=float)
    candidate_matrix_raw = candidate_feature_df[feature_columns].to_numpy(dtype=float)
    center = reference_matrix_raw.mean(axis=0)
    spread = reference_matrix_raw.std(axis=0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)

    reference_matrix = (reference_matrix_raw - center) / spread
    candidate_matrix = (candidate_matrix_raw - center) / spread
    distance_scale = float(np.sqrt(max(len(feature_columns), 1)))
    effective_k = min(int(support_metadata['bn_support_k_neighbors']), len(reference_df))

    reference_neighbors = NearestNeighbors(metric='euclidean', n_neighbors=effective_k)
    reference_neighbors.fit(reference_matrix)
    candidate_distances, candidate_indices = reference_neighbors.kneighbors(
        candidate_matrix,
        n_neighbors=effective_k,
    )
    candidate_nearest_formulas = (
        reference_df.iloc[candidate_indices[:, 0]][formula_col]
        .astype(str)
        .reset_index(drop=True)
    )
    candidate_neighbor_formulas = [
        '|'.join(_ordered_values(reference_df.iloc[index_row][formula_col].astype(str).tolist()))
        for index_row in candidate_indices
    ]
    candidate_neighbor_formula_count = [
        len([value for value in neighbor_value.split('|') if value])
        for neighbor_value in candidate_neighbor_formulas
    ]
    candidate_nearest_distances = candidate_distances[:, 0] / distance_scale
    candidate_mean_k_distances = candidate_distances.mean(axis=1) / distance_scale

    if len(reference_df) > 1:
        reference_effective_k = min(int(support_metadata['bn_support_k_neighbors']), len(reference_df) - 1)
        reference_distances, _ = reference_neighbors.kneighbors(
            reference_matrix,
            n_neighbors=reference_effective_k + 1,
        )
        reference_mean_k_distances = reference_distances[:, 1:].mean(axis=1) / distance_scale
        candidate_percentiles = np.asarray([
            100.0 * float((reference_mean_k_distances >= value).mean())
            for value in candidate_mean_k_distances
        ])
    else:
        candidate_percentiles = np.where(
            np.isclose(candidate_mean_k_distances, 0.0),
            100.0,
            0.0,
        ).astype(float)

    out['bn_support_nearest_formula'] = candidate_nearest_formulas
    out['bn_support_neighbor_formulas'] = candidate_neighbor_formulas
    out['bn_support_neighbor_formula_count'] = np.asarray(candidate_neighbor_formula_count, dtype=int)
    out['bn_support_nearest_distance'] = candidate_nearest_distances.astype(float)
    out['bn_support_mean_k_distance'] = candidate_mean_k_distances.astype(float)
    out['bn_support_percentile'] = candidate_percentiles.astype(float)

    if bool(support_metadata['bn_support_penalty_enabled']):
        percentile_threshold = float(support_metadata['bn_support_penalize_below_percentile'])
        safe_threshold = max(percentile_threshold, 1e-12)
        low_support_gap = np.clip(
            (percentile_threshold - np.nan_to_num(candidate_percentiles, nan=0.0)) / safe_threshold,
            a_min=0.0,
            a_max=None,
        )
        out['bn_support_penalty'] = (
            float(support_metadata['bn_support_penalty_weight']) * low_support_gap
        ).astype(float)

    return out


def annotate_candidate_bn_analog_evidence(
    candidate_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    split_masks,
    cfg: dict | None = None,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    evidence_metadata = _bn_analog_evidence_config(cfg)
    band_gap_alignment_metadata = _bn_band_gap_alignment_config(cfg)
    out = pd.DataFrame({
        formula_col: candidate_df[formula_col].astype(str).reset_index(drop=True),
    })
    out['bn_analog_evidence_enabled'] = bool(evidence_metadata['enabled'])
    out['bn_analog_evidence_aggregation'] = evidence_metadata['aggregation']
    out['bn_analog_evidence_reference_split'] = evidence_metadata['reference_split']
    out['bn_analog_evidence_exfoliation_reference'] = evidence_metadata['exfoliation_reference']
    out['bn_analog_evidence_note'] = evidence_metadata['note']
    out['bn_analog_reference_formula_count'] = 0
    out['bn_analog_reference_band_gap_median'] = np.nan
    out['bn_analog_reference_band_gap_iqr'] = np.nan
    out['bn_analog_reference_exfoliation_energy_median'] = np.nan
    out['bn_analog_reference_energy_per_atom_median'] = np.nan
    out['bn_analog_reference_abs_total_magnetization_median'] = np.nan
    out['bn_analog_nearest_formula'] = ''
    out['bn_analog_neighbor_formulas'] = ''
    out['bn_analog_neighbor_formula_count'] = 0
    out['bn_analog_nearest_band_gap'] = np.nan
    out['bn_analog_nearest_energy_per_atom'] = np.nan
    out['bn_analog_nearest_exfoliation_energy_per_atom'] = np.nan
    out['bn_analog_nearest_abs_total_magnetization'] = np.nan
    out['bn_analog_neighbor_band_gap_mean'] = np.nan
    out['bn_analog_neighbor_band_gap_min'] = np.nan
    out['bn_analog_neighbor_band_gap_max'] = np.nan
    out['bn_analog_neighbor_band_gap_std'] = np.nan
    out['bn_analog_neighbor_energy_per_atom_mean'] = np.nan
    out['bn_analog_neighbor_exfoliation_energy_per_atom_mean'] = np.nan
    out['bn_analog_neighbor_abs_total_magnetization_mean'] = np.nan
    out['bn_analog_neighbor_exfoliation_available_formula_count'] = 0
    out['bn_analog_exfoliation_support_label'] = 'unknown'
    out['bn_analog_energy_support_label'] = 'unknown'
    out['bn_analog_abs_total_magnetization_support_label'] = 'unknown'
    out['bn_analog_support_vote_count'] = 0
    out['bn_analog_support_available_metric_count'] = 0
    out['bn_analog_validation_label'] = 'unknown'
    out['bn_band_gap_alignment_enabled'] = bool(band_gap_alignment_metadata['enabled'])
    out['bn_band_gap_alignment_method'] = band_gap_alignment_metadata['method']
    out['bn_band_gap_alignment_reference_split'] = (
        band_gap_alignment_metadata['reference_split']
    )
    out['bn_band_gap_alignment_note'] = band_gap_alignment_metadata['note']
    out['bn_band_gap_alignment_neighbor_available_formula_count'] = 0
    out['bn_band_gap_alignment_window_lower'] = np.nan
    out['bn_band_gap_alignment_window_upper'] = np.nan
    out['bn_band_gap_alignment_distance_to_window'] = np.nan
    out['bn_band_gap_alignment_relative_distance'] = np.nan
    out['bn_band_gap_alignment_penalty_eligible'] = False
    out['bn_band_gap_alignment_label'] = 'unknown'

    if not bool(evidence_metadata['enabled']):
        return out
    if split_masks is None:
        raise ValueError('BN analog evidence annotation requires split masks for the reference dataset')
    if 'bn_support_nearest_formula' not in candidate_df.columns:
        raise KeyError('BN analog evidence annotation requires bn_support_nearest_formula')
    if 'bn_support_neighbor_formulas' not in candidate_df.columns:
        raise KeyError('BN analog evidence annotation requires bn_support_neighbor_formulas')

    train_plus_val_mask = np.asarray(split_masks['train']) | np.asarray(split_masks['val'])
    if len(train_plus_val_mask) != len(dataset_df):
        raise ValueError('Dataset length does not match split masks for BN analog evidence annotation')

    reference_df = dataset_df.loc[train_plus_val_mask].copy()
    for column in REFERENCE_PROPERTY_COLUMNS:
        if column not in reference_df.columns:
            reference_df[column] = np.nan
    if 'abs_total_magnetization' in reference_df.columns and reference_df['abs_total_magnetization'].isna().all():
        reference_df['abs_total_magnetization'] = reference_df['total_magnetization'].abs()
    reference_df = filter_bn(reference_df, formula_col=formula_col)
    reference_df[formula_col] = reference_df[formula_col].astype(str)
    out['bn_analog_reference_formula_count'] = int(reference_df[formula_col].nunique())

    aggregated = (
        reference_df.groupby(formula_col, as_index=True)
        .agg(
            target=('target', 'mean'),
            energy_per_atom=('energy_per_atom', 'mean'),
            exfoliation_energy_per_atom=('exfoliation_energy_per_atom', 'mean'),
            abs_total_magnetization=('abs_total_magnetization', 'mean'),
            reference_row_count=(formula_col, 'size'),
        )
        .sort_index()
    )
    reference_band_gap_values = aggregated['target'].dropna().astype(float)
    reference_band_gap_median = np.nan
    reference_band_gap_iqr = np.nan
    if not reference_band_gap_values.empty:
        reference_band_gap_median = float(reference_band_gap_values.median())
        q1 = float(reference_band_gap_values.quantile(0.25))
        q3 = float(reference_band_gap_values.quantile(0.75))
        reference_band_gap_iqr = float(q3 - q1)
        out['bn_analog_reference_band_gap_median'] = reference_band_gap_median
        out['bn_analog_reference_band_gap_iqr'] = reference_band_gap_iqr
    reference_exfoliation_values = aggregated['exfoliation_energy_per_atom'].dropna()
    if not reference_exfoliation_values.empty:
        out['bn_analog_reference_exfoliation_energy_median'] = float(
            reference_exfoliation_values.median()
        )
    reference_energy_values = aggregated['energy_per_atom'].dropna()
    if not reference_energy_values.empty:
        out['bn_analog_reference_energy_per_atom_median'] = float(
            reference_energy_values.median()
        )
    reference_abs_mag_values = aggregated['abs_total_magnetization'].dropna()
    if not reference_abs_mag_values.empty:
        out['bn_analog_reference_abs_total_magnetization_median'] = float(
            reference_abs_mag_values.median()
        )

    def _lookup_metric(formulas: list[str], column: str) -> tuple[float, int]:
        values = []
        for formula in formulas:
            if formula in aggregated.index:
                value = aggregated.at[formula, column]
                if pd.notna(value):
                    values.append(float(value))
        if not values:
            return np.nan, 0
        return float(np.mean(values)), len(values)

    def _lookup_metric_values(formulas: list[str], column: str) -> list[float]:
        values: list[float] = []
        for formula in formulas:
            if formula not in aggregated.index:
                continue
            value = aggregated.at[formula, column]
            if pd.notna(value):
                values.append(float(value))
        return values

    nearest_formulas = candidate_df['bn_support_nearest_formula'].astype(str).reset_index(drop=True)
    neighbor_formula_strings = candidate_df['bn_support_neighbor_formulas'].astype(str).reset_index(drop=True)
    predicted_band_gap_series = pd.to_numeric(
        candidate_df.get(
            'predicted_band_gap',
            pd.Series(np.nan, index=candidate_df.index, dtype=float),
        ),
        errors='coerce',
    ).reset_index(drop=True)
    reference_exfoliation_median = (
        float(reference_exfoliation_values.median()) if not reference_exfoliation_values.empty else np.nan
    )
    reference_energy_median = (
        float(reference_energy_values.median()) if not reference_energy_values.empty else np.nan
    )
    reference_abs_mag_median = (
        float(reference_abs_mag_values.median()) if not reference_abs_mag_values.empty else np.nan
    )

    def _compare_against_reference(mean_value: float, reference_value: float) -> tuple[str, int, int]:
        if not np.isfinite(mean_value) or not np.isfinite(reference_value):
            return 'unknown', 0, 0
        favorable = int(mean_value <= reference_value)
        label = (
            'lower_or_equal_bn_reference_median'
            if favorable
            else 'higher_than_bn_reference_median'
        )
        return label, 1, favorable

    for idx, nearest_formula in enumerate(nearest_formulas):
        neighbor_formulas = _ordered_values([
            value for value in neighbor_formula_strings.iloc[idx].split('|') if value
        ])
        out.at[idx, 'bn_analog_nearest_formula'] = nearest_formula
        out.at[idx, 'bn_analog_neighbor_formulas'] = '|'.join(neighbor_formulas)
        out.at[idx, 'bn_analog_neighbor_formula_count'] = int(len(neighbor_formulas))

        if nearest_formula in aggregated.index:
            out.at[idx, 'bn_analog_nearest_band_gap'] = float(aggregated.at[nearest_formula, 'target'])
            out.at[idx, 'bn_analog_nearest_energy_per_atom'] = float(
                aggregated.at[nearest_formula, 'energy_per_atom']
            ) if pd.notna(aggregated.at[nearest_formula, 'energy_per_atom']) else np.nan
            out.at[idx, 'bn_analog_nearest_exfoliation_energy_per_atom'] = float(
                aggregated.at[nearest_formula, 'exfoliation_energy_per_atom']
            ) if pd.notna(aggregated.at[nearest_formula, 'exfoliation_energy_per_atom']) else np.nan
            out.at[idx, 'bn_analog_nearest_abs_total_magnetization'] = float(
                aggregated.at[nearest_formula, 'abs_total_magnetization']
            ) if pd.notna(aggregated.at[nearest_formula, 'abs_total_magnetization']) else np.nan

        neighbor_band_gap_values = _lookup_metric_values(neighbor_formulas, 'target')
        neighbor_band_gap_mean, _ = _lookup_metric(neighbor_formulas, 'target')
        neighbor_energy_mean, _ = _lookup_metric(neighbor_formulas, 'energy_per_atom')
        neighbor_exfoliation_mean, neighbor_exfoliation_count = _lookup_metric(
            neighbor_formulas,
            'exfoliation_energy_per_atom',
        )
        neighbor_abs_mag_mean, _ = _lookup_metric(neighbor_formulas, 'abs_total_magnetization')
        out.at[idx, 'bn_analog_neighbor_band_gap_mean'] = neighbor_band_gap_mean
        out.at[idx, 'bn_band_gap_alignment_neighbor_available_formula_count'] = int(
            len(neighbor_band_gap_values)
        )
        if neighbor_band_gap_values:
            out.at[idx, 'bn_analog_neighbor_band_gap_min'] = float(min(neighbor_band_gap_values))
            out.at[idx, 'bn_analog_neighbor_band_gap_max'] = float(max(neighbor_band_gap_values))
            out.at[idx, 'bn_analog_neighbor_band_gap_std'] = float(
                np.std(neighbor_band_gap_values, ddof=0)
            )
        out.at[idx, 'bn_analog_neighbor_energy_per_atom_mean'] = neighbor_energy_mean
        out.at[idx, 'bn_analog_neighbor_exfoliation_energy_per_atom_mean'] = neighbor_exfoliation_mean
        out.at[idx, 'bn_analog_neighbor_abs_total_magnetization_mean'] = neighbor_abs_mag_mean
        out.at[idx, 'bn_analog_neighbor_exfoliation_available_formula_count'] = int(
            neighbor_exfoliation_count
        )
        if bool(band_gap_alignment_metadata['enabled']) and neighbor_band_gap_values:
            local_min = float(min(neighbor_band_gap_values))
            local_max = float(max(neighbor_band_gap_values))
            window_expansion = (
                float(band_gap_alignment_metadata['window_expansion_iqr_factor'])
                * float(np.nan_to_num(reference_band_gap_iqr, nan=0.0))
            )
            window_lower = local_min - window_expansion
            window_upper = local_max + window_expansion
            predicted_band_gap = predicted_band_gap_series.iloc[idx]
            out.at[idx, 'bn_band_gap_alignment_window_lower'] = float(window_lower)
            out.at[idx, 'bn_band_gap_alignment_window_upper'] = float(window_upper)
            penalty_eligible = bool(
                len(neighbor_band_gap_values)
                >= int(
                    band_gap_alignment_metadata[
                        'minimum_neighbor_formula_count_for_penalty'
                    ]
                )
                and np.isfinite(reference_band_gap_iqr)
                and float(reference_band_gap_iqr) > 0.0
            )
            out.at[idx, 'bn_band_gap_alignment_penalty_eligible'] = penalty_eligible
            if np.isfinite(predicted_band_gap):
                if predicted_band_gap < window_lower:
                    distance_to_window = float(window_lower - predicted_band_gap)
                    label = 'below_local_bn_analog_band_gap_window'
                elif predicted_band_gap > window_upper:
                    distance_to_window = float(predicted_band_gap - window_upper)
                    label = 'above_local_bn_analog_band_gap_window'
                else:
                    distance_to_window = 0.0
                    label = 'within_local_bn_analog_band_gap_window'
                out.at[idx, 'bn_band_gap_alignment_distance_to_window'] = distance_to_window
                if np.isfinite(reference_band_gap_iqr) and float(reference_band_gap_iqr) > 0.0:
                    out.at[idx, 'bn_band_gap_alignment_relative_distance'] = float(
                        distance_to_window / float(reference_band_gap_iqr)
                    )
                out.at[idx, 'bn_band_gap_alignment_label'] = label

        energy_label, energy_available, energy_vote = _compare_against_reference(
            neighbor_energy_mean,
            reference_energy_median,
        )
        out.at[idx, 'bn_analog_energy_support_label'] = energy_label

        if neighbor_exfoliation_count <= 0 or not np.isfinite(reference_exfoliation_median):
            exfoliation_label = 'unknown'
            exfoliation_available = 0
            exfoliation_vote = 0
        elif neighbor_exfoliation_mean <= reference_exfoliation_median:
            exfoliation_label = 'lower_or_equal_bn_reference_median'
            exfoliation_available = 1
            exfoliation_vote = 1
        else:
            exfoliation_label = 'higher_than_bn_reference_median'
            exfoliation_available = 1
            exfoliation_vote = 0
        out.at[idx, 'bn_analog_exfoliation_support_label'] = exfoliation_label

        abs_mag_label, abs_mag_available, abs_mag_vote = _compare_against_reference(
            neighbor_abs_mag_mean,
            reference_abs_mag_median,
        )
        out.at[idx, 'bn_analog_abs_total_magnetization_support_label'] = abs_mag_label

        support_available = int(energy_available + exfoliation_available + abs_mag_available)
        support_votes = int(energy_vote + exfoliation_vote + abs_mag_vote)
        out.at[idx, 'bn_analog_support_vote_count'] = support_votes
        out.at[idx, 'bn_analog_support_available_metric_count'] = support_available
        if support_available <= 0:
            validation_label = 'unknown'
        elif support_votes == support_available:
            validation_label = 'reference_like_on_available_metrics'
        elif support_votes == 0:
            validation_label = 'reference_divergent_on_available_metrics'
        else:
            validation_label = 'mixed_reference_alignment'
        out.at[idx, 'bn_analog_validation_label'] = validation_label

    return out


def screen_candidates(
    candidate_df: pd.DataFrame,
    model,
    feature_columns: list[str],
    cfg: dict,
    feature_set: str,
    model_type: str,
    best_overall_feature_set: str | None = None,
    best_overall_model_type: str | None = None,
    screening_selection_note: str | None = None,
    dataset_df: pd.DataFrame | None = None,
    split_masks=None,
    ensemble_prediction_df: pd.DataFrame | None = None,
    grouped_robustness_prediction_df: pd.DataFrame | None = None,
    reference_feature_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    annotated_candidate_df = annotate_candidate_chemical_plausibility(candidate_df, cfg=cfg, formula_col='formula')
    feature_df = build_feature_table(annotated_candidate_df, formula_col='formula', feature_set=feature_set)
    feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
    if not feature_info['selection_eligible']:
        raise ValueError(
            'Candidate ranking aborted because the selected feature set could not featurize '
            f'candidate formulas: {feature_info["failed_formula_examples"]}'
        )

    pred = model.predict(feature_df[feature_columns])
    out = annotated_candidate_df.copy().reset_index(drop=True)
    out['predicted_band_gap'] = pred
    if ensemble_prediction_df is not None:
        out = out.merge(ensemble_prediction_df, on='formula', how='left')
        if out['ensemble_predicted_band_gap_mean'].isna().any():
            missing_formulas = out.loc[
                out['ensemble_predicted_band_gap_mean'].isna(), 'formula'
            ].astype(str).tolist()
            raise ValueError(
                'Candidate ranking aborted because ensemble predictions were missing for formulas: '
                f'{missing_formulas}'
            )
    else:
        out['ensemble_predicted_band_gap_mean'] = out['predicted_band_gap']
        out['ensemble_predicted_band_gap_std'] = 0.0
        out['ensemble_member_count'] = 1

    grouped_robustness_cfg = _grouped_robustness_uncertainty_config(cfg)
    if grouped_robustness_prediction_df is not None:
        out = out.merge(grouped_robustness_prediction_df, on='formula', how='left')
        if out['grouped_robustness_prediction_enabled'].isna().any():
            missing_formulas = out.loc[
                out['grouped_robustness_prediction_enabled'].isna(), 'formula'
            ].astype(str).tolist()
            raise ValueError(
                'Candidate ranking aborted because grouped robustness predictions were missing '
                f'for formulas: {missing_formulas}'
            )
    else:
        out['grouped_robustness_prediction_enabled'] = False
        out['grouped_robustness_prediction_method'] = grouped_robustness_cfg['method']
        out['grouped_robustness_prediction_note'] = grouped_robustness_cfg['note']
        out['grouped_robustness_prediction_feature_set'] = feature_set
        out['grouped_robustness_prediction_model_type'] = model_type
        out['grouped_robustness_prediction_fold_count'] = 0
        out['grouped_robustness_predicted_band_gap_mean'] = np.nan
        out['grouped_robustness_predicted_band_gap_std'] = 0.0

    ranking_config_metadata = get_screening_ranking_metadata(cfg)
    uncertainty_penalty = float(ranking_config_metadata['ranking_uncertainty_penalty'])
    use_model_disagreement = bool(cfg['screening'].get('use_model_disagreement', False))
    target_property = cfg['screening'].get(
        'objective_target_property',
        (cfg.get('data') or {}).get('target_column', 'band_gap'),
    )
    out['ranking_signal_property'] = target_property
    out['ranking_signal_direction'] = cfg['screening'].get('objective_target_direction', 'maximize')
    out['ranking_signal_source'] = (
        'ensemble_predicted_band_gap_mean'
        if use_model_disagreement and ensemble_prediction_df is not None
        else 'predicted_band_gap'
    )
    if use_model_disagreement and ensemble_prediction_df is not None:
        out['ranking_signal_value'] = out['ensemble_predicted_band_gap_mean']
        out['ranking_uncertainty_penalty_component'] = (
            uncertainty_penalty * out['ensemble_predicted_band_gap_std']
        )
        out['ranking_score'] = (
            out['ranking_signal_value']
            - out['ranking_uncertainty_penalty_component']
        )
    else:
        out['ranking_signal_value'] = out['predicted_band_gap']
        out['ranking_uncertainty_penalty_component'] = 0.0
        out['ranking_score'] = out['ranking_signal_value']
    out['ranking_score_before_grouped_robustness_penalty'] = out['ranking_score']
    grouped_robustness_penalty_enabled = bool(
        ranking_config_metadata['grouped_robustness_penalty_enabled']
    )
    out['grouped_robustness_uncertainty_enabled'] = bool(
        ranking_config_metadata['grouped_robustness_uncertainty_enabled']
    )
    out['grouped_robustness_uncertainty_method'] = (
        ranking_config_metadata['grouped_robustness_uncertainty_method']
    )
    out['grouped_robustness_uncertainty_note'] = (
        ranking_config_metadata['grouped_robustness_uncertainty_note']
    )
    out['grouped_robustness_uncertainty_penalty'] = 0.0
    if grouped_robustness_penalty_enabled and grouped_robustness_prediction_df is not None:
        out['grouped_robustness_uncertainty_penalty'] = (
            float(ranking_config_metadata['grouped_robustness_penalty_weight'])
            * out['grouped_robustness_predicted_band_gap_std'].fillna(0.0)
        )
        out['ranking_score'] = (
            out['ranking_score'] - out['grouped_robustness_uncertainty_penalty']
        )
    out['ranking_score_before_domain_support_penalty'] = out['ranking_score']
    out['ranking_score_before_bn_support_penalty'] = out['ranking_score']
    out['ranking_score_before_bn_band_gap_alignment_penalty'] = out['ranking_score']
    out['ranking_score_before_bn_analog_validation_penalty'] = out['ranking_score']

    reference_feature_df = reference_feature_df
    if bool(ranking_config_metadata['domain_support_enabled']):
        if reference_feature_df is None:
            if dataset_df is None:
                raise ValueError(
                    'Domain-support annotation requires either reference_feature_df or dataset_df'
                )
            reference_feature_df = build_feature_table(
                dataset_df,
                formula_col=(cfg.get('data') or {}).get('formula_column', 'formula'),
                feature_set=feature_set,
            )
        domain_support_df = annotate_candidate_domain_support(
            candidate_feature_df=feature_df,
            reference_feature_df=reference_feature_df,
            split_masks=split_masks,
            feature_columns=feature_columns,
            cfg=cfg,
            formula_col='formula',
        )
        out = pd.concat(
            [out.reset_index(drop=True), domain_support_df.drop(columns=['formula'])],
            axis=1,
        )
        if bool(ranking_config_metadata['domain_support_penalty_enabled']):
            out['ranking_score'] = out['ranking_score'] - out['domain_support_penalty'].fillna(0.0)
    else:
        out['domain_support_enabled'] = False
        out['domain_support_method'] = ranking_config_metadata['domain_support_method']
        out['domain_support_distance_metric'] = ranking_config_metadata['domain_support_distance_metric']
        out['domain_support_reference_split'] = ranking_config_metadata['domain_support_reference_split']
        out['domain_support_reference_formula_count'] = 0
        out['domain_support_k_neighbors'] = int(ranking_config_metadata['domain_support_k_neighbors'])
        out['domain_support_nearest_formula'] = ''
        out['domain_support_nearest_distance'] = np.nan
        out['domain_support_mean_k_distance'] = np.nan
        out['domain_support_percentile'] = np.nan
        out['domain_support_penalty'] = 0.0

    out['ranking_score_before_bn_support_penalty'] = out['ranking_score']
    if bool(ranking_config_metadata['bn_support_enabled']):
        if reference_feature_df is None:
            if dataset_df is None:
                raise ValueError(
                    'BN-support annotation requires either reference_feature_df or dataset_df'
                )
            reference_feature_df = build_feature_table(
                dataset_df,
                formula_col=(cfg.get('data') or {}).get('formula_column', 'formula'),
                feature_set=feature_set,
            )
        bn_support_df = annotate_candidate_bn_support(
            candidate_feature_df=feature_df,
            reference_feature_df=reference_feature_df,
            split_masks=split_masks,
            feature_columns=feature_columns,
            cfg=cfg,
            formula_col='formula',
        )
        out = pd.concat(
            [out.reset_index(drop=True), bn_support_df.drop(columns=['formula'])],
            axis=1,
        )
        if bool(ranking_config_metadata['bn_support_penalty_enabled']):
            out['ranking_score'] = out['ranking_score'] - out['bn_support_penalty'].fillna(0.0)
    else:
        out['bn_support_enabled'] = False
        out['bn_support_method'] = ranking_config_metadata['bn_support_method']
        out['bn_support_distance_metric'] = ranking_config_metadata['bn_support_distance_metric']
        out['bn_support_reference_split'] = ranking_config_metadata['bn_support_reference_split']
        out['bn_support_reference_formula_count'] = 0
        out['bn_support_k_neighbors'] = int(ranking_config_metadata['bn_support_k_neighbors'])
        out['bn_support_nearest_formula'] = ''
        out['bn_support_neighbor_formulas'] = ''
        out['bn_support_neighbor_formula_count'] = 0
        out['bn_support_nearest_distance'] = np.nan
        out['bn_support_mean_k_distance'] = np.nan
        out['bn_support_percentile'] = np.nan
        out['bn_support_penalty'] = 0.0

    bn_analog_evidence_cfg = _bn_analog_evidence_config(cfg)
    bn_band_gap_alignment_cfg = _bn_band_gap_alignment_config(cfg)
    bn_analog_validation_cfg = _bn_analog_validation_config(cfg)
    out['bn_band_gap_alignment_penalty'] = 0.0
    out['bn_analog_validation_enabled'] = bool(bn_analog_validation_cfg['enabled'])
    out['bn_analog_validation_method'] = bn_analog_validation_cfg['method']
    out['bn_analog_validation_note'] = bn_analog_validation_cfg['note']
    out['bn_analog_validation_support_fraction'] = np.nan
    out['bn_analog_validation_penalty'] = 0.0
    out['ranking_score_before_bn_band_gap_alignment_penalty'] = out['ranking_score']
    out['ranking_score_before_bn_analog_validation_penalty'] = out['ranking_score']
    if bool(bn_analog_evidence_cfg['enabled']):
        if dataset_df is None:
            raise ValueError('BN analog evidence annotation requires dataset_df')
        bn_analog_df = annotate_candidate_bn_analog_evidence(
            out,
            dataset_df,
            split_masks=split_masks,
            cfg=cfg,
            formula_col='formula',
        )
        out = pd.concat(
            [out.reset_index(drop=True), bn_analog_df.drop(columns=['formula'])],
            axis=1,
        )
        if bool(
            bn_band_gap_alignment_cfg['enabled']
            and bn_band_gap_alignment_cfg['ranking_penalty_enabled']
        ):
            relative_distance = pd.to_numeric(
                out['bn_band_gap_alignment_relative_distance'],
                errors='coerce',
            )
            penalty_eligible = (
                out['bn_band_gap_alignment_penalty_eligible'].fillna(False).astype(bool)
            )
            capped_relative_distance = np.minimum(
                np.nan_to_num(relative_distance.to_numpy(dtype=float), nan=0.0),
                1.0,
            )
            out['bn_band_gap_alignment_penalty'] = np.where(
                penalty_eligible.to_numpy(dtype=bool),
                float(bn_band_gap_alignment_cfg['ranking_penalty_weight'])
                * capped_relative_distance,
                0.0,
            )
            out['ranking_score'] = (
                out['ranking_score'] - out['bn_band_gap_alignment_penalty'].fillna(0.0)
            )
        out['ranking_score_before_bn_analog_validation_penalty'] = out['ranking_score']
        available_metric_count = out['bn_analog_support_available_metric_count'].fillna(0.0)
        vote_count = out['bn_analog_support_vote_count'].fillna(0.0)
        support_fraction = np.where(
            available_metric_count.to_numpy(dtype=float) > 0.0,
            vote_count.to_numpy(dtype=float) / available_metric_count.to_numpy(dtype=float),
            np.nan,
        )
        out['bn_analog_validation_support_fraction'] = support_fraction
        if bool(bn_analog_validation_cfg['enabled'] and bn_analog_validation_cfg['ranking_penalty_enabled']):
            out['bn_analog_validation_penalty'] = np.where(
                np.isfinite(support_fraction),
                float(bn_analog_validation_cfg['ranking_penalty_weight']) * (1.0 - support_fraction),
                0.0,
            )
            out['ranking_score'] = out['ranking_score'] - out['bn_analog_validation_penalty'].fillna(0.0)

    ranking_metadata = get_screening_ranking_metadata(
        cfg,
        domain_support_penalty_applied=bool(out['domain_support_penalty'].fillna(0.0).gt(0.0).any()),
        bn_support_penalty_applied=bool(out['bn_support_penalty'].fillna(0.0).gt(0.0).any()),
        grouped_robustness_penalty_applied=bool(
            out['grouped_robustness_uncertainty_penalty'].fillna(0.0).gt(0.0).any()
        ),
        bn_band_gap_alignment_penalty_applied=bool(
            out['bn_band_gap_alignment_penalty'].fillna(0.0).gt(0.0).any()
        ),
        bn_analog_validation_penalty_applied=bool(
            out['bn_analog_validation_penalty'].fillna(0.0).gt(0.0).any()
        ),
    )

    if dataset_df is not None:
        out = out.merge(
            annotate_candidate_dataset_overlap(
                candidate_df,
                dataset_df,
                split_masks=split_masks,
                formula_col='formula',
            ),
            on='formula',
            how='left',
        )
        if split_masks is not None and 'seen_in_train_plus_val' in out.columns:
            novelty_df = annotate_candidate_novelty(out, formula_col='formula')
            out = pd.concat(
                [out.reset_index(drop=True), novelty_df.drop(columns=['formula'])],
                axis=1,
            )

    out['ranking_label'] = cfg['screening'].get('ranking_label', 'demo_candidate_ranking')
    out['ranking_basis'] = ranking_metadata['ranking_basis']
    out['ranking_note'] = ranking_metadata['ranking_note']
    out['ranking_feature_set'] = feature_set
    out['ranking_model_type'] = model_type
    out['ranking_feature_family'] = get_feature_family(feature_set)
    out['ranking_uncertainty_method'] = ranking_metadata['ranking_uncertainty_method']
    out['ranking_uncertainty_penalty'] = uncertainty_penalty
    out['objective_name'] = cfg['screening'].get(
        'objective_name',
        'bn_themed_formula_level_wide_gap_followup_prioritization',
    )
    out['objective_target_property'] = target_property
    out['objective_target_direction'] = cfg['screening'].get('objective_target_direction', 'maximize')
    out['objective_decision_unit'] = cfg['screening'].get(
        'objective_decision_unit',
        'formula_level_candidate',
    )
    out['objective_decision_consequence'] = cfg['screening'].get(
        'objective_decision_consequence',
        'low_confidence_prioritization_for_structure_followup',
    )
    out['objective_note'] = cfg['screening'].get(
        'objective_note',
        'The screening objective is to prioritize BN-themed formula-level candidates with '
        'wide predicted band gaps for low-confidence downstream follow-up, not to claim '
        'validated discovery.',
    )
    out['ranking_total_penalty'] = (
        out['ranking_uncertainty_penalty_component'].fillna(0.0)
        + out['grouped_robustness_uncertainty_penalty'].fillna(0.0)
        + out['domain_support_penalty'].fillna(0.0)
        + out['bn_support_penalty'].fillna(0.0)
        + out['bn_band_gap_alignment_penalty'].fillna(0.0)
        + out['bn_analog_validation_penalty'].fillna(0.0)
    )
    out['ranking_score_formula'] = (
        'ranking_signal_value - ranking_uncertainty_penalty_component - '
        'grouped_robustness_uncertainty_penalty - domain_support_penalty - '
        'bn_support_penalty - bn_band_gap_alignment_penalty - '
        'bn_analog_validation_penalty'
    )
    out['ranking_active_penalty_terms'] = out.apply(_ranking_active_penalty_terms, axis=1)
    out['ranking_main_penalty_driver'] = out.apply(_ranking_main_penalty_driver, axis=1)
    out['best_overall_evaluation_feature_set'] = best_overall_feature_set or feature_set
    out['best_overall_evaluation_model_type'] = best_overall_model_type or model_type
    out['screening_matches_best_overall_evaluation'] = bool(
        (best_overall_feature_set or feature_set) == feature_set
        and (best_overall_model_type or model_type) == model_type
    )
    out['screening_selection_note'] = screening_selection_note or _screening_selection_note(
        selected_feature_set=best_overall_feature_set or feature_set,
        selected_model_type=best_overall_model_type or model_type,
        screening_feature_set=feature_set,
        screening_model_type=model_type,
    )
    if bool(ranking_metadata['domain_support_enabled']):
        out['ranking_note'] = out['ranking_note'] + ' ' + DOMAIN_SUPPORT_RANKING_NOTE
    if bool(ranking_metadata['bn_support_enabled']):
        out['ranking_note'] = out['ranking_note'] + ' ' + BN_SUPPORT_RANKING_NOTE
    if bool(ranking_metadata['bn_band_gap_alignment_enabled']):
        out['ranking_note'] = out['ranking_note'] + ' ' + BN_BAND_GAP_ALIGNMENT_RANKING_NOTE
    if 'bn_analog_evidence_enabled' in out.columns and bool(out['bn_analog_evidence_enabled'].fillna(False).any()):
        out['ranking_note'] = out['ranking_note'] + ' ' + BN_ANALOG_EVIDENCE_RANKING_NOTE
    if 'candidate_novelty_bucket' in out.columns:
        out['ranking_note'] = out['ranking_note'] + ' ' + NOVELTY_ANNOTATION_RANKING_NOTE
    signal_rank_df = out[['formula', 'ranking_signal_value', 'predicted_band_gap']].copy()
    signal_rank_df = signal_rank_df.sort_values(
        ['ranking_signal_value', 'predicted_band_gap', 'formula'],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    signal_rank_df['ranking_signal_rank'] = np.arange(1, len(signal_rank_df) + 1, dtype=int)
    out = out.merge(signal_rank_df[['formula', 'ranking_signal_rank']], on='formula', how='left')
    out['ranking_signal_selected_for_top_k'] = (
        out['ranking_signal_rank'] <= int(cfg['screening']['top_k'])
    )
    chemical_plausibility_enabled = bool(out.get('chemical_plausibility_enabled', True).fillna(True).all())
    if chemical_plausibility_enabled:
        out['chemical_plausibility_pass'] = out['chemical_plausibility_pass'].fillna(False).astype(bool)
        out['chemical_plausibility_rank_priority'] = out['chemical_plausibility_pass'].astype(int)
        sort_columns = ['chemical_plausibility_rank_priority', 'ranking_score', 'predicted_band_gap']
        ascending = [False, False, False]
        out['ranking_note'] = out['ranking_note'] + ' ' + CHEMICAL_PLAUSIBILITY_SCREENING_NOTE
    else:
        sort_columns = ['ranking_score', 'predicted_band_gap']
        ascending = [False, False]

    out = out.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    out['ranking_rank'] = np.arange(1, len(out) + 1, dtype=int)
    if 'candidate_novelty_bucket' in out.columns:
        out['novelty_rank_within_bucket'] = (
            out.groupby('candidate_novelty_bucket').cumcount() + 1
        ).astype(int)
        out['novel_formula_rank'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
        novel_mask = out['candidate_is_formula_level_extrapolation'].fillna(False).astype(bool)
        out.loc[novel_mask, 'novel_formula_rank'] = (
            out.loc[novel_mask, 'novelty_rank_within_bucket'].astype(int).to_numpy()
        )
    out['screening_selected_for_top_k'] = out['ranking_rank'] <= int(cfg['screening']['top_k'])
    out['ranking_penalty_rank_shift'] = (
        out['ranking_rank'].astype(int) - out['ranking_signal_rank'].astype(int)
    )
    out['ranking_penalty_impact_label'] = 'rank_unchanged_after_penalties'
    out.loc[
        out['ranking_penalty_rank_shift'] > 0,
        'ranking_penalty_impact_label',
    ] = 'moved_down_after_penalties'
    out.loc[
        out['ranking_penalty_rank_shift'] < 0,
        'ranking_penalty_impact_label',
    ] = 'moved_up_after_penalties'
    out['screening_selection_decision'] = 'not_selected_top_k'
    out.loc[out['screening_selected_for_top_k'], 'screening_selection_decision'] = 'selected_top_k'
    if chemical_plausibility_enabled:
        out.loc[
            ~out['screening_selected_for_top_k'] & ~out['chemical_plausibility_pass'],
            'screening_selection_decision',
        ] = 'failed_chemical_plausibility'
    out['ranking_decision_summary'] = out.apply(_ranking_decision_summary, axis=1)
    out = annotate_candidate_proposal_shortlist(out, cfg=cfg)
    out = annotate_candidate_extrapolation_shortlist(out, cfg=cfg)

    return out

