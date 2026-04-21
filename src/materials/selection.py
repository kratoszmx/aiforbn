from __future__ import annotations

import numpy as np
import pandas as pd

from materials.constants import *
from materials.candidate_space import *
from materials.candidate_space import _ordered_values
from materials.feature_building import *
from materials.feature_building import _default_model_type_for_feature_set
from materials.modeling import *

def _ordered_model_types(values: list[str]) -> list[str]:
    return _ordered_values(values)


def _metric_key(value):
    if value is None:
        return np.inf
    return value


def _select_best_validation_result(
    validation_results: list[dict],
    selection_metric: str,
    allowed_feature_sets: list[str] | None = None,
) -> dict | None:
    allowed = set(allowed_feature_sets) if allowed_feature_sets is not None else None
    best_result = None
    for result in validation_results:
        if result.get('status') != 'ok':
            continue
        if allowed is not None and result.get('feature_set') not in allowed:
            continue
        if best_result is None:
            best_result = result
            continue

        if selection_metric == 'r2':
            is_better = _metric_key(result.get('r2')) > _metric_key(best_result.get('r2'))
        else:
            is_better = _metric_key(result.get(selection_metric)) < _metric_key(best_result.get(selection_metric))
        if is_better:
            best_result = result
    return best_result


def _screening_selection_note(
    selected_feature_set: str,
    selected_model_type: str,
    screening_feature_set: str,
    screening_model_type: str,
) -> str:
    if selected_feature_set == screening_feature_set and selected_model_type == screening_model_type:
        return 'Best overall validation combo is candidate-compatible, so screening reuses it.'
    return (
        'Best overall validation combo requires structure-derived inputs, so formula-only candidate '
        'screening falls back to the best candidate-compatible validation combo.'
    )


def _format_optional_float(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return 'na'
    return f'{float(value):.{digits}f}'


def _ranking_penalty_items(row: pd.Series) -> list[tuple[str, float]]:
    penalty_columns = [
        ('model_disagreement_penalty', 'ranking_uncertainty_penalty_component'),
        ('grouped_robustness_penalty', 'grouped_robustness_uncertainty_penalty'),
        ('domain_support_penalty', 'domain_support_penalty'),
        ('bn_support_penalty', 'bn_support_penalty'),
        ('bn_band_gap_alignment_penalty', 'bn_band_gap_alignment_penalty'),
        ('bn_analog_validation_penalty', 'bn_analog_validation_penalty'),
    ]
    items = []
    for label, column in penalty_columns:
        value = pd.to_numeric(pd.Series([row.get(column, 0.0)]), errors='coerce').iloc[0]
        items.append((label, 0.0 if pd.isna(value) else float(value)))
    return items


def _ranking_active_penalty_terms(row: pd.Series) -> str:
    active_terms = [label for label, value in _ranking_penalty_items(row) if value > 0.0]
    return '|'.join(active_terms) if active_terms else 'none'


def _ranking_main_penalty_driver(row: pd.Series) -> str:
    items = _ranking_penalty_items(row)
    if not items:
        return 'none'
    label, value = max(items, key=lambda item: item[1])
    return label if value > 0.0 else 'none'


def _ranking_decision_summary(row: pd.Series) -> str:
    plausibility_label = (
        'pass' if bool(row.get('chemical_plausibility_pass', True)) else 'fail'
    )
    novelty_bucket = row.get('candidate_novelty_bucket', 'unannotated')
    selection_decision = row.get('screening_selection_decision', 'unassigned')
    return (
        f"rank={int(row.get('ranking_rank', -1))}; "
        f"signal_rank={int(row.get('ranking_signal_rank', -1))}; "
        f"rank_shift={int(row.get('ranking_penalty_rank_shift', 0))}; "
        f"score={_format_optional_float(row.get('ranking_score'))}; "
        f"signal={str(row.get('ranking_signal_source', 'predicted_band_gap'))}"
        f"({_format_optional_float(row.get('ranking_signal_value'))}); "
        f"penalty_total={_format_optional_float(row.get('ranking_total_penalty'))}; "
        f"main_penalty={str(row.get('ranking_main_penalty_driver', 'none'))}; "
        f"active_penalties={str(row.get('ranking_active_penalty_terms', 'none'))}; "
        f"selection={selection_decision}; novelty={novelty_bucket}; plausibility={plausibility_label}"
    )


def select_feature_model_combo(feature_tables: dict[str, pd.DataFrame], split_masks, cfg: dict) -> dict:
    candidate_feature_sets = [value for value in get_candidate_feature_sets(cfg) if value in feature_tables]
    candidate_model_types = get_candidate_model_types(cfg)
    selection_metric = cfg['model'].get('selection_metric', 'mae')
    use_validation_selection = bool(cfg['model'].get('use_validation_selection', True))
    default_feature_set = cfg['features'].get('feature_set', candidate_feature_sets[0])
    default_model_type = cfg['model'].get('type', candidate_model_types[0])

    feature_set_results = [
        summarize_feature_table(feature_tables[feature_set], feature_set=feature_set)
        for feature_set in candidate_feature_sets
    ]
    eligible_feature_sets = [
        item['feature_set']
        for item in feature_set_results
        if item['selection_eligible']
    ]
    screening_candidate_feature_sets = [
        item['feature_set']
        for item in feature_set_results
        if item['selection_eligible'] and item['candidate_compatible']
    ]

    if not eligible_feature_sets:
        raise ValueError('No candidate feature set could featurize the full dataset')
    if not screening_candidate_feature_sets:
        raise ValueError('No candidate-compatible feature set is available for formula-only screening')
    if not any(
        compatible_model_types_for_feature_set(cfg, feature_set)
        for feature_set in eligible_feature_sets
    ):
        raise ValueError('No compatible feature/model combination could featurize the full dataset')
    if not any(
        compatible_model_types_for_feature_set(cfg, feature_set)
        for feature_set in screening_candidate_feature_sets
    ):
        raise ValueError(
            'No candidate-compatible feature/model combination is available for formula-only screening'
        )

    if default_feature_set not in eligible_feature_sets:
        default_feature_set = eligible_feature_sets[0]
    screening_default_feature_set = (
        default_feature_set
        if default_feature_set in screening_candidate_feature_sets
        else screening_candidate_feature_sets[0]
    )
    default_model_type = _default_model_type_for_feature_set(cfg, default_feature_set)
    screening_default_model_type = _default_model_type_for_feature_set(
        cfg,
        screening_default_feature_set,
    )

    summary = {
        'selection_space': 'feature_set_and_model_type',
        'selection_scope': 'all_configured_feature_sets',
        'candidate_feature_sets': candidate_feature_sets,
        'candidate_model_types': candidate_model_types,
        'selection_metric': selection_metric,
        'used_validation_selection': False,
        'selected_feature_set': default_feature_set,
        'selected_model_type': default_model_type,
        'selected_feature_family': get_feature_family(default_feature_set),
        'selected_feature_count': int(
            summarize_feature_table(feature_tables[default_feature_set], feature_set=default_feature_set)['n_features']
        ),
        'screening_selection_scope': FORMULA_ONLY_SCREENING_SCOPE,
        'screening_candidate_feature_sets': screening_candidate_feature_sets,
        'screening_selected_feature_set': screening_default_feature_set,
        'screening_selected_model_type': screening_default_model_type,
        'screening_selected_feature_family': get_feature_family(screening_default_feature_set),
        'screening_selected_feature_count': int(
            summarize_feature_table(
                feature_tables[screening_default_feature_set],
                feature_set=screening_default_feature_set,
            )['n_features']
        ),
        'screening_selection_matches_overall': bool(
            screening_default_feature_set == default_feature_set
            and screening_default_model_type == default_model_type
        ),
        'feature_set_results': feature_set_results,
        'validation_results': [],
    }
    summary['screening_selection_note'] = _screening_selection_note(
        selected_feature_set=summary['selected_feature_set'],
        selected_model_type=summary['selected_model_type'],
        screening_feature_set=summary['screening_selected_feature_set'],
        screening_model_type=summary['screening_selected_model_type'],
    )

    if not use_validation_selection or len(candidate_feature_sets) * len(candidate_model_types) == 1:
        return summary

    if int(np.asarray(split_masks['val']).sum()) == 0:
        summary['selection_note'] = 'validation_split_empty'
        return summary

    for feature_set in candidate_feature_sets:
        feature_df = feature_tables[feature_set]
        feature_info = summarize_feature_table(feature_df, feature_set=feature_set)
        for model_type in candidate_model_types:
            result = {
                'feature_set': feature_set,
                'feature_family': feature_info['feature_family'],
                'candidate_compatible': feature_info['candidate_compatible'],
                'model_type': model_type,
                'n_features': feature_info['n_features'],
                'status': 'ok',
                'note': '',
                'mae': None,
                'rmse': None,
                'r2': None,
            }

            if not feature_info['selection_eligible']:
                result['status'] = 'skipped_featurization_failure'
                result['note'] = (
                    'Skipped because this feature set could not featurize every dataset formula.'
                )
                summary['validation_results'].append(result)
                continue
            if not model_type_supports_feature_set(model_type, feature_set):
                result['status'] = 'skipped_model_feature_incompatible'
                result['note'] = incompatible_model_feature_note(model_type, feature_set)
                summary['validation_results'].append(result)
                continue

            try:
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
                    split_name='val',
                )
            except Exception as exc:
                result['status'] = 'evaluation_failed'
                result['note'] = f'{type(exc).__name__}: {exc}'
                summary['validation_results'].append(result)
                continue

            result.update(metrics)
            summary['validation_results'].append(result)

    best_result = _select_best_validation_result(
        summary['validation_results'],
        selection_metric=selection_metric,
    )
    if best_result is None:
        raise ValueError('Validation selection failed for every candidate feature/model combination')
    best_screening_result = _select_best_validation_result(
        summary['validation_results'],
        selection_metric=selection_metric,
        allowed_feature_sets=screening_candidate_feature_sets,
    )
    if best_screening_result is None:
        raise ValueError('Validation selection failed for every candidate-compatible screening combo')

    summary['used_validation_selection'] = True
    summary['selected_feature_set'] = best_result['feature_set']
    summary['selected_model_type'] = best_result['model_type']
    summary['selected_feature_family'] = get_feature_family(best_result['feature_set'])
    summary['selected_feature_count'] = int(
        summarize_feature_table(
            feature_tables[best_result['feature_set']],
            feature_set=best_result['feature_set'],
        )['n_features']
    )
    summary['screening_selected_feature_set'] = best_screening_result['feature_set']
    summary['screening_selected_model_type'] = best_screening_result['model_type']
    summary['screening_selected_feature_family'] = get_feature_family(best_screening_result['feature_set'])
    summary['screening_selected_feature_count'] = int(
        summarize_feature_table(
            feature_tables[best_screening_result['feature_set']],
            feature_set=best_screening_result['feature_set'],
        )['n_features']
    )
    summary['screening_selection_matches_overall'] = bool(
        summary['selected_feature_set'] == summary['screening_selected_feature_set']
        and summary['selected_model_type'] == summary['screening_selected_model_type']
    )
    summary['screening_selection_note'] = _screening_selection_note(
        selected_feature_set=summary['selected_feature_set'],
        selected_model_type=summary['selected_model_type'],
        screening_feature_set=summary['screening_selected_feature_set'],
        screening_model_type=summary['screening_selected_model_type'],
    )
    return summary


def select_model_type(feature_tables: dict[str, pd.DataFrame], split_masks, cfg: dict) -> dict:
    return select_feature_model_combo(feature_tables, split_masks, cfg)

