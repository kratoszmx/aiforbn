from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from materials.constants import *

def extract_elements(formula: str) -> list[str]:
    return re.findall(r'[A-Z][a-z]?', formula or '')


def filter_bn(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    mask = df[formula_col].astype(str).apply(lambda x: {'B', 'N'}.issubset(set(extract_elements(x))))
    out = df.loc[mask].copy()
    out['elements'] = out[formula_col].astype(str).apply(extract_elements)
    return out


def _bn_family_label(
    formula: str,
    *,
    grouping_method: str = DEFAULT_BN_FAMILY_GROUPING_METHOD,
) -> str:
    if grouping_method != 'reduced_bn_chemical_system':
        raise ValueError(f'Unsupported BN family grouping method: {grouping_method}')

    try:
        composition = Composition(str(formula)).reduced_composition
        elements = sorted(str(element) for element in composition.elements)
    except Exception:
        elements = sorted(set(extract_elements(str(formula))))

    if not {'B', 'N'}.issubset(set(elements)):
        return 'non_bn'
    return '-'.join(elements)


def annotate_bn_families(
    df: pd.DataFrame,
    *,
    formula_col: str = 'formula',
    grouping_method: str = DEFAULT_BN_FAMILY_GROUPING_METHOD,
) -> pd.DataFrame:
    out = df.copy()
    out['bn_family'] = out[formula_col].astype(str).apply(
        lambda value: _bn_family_label(value, grouping_method=grouping_method)
    )
    out['bn_family_grouping_method'] = grouping_method
    return out


def _candidate_space_metadata(cfg: dict | None = None) -> dict[str, str]:
    screening_cfg = (cfg or {}).get('screening', {})
    strategy = screening_cfg.get(
        'candidate_generation_strategy',
        DEFAULT_CANDIDATE_GENERATION_STRATEGY,
    )
    if strategy not in CANDIDATE_SPACE_DEFAULTS:
        raise ValueError(f'Unsupported candidate_generation_strategy: {strategy}')

    defaults = CANDIDATE_SPACE_DEFAULTS[strategy]
    return {
        'candidate_generation_strategy': strategy,
        'candidate_space_name': screening_cfg.get(
            'candidate_space_name', defaults['candidate_space_name']
        ),
        'candidate_space_kind': screening_cfg.get(
            'candidate_space_kind', defaults['candidate_space_kind']
        ),
        'candidate_space_note': screening_cfg.get(
            'candidate_space_note', defaults['candidate_space_note']
        ),
    }


def _canonical_formula(formula: str) -> str:
    return Composition(str(formula)).reduced_formula


def _candidate_row(
    formula: str,
    metadata: dict[str, str],
    candidate_family: str,
    candidate_template: str,
    candidate_family_note: str,
) -> dict[str, str]:
    return {
        'formula': _canonical_formula(formula),
        'candidate_space_name': metadata['candidate_space_name'],
        'candidate_space_kind': metadata['candidate_space_kind'],
        'candidate_generation_strategy': metadata['candidate_generation_strategy'],
        'candidate_space_note': metadata['candidate_space_note'],
        'candidate_family': candidate_family,
        'candidate_template': candidate_template,
        'candidate_family_note': candidate_family_note,
    }


def _generate_toy_iii_v_candidates(metadata: dict[str, str]) -> list[dict[str, str]]:
    group13 = ['B', 'Al', 'Ga', 'In', 'Tl']
    group15 = ['N', 'P', 'As', 'Sb', 'Bi']
    family_note = (
        'Legacy control grid copied from the 2DMatPedia-style Group 13/15 binary substitution '
        'idea. Useful as a transparent control candidate space, but weakly aligned with BN as a '
        'research target.'
    )
    rows: list[dict[str, str]] = []
    for left in group13:
        for right in group15:
            rows.append(
                _candidate_row(
                    formula=f'{left}{right}',
                    metadata=metadata,
                    candidate_family='group13_group15_binary_analog',
                    candidate_template='A1B1',
                    candidate_family_note=family_note,
                )
            )
    return rows


def _generate_bn_anchored_candidates(metadata: dict[str, str]) -> list[dict[str, str]]:
    group14 = ['C', 'Si', 'Ge', 'Sn']
    group13 = ['Al', 'Ga', 'In', 'Tl']
    rows: list[dict[str, str]] = [
        _candidate_row(
            formula='BN',
            metadata=metadata,
            candidate_family='bn_binary_anchor',
            candidate_template='B1N1',
            candidate_family_note='Binary BN anchor/control inside the BN-containing candidate grid.',
        )
    ]

    group14_families = [
        (
            'group14_bn_111_family',
            'B1X1N1',
            lambda element: f'B{element}N',
            'BN-containing Group-IV ternary family anchored by BCN / h-BCN-style literature motifs.',
        ),
        (
            'group14_bn_211_family',
            'B1X2N1',
            lambda element: f'B{element}2N',
            'BN-containing Group-IV ternary family anchored by BC2N-style literature motifs and by Si2BN-like dataset motifs.',
        ),
        (
            'group14_bn_121_family',
            'B1X1N2',
            lambda element: f'B{element}N2',
            'BN-containing Group-IV ternary family that preserves a BN-centered local template while exploring more N-rich stoichiometries.',
        ),
    ]
    for family_name, template, formula_builder, family_note in group14_families:
        for element in group14:
            rows.append(
                _candidate_row(
                    formula=formula_builder(element),
                    metadata=metadata,
                    candidate_family=family_name,
                    candidate_template=template,
                    candidate_family_note=family_note,
                )
            )

    group13_families = [
        (
            'group13_bn_111_family',
            'X1B1N1',
            lambda element: f'{element}BN',
            'BN-containing Group-III ternary family used as a BN-local exploratory extension. Some members fail the lightweight oxidation-state plausibility screen.',
        ),
        (
            'group13_bn_211_family',
            'X2B1N1',
            lambda element: f'{element}2BN',
            'BN-containing Group-III ternary family that keeps B and N explicit while moving to a more cation-rich stoichiometric template.',
        ),
        (
            'group13_bn_121_family',
            'X1B1N2',
            lambda element: f'{element}BN2',
            'BN-containing Group-III ternary family with a simple charge-balanced nitride-like template.',
        ),
    ]
    for family_name, template, formula_builder, family_note in group13_families:
        for element in group13:
            rows.append(
                _candidate_row(
                    formula=formula_builder(element),
                    metadata=metadata,
                    candidate_family=family_name,
                    candidate_template=template,
                    candidate_family_note=family_note,
                )
            )

    deduped_rows: list[dict[str, str]] = []
    seen_formulas: set[str] = set()
    for row in rows:
        formula = row['formula']
        if formula in seen_formulas:
            continue
        deduped_rows.append(row)
        seen_formulas.add(formula)
    return deduped_rows


def generate_bn_candidates(cfg: dict | None = None) -> pd.DataFrame:
    metadata = _candidate_space_metadata(cfg)
    strategy = metadata['candidate_generation_strategy']
    if strategy == TOY_CANDIDATE_GENERATION_STRATEGY:
        rows = _generate_toy_iii_v_candidates(metadata)
    elif strategy == BN_ANCHORED_CANDIDATE_GENERATION_STRATEGY:
        rows = _generate_bn_anchored_candidates(metadata)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported candidate_generation_strategy: {strategy}')
    return annotate_candidate_chemical_plausibility(pd.DataFrame(rows))


def _ordered_values(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _chemical_plausibility_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    plausibility_cfg = screening_cfg.get('chemical_plausibility', {})
    return {
        'enabled': bool(plausibility_cfg.get('enabled', True)),
        'method': plausibility_cfg.get('method', DEFAULT_CHEMICAL_PLAUSIBILITY_METHOD),
        'selection_policy': plausibility_cfg.get(
            'selection_policy',
            'annotate_and_prioritize_passing_candidates',
        ),
        'note': plausibility_cfg.get('note', DEFAULT_CHEMICAL_PLAUSIBILITY_NOTE),
    }


def _proposal_shortlist_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    shortlist_cfg = screening_cfg.get('proposal_shortlist', {})
    top_k = int(screening_cfg.get('top_k', 20))
    shortlist_size = int(shortlist_cfg.get('shortlist_size', min(top_k, 10)))
    family_cap = int(shortlist_cfg.get('max_per_candidate_family', 2))
    if shortlist_size <= 0:
        raise ValueError('proposal_shortlist.shortlist_size must be positive')
    if family_cap <= 0:
        raise ValueError('proposal_shortlist.max_per_candidate_family must be positive')
    return {
        'enabled': bool(shortlist_cfg.get('enabled', True)),
        'label': str(shortlist_cfg.get('label', DEFAULT_PROPOSAL_SHORTLIST_LABEL)),
        'method': str(shortlist_cfg.get('method', DEFAULT_PROPOSAL_SHORTLIST_METHOD)),
        'shortlist_size': shortlist_size,
        'max_per_candidate_family': family_cap,
        'chemical_plausibility_priority': bool(
            shortlist_cfg.get('chemical_plausibility_priority', True)
        ),
        'note': str(shortlist_cfg.get('note', DEFAULT_PROPOSAL_SHORTLIST_NOTE)),
    }


def _extrapolation_shortlist_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    shortlist_cfg = screening_cfg.get('extrapolation_shortlist', {})
    top_k = int(screening_cfg.get('top_k', 20))
    shortlist_size = int(shortlist_cfg.get('shortlist_size', min(top_k, 5)))
    family_cap = int(shortlist_cfg.get('max_per_candidate_family', 1))
    required_novelty_bucket = str(
        shortlist_cfg.get(
            'required_novelty_bucket',
            NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION,
        )
    )
    if shortlist_size <= 0:
        raise ValueError('extrapolation_shortlist.shortlist_size must be positive')
    if family_cap <= 0:
        raise ValueError('extrapolation_shortlist.max_per_candidate_family must be positive')
    if required_novelty_bucket not in NOVELTY_BUCKET_PRIORITY:
        raise ValueError(
            'extrapolation_shortlist.required_novelty_bucket must be one of '
            f'{sorted(NOVELTY_BUCKET_PRIORITY)}'
        )
    return {
        'enabled': bool(shortlist_cfg.get('enabled', True)),
        'label': str(shortlist_cfg.get('label', DEFAULT_EXTRAPOLATION_SHORTLIST_LABEL)),
        'method': str(shortlist_cfg.get('method', DEFAULT_EXTRAPOLATION_SHORTLIST_METHOD)),
        'shortlist_size': shortlist_size,
        'max_per_candidate_family': family_cap,
        'chemical_plausibility_priority': bool(
            shortlist_cfg.get('chemical_plausibility_priority', True)
        ),
        'required_novelty_bucket': required_novelty_bucket,
        'note': str(shortlist_cfg.get('note', DEFAULT_EXTRAPOLATION_SHORTLIST_NOTE)),
    }


def _structure_generation_seed_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    seed_cfg = screening_cfg.get('structure_generation_seeds', {})
    proposal_cfg = _proposal_shortlist_config(cfg)
    per_candidate_seed_limit = int(seed_cfg.get('per_candidate_seed_limit', 3))
    bn_centered_top_n = int(seed_cfg.get('bn_centered_top_n', proposal_cfg['shortlist_size']))
    if per_candidate_seed_limit <= 0:
        raise ValueError('structure_generation_seeds.per_candidate_seed_limit must be positive')
    if bn_centered_top_n <= 0:
        raise ValueError('structure_generation_seeds.bn_centered_top_n must be positive')
    return {
        'enabled': bool(seed_cfg.get('enabled', True)),
        'label': str(seed_cfg.get('label', DEFAULT_STRUCTURE_GENERATION_SEED_LABEL)),
        'method': str(seed_cfg.get('method', DEFAULT_STRUCTURE_GENERATION_SEED_METHOD)),
        'candidate_scope': str(
            seed_cfg.get(
                'candidate_scope',
                DEFAULT_STRUCTURE_GENERATION_SEED_CANDIDATE_SCOPE,
            )
        ),
        'per_candidate_seed_limit': per_candidate_seed_limit,
        'bn_centered_top_n': bn_centered_top_n,
        'note': str(seed_cfg.get('note', DEFAULT_STRUCTURE_GENERATION_SEED_NOTE)),
    }


def _formula_amount_map(formula: str) -> dict[str, float]:
    comp = Composition(str(formula)).reduced_composition
    return {str(key): float(value) for key, value in comp.get_el_amt_dict().items()}


def _structure_seed_edit_metadata(candidate_formula: str, seed_formula: str) -> dict[str, object]:
    candidate_amounts = _formula_amount_map(candidate_formula)
    seed_amounts = _formula_amount_map(seed_formula)
    candidate_elements = _ordered_values(extract_elements(candidate_formula))
    seed_elements = _ordered_values(extract_elements(seed_formula))
    shared_elements = [element for element in candidate_elements if element in seed_amounts]
    candidate_only_elements = [element for element in candidate_elements if element not in seed_amounts]
    seed_only_elements = [element for element in seed_elements if element not in candidate_amounts]
    all_elements = sorted(set(candidate_amounts) | set(seed_amounts))
    element_count_l1_distance = float(
        sum(
            abs(candidate_amounts.get(element, 0.0) - seed_amounts.get(element, 0.0))
            for element in all_elements
        )
    )
    exact_element_match = not candidate_only_elements and not seed_only_elements

    if exact_element_match and element_count_l1_distance == 0.0:
        edit_strategy = 'same_reduced_formula_reference'
    elif exact_element_match:
        edit_strategy = 'same_elements_stoichiometry_adjustment'
    elif candidate_only_elements and seed_only_elements:
        edit_strategy = 'element_substitution_or_decoration'
    elif candidate_only_elements:
        edit_strategy = 'element_insertion_or_decoration'
    elif seed_only_elements:
        edit_strategy = 'element_removal_or_vacancy'
    else:
        edit_strategy = 'mixed_formula_edit'

    return {
        'seed_formula_exact_element_match': bool(exact_element_match),
        'seed_formula_shared_elements': '|'.join(shared_elements),
        'seed_formula_candidate_only_elements': '|'.join(candidate_only_elements),
        'seed_formula_seed_only_elements': '|'.join(seed_only_elements),
        'seed_formula_element_count_l1_distance': element_count_l1_distance,
        'seed_formula_edit_strategy': edit_strategy,
    }


def _annotate_ranked_family_capped_shortlist(
    ranked_candidate_df: pd.DataFrame,
    *,
    prefix: str,
    shortlist_cfg: dict[str, object],
    required_novelty_bucket: str | None = None,
) -> pd.DataFrame:
    out = ranked_candidate_df.copy()
    out[f'{prefix}_enabled'] = bool(shortlist_cfg['enabled'])
    out[f'{prefix}_label'] = str(shortlist_cfg['label'])
    out[f'{prefix}_method'] = str(shortlist_cfg['method'])
    out[f'{prefix}_note'] = str(shortlist_cfg['note'])
    out[f'{prefix}_size'] = int(shortlist_cfg['shortlist_size'])
    out[f'{prefix}_family_cap'] = int(shortlist_cfg['max_per_candidate_family'])
    out[f'{prefix}_chemical_plausibility_priority'] = bool(
        shortlist_cfg['chemical_plausibility_priority']
    )
    out[f'{prefix}_family_count_before_selection'] = 0
    out[f'{prefix}_selected'] = False
    out[f'{prefix}_rank'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    if required_novelty_bucket is not None:
        out[f'{prefix}_target_novelty_bucket'] = required_novelty_bucket
    if not bool(shortlist_cfg['enabled']):
        out[f'{prefix}_decision'] = f'{prefix}_disabled'
        return out

    if 'candidate_family' not in out.columns:
        raise ValueError(
            f'{prefix} requires candidate_family so the family-aware cap is explicit.'
        )
    if required_novelty_bucket is not None and 'candidate_novelty_bucket' not in out.columns:
        raise ValueError(
            f'{prefix} requires candidate_novelty_bucket so the novelty filter is explicit.'
        )

    ranked_index = (
        out.sort_values('ranking_rank', ascending=True, kind='stable').index.tolist()
        if 'ranking_rank' in out.columns
        else out.index.tolist()
    )
    if (
        bool(shortlist_cfg['chemical_plausibility_priority'])
        and 'chemical_plausibility_pass' in out.columns
    ):
        plausibility_mask = out['chemical_plausibility_pass'].fillna(False).astype(bool)
        passing_index = (
            out.loc[plausibility_mask]
            .sort_values('ranking_rank', ascending=True, kind='stable')
            .index.tolist()
            if 'ranking_rank' in out.columns
            else out.index[plausibility_mask].tolist()
        )
        failing_index = (
            out.loc[~plausibility_mask]
            .sort_values('ranking_rank', ascending=True, kind='stable')
            .index.tolist()
            if 'ranking_rank' in out.columns
            else out.index[~plausibility_mask].tolist()
        )
        ranked_index = passing_index + failing_index

    selected_count = 0
    family_counts: dict[str, int] = {}
    out[f'{prefix}_decision'] = f'not_selected_for_{prefix}'
    for idx in ranked_index:
        family_name = str(out.at[idx, 'candidate_family'])
        family_count_before = int(family_counts.get(family_name, 0))
        out.at[idx, f'{prefix}_family_count_before_selection'] = family_count_before

        if required_novelty_bucket is not None and (
            str(out.at[idx, 'candidate_novelty_bucket']) != required_novelty_bucket
        ):
            out.at[idx, f'{prefix}_decision'] = 'not_selected_novelty_bucket_mismatch'
            continue
        if (
            bool(shortlist_cfg['chemical_plausibility_priority'])
            and 'chemical_plausibility_pass' in out.columns
            and not bool(out.at[idx, 'chemical_plausibility_pass'])
        ):
            out.at[idx, f'{prefix}_decision'] = (
                'not_selected_failed_chemical_plausibility'
            )
            continue
        if selected_count >= int(shortlist_cfg['shortlist_size']):
            out.at[idx, f'{prefix}_decision'] = 'not_selected_shortlist_full'
            continue
        if family_count_before >= int(shortlist_cfg['max_per_candidate_family']):
            out.at[idx, f'{prefix}_decision'] = 'not_selected_family_cap_reached'
            continue

        selected_count += 1
        family_counts[family_name] = family_count_before + 1
        out.at[idx, f'{prefix}_selected'] = True
        out.at[idx, f'{prefix}_rank'] = selected_count
        out.at[idx, f'{prefix}_decision'] = f'selected_for_{prefix}'

    return out


def annotate_candidate_proposal_shortlist(
    ranked_candidate_df: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    shortlist_cfg = _proposal_shortlist_config(cfg)
    return _annotate_ranked_family_capped_shortlist(
        ranked_candidate_df,
        prefix='proposal_shortlist',
        shortlist_cfg=shortlist_cfg,
    )


def annotate_candidate_extrapolation_shortlist(
    ranked_candidate_df: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    shortlist_cfg = _extrapolation_shortlist_config(cfg)
    return _annotate_ranked_family_capped_shortlist(
        ranked_candidate_df,
        prefix='extrapolation_shortlist',
        shortlist_cfg=shortlist_cfg,
        required_novelty_bucket=str(shortlist_cfg['required_novelty_bucket']),
    )


def _domain_support_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    support_cfg = screening_cfg.get('domain_support', {})
    k_neighbors = max(1, int(support_cfg.get('k_neighbors', 5)))
    ranking_penalty_weight = max(0.0, float(support_cfg.get('ranking_penalty_weight', 0.15)))
    penalize_below_percentile = float(support_cfg.get('penalize_below_percentile', 25.0))
    penalize_below_percentile = min(max(penalize_below_percentile, 0.0), 100.0)
    enabled = bool(support_cfg.get('enabled', True))
    return {
        'enabled': enabled,
        'method': support_cfg.get('method', DEFAULT_DOMAIN_SUPPORT_METHOD),
        'distance_metric': support_cfg.get(
            'distance_metric',
            DEFAULT_DOMAIN_SUPPORT_DISTANCE_METRIC,
        ),
        'k_neighbors': k_neighbors,
        'ranking_penalty_enabled': enabled and bool(
            support_cfg.get('ranking_penalty_enabled', True)
        ),
        'ranking_penalty_weight': ranking_penalty_weight,
        'penalize_below_percentile': penalize_below_percentile,
        'note': support_cfg.get('note', DEFAULT_DOMAIN_SUPPORT_NOTE),
    }


def _bn_support_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    support_cfg = screening_cfg.get('bn_support', {})
    k_neighbors = max(1, int(support_cfg.get('k_neighbors', 3)))
    ranking_penalty_weight = max(0.0, float(support_cfg.get('ranking_penalty_weight', 0.1)))
    penalize_below_percentile = float(support_cfg.get('penalize_below_percentile', 25.0))
    penalize_below_percentile = min(max(penalize_below_percentile, 0.0), 100.0)
    enabled = bool(support_cfg.get('enabled', True))
    return {
        'enabled': enabled,
        'method': support_cfg.get('method', DEFAULT_BN_SUPPORT_METHOD),
        'distance_metric': support_cfg.get(
            'distance_metric',
            DEFAULT_BN_SUPPORT_DISTANCE_METRIC,
        ),
        'k_neighbors': k_neighbors,
        'ranking_penalty_enabled': enabled and bool(
            support_cfg.get('ranking_penalty_enabled', True)
        ),
        'ranking_penalty_weight': ranking_penalty_weight,
        'penalize_below_percentile': penalize_below_percentile,
        'note': support_cfg.get('note', DEFAULT_BN_SUPPORT_NOTE),
    }


def _bn_analog_evidence_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    evidence_cfg = screening_cfg.get('bn_analog_evidence', {})
    return {
        'enabled': bool(evidence_cfg.get('enabled', True)),
        'aggregation': evidence_cfg.get(
            'aggregation',
            'mean_over_k_nearest_bn_formulas',
        ),
        'reference_split': evidence_cfg.get(
            'reference_split',
            BN_ANALOG_EVIDENCE_REFERENCE_SPLIT,
        ),
        'exfoliation_reference': evidence_cfg.get(
            'exfoliation_reference',
            'train_plus_val_bn_formula_median',
        ),
        'note': evidence_cfg.get('note', DEFAULT_BN_ANALOG_EVIDENCE_NOTE),
    }


def _bn_band_gap_alignment_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    alignment_cfg = screening_cfg.get('bn_band_gap_alignment', {})
    enabled = bool(alignment_cfg.get('enabled', True))
    return {
        'enabled': enabled,
        'method': alignment_cfg.get(
            'method',
            'predicted_band_gap_vs_local_bn_analog_window',
        ),
        'reference_split': alignment_cfg.get(
            'reference_split',
            BN_BAND_GAP_ALIGNMENT_REFERENCE_SPLIT,
        ),
        'window_expansion_iqr_factor': max(
            0.0,
            float(alignment_cfg.get('window_expansion_iqr_factor', 0.5)),
        ),
        'minimum_neighbor_formula_count_for_penalty': max(
            1,
            int(alignment_cfg.get('minimum_neighbor_formula_count_for_penalty', 2)),
        ),
        'ranking_penalty_enabled': enabled and bool(
            alignment_cfg.get('ranking_penalty_enabled', True)
        ),
        'ranking_penalty_weight': max(
            0.0,
            float(alignment_cfg.get('ranking_penalty_weight', 0.08)),
        ),
        'note': alignment_cfg.get('note', DEFAULT_BN_BAND_GAP_ALIGNMENT_NOTE),
    }


def _robustness_config(cfg: dict | None = None) -> dict:
    resolved_cfg = cfg or {}
    robustness_cfg = resolved_cfg.get('robustness', {})
    split_cfg = resolved_cfg.get('split', {})
    return {
        'enabled': bool(robustness_cfg.get('enabled', False)),
        'method': robustness_cfg.get('method', DEFAULT_ROBUSTNESS_METHOD),
        'group_column': robustness_cfg.get(
            'group_column',
            split_cfg.get('group_column', 'formula'),
        ),
        'n_splits': max(2, int(robustness_cfg.get('n_splits', 5))),
        'note': robustness_cfg.get('note', DEFAULT_ROBUSTNESS_NOTE),
    }


def _bn_slice_benchmark_config(cfg: dict | None = None) -> dict:
    resolved_cfg = cfg or {}
    benchmark_cfg = resolved_cfg.get('bn_slice_benchmark', {})
    return {
        'enabled': bool(benchmark_cfg.get('enabled', False)),
        'method': benchmark_cfg.get('method', DEFAULT_BN_SLICE_BENCHMARK_METHOD),
        'k_neighbors': max(1, int(benchmark_cfg.get('k_neighbors', 3))),
        'note': benchmark_cfg.get('note', DEFAULT_BN_SLICE_BENCHMARK_NOTE),
    }


def _bn_family_benchmark_config(cfg: dict | None = None) -> dict:
    resolved_cfg = cfg or {}
    benchmark_cfg = resolved_cfg.get('bn_family_benchmark', {})
    return {
        'enabled': bool(benchmark_cfg.get('enabled', True)),
        'method': benchmark_cfg.get('method', DEFAULT_BN_FAMILY_BENCHMARK_METHOD),
        'grouping_method': benchmark_cfg.get(
            'grouping_method',
            DEFAULT_BN_FAMILY_GROUPING_METHOD,
        ),
        'k_neighbors': max(1, int(benchmark_cfg.get('k_neighbors', 3))),
        'note': benchmark_cfg.get('note', DEFAULT_BN_FAMILY_BENCHMARK_NOTE),
    }


def _bn_stratified_error_config(cfg: dict | None = None) -> dict:
    resolved_cfg = cfg or {}
    benchmark_cfg = resolved_cfg.get('bn_stratified_error', {})
    split_cfg = resolved_cfg.get('split', {})
    robustness_cfg = resolved_cfg.get('robustness', {})
    return {
        'enabled': bool(benchmark_cfg.get('enabled', True)),
        'method': benchmark_cfg.get('method', DEFAULT_BN_STRATIFIED_ERROR_METHOD),
        'group_column': benchmark_cfg.get(
            'group_column',
            robustness_cfg.get('group_column', split_cfg.get('group_column', 'formula')),
        ),
        'n_splits': max(
            2,
            int(benchmark_cfg.get('n_splits', robustness_cfg.get('n_splits', 5))),
        ),
        'note': benchmark_cfg.get('note', DEFAULT_BN_STRATIFIED_ERROR_NOTE),
    }


def _bn_analog_validation_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    validation_cfg = screening_cfg.get('bn_analog_validation', {})
    return {
        'enabled': bool(validation_cfg.get('enabled', True)),
        'method': validation_cfg.get(
            'method',
            'bn_analog_alignment_vote_fraction',
        ),
        'ranking_penalty_enabled': bool(validation_cfg.get('ranking_penalty_enabled', True)),
        'ranking_penalty_weight': max(
            0.0,
            float(validation_cfg.get('ranking_penalty_weight', 0.12)),
        ),
        'note': validation_cfg.get('note', DEFAULT_BN_ANALOG_VALIDATION_NOTE),
    }


def _grouped_robustness_uncertainty_config(cfg: dict | None = None) -> dict:
    screening_cfg = (cfg or {}).get('screening', {})
    robustness_cfg = screening_cfg.get('grouped_robustness_uncertainty', {})
    return {
        'enabled': bool(robustness_cfg.get('enabled', False)),
        'method': robustness_cfg.get(
            'method',
            DEFAULT_GROUPED_ROBUSTNESS_UNCERTAINTY_METHOD,
        ),
        'ranking_penalty_enabled': bool(robustness_cfg.get('ranking_penalty_enabled', True)),
        'ranking_penalty_weight': max(
            0.0,
            float(robustness_cfg.get('ranking_penalty_weight', 0.15)),
        ),
        'note': robustness_cfg.get(
            'note',
            DEFAULT_GROUPED_ROBUSTNESS_UNCERTAINTY_NOTE,
        ),
    }


def _append_grouped_robustness_basis(ranking_basis: str) -> str:
    if ranking_basis.endswith('_penalties'):
        return (
            ranking_basis[: -len('_penalties')]
            + '_and_grouped_robustness_penalties'
        )
    if ranking_basis.endswith('_penalty'):
        return (
            ranking_basis[: -len('_penalty')]
            + '_and_grouped_robustness_penalties'
        )
    return f'{ranking_basis}_minus_grouped_robustness_penalty'


def _append_bn_band_gap_alignment_basis(ranking_basis: str) -> str:
    if ranking_basis.endswith('_penalties'):
        return (
            ranking_basis[: -len('_penalties')]
            + '_and_bn_band_gap_alignment_penalties'
        )
    if ranking_basis.endswith('_penalty'):
        return (
            ranking_basis[: -len('_penalty')]
            + '_and_bn_band_gap_alignment_penalties'
        )
    return f'{ranking_basis}_minus_bn_band_gap_alignment_penalty'


def _append_bn_analog_validation_basis(ranking_basis: str) -> str:
    if ranking_basis.endswith('_penalties'):
        return (
            ranking_basis[: -len('_penalties')]
            + '_and_bn_analog_validation_penalties'
        )
    if ranking_basis.endswith('_penalty'):
        return (
            ranking_basis[: -len('_penalty')]
            + '_and_bn_analog_validation_penalties'
        )
    return f'{ranking_basis}_minus_bn_analog_validation_penalty'


def get_screening_ranking_metadata(
    cfg: dict | None = None,
    domain_support_penalty_applied: bool | None = None,
    bn_support_penalty_applied: bool | None = None,
    grouped_robustness_penalty_applied: bool | None = None,
    bn_band_gap_alignment_penalty_applied: bool | None = None,
    bn_analog_validation_penalty_applied: bool | None = None,
) -> dict[str, object]:
    screening_cfg = (cfg or {}).get('screening', {})
    support_cfg = _domain_support_config(cfg)
    bn_support_cfg = _bn_support_config(cfg)
    grouped_robustness_cfg = _grouped_robustness_uncertainty_config(cfg)
    bn_band_gap_alignment_cfg = _bn_band_gap_alignment_config(cfg)
    bn_analog_validation_cfg = _bn_analog_validation_config(cfg)
    use_model_disagreement = bool(screening_cfg.get('use_model_disagreement', False))
    domain_support_penalty_enabled = bool(
        support_cfg['enabled'] and support_cfg['ranking_penalty_enabled']
    )
    domain_support_penalty_active = bool(
        domain_support_penalty_enabled
        and (
            True
            if domain_support_penalty_applied is None
            else domain_support_penalty_applied
        )
    )
    bn_support_penalty_enabled = bool(
        bn_support_cfg['enabled'] and bn_support_cfg['ranking_penalty_enabled']
    )
    bn_support_penalty_active = bool(
        bn_support_penalty_enabled
        and (
            True
            if bn_support_penalty_applied is None
            else bn_support_penalty_applied
        )
    )

    grouped_robustness_penalty_enabled = bool(
        grouped_robustness_cfg['enabled'] and grouped_robustness_cfg['ranking_penalty_enabled']
    )
    grouped_robustness_penalty_active = bool(
        grouped_robustness_penalty_enabled
        and (
            True
            if grouped_robustness_penalty_applied is None
            else grouped_robustness_penalty_applied
        )
    )

    bn_band_gap_alignment_penalty_enabled = bool(
        bn_band_gap_alignment_cfg['enabled']
        and bn_band_gap_alignment_cfg['ranking_penalty_enabled']
    )
    bn_band_gap_alignment_penalty_active = bool(
        bn_band_gap_alignment_penalty_enabled
        and (
            True
            if bn_band_gap_alignment_penalty_applied is None
            else bn_band_gap_alignment_penalty_applied
        )
    )

    bn_analog_validation_penalty_enabled = bool(
        bn_analog_validation_cfg['enabled'] and bn_analog_validation_cfg['ranking_penalty_enabled']
    )
    bn_analog_validation_penalty_active = bool(
        bn_analog_validation_penalty_enabled
        and (
            True
            if bn_analog_validation_penalty_applied is None
            else bn_analog_validation_penalty_applied
        )
    )

    if use_model_disagreement and domain_support_penalty_active and bn_support_penalty_active:
        ranking_basis = DISAGREEMENT_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_BASIS
        ranking_note = DISAGREEMENT_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_NOTE
    elif use_model_disagreement and domain_support_penalty_active:
        ranking_basis = DISAGREEMENT_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_BASIS
        ranking_note = DISAGREEMENT_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_NOTE
    elif use_model_disagreement and bn_support_penalty_active:
        ranking_basis = DISAGREEMENT_WITH_BN_SUPPORT_PENALTY_RANKING_BASIS
        ranking_note = DISAGREEMENT_WITH_BN_SUPPORT_PENALTY_RANKING_NOTE
    elif use_model_disagreement:
        ranking_basis = DISAGREEMENT_RANKING_BASIS
        ranking_note = DISAGREEMENT_RANKING_NOTE
    elif domain_support_penalty_active and bn_support_penalty_active:
        ranking_basis = SELECTED_MODEL_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_BASIS
        ranking_note = SELECTED_MODEL_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_NOTE
    elif domain_support_penalty_active:
        ranking_basis = SELECTED_MODEL_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_BASIS
        ranking_note = SELECTED_MODEL_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_NOTE
    elif bn_support_penalty_active:
        ranking_basis = SELECTED_MODEL_WITH_BN_SUPPORT_PENALTY_RANKING_BASIS
        ranking_note = SELECTED_MODEL_WITH_BN_SUPPORT_PENALTY_RANKING_NOTE
    else:
        ranking_basis = SELECTED_MODEL_RANKING_BASIS
        ranking_note = SELECTED_MODEL_RANKING_NOTE

    if grouped_robustness_penalty_active:
        ranking_basis = _append_grouped_robustness_basis(ranking_basis)
        ranking_note = f'{ranking_note} {GROUPED_ROBUSTNESS_UNCERTAINTY_RANKING_NOTE}'

    if bn_band_gap_alignment_penalty_active:
        ranking_basis = _append_bn_band_gap_alignment_basis(ranking_basis)
        ranking_note = f'{ranking_note} {BN_BAND_GAP_ALIGNMENT_PENALTY_RANKING_NOTE}'

    if bn_analog_validation_penalty_active:
        ranking_basis = _append_bn_analog_validation_basis(ranking_basis)
        ranking_note = f'{ranking_note} {BN_ANALOG_VALIDATION_RANKING_NOTE}'

    return {
        'ranking_basis': ranking_basis,
        'ranking_note': ranking_note,
        'ranking_uncertainty_method': screening_cfg.get('uncertainty_method', 'disabled'),
        'ranking_uncertainty_penalty': float(screening_cfg.get('uncertainty_penalty', 0.0)),
        'domain_support_enabled': bool(support_cfg['enabled']),
        'domain_support_method': support_cfg['method'],
        'domain_support_distance_metric': support_cfg['distance_metric'],
        'domain_support_reference_split': DOMAIN_SUPPORT_REFERENCE_SPLIT,
        'domain_support_k_neighbors': int(support_cfg['k_neighbors']),
        'domain_support_note': support_cfg['note'],
        'domain_support_penalty_enabled': domain_support_penalty_enabled,
        'domain_support_penalty_active': domain_support_penalty_active,
        'domain_support_penalty_weight': float(support_cfg['ranking_penalty_weight']),
        'domain_support_penalize_below_percentile': float(support_cfg['penalize_below_percentile']),
        'bn_support_enabled': bool(bn_support_cfg['enabled']),
        'bn_support_method': bn_support_cfg['method'],
        'bn_support_distance_metric': bn_support_cfg['distance_metric'],
        'bn_support_reference_split': BN_SUPPORT_REFERENCE_SPLIT,
        'bn_support_k_neighbors': int(bn_support_cfg['k_neighbors']),
        'bn_support_note': bn_support_cfg['note'],
        'bn_support_penalty_enabled': bn_support_penalty_enabled,
        'bn_support_penalty_active': bn_support_penalty_active,
        'bn_support_penalty_weight': float(bn_support_cfg['ranking_penalty_weight']),
        'bn_support_penalize_below_percentile': float(bn_support_cfg['penalize_below_percentile']),
        'grouped_robustness_uncertainty_enabled': bool(grouped_robustness_cfg['enabled']),
        'grouped_robustness_uncertainty_method': grouped_robustness_cfg['method'],
        'grouped_robustness_uncertainty_note': grouped_robustness_cfg['note'],
        'grouped_robustness_penalty_enabled': grouped_robustness_penalty_enabled,
        'grouped_robustness_penalty_active': grouped_robustness_penalty_active,
        'grouped_robustness_penalty_weight': float(
            grouped_robustness_cfg['ranking_penalty_weight']
        ),
        'bn_band_gap_alignment_enabled': bool(bn_band_gap_alignment_cfg['enabled']),
        'bn_band_gap_alignment_method': bn_band_gap_alignment_cfg['method'],
        'bn_band_gap_alignment_reference_split': bn_band_gap_alignment_cfg['reference_split'],
        'bn_band_gap_alignment_window_expansion_iqr_factor': float(
            bn_band_gap_alignment_cfg['window_expansion_iqr_factor']
        ),
        'bn_band_gap_alignment_minimum_neighbor_formula_count_for_penalty': int(
            bn_band_gap_alignment_cfg['minimum_neighbor_formula_count_for_penalty']
        ),
        'bn_band_gap_alignment_note': bn_band_gap_alignment_cfg['note'],
        'bn_band_gap_alignment_penalty_enabled': bn_band_gap_alignment_penalty_enabled,
        'bn_band_gap_alignment_penalty_active': bn_band_gap_alignment_penalty_active,
        'bn_band_gap_alignment_penalty_weight': float(
            bn_band_gap_alignment_cfg['ranking_penalty_weight']
        ),
        'bn_analog_validation_enabled': bool(bn_analog_validation_cfg['enabled']),
        'bn_analog_validation_method': bn_analog_validation_cfg['method'],
        'bn_analog_validation_note': bn_analog_validation_cfg['note'],
        'bn_analog_validation_penalty_enabled': bn_analog_validation_penalty_enabled,
        'bn_analog_validation_penalty_active': bn_analog_validation_penalty_active,
        'bn_analog_validation_penalty_weight': float(
            bn_analog_validation_cfg['ranking_penalty_weight']
        ),
    }


def _format_oxidation_state_value(value: float) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        numeric_value = int(numeric_value)
    if numeric_value > 0:
        return f'+{numeric_value}'
    return str(numeric_value)


def _format_oxidation_state_guess(guess: dict[str, float], element_order: list[str]) -> str:
    ordered_symbols = _ordered_values(element_order + sorted(guess))
    parts = []
    for symbol in ordered_symbols:
        if symbol not in guess:
            continue
        parts.append(f'{symbol}({_format_oxidation_state_value(guess[symbol])})')
    return ', '.join(parts)


def _chemical_plausibility_row(formula: str, method: str, note: str) -> dict[str, object]:
    formula_str = str(formula)
    element_order = _ordered_values(extract_elements(formula_str))
    try:
        composition = Composition(formula_str)
        chemical_system = '-'.join(sorted({element.symbol for element in composition.elements}))
        reduced_formula = composition.reduced_formula
        electronegativity_values = [float(Element(symbol).X) for symbol in element_order if Element(symbol).X is not None]
        electronegativity_spread = (
            max(electronegativity_values) - min(electronegativity_values)
            if electronegativity_values
            else np.nan
        )
        oxidation_state_guesses = composition.oxi_state_guesses()
    except Exception as exc:  # pragma: no cover - candidate formulas are expected to be valid
        return {
            'formula': formula_str,
            'candidate_reduced_formula': formula_str,
            'candidate_chemical_system': '-'.join(sorted(set(element_order))),
            'candidate_n_unique_elements': int(len(set(element_order))),
            'candidate_element_list': '|'.join(element_order),
            'candidate_max_element_electronegativity_delta': np.nan,
            'chemical_plausibility_enabled': True,
            'chemical_plausibility_method': method,
            'chemical_plausibility_pass': False,
            'chemical_plausibility_guess_count': 0,
            'chemical_plausibility_primary_oxidation_state_guess': '',
            'chemical_plausibility_guess_preview': '',
            'chemical_plausibility_note': (
                f'Formula parsing failed before plausibility screening: {type(exc).__name__}: {exc}. {note}'
            ),
        }

    guess_count = len(oxidation_state_guesses)
    primary_guess = (
        _format_oxidation_state_guess(oxidation_state_guesses[0], element_order)
        if guess_count
        else ''
    )
    guess_preview = ' | '.join(
        _format_oxidation_state_guess(guess, element_order)
        for guess in oxidation_state_guesses[:3]
    )
    passes_plausibility = bool(guess_count > 0)
    if passes_plausibility:
        plausibility_note = (
            f'Found {guess_count} charge-balanced oxidation-state guess(es); top guess: {primary_guess}. {note}'
        )
    else:
        plausibility_note = (
            'No charge-balanced common oxidation-state assignment was found by pymatgen for this '
            f'formula. {note}'
        )

    return {
        'formula': formula_str,
        'candidate_reduced_formula': reduced_formula,
        'candidate_chemical_system': chemical_system,
        'candidate_n_unique_elements': int(len(composition.elements)),
        'candidate_element_list': '|'.join(element_order),
        'candidate_max_element_electronegativity_delta': (
            float(electronegativity_spread)
            if not pd.isna(electronegativity_spread)
            else np.nan
        ),
        'chemical_plausibility_enabled': True,
        'chemical_plausibility_method': method,
        'chemical_plausibility_pass': passes_plausibility,
        'chemical_plausibility_guess_count': int(guess_count),
        'chemical_plausibility_primary_oxidation_state_guess': primary_guess,
        'chemical_plausibility_guess_preview': guess_preview,
        'chemical_plausibility_note': plausibility_note,
    }


def annotate_candidate_chemical_plausibility(
    candidate_df: pd.DataFrame,
    cfg: dict | None = None,
    formula_col: str = 'formula',
) -> pd.DataFrame:
    if formula_col not in candidate_df.columns:
        raise KeyError(f'Formula column not found: {formula_col}')

    plausibility_cfg = _chemical_plausibility_config(cfg)
    out = candidate_df.copy()
    if not plausibility_cfg['enabled']:
        out['chemical_plausibility_enabled'] = False
        out['chemical_plausibility_method'] = plausibility_cfg['method']
        out['chemical_plausibility_pass'] = True
        out['chemical_plausibility_guess_count'] = 0
        out['chemical_plausibility_primary_oxidation_state_guess'] = ''
        out['chemical_plausibility_guess_preview'] = ''
        out['chemical_plausibility_note'] = 'Chemical plausibility screening disabled in config.'
        out['candidate_reduced_formula'] = out[formula_col].astype(str)
        out['candidate_chemical_system'] = out[formula_col].astype(str)
        out['candidate_n_unique_elements'] = out[formula_col].astype(str).apply(
            lambda formula: len(set(extract_elements(formula)))
        )
        out['candidate_element_list'] = out[formula_col].astype(str).apply(
            lambda formula: '|'.join(_ordered_values(extract_elements(formula)))
        )
        out['candidate_max_element_electronegativity_delta'] = np.nan
        return out

    annotation_df = pd.DataFrame([
        _chemical_plausibility_row(
            formula=formula,
            method=plausibility_cfg['method'],
            note=plausibility_cfg['note'],
        )
        for formula in out[formula_col].astype(str)
    ])
    preserved_columns = [column for column in out.columns if column not in annotation_df.columns or column == formula_col]
    return out[preserved_columns].merge(annotation_df, on=formula_col, how='left')

