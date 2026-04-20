from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure

from core.io_utils import make_json_safe
from pipeline.data import STRUCTURE_SUMMARY_COLUMNS, _structure_summary_from_atoms
from pipeline.features import (
    STRUCTURE_AWARE_FEATURE_SET,
    _formula_amount_map,
    build_feature_table,
)
from pipeline.reporting import (
    _build_structure_generation_first_pass_queue_payload,
    _build_structure_generation_followup_shortlist_df,
    _build_structure_generation_reference_record_payload,
    _structure_followup_shortlist_config,
    _structure_generation_seed_config,
)


DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_LABEL = 'prototype_first_pass_execution'
DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_METHOD = (
    'deterministic_unrelaxed_reference_reuse_species_edit'
)
DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_NOTE = (
    'Materializes low-complexity structure-follow-up candidates by reusing reference cells and '
    'applying deterministic species relabeling and/or vacancy edits when the reduced formula '
    'scales cleanly to the reference record. This is first-pass prototype execution only. No '
    'ionic/cell relaxation, stability calculation, or discovery claim is made here.'
)
DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_ARTIFACT = (
    'demo_candidate_structure_generation_first_pass_execution.json'
)
DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_SUMMARY_ARTIFACT = (
    'demo_candidate_structure_generation_first_pass_execution_summary.csv'
)
DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_VARIANTS_ARTIFACT = (
    'demo_candidate_structure_generation_first_pass_execution_variants.csv'
)
DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_STRUCTURE_DIR = (
    'demo_candidate_structure_generation_first_pass_structures'
)


def _structure_first_pass_execution_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    execution_cfg = screening_cfg.get('structure_first_pass_execution', {})
    out = {
        'enabled': bool(execution_cfg.get('enabled', True)),
        'label': str(
            execution_cfg.get(
                'label',
                DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_LABEL,
            )
        ),
        'method': str(
            execution_cfg.get(
                'method',
                DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_METHOD,
            )
        ),
        'max_candidates': int(execution_cfg.get('max_candidates', 5)),
        'max_variants_per_candidate': int(execution_cfg.get('max_variants_per_candidate', 3)),
        'geometry_min_distance_ratio_pass_threshold': float(
            execution_cfg.get('geometry_min_distance_ratio_pass_threshold', 0.75)
        ),
        'geometry_min_distance_ratio_overlap_threshold': float(
            execution_cfg.get('geometry_min_distance_ratio_overlap_threshold', 0.6)
        ),
        'note': str(
            execution_cfg.get(
                'note',
                DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_NOTE,
            )
        ),
        'artifact': str(
            execution_cfg.get(
                'artifact',
                DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_ARTIFACT,
            )
        ),
        'summary_artifact': str(
            execution_cfg.get(
                'summary_artifact',
                DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_SUMMARY_ARTIFACT,
            )
        ),
        'variants_artifact': str(
            execution_cfg.get(
                'variants_artifact',
                DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_VARIANTS_ARTIFACT,
            )
        ),
        'structure_dir': str(
            execution_cfg.get(
                'structure_dir',
                DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_STRUCTURE_DIR,
            )
        ),
    }
    if out['max_candidates'] <= 0:
        raise ValueError('structure_first_pass_execution.max_candidates must be positive')
    if out['max_variants_per_candidate'] <= 0:
        raise ValueError('structure_first_pass_execution.max_variants_per_candidate must be positive')
    overlap_threshold = float(out['geometry_min_distance_ratio_overlap_threshold'])
    pass_threshold = float(out['geometry_min_distance_ratio_pass_threshold'])
    if not 0.0 < overlap_threshold <= pass_threshold:
        raise ValueError(
            'structure_first_pass_execution geometry thresholds must satisfy '
            '0 < overlap_threshold <= pass_threshold'
        )
    return out


def _canonical_formula(formula: str | None) -> str | None:
    if formula is None:
        return None
    value = str(formula).strip()
    if not value:
        return None
    return Composition(value).reduced_formula


def _json_safe_value(value):
    return make_json_safe(value)


def _structure_from_atoms(atoms: dict) -> Structure:
    return Structure(
        atoms['lattice_mat'],
        atoms['elements'],
        atoms['coords'],
        coords_are_cartesian=bool(atoms.get('cartesian', False)),
        to_unit_cell=True,
    )


def _structure_to_atoms(structure: Structure, *, cartesian: bool = False) -> dict[str, object]:
    coords = structure.cart_coords if cartesian else structure.frac_coords
    lattice = structure.lattice
    return {
        'elements': [site.specie.symbol for site in structure],
        'coords': coords.tolist(),
        'lattice_mat': lattice.matrix.tolist(),
        'abc': [float(value) for value in lattice.abc],
        'angles': [float(value) for value in lattice.angles],
        'cartesian': bool(cartesian),
    }


def _pair_distance_statistics(
    structure: Structure,
    *,
    overlap_threshold: float,
) -> tuple[float | None, float | None, int, float]:
    if len(structure) <= 1:
        return None, None, 0, 0.0

    distance_matrix = np.asarray(structure.distance_matrix, dtype=float)
    triu_i, triu_j = np.triu_indices(len(structure), k=1)
    distances = distance_matrix[triu_i, triu_j]
    if distances.size == 0:
        return None, None, 0, 0.0

    ratios = []
    overlap_count = 0
    for site_i, site_j, distance in zip(triu_i, triu_j, distances, strict=False):
        distance_value = float(distance)
        if not np.isfinite(distance_value) or distance_value <= 0:
            continue
        elem_i = Element(structure[site_i].specie.symbol)
        elem_j = Element(structure[site_j].specie.symbol)
        radius_i = float(
            elem_i.atomic_radius_calculated or elem_i.atomic_radius or 0.0
        )
        radius_j = float(
            elem_j.atomic_radius_calculated or elem_j.atomic_radius or 0.0
        )
        denom = radius_i + radius_j
        if denom > 0:
            ratio = distance_value / denom
            ratios.append(ratio)
            if ratio < overlap_threshold:
                overlap_count += 1
    min_distance = float(np.min(distances)) if len(distances) else None
    min_distance_ratio = float(np.min(ratios)) if ratios else None
    mean_distance = float(np.mean(distances)) if len(distances) else 0.0
    return min_distance, min_distance_ratio, overlap_count, mean_distance


def _score_site_index_tuple(structure: Structure, indices: tuple[int, ...]) -> tuple[float, float, tuple[int, ...]]:
    if not indices:
        return 0.0, 0.0, tuple()
    if len(indices) == 1:
        return 0.0, 0.0, tuple(indices)
    distance_matrix = np.asarray(structure.distance_matrix, dtype=float)
    pair_distances = [
        float(distance_matrix[i, j])
        for i, j in combinations(indices, 2)
    ]
    return (
        float(min(pair_distances)) if pair_distances else 0.0,
        float(np.mean(pair_distances)) if pair_distances else 0.0,
        tuple(indices),
    )


def _rank_index_combinations(
    structure: Structure,
    candidate_indices: list[int],
    select_count: int,
    *,
    max_variants: int,
) -> list[tuple[int, ...]]:
    if select_count <= 0:
        return [tuple()]
    if select_count > len(candidate_indices):
        return []

    if comb(len(candidate_indices), select_count) <= 128:
        combos = list(combinations(candidate_indices, select_count))
    else:
        ordered = sorted(candidate_indices)
        combos = [tuple(ordered[:select_count])]
        if len(ordered) > select_count:
            combos.append(tuple(ordered[-select_count:]))
        if len(ordered) >= select_count + 1:
            midpoint = len(ordered) // 2
            window = ordered[max(midpoint - select_count // 2, 0):]
            combos.append(tuple(window[:select_count]))
        combos = list(dict.fromkeys(combos))

    ranked = sorted(
        combos,
        key=lambda item: (
            -_score_site_index_tuple(structure, item)[0],
            -_score_site_index_tuple(structure, item)[1],
            item,
        ),
    )
    return ranked[:max_variants]


def _infer_reference_formula_multiplier(atoms: dict, seed_formula: str) -> int | None:
    actual_counts = Counter(str(element) for element in atoms.get('elements', []))
    if not actual_counts:
        return None
    reduced_counts = _formula_amount_map(seed_formula)
    if not reduced_counts:
        return None

    ratios = []
    for element, amount in reduced_counts.items():
        actual = actual_counts.get(element)
        if actual is None:
            return None
        ratio = float(actual) / float(amount)
        ratios.append(ratio)

    if not ratios:
        return None
    reference_ratio = ratios[0]
    if any(abs(ratio - reference_ratio) > 1e-6 for ratio in ratios[1:]):
        return None
    rounded = int(round(reference_ratio))
    if rounded <= 0 or abs(reference_ratio - rounded) > 1e-6:
        return None
    return rounded


def _scaled_formula_counts(formula: str, scale_factor: int) -> dict[str, int] | None:
    counts = _formula_amount_map(formula)
    out = {}
    for element, amount in counts.items():
        scaled = float(amount) * int(scale_factor)
        rounded = int(round(scaled))
        if abs(scaled - rounded) > 1e-6:
            return None
        out[str(element)] = rounded
    return out


def _build_variant_plans(
    structure: Structure,
    current_counts: dict[str, int],
    target_counts: dict[str, int],
    *,
    max_variants: int,
) -> tuple[list[dict[str, object]], str | None]:
    delta_map = {
        element: int(target_counts.get(element, 0) - current_counts.get(element, 0))
        for element in sorted(set(current_counts) | set(target_counts))
        if int(target_counts.get(element, 0) - current_counts.get(element, 0)) != 0
    }
    if not delta_map:
        return [
            {
                'plan_type': 'reference_reuse',
                'relabel_indices': tuple(),
                'relabel_targets': tuple(),
                'remove_indices': tuple(),
                'variant_selection_score': 0.0,
            }
        ], None

    total_site_delta = int(sum(delta_map.values()))
    donor_elements = {element: -delta for element, delta in delta_map.items() if delta < 0}
    recipient_elements = {
        element: delta for element, delta in delta_map.items() if delta > 0
    }

    if total_site_delta > 0:
        return [], 'requires_atom_insertion'

    if len(donor_elements) > 1:
        return [], 'multiple_donor_species_not_supported'

    donor_element = next(iter(donor_elements), None)
    if donor_element is None:
        return [], 'no_donor_species_found'

    donor_indices = [
        index for index, site in enumerate(structure)
        if site.specie.symbol == donor_element
    ]
    donor_surplus = int(donor_elements[donor_element])
    relabel_count = int(sum(recipient_elements.values()))
    remove_count = max(donor_surplus - relabel_count, 0)
    if relabel_count < 0 or remove_count < 0:
        return [], 'invalid_edit_counts'
    if relabel_count + remove_count > len(donor_indices):
        return [], 'insufficient_donor_sites'

    relabel_combos = _rank_index_combinations(
        structure,
        donor_indices,
        relabel_count,
        max_variants=max(max_variants * 2, 1),
    )
    if not relabel_combos:
        relabel_combos = [tuple()]

    recipient_sequence = tuple(
        element
        for element in sorted(recipient_elements)
        for _ in range(int(recipient_elements[element]))
    )
    plans: list[dict[str, object]] = []
    for relabel_indices in relabel_combos:
        remaining_donor_indices = [
            index for index in donor_indices if index not in set(relabel_indices)
        ]
        remove_combos = _rank_index_combinations(
            structure,
            remaining_donor_indices,
            remove_count,
            max_variants=max(max_variants * 2, 1),
        )
        if not remove_combos:
            remove_combos = [tuple()]
        for remove_indices in remove_combos:
            relabel_score = _score_site_index_tuple(structure, tuple(sorted(relabel_indices)))
            remove_score = _score_site_index_tuple(structure, tuple(sorted(remove_indices)))
            plans.append(
                {
                    'plan_type': 'edited_structure',
                    'relabel_indices': tuple(sorted(relabel_indices)),
                    'relabel_targets': recipient_sequence,
                    'remove_indices': tuple(sorted(remove_indices)),
                    'variant_selection_score': float(relabel_score[0] + remove_score[0]),
                }
            )

    unique_plans: list[dict[str, object]] = []
    seen_keys: set[tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]] = set()
    for plan in plans:
        key = (
            tuple(plan['relabel_indices']),
            tuple(plan['relabel_targets']),
            tuple(plan['remove_indices']),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_plans.append(plan)

    ranked_plans = sorted(
        unique_plans,
        key=lambda item: (
            -float(item['variant_selection_score']),
            item['relabel_indices'],
            item['remove_indices'],
        ),
    )
    return ranked_plans[:max_variants], None


def _predict_structure_band_gap_proxy(
    *,
    candidate_formula: str,
    atoms: dict[str, object],
    structure_model,
    structure_feature_columns: list[str] | None,
    structure_feature_set: str | None,
) -> tuple[float | None, str | None]:
    if (
        structure_model is None
        or not structure_feature_columns
        or structure_feature_set != STRUCTURE_AWARE_FEATURE_SET
    ):
        return None, 'no_structure_model'

    summary = _structure_summary_from_atoms(atoms)
    feature_input = pd.DataFrame([
        {
            'formula': str(candidate_formula),
            **summary,
        }
    ])
    feature_df = build_feature_table(
        feature_input,
        formula_col='formula',
        feature_set=structure_feature_set,
    )
    if feature_df['feature_generation_failed'].fillna(False).astype(bool).any():
        error_value = feature_df['feature_generation_error'].iloc[0]
        return None, str(error_value) if error_value is not None else 'feature_generation_failed'
    if any(column not in feature_df.columns for column in structure_feature_columns):
        return None, 'missing_structure_feature_columns'

    try:
        prediction = float(structure_model.predict(feature_df[structure_feature_columns])[0])
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return None, f'{type(exc).__name__}: {exc}'
    return prediction, None


def _apply_variant_plan(
    reference_structure: Structure,
    *,
    relabel_indices: tuple[int, ...],
    relabel_targets: tuple[str, ...],
    remove_indices: tuple[int, ...],
) -> Structure:
    structure = reference_structure.copy()
    for index, element in zip(relabel_indices, relabel_targets, strict=False):
        structure.replace(index, element)
    if remove_indices:
        structure.remove_sites(sorted(remove_indices, reverse=True))
    return structure


def _clean_variant_basename(candidate_formula: str, variant_rank: int) -> str:
    safe_formula = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(candidate_formula))
    safe_formula = safe_formula.strip('_') or 'candidate'
    return f'{safe_formula}__variant_{variant_rank:02d}'


def build_structure_first_pass_execution_artifacts(
    structure_generation_seed_df: pd.DataFrame,
    *,
    cfg: dict,
    formula_col: str = 'formula',
    structure_model=None,
    structure_feature_columns: list[str] | None = None,
    structure_feature_set: str | None = None,
    structure_model_type: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    execution_cfg = _structure_first_pass_execution_config(cfg)
    empty_variant_df = pd.DataFrame()
    empty_summary_df = pd.DataFrame()
    empty_payload = {
        'enabled': bool(execution_cfg['enabled']),
        'label': execution_cfg['label'],
        'method': execution_cfg['method'],
        'note': execution_cfg['note'],
        'artifact': execution_cfg['artifact'],
        'summary_artifact': execution_cfg['summary_artifact'],
        'variants_artifact': execution_cfg['variants_artifact'],
        'structure_dir': execution_cfg['structure_dir'],
        'candidate_count': 0,
        'variant_count': 0,
        'successful_variant_count': 0,
        'status_counts': {},
        'executed_formulas': [],
        'model_feature_set': structure_feature_set,
        'model_type': structure_model_type,
        'model_available': bool(
            structure_model is not None
            and structure_feature_columns
            and structure_feature_set == STRUCTURE_AWARE_FEATURE_SET
        ),
        'candidates': [],
    }
    if (
        not bool(execution_cfg['enabled'])
        or structure_generation_seed_df is None
        or structure_generation_seed_df.empty
    ):
        return empty_variant_df, empty_summary_df, empty_payload

    seed_cfg = _structure_generation_seed_config(cfg)
    followup_cfg = _structure_followup_shortlist_config(cfg)
    reference_payload = _build_structure_generation_reference_record_payload(
        structure_generation_seed_df,
        cfg=cfg,
    )
    queue_payload = _build_structure_generation_first_pass_queue_payload(
        structure_generation_seed_df,
        formula_col=formula_col,
        cfg_defaults=seed_cfg,
    )
    followup_df = _build_structure_generation_followup_shortlist_df(
        queue_payload,
        formula_col=formula_col,
        cfg_defaults=followup_cfg,
    )
    if followup_df.empty:
        return empty_variant_df, empty_summary_df, empty_payload

    selected_followup_df = followup_df.loc[
        followup_df['structure_followup_shortlist_selected'].fillna(False).astype(bool)
    ].copy()
    if selected_followup_df.empty:
        return empty_variant_df, empty_summary_df, empty_payload

    selected_followup_df = selected_followup_df.sort_values(
        'structure_followup_shortlist_rank',
        ascending=True,
        kind='stable',
    ).head(int(execution_cfg['max_candidates']))
    queue_rows = pd.DataFrame(queue_payload.get('queue', [])).copy()
    reference_records = {
        str(record['record_id']): record
        for record in reference_payload.get('reference_records', [])
        if isinstance(record, dict) and record.get('record_id') is not None
    }

    variant_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    payload_candidates: list[dict[str, object]] = []

    for _, shortlist_row in selected_followup_df.iterrows():
        candidate_formula = str(shortlist_row[formula_col])
        candidate_queue_df = queue_rows.loc[
            queue_rows['candidate_formula'].astype(str).eq(candidate_formula)
        ].copy() if not queue_rows.empty else pd.DataFrame()
        if not candidate_queue_df.empty and 'queue_rank' in candidate_queue_df.columns:
            candidate_queue_df = candidate_queue_df.sort_values(
                ['queue_rank', 'edit_complexity_score', 'job_id'],
                ascending=[True, True, True],
                kind='stable',
            )

        best_queue_rank = shortlist_row.get('structure_followup_best_queue_rank')
        if best_queue_rank is not None and not pd.isna(best_queue_rank) and not candidate_queue_df.empty:
            matching_best = candidate_queue_df.loc[
                candidate_queue_df['queue_rank'].astype(float).eq(float(best_queue_rank))
            ]
            best_queue_row = matching_best.iloc[0] if not matching_best.empty else candidate_queue_df.iloc[0]
        elif not candidate_queue_df.empty:
            best_queue_row = candidate_queue_df.iloc[0]
        else:
            best_queue_row = pd.Series(dtype='object')

        record_id = str(
            _json_safe_value(
                shortlist_row.get('structure_followup_best_seed_reference_record_id')
                or best_queue_row.get('seed_reference_record_id')
                or ''
            )
            or ''
        )
        reference_record = reference_records.get(record_id)
        seed_formula = str(
            _json_safe_value(
                shortlist_row.get('structure_followup_best_seed_reference_formula')
                or best_queue_row.get('seed_reference_formula')
                or ''
            )
            or ''
        )
        action_label = str(
            _json_safe_value(
                shortlist_row.get('structure_followup_best_action_label')
                or best_queue_row.get('job_action_label')
                or 'unavailable'
            )
        )

        candidate_payload = {
            formula_col: candidate_formula,
            'structure_followup_shortlist_rank': _json_safe_value(
                shortlist_row.get('structure_followup_shortlist_rank')
            ),
            'seed_reference_formula': seed_formula or None,
            'seed_reference_record_id': record_id or None,
            'job_action_label': action_label,
            'variants': [],
        }

        if reference_record is None or not isinstance(reference_record.get('atoms'), dict):
            summary_rows.append(
                {
                    formula_col: candidate_formula,
                    'structure_followup_shortlist_rank': _json_safe_value(
                        shortlist_row.get('structure_followup_shortlist_rank')
                    ),
                    'structure_followup_best_action_label': action_label,
                    'structure_followup_best_seed_reference_formula': seed_formula or None,
                    'structure_followup_best_seed_reference_record_id': record_id or None,
                    'first_pass_execution_variant_count': 0,
                    'first_pass_execution_successful_variant_count': 0,
                    'first_pass_execution_geometry_pass_variant_count': 0,
                    'first_pass_execution_status': 'missing_reference_record',
                    'first_pass_execution_selected_variant_id': None,
                    'first_pass_execution_selected_variant_rank': None,
                    'first_pass_execution_selected_cif_path': None,
                    'first_pass_execution_selected_generated_formula': None,
                    'first_pass_execution_selected_structure_n_sites': None,
                    'first_pass_execution_selected_min_distance': None,
                    'first_pass_execution_selected_min_distance_ratio': None,
                    'first_pass_execution_selected_band_gap_proxy': None,
                    'first_pass_execution_selected_relaxation_status': None,
                    'first_pass_execution_selected_final_status': 'not_executed',
                }
            )
            candidate_payload['candidate_status'] = 'missing_reference_record'
            payload_candidates.append(candidate_payload)
            continue

        reference_atoms = reference_record['atoms']
        try:
            reference_structure = _structure_from_atoms(reference_atoms)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            summary_rows.append(
                {
                    formula_col: candidate_formula,
                    'structure_followup_shortlist_rank': _json_safe_value(
                        shortlist_row.get('structure_followup_shortlist_rank')
                    ),
                    'structure_followup_best_action_label': action_label,
                    'structure_followup_best_seed_reference_formula': seed_formula or None,
                    'structure_followup_best_seed_reference_record_id': record_id or None,
                    'first_pass_execution_variant_count': 0,
                    'first_pass_execution_successful_variant_count': 0,
                    'first_pass_execution_geometry_pass_variant_count': 0,
                    'first_pass_execution_status': 'invalid_reference_structure',
                    'first_pass_execution_selected_variant_id': None,
                    'first_pass_execution_selected_variant_rank': None,
                    'first_pass_execution_selected_cif_path': None,
                    'first_pass_execution_selected_generated_formula': None,
                    'first_pass_execution_selected_structure_n_sites': None,
                    'first_pass_execution_selected_min_distance': None,
                    'first_pass_execution_selected_min_distance_ratio': None,
                    'first_pass_execution_selected_band_gap_proxy': None,
                    'first_pass_execution_selected_relaxation_status': None,
                    'first_pass_execution_selected_final_status': f'{type(exc).__name__}: {exc}',
                }
            )
            candidate_payload['candidate_status'] = 'invalid_reference_structure'
            payload_candidates.append(candidate_payload)
            continue

        scale_factor = _infer_reference_formula_multiplier(reference_atoms, seed_formula)
        if scale_factor is None:
            summary_rows.append(
                {
                    formula_col: candidate_formula,
                    'structure_followup_shortlist_rank': _json_safe_value(
                        shortlist_row.get('structure_followup_shortlist_rank')
                    ),
                    'structure_followup_best_action_label': action_label,
                    'structure_followup_best_seed_reference_formula': seed_formula or None,
                    'structure_followup_best_seed_reference_record_id': record_id or None,
                    'first_pass_execution_variant_count': 0,
                    'first_pass_execution_successful_variant_count': 0,
                    'first_pass_execution_geometry_pass_variant_count': 0,
                    'first_pass_execution_status': 'unresolved_reference_scale_factor',
                    'first_pass_execution_selected_variant_id': None,
                    'first_pass_execution_selected_variant_rank': None,
                    'first_pass_execution_selected_cif_path': None,
                    'first_pass_execution_selected_generated_formula': None,
                    'first_pass_execution_selected_structure_n_sites': None,
                    'first_pass_execution_selected_min_distance': None,
                    'first_pass_execution_selected_min_distance_ratio': None,
                    'first_pass_execution_selected_band_gap_proxy': None,
                    'first_pass_execution_selected_relaxation_status': None,
                    'first_pass_execution_selected_final_status': 'not_executed',
                }
            )
            candidate_payload['candidate_status'] = 'unresolved_reference_scale_factor'
            payload_candidates.append(candidate_payload)
            continue

        target_counts = _scaled_formula_counts(candidate_formula, scale_factor)
        if target_counts is None:
            summary_rows.append(
                {
                    formula_col: candidate_formula,
                    'structure_followup_shortlist_rank': _json_safe_value(
                        shortlist_row.get('structure_followup_shortlist_rank')
                    ),
                    'structure_followup_best_action_label': action_label,
                    'structure_followup_best_seed_reference_formula': seed_formula or None,
                    'structure_followup_best_seed_reference_record_id': record_id or None,
                    'first_pass_execution_variant_count': 0,
                    'first_pass_execution_successful_variant_count': 0,
                    'first_pass_execution_geometry_pass_variant_count': 0,
                    'first_pass_execution_status': 'candidate_formula_does_not_scale_to_reference_cell',
                    'first_pass_execution_selected_variant_id': None,
                    'first_pass_execution_selected_variant_rank': None,
                    'first_pass_execution_selected_cif_path': None,
                    'first_pass_execution_selected_generated_formula': None,
                    'first_pass_execution_selected_structure_n_sites': None,
                    'first_pass_execution_selected_min_distance': None,
                    'first_pass_execution_selected_min_distance_ratio': None,
                    'first_pass_execution_selected_band_gap_proxy': None,
                    'first_pass_execution_selected_relaxation_status': None,
                    'first_pass_execution_selected_final_status': 'not_executed',
                }
            )
            candidate_payload['candidate_status'] = 'candidate_formula_does_not_scale_to_reference_cell'
            payload_candidates.append(candidate_payload)
            continue

        current_counts = Counter(site.specie.symbol for site in reference_structure)
        variant_plans, plan_error = _build_variant_plans(
            reference_structure,
            current_counts,
            target_counts,
            max_variants=int(execution_cfg['max_variants_per_candidate']),
        )
        if plan_error or not variant_plans:
            summary_rows.append(
                {
                    formula_col: candidate_formula,
                    'structure_followup_shortlist_rank': _json_safe_value(
                        shortlist_row.get('structure_followup_shortlist_rank')
                    ),
                    'structure_followup_best_action_label': action_label,
                    'structure_followup_best_seed_reference_formula': seed_formula or None,
                    'structure_followup_best_seed_reference_record_id': record_id or None,
                    'first_pass_execution_variant_count': 0,
                    'first_pass_execution_successful_variant_count': 0,
                    'first_pass_execution_geometry_pass_variant_count': 0,
                    'first_pass_execution_status': str(plan_error or 'no_variant_plan_generated'),
                    'first_pass_execution_selected_variant_id': None,
                    'first_pass_execution_selected_variant_rank': None,
                    'first_pass_execution_selected_cif_path': None,
                    'first_pass_execution_selected_generated_formula': None,
                    'first_pass_execution_selected_structure_n_sites': None,
                    'first_pass_execution_selected_min_distance': None,
                    'first_pass_execution_selected_min_distance_ratio': None,
                    'first_pass_execution_selected_band_gap_proxy': None,
                    'first_pass_execution_selected_relaxation_status': None,
                    'first_pass_execution_selected_final_status': 'not_executed',
                }
            )
            candidate_payload['candidate_status'] = str(plan_error or 'no_variant_plan_generated')
            payload_candidates.append(candidate_payload)
            continue

        for variant_rank, plan in enumerate(variant_plans, start=1):
            variant_id = _clean_variant_basename(candidate_formula, variant_rank)
            cif_path = f"{execution_cfg['structure_dir']}/{variant_id}.cif"
            try:
                variant_structure = _apply_variant_plan(
                    reference_structure,
                    relabel_indices=tuple(plan['relabel_indices']),
                    relabel_targets=tuple(plan['relabel_targets']),
                    remove_indices=tuple(plan['remove_indices']),
                )
                variant_atoms = _structure_to_atoms(variant_structure)
                generated_formula = _canonical_formula(variant_structure.composition.reduced_formula)
                formula_matches_candidate = generated_formula == _canonical_formula(candidate_formula)
                min_distance, min_distance_ratio, overlap_pair_count, mean_distance = (
                    _pair_distance_statistics(
                        variant_structure,
                        overlap_threshold=float(
                            execution_cfg['geometry_min_distance_ratio_overlap_threshold']
                        ),
                    )
                )
                geometry_sanity_pass = bool(
                    overlap_pair_count == 0
                    and (
                        min_distance_ratio is None
                        or min_distance_ratio
                        >= float(execution_cfg['geometry_min_distance_ratio_pass_threshold'])
                    )
                )
                structure_summary = _structure_summary_from_atoms(variant_atoms)
                predicted_band_gap_proxy, proxy_error = _predict_structure_band_gap_proxy(
                    candidate_formula=candidate_formula,
                    atoms=variant_atoms,
                    structure_model=structure_model,
                    structure_feature_columns=structure_feature_columns,
                    structure_feature_set=structure_feature_set,
                )
                relaxation_status = (
                    'not_run_reference_geometry_reused'
                    if not plan['relabel_indices'] and not plan['remove_indices']
                    else 'not_run_unrelaxed_species_edit'
                )
                if not formula_matches_candidate:
                    final_status = 'formula_mismatch_after_edit'
                elif not geometry_sanity_pass:
                    final_status = 'geometry_sanity_failed'
                elif relaxation_status == 'not_run_reference_geometry_reused':
                    final_status = 'reference_control_ready'
                else:
                    final_status = 'ready_for_external_relaxation'
                execution_status = 'ok'
                cif_text = variant_structure.to(fmt='cif')
                error_message = None
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                variant_atoms = None
                generated_formula = None
                formula_matches_candidate = False
                min_distance = None
                min_distance_ratio = None
                overlap_pair_count = 0
                mean_distance = 0.0
                geometry_sanity_pass = False
                structure_summary = {column: None for column in STRUCTURE_SUMMARY_COLUMNS}
                predicted_band_gap_proxy = None
                proxy_error = None
                relaxation_status = 'not_run_due_to_execution_error'
                final_status = 'execution_error'
                execution_status = 'error'
                cif_text = None
                error_message = f'{type(exc).__name__}: {exc}'

            variant_row = {
                formula_col: candidate_formula,
                'structure_followup_shortlist_rank': _json_safe_value(
                    shortlist_row.get('structure_followup_shortlist_rank')
                ),
                'queue_rank': _json_safe_value(best_queue_row.get('queue_rank')),
                'seed_reference_formula': seed_formula or None,
                'seed_reference_record_id': record_id or None,
                'job_action_label': action_label,
                'execution_variant_id': variant_id,
                'execution_variant_rank': variant_rank,
                'execution_status': execution_status,
                'execution_message': error_message,
                'execution_plan_type': plan['plan_type'],
                'execution_variant_selection_score': float(plan['variant_selection_score']),
                'relabel_site_indices': '|'.join(str(index) for index in plan['relabel_indices']),
                'relabel_target_elements': '|'.join(plan['relabel_targets']),
                'removed_site_indices': '|'.join(str(index) for index in plan['remove_indices']),
                'relabeled_site_count': int(len(plan['relabel_indices'])),
                'removed_site_count': int(len(plan['remove_indices'])),
                'generated_formula': generated_formula,
                'formula_matches_candidate': bool(formula_matches_candidate),
                'generated_structure_n_sites': _json_safe_value(structure_summary['structure_n_sites']),
                'geometry_min_distance': min_distance,
                'geometry_mean_distance': mean_distance,
                'geometry_min_distance_ratio': min_distance_ratio,
                'geometry_overlap_pair_count': int(overlap_pair_count),
                'geometry_sanity_pass': bool(geometry_sanity_pass),
                'structure_band_gap_proxy': predicted_band_gap_proxy,
                'structure_band_gap_proxy_error': proxy_error,
                'structure_band_gap_proxy_feature_set': structure_feature_set,
                'structure_band_gap_proxy_model_type': structure_model_type,
                'relaxation_status': relaxation_status,
                'final_status': final_status,
                'generated_structure_cif_path': cif_path,
                **structure_summary,
            }
            variant_rows.append(variant_row)
            candidate_payload['variants'].append(
                {
                    **variant_row,
                    'atoms': variant_atoms,
                    '_cif_text': cif_text,
                }
            )

        candidate_variant_df = pd.DataFrame(candidate_payload['variants'])
        successful_variant_df = candidate_variant_df.loc[
            candidate_variant_df['execution_status'].astype(str).eq('ok')
        ].copy() if not candidate_variant_df.empty else pd.DataFrame()
        geometry_pass_variant_df = successful_variant_df.loc[
            successful_variant_df['geometry_sanity_pass'].fillna(False).astype(bool)
        ].copy() if not successful_variant_df.empty else pd.DataFrame()

        selected_variant = None
        if not successful_variant_df.empty:
            ranked_successful = successful_variant_df.sort_values(
                [
                    'geometry_sanity_pass',
                    'formula_matches_candidate',
                    'structure_band_gap_proxy',
                    'execution_variant_selection_score',
                    'execution_variant_rank',
                ],
                ascending=[False, False, False, False, True],
                kind='stable',
                na_position='last',
            )
            selected_variant = ranked_successful.iloc[0]

        candidate_status = 'executed' if selected_variant is not None else 'no_successful_variant'
        summary_rows.append(
            {
                formula_col: candidate_formula,
                'structure_followup_shortlist_rank': _json_safe_value(
                    shortlist_row.get('structure_followup_shortlist_rank')
                ),
                'structure_followup_best_action_label': action_label,
                'structure_followup_best_seed_reference_formula': seed_formula or None,
                'structure_followup_best_seed_reference_record_id': record_id or None,
                'first_pass_execution_variant_count': int(len(candidate_variant_df)),
                'first_pass_execution_successful_variant_count': int(len(successful_variant_df)),
                'first_pass_execution_geometry_pass_variant_count': int(len(geometry_pass_variant_df)),
                'first_pass_execution_status': candidate_status,
                'first_pass_execution_selected_variant_id': (
                    _json_safe_value(selected_variant.get('execution_variant_id'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_variant_rank': (
                    _json_safe_value(selected_variant.get('execution_variant_rank'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_cif_path': (
                    _json_safe_value(selected_variant.get('generated_structure_cif_path'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_generated_formula': (
                    _json_safe_value(selected_variant.get('generated_formula'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_structure_n_sites': (
                    _json_safe_value(selected_variant.get('generated_structure_n_sites'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_min_distance': (
                    _json_safe_value(selected_variant.get('geometry_min_distance'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_min_distance_ratio': (
                    _json_safe_value(selected_variant.get('geometry_min_distance_ratio'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_band_gap_proxy': (
                    _json_safe_value(selected_variant.get('structure_band_gap_proxy'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_relaxation_status': (
                    _json_safe_value(selected_variant.get('relaxation_status'))
                    if selected_variant is not None
                    else None
                ),
                'first_pass_execution_selected_final_status': (
                    _json_safe_value(selected_variant.get('final_status'))
                    if selected_variant is not None
                    else 'not_executed'
                ),
            }
        )
        candidate_payload['candidate_status'] = candidate_status
        candidate_payload['selected_variant_id'] = (
            _json_safe_value(selected_variant.get('execution_variant_id'))
            if selected_variant is not None
            else None
        )
        payload_candidates.append(candidate_payload)

    variant_df = pd.DataFrame(variant_rows)
    summary_df = pd.DataFrame(summary_rows)
    status_counts = (
        summary_df['first_pass_execution_status'].astype(str).value_counts().to_dict()
        if not summary_df.empty
        else {}
    )
    successful_variant_count = int(
        variant_df['execution_status'].astype(str).eq('ok').sum()
    ) if not variant_df.empty else 0

    payload = {
        'enabled': bool(execution_cfg['enabled']),
        'label': execution_cfg['label'],
        'method': execution_cfg['method'],
        'note': execution_cfg['note'],
        'artifact': execution_cfg['artifact'],
        'summary_artifact': execution_cfg['summary_artifact'],
        'variants_artifact': execution_cfg['variants_artifact'],
        'structure_dir': execution_cfg['structure_dir'],
        'candidate_count': int(len(summary_df)),
        'variant_count': int(len(variant_df)),
        'successful_variant_count': successful_variant_count,
        'status_counts': {str(key): int(value) for key, value in status_counts.items()},
        'executed_formulas': summary_df.loc[
            summary_df['first_pass_execution_status'].astype(str).eq('executed'),
            formula_col,
        ].astype(str).tolist() if not summary_df.empty else [],
        'model_feature_set': structure_feature_set,
        'model_type': structure_model_type,
        'model_available': bool(
            structure_model is not None
            and structure_feature_columns
            and structure_feature_set == STRUCTURE_AWARE_FEATURE_SET
        ),
        'candidates': payload_candidates,
    }
    return variant_df, summary_df, payload
