from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure

from runtime.io_utils import make_json_safe
from materials.data import STRUCTURE_SUMMARY_COLUMNS, _structure_summary_from_atoms
from materials.candidate_space import _formula_amount_map, _structure_generation_seed_config
from materials.constants import STRUCTURE_AWARE_FEATURE_SET
from materials.feature_building import build_feature_table
from materials.common import _structure_followup_shortlist_config
from materials.structure_artifacts import (
    _build_structure_generation_first_pass_queue_payload,
    _build_structure_generation_followup_shortlist_df,
    _build_structure_generation_reference_record_payload,
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

