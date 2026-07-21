from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure

from runtime.io_utils import make_json_safe
from runtime.schema import STRUCTURE_EXECUTION_OUTPUT_ROLES
from materials.data import STRUCTURE_SUMMARY_COLUMNS, _structure_summary_from_atoms
from materials.candidate_space import _formula_amount_map, _structure_generation_seed_config
from materials.constants import STRUCTURE_AWARE_FEATURE_SET
from materials.feature_building import build_feature_table
from materials.common import _artifact_relative_path, _structure_followup_shortlist_config
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
_STRUCTURE_FIRST_PASS_EXECUTION_OUTPUT_DEFAULTS = {
    config_field: default_path
    for _artifact_key, _summary_field, config_field, _suffix, default_path
    in STRUCTURE_EXECUTION_OUTPUT_ROLES
}
DEFAULT_STRUCTURE_FIRST_PASS_EXECUTION_STRUCTURE_DIR = (
    'demo_candidate_structure_generation_first_pass_structures'
)
_STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS = (
    ('first_pass_execution_selected_variant_id', 'execution_variant_id'),
    ('first_pass_execution_selected_variant_rank', 'execution_variant_rank'),
    ('first_pass_execution_selected_cif_path', 'generated_structure_cif_path'),
    ('first_pass_execution_selected_generated_formula', 'generated_formula'),
    (
        'first_pass_execution_selected_structure_n_sites',
        'generated_structure_n_sites',
    ),
    ('first_pass_execution_selected_min_distance', 'geometry_min_distance'),
    (
        'first_pass_execution_selected_min_distance_ratio',
        'geometry_min_distance_ratio',
    ),
    ('first_pass_execution_selected_band_gap_proxy', 'structure_band_gap_proxy'),
    ('first_pass_execution_selected_relaxation_status', 'relaxation_status'),
    ('first_pass_execution_selected_final_status', 'final_status'),
)
_STRUCTURE_EXECUTION_VARIANT_SELECTION_FIELDS = (
    'geometry_sanity_pass',
    'formula_matches_candidate',
    'structure_band_gap_proxy',
    'execution_variant_selection_score',
    'execution_variant_rank',
)
_STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH = {
    'missing_reference': 'missing_reference_record',
    'invalid_reference': 'invalid_reference_structure',
    'unresolved_reference_scale': 'unresolved_reference_scale_factor',
    'unscalable_candidate_formula': (
        'candidate_formula_does_not_scale_to_reference_cell'
    ),
    'requires_atom_insertion': 'requires_atom_insertion',
    'multiple_donor_species': 'multiple_donor_species_not_supported',
    'no_donor_species': 'no_donor_species_found',
    'invalid_edit_counts': 'invalid_edit_counts',
    'insufficient_donor_sites': 'insufficient_donor_sites',
    'no_variant_plan': 'no_variant_plan_generated',
    'executed': 'executed',
    'no_successful_variant': 'no_successful_variant',
}
_STRUCTURE_EXECUTION_ZERO_VARIANT_STATUSES = frozenset(
    status
    for branch, status in _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH.items()
    if branch not in {'executed', 'no_successful_variant'}
)
_STRUCTURE_EXECUTION_VARIANT_STATUS_BY_BRANCH = {
    'execution_ok': 'ok',
    'execution_error': 'error',
    'relaxation_reference_geometry_reused': 'not_run_reference_geometry_reused',
    'relaxation_unrelaxed_species_edit': 'not_run_unrelaxed_species_edit',
    'relaxation_execution_error': 'not_run_due_to_execution_error',
    'final_formula_mismatch': 'formula_mismatch_after_edit',
    'final_geometry_failure': 'geometry_sanity_failed',
    'final_reference_control': 'reference_control_ready',
    'final_external_relaxation': 'ready_for_external_relaxation',
    'final_execution_error': 'execution_error',
}


def _select_structure_execution_variant(
    variant_df: pd.DataFrame,
) -> pd.Series | None:
    if variant_df.empty:
        return None
    successful_variant_df = variant_df.loc[
        variant_df['execution_status'].astype(str).eq('ok')
    ].copy()
    if successful_variant_df.empty:
        return None
    return successful_variant_df.sort_values(
        list(_STRUCTURE_EXECUTION_VARIANT_SELECTION_FIELDS),
        ascending=[False, False, False, False, True],
        kind='stable',
        na_position='last',
    ).iloc[0]


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
                _STRUCTURE_FIRST_PASS_EXECUTION_OUTPUT_DEFAULTS['artifact'],
            )
        ),
        'summary_artifact': str(
            execution_cfg.get(
                'summary_artifact',
                _STRUCTURE_FIRST_PASS_EXECUTION_OUTPUT_DEFAULTS['summary_artifact'],
            )
        ),
        'variants_artifact': str(
            execution_cfg.get(
                'variants_artifact',
                _STRUCTURE_FIRST_PASS_EXECUTION_OUTPUT_DEFAULTS['variants_artifact'],
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
    for artifact_field in (
        'artifact',
        'summary_artifact',
        'variants_artifact',
        'structure_dir',
    ):
        out[artifact_field] = _artifact_relative_path(
            out[artifact_field],
            field_name=f'structure_first_pass_execution.{artifact_field}',
        )
    return out


def _canonical_formula(formula: str | None) -> str | None:
    if formula is None:
        return None
    value = str(formula).strip()
    if not value:
        return None
    return Composition(value).reduced_formula


def _validate_structure_execution_variant_identity(
    *,
    atoms,
    cif_text,
    generated_formula,
    generated_structure_n_sites,
    structure_summary,
    geometry_min_distance,
    geometry_mean_distance,
    geometry_min_distance_ratio,
    geometry_overlap_pair_count,
    geometry_min_distance_ratio_overlap_threshold,
) -> None:
    """Validate one builder-produced structure across atoms, metadata, and CIF."""

    def reject(detail):
        raise ValueError(
            f'structure_first_pass_execution variant structure identity {detail}'
        )

    def finite_vector(value, field_name):
        try:
            vector = np.asarray(value, dtype=float)
        except Exception:
            reject(f'{field_name} must be a finite three-vector')
        if vector.shape != (3,) or not np.isfinite(vector).all():
            reject(f'{field_name} must be a finite three-vector')
        return vector

    def same_optional_number(actual, expected, field_name, *, tolerance=1e-8):
        actual = make_json_safe(actual)
        expected = make_json_safe(expected)
        if actual is None or expected is None:
            if actual is not expected:
                reject(f'{field_name} disagrees with atoms evidence')
            return
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not np.isfinite(actual)
            or not np.isclose(actual, expected, rtol=tolerance, atol=tolerance)
        ):
            reject(f'{field_name} disagrees with atoms evidence')

    if not isinstance(atoms, dict):
        reject('requires an atoms object')
    if not isinstance(atoms.get('cartesian'), bool):
        reject('atoms.cartesian must be boolean')
    if not isinstance(structure_summary, dict):
        reject('requires structure summary metadata')
    missing_summary_fields = set(STRUCTURE_SUMMARY_COLUMNS) - set(structure_summary)
    if missing_summary_fields:
        reject(
            'is missing structure summary fields: '
            f'{sorted(missing_summary_fields)}'
        )

    try:
        atoms_structure = _structure_from_atoms(atoms)
    except Exception as exc:
        reject(f'atoms cannot construct a structure: {type(exc).__name__}')
    if not len(atoms_structure):
        reject('atoms must contain at least one site')

    declared_abc = finite_vector(atoms.get('abc'), 'atoms.abc')
    declared_angles = finite_vector(atoms.get('angles'), 'atoms.angles')
    if not np.allclose(
        declared_abc,
        np.asarray(atoms_structure.lattice.abc),
        rtol=1e-8,
        atol=1e-8,
    ):
        reject('atoms.abc disagrees with atoms.lattice_mat')
    if not np.allclose(
        declared_angles,
        np.asarray(atoms_structure.lattice.angles),
        rtol=1e-8,
        atol=1e-8,
    ):
        reject('atoms.angles disagrees with atoms.lattice_mat')

    atoms_formula = _canonical_formula(atoms_structure.composition.reduced_formula)
    if _canonical_formula(generated_formula) != atoms_formula:
        reject('generated_formula disagrees with atoms evidence')
    if int(generated_structure_n_sites) != len(atoms_structure):
        reject('generated_structure_n_sites disagrees with atoms evidence')

    canonical_atoms = _structure_to_atoms(atoms_structure)
    expected_summary = _structure_summary_from_atoms(canonical_atoms)
    for field_name in STRUCTURE_SUMMARY_COLUMNS:
        same_optional_number(
            structure_summary.get(field_name),
            expected_summary.get(field_name),
            field_name,
        )

    try:
        overlap_threshold = float(
            make_json_safe(geometry_min_distance_ratio_overlap_threshold)
        )
    except (TypeError, ValueError):
        reject('geometry overlap threshold must be numeric')
    if not np.isfinite(overlap_threshold) or overlap_threshold <= 0:
        reject('geometry overlap threshold must be positive and finite')
    (
        expected_min_distance,
        expected_min_distance_ratio,
        expected_overlap_pair_count,
        expected_mean_distance,
    ) = _pair_distance_statistics(
        atoms_structure,
        overlap_threshold=overlap_threshold,
    )
    same_optional_number(
        geometry_min_distance,
        expected_min_distance,
        'geometry_min_distance',
    )
    same_optional_number(
        geometry_mean_distance,
        expected_mean_distance,
        'geometry_mean_distance',
    )
    same_optional_number(
        geometry_min_distance_ratio,
        expected_min_distance_ratio,
        'geometry_min_distance_ratio',
    )
    if int(geometry_overlap_pair_count) != expected_overlap_pair_count:
        reject('geometry_overlap_pair_count disagrees with atoms evidence')

    try:
        cif_structure = Structure.from_str(cif_text, fmt='cif')
    except Exception as exc:
        reject(f'CIF bytes cannot construct a structure: {type(exc).__name__}')
    if len(cif_structure) != len(atoms_structure):
        reject('CIF site count disagrees with atoms evidence')
    if not np.allclose(
        np.asarray(cif_structure.lattice.abc),
        np.asarray(atoms_structure.lattice.abc),
        rtol=1e-6,
        atol=1e-5,
    ) or not np.allclose(
        np.asarray(cif_structure.lattice.angles),
        np.asarray(atoms_structure.lattice.angles),
        rtol=1e-6,
        atol=1e-5,
    ):
        reject('CIF lattice disagrees with atoms evidence')

    unmatched_cif_sites = list(range(len(cif_structure)))
    for atoms_site in atoms_structure:
        matched_index = None
        for cif_index in unmatched_cif_sites:
            cif_site = cif_structure[cif_index]
            if atoms_site.species_string != cif_site.species_string:
                continue
            coordinate_delta = np.asarray(
                atoms_site.frac_coords - cif_site.frac_coords,
                dtype=float,
            )
            coordinate_delta -= np.round(coordinate_delta)
            if np.allclose(
                coordinate_delta,
                np.zeros(3),
                rtol=0.0,
                atol=1e-5,
            ):
                matched_index = cif_index
                break
        if matched_index is None:
            reject('CIF species/coordinates disagree with atoms evidence')
        unmatched_cif_sites.remove(matched_index)


def _structure_execution_variant_expected_state(
    *,
    candidate_formula,
    execution_status,
    execution_message,
    atoms,
    generated_formula,
    formula_matches_candidate,
    geometry_min_distance,
    geometry_mean_distance,
    geometry_min_distance_ratio,
    geometry_overlap_pair_count,
    geometry_sanity_pass,
    geometry_min_distance_ratio_pass_threshold,
    geometry_min_distance_ratio_overlap_threshold,
    relabeled_site_count,
    removed_site_count,
    generated_structure_n_sites,
    structure_summary,
    cif_text,
) -> tuple[str, str]:
    """Return the builder-owned relaxation/final state for one variant."""

    def reject(detail):
        raise ValueError(
            f'structure_first_pass_execution variant state {detail}'
        )

    def nonnegative_integer(value, field_name):
        value = make_json_safe(value)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or float(value) != int(value)
            or value < 0
        ):
            reject(f'{field_name} must be a non-negative integer')
        return int(value)

    statuses = _STRUCTURE_EXECUTION_VARIANT_STATUS_BY_BRANCH
    execution_status = make_json_safe(execution_status)
    execution_message = make_json_safe(execution_message)
    generated_formula = make_json_safe(generated_formula)
    formula_matches_candidate = make_json_safe(formula_matches_candidate)
    geometry_min_distance = make_json_safe(geometry_min_distance)
    geometry_mean_distance = make_json_safe(geometry_mean_distance)
    geometry_min_distance_ratio = make_json_safe(geometry_min_distance_ratio)
    geometry_sanity_pass = make_json_safe(geometry_sanity_pass)
    generated_structure_n_sites = make_json_safe(generated_structure_n_sites)
    geometry_overlap_pair_count = nonnegative_integer(
        geometry_overlap_pair_count,
        'geometry_overlap_pair_count',
    )
    geometry_min_distance_ratio_pass_threshold = make_json_safe(
        geometry_min_distance_ratio_pass_threshold
    )
    if (
        isinstance(geometry_min_distance_ratio_pass_threshold, bool)
        or not isinstance(geometry_min_distance_ratio_pass_threshold, (int, float))
        or not np.isfinite(geometry_min_distance_ratio_pass_threshold)
        or geometry_min_distance_ratio_pass_threshold <= 0
    ):
        reject('geometry pass threshold must be a positive finite number')
    if geometry_min_distance_ratio is not None and (
        isinstance(geometry_min_distance_ratio, bool)
        or not isinstance(geometry_min_distance_ratio, (int, float))
        or not np.isfinite(geometry_min_distance_ratio)
        or geometry_min_distance_ratio <= 0
    ):
        reject('geometry_min_distance_ratio must be null or a positive finite number')
    relabeled_site_count = nonnegative_integer(
        relabeled_site_count,
        'relabeled_site_count',
    )
    removed_site_count = nonnegative_integer(
        removed_site_count,
        'removed_site_count',
    )
    if not isinstance(formula_matches_candidate, bool):
        reject('formula_matches_candidate must be boolean')
    if not isinstance(geometry_sanity_pass, bool):
        reject('geometry_sanity_pass must be boolean')

    if execution_status == statuses['execution_error']:
        if not isinstance(execution_message, str) or not execution_message:
            reject('execution errors require descriptive execution_message evidence')
        if formula_matches_candidate or geometry_sanity_pass:
            reject('execution errors cannot claim formula or geometry success')
        if generated_formula is not None or generated_structure_n_sites is not None:
            reject('execution errors cannot claim generated structure evidence')
        if atoms is not None:
            reject('execution errors cannot publish atoms evidence')
        if not isinstance(structure_summary, dict) or any(
            make_json_safe(structure_summary.get(field_name)) is not None
            for field_name in STRUCTURE_SUMMARY_COLUMNS
        ):
            reject('execution errors cannot claim structure summary evidence')
        if (
            geometry_min_distance is not None
            or geometry_mean_distance not in (None, 0, 0.0)
            or geometry_min_distance_ratio is not None
            or geometry_overlap_pair_count != 0
        ):
            reject('execution errors cannot claim generated geometry evidence')
        if cif_text is not None:
            reject('execution errors cannot publish CIF bytes')
        return (
            statuses['relaxation_execution_error'],
            statuses['final_execution_error'],
        )

    if execution_status != statuses['execution_ok']:
        reject('execution_status is outside the builder-owned finite vocabulary')
    if execution_message is not None:
        reject('successful execution cannot carry execution error detail')
    if not isinstance(cif_text, str) or not cif_text.strip():
        reject('successful execution requires generated CIF bytes')
    if not isinstance(generated_formula, str) or not generated_formula.strip():
        reject('successful execution requires a generated formula')
    if (
        isinstance(generated_structure_n_sites, bool)
        or not isinstance(generated_structure_n_sites, (int, float))
        or not np.isfinite(generated_structure_n_sites)
        or float(generated_structure_n_sites) != int(generated_structure_n_sites)
        or generated_structure_n_sites <= 0
    ):
        reject('successful execution requires a positive generated site count')
    try:
        formula_evidence_matches = (
            _canonical_formula(generated_formula)
            == _canonical_formula(candidate_formula)
        )
    except Exception:
        reject('generated formula evidence must be a valid composition')
    if formula_matches_candidate is not formula_evidence_matches:
        reject('formula_matches_candidate disagrees with generated formula evidence')
    _validate_structure_execution_variant_identity(
        atoms=atoms,
        cif_text=cif_text,
        generated_formula=generated_formula,
        generated_structure_n_sites=generated_structure_n_sites,
        structure_summary=structure_summary,
        geometry_min_distance=geometry_min_distance,
        geometry_mean_distance=geometry_mean_distance,
        geometry_min_distance_ratio=geometry_min_distance_ratio,
        geometry_overlap_pair_count=geometry_overlap_pair_count,
        geometry_min_distance_ratio_overlap_threshold=(
            geometry_min_distance_ratio_overlap_threshold
        ),
    )
    expected_geometry_sanity_pass = bool(
        geometry_overlap_pair_count == 0
        and (
            geometry_min_distance_ratio is None
            or geometry_min_distance_ratio
            >= geometry_min_distance_ratio_pass_threshold
        )
    )
    if geometry_sanity_pass is not expected_geometry_sanity_pass:
        reject('geometry_sanity_pass disagrees with distance-ratio/overlap evidence')

    relaxation_status = (
        statuses['relaxation_reference_geometry_reused']
        if relabeled_site_count == 0 and removed_site_count == 0
        else statuses['relaxation_unrelaxed_species_edit']
    )
    if not formula_matches_candidate:
        final_status = statuses['final_formula_mismatch']
    elif not geometry_sanity_pass:
        final_status = statuses['final_geometry_failure']
    elif relaxation_status == statuses['relaxation_reference_geometry_reused']:
        final_status = statuses['final_reference_control']
    else:
        final_status = statuses['final_external_relaxation']
    return relaxation_status, final_status


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
        return [], _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH[
            'requires_atom_insertion'
        ]

    if len(donor_elements) > 1:
        return [], _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH[
            'multiple_donor_species'
        ]

    donor_element = next(iter(donor_elements), None)
    if donor_element is None:
        return [], _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH[
            'no_donor_species'
        ]

    donor_indices = [
        index for index, site in enumerate(structure)
        if site.specie.symbol == donor_element
    ]
    donor_surplus = int(donor_elements[donor_element])
    relabel_count = int(sum(recipient_elements.values()))
    remove_count = max(donor_surplus - relabel_count, 0)
    if relabel_count < 0 or remove_count < 0:
        return [], _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH[
            'invalid_edit_counts'
        ]
    if relabel_count + remove_count > len(donor_indices):
        return [], _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH[
            'insufficient_donor_sites'
        ]

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
