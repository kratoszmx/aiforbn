from __future__ import annotations

from functools import lru_cache
import os
import re

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition, Element
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

from pipeline.data import REFERENCE_PROPERTY_COLUMNS, STRUCTURE_SUMMARY_COLUMNS


ATOMIC_NUMBERS = {
    'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'Al': 13, 'Si': 14, 'P': 15,
    'Ga': 31, 'Ge': 32, 'As': 33, 'In': 49, 'Sn': 50, 'Sb': 51, 'Tl': 81, 'Pb': 82, 'Bi': 83,
}
TOY_CANDIDATE_GENERATION_STRATEGY = 'toy_iii_v_demo_grid'
BN_ANCHORED_CANDIDATE_GENERATION_STRATEGY = 'bn_anchored_formula_family_grid'
DEFAULT_CANDIDATE_GENERATION_STRATEGY = BN_ANCHORED_CANDIDATE_GENERATION_STRATEGY
CANDIDATE_SPACE_DEFAULTS = {
    TOY_CANDIDATE_GENERATION_STRATEGY: {
        'candidate_space_name': 'toy_iii_v_demo_grid',
        'candidate_space_kind': 'toy_demo',
        'candidate_space_note': (
            'Formula-only Group 13/15 enumeration without stability, structure, or synthesis constraints.'
        ),
    },
    BN_ANCHORED_CANDIDATE_GENERATION_STRATEGY: {
        'candidate_space_name': 'bn_anchored_formula_family_grid',
        'candidate_space_kind': 'bn_family_demo',
        'candidate_space_note': (
            'BN-containing formula-family grid anchored by BCN / BC2N-style ternary motifs from '
            'the literature and by Si2BN-like motifs observed in the dataset. This is still '
            'formula-only and does not establish structure stability, synthesis feasibility, or '
            'real discovery.'
        ),
    },
}
DEFAULT_CHEMICAL_PLAUSIBILITY_METHOD = 'pymatgen_common_oxidation_state_balance'
DEFAULT_CHEMICAL_PLAUSIBILITY_NOTE = (
    'Formula-level plausibility annotation using pymatgen oxidation-state guesses. This is not a '
    'structure, thermodynamic stability, phonon stability, or synthesis feasibility filter.'
)
DEFAULT_DOMAIN_SUPPORT_METHOD = 'train_plus_val_knn_feature_space_support'
DEFAULT_DOMAIN_SUPPORT_DISTANCE_METRIC = 'z_scored_euclidean_rms'
DEFAULT_DOMAIN_SUPPORT_NOTE = (
    'Support is measured in the selected formula-only screening feature space using z-scored '
    'distances to unique train+val formulas. This is a lightweight transparency heuristic, not '
    'a calibrated discovery confidence, uncertainty, or stability estimate.'
)
DOMAIN_SUPPORT_REFERENCE_SPLIT = 'train_plus_val_unique_formulas'
DEFAULT_BN_SUPPORT_METHOD = 'train_plus_val_bn_knn_feature_space_support'
DEFAULT_BN_SUPPORT_DISTANCE_METRIC = 'z_scored_euclidean_rms'
DEFAULT_BN_SUPPORT_NOTE = (
    'Support is measured relative to known BN-containing train+val formulas in the selected '
    'formula-only screening feature space. This is a BN-theme alignment heuristic, not a '
    'calibrated discovery confidence, uncertainty, or structure/stability estimate.'
)
BN_SUPPORT_REFERENCE_SPLIT = 'train_plus_val_bn_unique_formulas'
DEFAULT_BN_ANALOG_EVIDENCE_NOTE = (
    'Observed-property evidence is retrieved from nearby BN-containing train+val formulas. This '
    'is an analog-evidence layer, not a predicted structure, thermodynamic stability, or synthesis '
    'feasibility estimate.'
)
BN_ANALOG_EVIDENCE_REFERENCE_SPLIT = 'train_plus_val_bn_unique_formulas'
BN_ANALOG_EVIDENCE_RANKING_NOTE = (
    'Candidates are also paired with observed-property evidence from nearby BN-containing '
    'train+val formulas, including analog band gap, energy-per-atom, exfoliation-energy, and '
    'magnetization summaries. A lightweight analog-validation label then summarizes whether those '
    'nearby BN references stay in more reference-like or more divergent property regimes on the '
    'available metrics. This is analog retrieval, not direct candidate property validation.'
)
BASIC_FEATURE_SET = 'basic_formula_composition'
MATMINER_FEATURE_SET = 'matminer_composition'
STRUCTURE_AWARE_FEATURE_SET = 'matminer_composition_plus_structure_summary'
COMPOSITION_ONLY_FAMILY = 'composition_only'
STRUCTURE_AWARE_FAMILY = 'structure_aware'
DUMMY_FEATURE_SET = 'feature_agnostic_dummy'
FORMULA_ONLY_SCREENING_SCOPE = 'candidate_compatible_formula_only'
SELECTED_MODEL_RANKING_BASIS = 'composition_only_selected_model_band_gap'
SELECTED_MODEL_RANKING_NOTE = (
    'Composition-only ranking from the selected formula-based model; not structure-aware.'
)
DISAGREEMENT_RANKING_BASIS = 'composition_only_mean_band_gap_minus_model_disagreement_penalty'
DISAGREEMENT_RANKING_NOTE = (
    'Composition-only demo ranking using a tiny feature/model candidate pool. The score is the '
    'ensemble mean band gap minus a small model-disagreement penalty; disagreement is heuristic '
    'and not calibrated physical uncertainty.'
)
SELECTED_MODEL_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_BASIS = (
    'composition_only_selected_model_band_gap_minus_low_support_penalty'
)
SELECTED_MODEL_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_NOTE = (
    'Composition-only ranking from the selected formula-based model with a mild low-support '
    'penalty in the selected screening feature space; this remains a formula-only heuristic and '
    'not a structure-aware stability estimate.'
)
DISAGREEMENT_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_BASIS = (
    'composition_only_mean_band_gap_minus_model_disagreement_and_low_support_penalties'
)
DISAGREEMENT_WITH_DOMAIN_SUPPORT_PENALTY_RANKING_NOTE = (
    'Composition-only demo ranking using a tiny feature/model candidate pool. The score is the '
    'ensemble mean band gap minus a small model-disagreement penalty and a mild low-support '
    'penalty in the selected screening feature space; both terms are heuristic and not calibrated '
    'physical uncertainty or stability estimates.'
)
SELECTED_MODEL_WITH_BN_SUPPORT_PENALTY_RANKING_BASIS = (
    'composition_only_selected_model_band_gap_minus_bn_support_penalty'
)
SELECTED_MODEL_WITH_BN_SUPPORT_PENALTY_RANKING_NOTE = (
    'Composition-only ranking from the selected formula-based model with a mild BN-local support '
    'penalty in the selected screening feature space; this remains a formula-only heuristic and '
    'not a structure-aware stability estimate.'
)
DISAGREEMENT_WITH_BN_SUPPORT_PENALTY_RANKING_BASIS = (
    'composition_only_mean_band_gap_minus_model_disagreement_and_bn_support_penalties'
)
DISAGREEMENT_WITH_BN_SUPPORT_PENALTY_RANKING_NOTE = (
    'Composition-only demo ranking using a tiny feature/model candidate pool. The score is the '
    'ensemble mean band gap minus a small model-disagreement penalty and a mild BN-local support '
    'penalty; both terms are heuristic and not calibrated physical uncertainty or stability '
    'estimates.'
)
SELECTED_MODEL_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_BASIS = (
    'composition_only_selected_model_band_gap_minus_low_support_and_bn_support_penalties'
)
SELECTED_MODEL_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_NOTE = (
    'Composition-only ranking from the selected formula-based model with mild low-support and '
    'BN-local support penalties in the selected screening feature space; these remain formula-only '
    'heuristics and not structure-aware stability estimates.'
)
DISAGREEMENT_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_BASIS = (
    'composition_only_mean_band_gap_minus_model_disagreement_low_support_and_bn_support_penalties'
)
DISAGREEMENT_WITH_DOMAIN_AND_BN_SUPPORT_PENALTY_RANKING_NOTE = (
    'Composition-only demo ranking using a tiny feature/model candidate pool. The score is the '
    'ensemble mean band gap minus a small model-disagreement penalty together with mild '
    'feature-space low-support and BN-local support penalties; all terms remain heuristic and not '
    'calibrated physical uncertainty or stability estimates.'
)
DOMAIN_SUPPORT_RANKING_NOTE = (
    'Candidates are also annotated with a train+val feature-space domain-support layer: support '
    'comes from z-scored distances to nearby known formulas in the selected screening feature '
    'space, contextualized by the leave-one-out train+val neighborhood-distance distribution.'
)
BN_SUPPORT_RANKING_NOTE = (
    'Candidates are also contextualized against the known BN slice: support comes from z-scored '
    'distances to nearby BN-containing train+val formulas in the selected screening feature '
    'space, contextualized by the leave-one-out BN-neighborhood-distance distribution.'
)
CHEMICAL_PLAUSIBILITY_SCREENING_NOTE = (
    'A lightweight formula-level chemical plausibility screen is applied first using pymatgen '
    'oxidation-state guesses; candidates that pass are prioritized ahead of candidates without a '
    'charge-balanced common oxidation-state assignment.'
)
NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY = 'train_plus_val_rediscovery'
NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA = 'held_out_known_formula'
NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION = 'formula_level_extrapolation'
NOVELTY_BUCKET_PRIORITY = {
    NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY: 1,
    NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA: 2,
    NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION: 3,
}
NOVELTY_BUCKET_NOTE = {
    NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY: (
        'Seen in train+val, so this is rediscovery / in-domain replay rather than formula-level '
        'extrapolation.'
    ),
    NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA: (
        'Seen elsewhere in the dataset but not in train+val, so this is a held-out-known formula '
        'rather than a new formula-level candidate.'
    ),
    NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION: (
        'Unseen in the dataset, so this is formula-level extrapolation within the current demo '
        'candidate space. This still does not establish real materials discovery.'
    ),
}
NOVELTY_ANNOTATION_RANKING_NOTE = (
    'Novelty is tracked only at the formula level: use the novelty bucket fields to separate '
    'train+val rediscovery, held-out-known formulas, and unseen formulas. This transparency layer '
    'does not turn the demo ranking into validated discovery.'
)
FEATURE_SET_METADATA = {
    BASIC_FEATURE_SET: (
        COMPOSITION_ONLY_FAMILY,
        True,
        'Hand-written control baseline from formula tokens and a small atomic-number lookup.',
    ),
    MATMINER_FEATURE_SET: (
        COMPOSITION_ONLY_FAMILY,
        True,
        'Curated matminer composition descriptors from pymatgen Composition objects using '
        'Stoichiometry plus selected Magpie elemental-property statistics.',
    ),
    STRUCTURE_AWARE_FEATURE_SET: (
        STRUCTURE_AWARE_FAMILY,
        False,
        'Matminer composition descriptors plus compact lattice/layer summary columns derived '
        'from cached atoms and lattice information.',
    ),
    DUMMY_FEATURE_SET: (
        'feature_agnostic_baseline',
        False,
        'Dummy regressor baseline that ignores composition features.',
    ),
}
BASIC_FEATURE_COLUMNS = (
    'n_elements',
    'sum_z',
    'max_z',
    'min_z',
    'mean_z',
    'contains_B',
    'contains_N',
)
MATMINER_SELECTED_RAW_LABELS = (
    '0-norm',
    '2-norm',
    '3-norm',
    'MagpieData mean Number',
    'MagpieData range Number',
    'MagpieData mean MendeleevNumber',
    'MagpieData range MendeleevNumber',
    'MagpieData mean AtomicWeight',
    'MagpieData range AtomicWeight',
    'MagpieData mean Row',
    'MagpieData range Row',
    'MagpieData mean Column',
    'MagpieData range Column',
    'MagpieData mean Electronegativity',
    'MagpieData range Electronegativity',
    'MagpieData mean NsValence',
    'MagpieData mean NpValence',
    'MagpieData mean NdValence',
    'MagpieData mean NfValence',
)
STRUCTURE_AWARE_REQUIRED_COLUMNS = STRUCTURE_SUMMARY_COLUMNS
BASE_PASSTHROUGH_COLUMNS = (
    'record_id',
    'source',
    'formula',
    'target',
    *REFERENCE_PROPERTY_COLUMNS,
    'candidate_space_name',
    'candidate_space_kind',
    'candidate_generation_strategy',
    'candidate_space_note',
    'candidate_family',
    'candidate_template',
    'candidate_family_note',
)


def extract_elements(formula: str) -> list[str]:
    return re.findall(r'[A-Z][a-z]?', formula or '')


def filter_bn(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    mask = df[formula_col].astype(str).apply(lambda x: {'B', 'N'}.issubset(set(extract_elements(x))))
    out = df.loc[mask].copy()
    out['elements'] = out[formula_col].astype(str).apply(extract_elements)
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


def get_screening_ranking_metadata(
    cfg: dict | None = None,
    domain_support_penalty_applied: bool | None = None,
    bn_support_penalty_applied: bool | None = None,
) -> dict[str, object]:
    screening_cfg = (cfg or {}).get('screening', {})
    support_cfg = _domain_support_config(cfg)
    bn_support_cfg = _bn_support_config(cfg)
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


def get_candidate_feature_sets(cfg: dict) -> list[str]:
    features_cfg = cfg.get('features', {})
    default_feature_set = features_cfg.get('feature_set', BASIC_FEATURE_SET)
    candidate_sets = list(features_cfg.get('candidate_sets', [default_feature_set]))
    return _ordered_values([default_feature_set] + candidate_sets)


def get_candidate_screening_feature_sets(cfg: dict) -> list[str]:
    return [
        feature_set
        for feature_set in get_candidate_feature_sets(cfg)
        if feature_set_supports_formula_only_screening(feature_set)
    ]


def get_candidate_model_types(cfg: dict) -> list[str]:
    model_cfg = cfg.get('model', {})
    default_model_type = model_cfg.get('type', 'hist_gradient_boosting')
    candidate_types = list(model_cfg.get('candidate_types', [default_model_type]))
    return _ordered_values([default_model_type] + candidate_types)


def get_feature_family(feature_set: str) -> str:
    return FEATURE_SET_METADATA.get(feature_set, ('unknown', False, ''))[0]


def feature_set_supports_formula_only_screening(feature_set: str) -> bool:
    return bool(FEATURE_SET_METADATA.get(feature_set, ('unknown', False, ''))[1])


def get_feature_note(feature_set: str) -> str:
    return FEATURE_SET_METADATA.get(feature_set, ('unknown', False, ''))[2]


def _basic_features(formula: str) -> tuple[dict[str, float], str | None]:
    elements = extract_elements(formula)
    if not elements:
        return (
            {column: np.nan for column in BASIC_FEATURE_COLUMNS},
            'ValueError: no elements parsed from formula',
        )

    z = [ATOMIC_NUMBERS.get(e, 0) for e in elements]
    return ({
        'n_elements': len(elements),
        'sum_z': sum(z),
        'max_z': max(z),
        'min_z': min(z),
        'mean_z': sum(z) / len(z),
        'contains_B': int('B' in elements),
        'contains_N': int('N' in elements),
    }, None)


def _clean_feature_label(label: str, prefix: str) -> str:
    normalized = re.sub(r'[^0-9a-zA-Z]+', '_', label).strip('_').lower()
    return f'{prefix}_{normalized}' if normalized else prefix


@lru_cache(maxsize=1)
def _matminer_featurizer() -> MultipleFeaturizer:
    featurizer = MultipleFeaturizer([
        Stoichiometry(),
        ElementProperty.from_preset('magpie'),
    ])
    featurizer.set_n_jobs(1)
    return featurizer


@lru_cache(maxsize=1)
def _matminer_feature_spec() -> tuple[tuple[int, str], ...]:
    selected_labels = set(MATMINER_SELECTED_RAW_LABELS)
    labels: list[tuple[int, str]] = []
    seen: set[str] = set()
    for idx, raw_label in enumerate(_matminer_featurizer().feature_labels()):
        if raw_label not in selected_labels:
            continue
        clean_label = _clean_feature_label(raw_label, prefix='matminer')
        if clean_label in seen:
            raise ValueError(f'Duplicate cleaned matminer feature label: {clean_label}')
        seen.add(clean_label)
        labels.append((idx, clean_label))

    missing = sorted(selected_labels - {raw for raw in _matminer_featurizer().feature_labels() if raw in selected_labels})
    if missing:
        raise ValueError(f'Missing configured matminer labels: {missing}')
    return tuple(labels)


def _matminer_features(formula: str) -> tuple[dict[str, float], str | None]:
    feature_spec = _matminer_feature_spec()
    labels = [label for _, label in feature_spec]
    try:
        values = _matminer_featurizer().featurize(Composition(str(formula)))
        selected_values = [values[idx] for idx, _ in feature_spec]
        values_array = np.asarray(selected_values, dtype=float)
    except Exception as exc:  # pragma: no cover - exact exception type depends on formula failure mode
        return {label: np.nan for label in labels}, f'{type(exc).__name__}: {exc}'

    if not np.isfinite(values_array).all():
        return {label: np.nan for label in labels}, 'ValueError: non-finite matminer features'

    return dict(zip(labels, values_array.tolist())), None


def _build_base_frame(df: pd.DataFrame, formula_col: str, feature_set: str) -> pd.DataFrame:
    if formula_col not in df.columns:
        raise KeyError(f'Formula column not found: {formula_col}')

    base = pd.DataFrame(index=df.index)
    for column in BASE_PASSTHROUGH_COLUMNS:
        if column == 'formula':
            base[column] = df[formula_col].astype(str)
        elif column in df.columns:
            base[column] = df[column]

    if feature_set == STRUCTURE_AWARE_FEATURE_SET:
        for column in STRUCTURE_AWARE_REQUIRED_COLUMNS:
            if column in df.columns:
                base[column] = df[column]
            else:
                base[column] = np.nan

    return base.reset_index(drop=True)


def _validate_structure_summary_features(row: pd.Series) -> str | None:
    missing_columns = [column for column in STRUCTURE_AWARE_REQUIRED_COLUMNS if column not in row.index]
    if missing_columns:
        return f'ValueError: missing structure summary columns: {missing_columns}'

    invalid_columns = []
    for column in STRUCTURE_AWARE_REQUIRED_COLUMNS:
        value = row[column]
        if pd.isna(value):
            invalid_columns.append(column)
            continue
        try:
            numeric_value = float(value)
        except Exception:
            invalid_columns.append(column)
            continue
        if not np.isfinite(numeric_value):
            invalid_columns.append(column)

    if invalid_columns:
        return f'ValueError: missing structure summary values: {invalid_columns}'
    return None


def _combine_feature_errors(*errors: str | None) -> str | None:
    messages = [error for error in errors if error]
    if not messages:
        return None
    return '; '.join(messages)


def build_feature_table(
    df: pd.DataFrame,
    formula_col: str = 'formula',
    feature_set: str = BASIC_FEATURE_SET,
) -> pd.DataFrame:
    base = _build_base_frame(df, formula_col=formula_col, feature_set=feature_set)
    formula_series = df[formula_col].astype(str).reset_index(drop=True)

    feature_dicts: list[dict[str, float]] = []
    errors: list[str | None] = []

    if feature_set == BASIC_FEATURE_SET:
        for formula in formula_series:
            feature_row, error = _basic_features(formula)
            feature_dicts.append(feature_row)
            errors.append(error)
    elif feature_set == MATMINER_FEATURE_SET:
        for formula in formula_series:
            feature_row, error = _matminer_features(formula)
            feature_dicts.append(feature_row)
            errors.append(error)
    elif feature_set == STRUCTURE_AWARE_FEATURE_SET:
        for formula, (_, row) in zip(formula_series, base.iterrows(), strict=False):
            matminer_row, matminer_error = _matminer_features(formula)
            structure_error = _validate_structure_summary_features(row)
            feature_dicts.append(matminer_row)
            errors.append(_combine_feature_errors(matminer_error, structure_error))
    else:
        raise ValueError(f'Unsupported feature set: {feature_set}')

    feature_rows = pd.DataFrame(feature_dicts)
    out = pd.concat([base, feature_rows], axis=1)
    out['feature_set'] = feature_set
    out['feature_generation_failed'] = pd.Series([error is not None for error in errors], dtype=bool)
    out['feature_generation_error'] = errors
    return out


def build_feature_tables(
    df: pd.DataFrame,
    cfg: dict,
    formula_col: str = 'formula',
) -> dict[str, pd.DataFrame]:
    return {
        feature_set: build_feature_table(df, formula_col=formula_col, feature_set=feature_set)
        for feature_set in get_candidate_feature_sets(cfg)
    }


def _ratio_counts(total: int, ratios: list[float]) -> list[int]:
    if total <= 0:
        return [0] * len(ratios)

    raw_ratios = np.asarray(ratios, dtype=float)
    if raw_ratios.sum() <= 0:
        raise ValueError('Split ratios must sum to a positive value')

    normalized = raw_ratios / raw_ratios.sum()
    raw_counts = normalized * total
    counts = np.floor(raw_counts).astype(int)

    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        for idx in order[:remainder]:
            counts[idx] += 1

    positive_indices = [idx for idx, ratio in enumerate(raw_ratios) if ratio > 0]
    if total >= len(positive_indices):
        for idx in positive_indices:
            if counts[idx] > 0:
                continue
            donor_order = np.argsort(counts)[::-1]
            for donor_idx in donor_order:
                if donor_idx != idx and counts[donor_idx] > 1:
                    counts[donor_idx] -= 1
                    counts[idx] += 1
                    break

    return counts.astype(int).tolist()


def _build_split_metadata(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    group_col: str,
    method: str,
    cfg: dict,
) -> dict:
    group_sets = {
        name: set(df.loc[masks[name], group_col].astype(str).tolist())
        for name in ('train', 'val', 'test')
    }
    return {
        'method': method,
        'group_column': group_col,
        'ratios': {
            'train': float(cfg['split']['train_ratio']),
            'val': float(cfg['split']['val_ratio']),
            'test': float(cfg['split']['test_ratio']),
        },
        'row_counts': {
            name: int(np.asarray(masks[name]).sum())
            for name in ('train', 'val', 'test')
        },
        'group_counts': {
            name: int(len(group_sets[name]))
            for name in ('train', 'val', 'test')
        },
        'group_overlap_counts': {
            'train_val': int(len(group_sets['train'] & group_sets['val'])),
            'train_test': int(len(group_sets['train'] & group_sets['test'])),
            'val_test': int(len(group_sets['val'] & group_sets['test'])),
        },
    }


def _make_random_row_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg['project']['random_seed'])
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(n * cfg['split']['train_ratio'])
    val_end = train_end + int(n * cfg['split']['val_ratio'])

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    masks = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        masks[name] = mask
    masks['metadata'] = _build_split_metadata(
        df=df,
        masks=masks,
        group_col=cfg['split'].get('group_column', 'formula'),
        method='random_row',
        cfg=cfg,
    )
    return masks


def _make_grouped_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    group_col = cfg['split'].get('group_column', 'formula')
    if group_col not in df.columns:
        raise KeyError(f'Group split column not found: {group_col}')

    rng = np.random.default_rng(cfg['project']['random_seed'])
    group_series = df[group_col].astype(str)
    unique_groups = group_series.drop_duplicates().to_numpy()
    rng.shuffle(unique_groups)

    train_count, val_count, test_count = _ratio_counts(
        total=len(unique_groups),
        ratios=[
            cfg['split']['train_ratio'],
            cfg['split']['val_ratio'],
            cfg['split']['test_ratio'],
        ],
    )

    train_groups = set(unique_groups[:train_count])
    val_groups = set(unique_groups[train_count:train_count + val_count])
    test_groups = set(unique_groups[train_count + val_count:train_count + val_count + test_count])

    masks = {
        'train': group_series.isin(train_groups).to_numpy(),
        'val': group_series.isin(val_groups).to_numpy(),
        'test': group_series.isin(test_groups).to_numpy(),
    }
    masks['metadata'] = _build_split_metadata(
        df=df,
        masks=masks,
        group_col=group_col,
        method='group_by_formula',
        cfg=cfg,
    )
    return masks


def make_split_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    method = cfg['split'].get('method', 'group_by_formula')
    if method == 'group_by_formula':
        return _make_grouped_split_masks(df, cfg)
    if method == 'random_row':
        return _make_random_row_split_masks(df, cfg)
    raise ValueError(f'Unsupported split method: {method}')


def _feature_columns(df: pd.DataFrame) -> list[str]:
    banned = {
        'record_id',
        'source',
        'formula',
        'target',
        *REFERENCE_PROPERTY_COLUMNS,
        'elements',
        'feature_set',
        'feature_generation_failed',
        'feature_generation_error',
        'candidate_space_name',
        'candidate_space_kind',
        'candidate_generation_strategy',
        'candidate_space_note',
        'candidate_family',
        'candidate_template',
        'candidate_family_note',
        'ranking_label',
        'ranking_basis',
        'ranking_note',
        'ranking_feature_set',
        'ranking_model_type',
    }
    return [c for c in df.columns if c not in banned]


def _feature_valid_mask(df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    if 'feature_generation_failed' in df.columns:
        mask &= ~df['feature_generation_failed'].fillna(False).astype(bool)
    if feature_columns:
        mask &= df[feature_columns].notna().all(axis=1)
    return mask


def summarize_feature_table(feature_df: pd.DataFrame, feature_set: str | None = None) -> dict:
    inferred_feature_set = feature_set
    if inferred_feature_set is None and 'feature_set' in feature_df.columns and not feature_df.empty:
        inferred_feature_set = str(feature_df['feature_set'].iloc[0])

    feature_columns = _feature_columns(feature_df)
    failed_mask = feature_df.get(
        'feature_generation_failed',
        pd.Series(False, index=feature_df.index, dtype=bool),
    ).astype(bool)
    formula_examples = []
    if 'formula' in feature_df.columns:
        formula_examples = feature_df.loc[failed_mask, 'formula'].astype(str).head(5).tolist()
    error_examples = []
    if 'feature_generation_error' in feature_df.columns:
        error_examples = (
            feature_df.loc[failed_mask, 'feature_generation_error']
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )
    status = 'ok'
    if not feature_columns:
        status = 'no_features'
    elif failed_mask.any():
        status = 'featurization_incomplete'

    return {
        'feature_set': inferred_feature_set,
        'feature_family': get_feature_family(inferred_feature_set or ''),
        'feature_note': get_feature_note(inferred_feature_set or ''),
        'candidate_compatible': feature_set_supports_formula_only_screening(inferred_feature_set or ''),
        'n_features': int(len(feature_columns)),
        'status': status,
        'selection_eligible': bool(status == 'ok'),
        'failed_formula_count': int(failed_mask.sum()),
        'failed_formula_examples': formula_examples,
        'failed_error_examples': error_examples,
    }


def make_model(cfg: dict, model_type: str | None = None):
    model_type = model_type or cfg['model']['type']
    if model_type == 'linear_regression':
        return LinearRegression(**cfg['model'].get('linear_regression', {}))
    if model_type == 'random_forest':
        return RandomForestRegressor(**cfg['model']['random_forest'])
    if model_type == 'hist_gradient_boosting':
        return HistGradientBoostingRegressor(**cfg['model']['hist_gradient_boosting'])
    if model_type == 'dummy_mean':
        return DummyRegressor(**cfg['model'].get('dummy_mean', {'strategy': 'mean'}))
    raise ValueError(f'Unsupported model type: {model_type}')


def train_baseline_model(
    df: pd.DataFrame,
    split_masks,
    cfg: dict,
    model_type: str | None = None,
    include_validation: bool = False,
) -> tuple[object, list[str]]:
    feature_columns = _feature_columns(df)
    training_mask = split_masks['train'] | split_masks['val'] if include_validation else split_masks['train']
    train_df = df.loc[training_mask].copy()
    train_df = train_df[train_df['target'].notna()].copy()
    train_df = train_df.loc[_feature_valid_mask(train_df, feature_columns)].copy()
    if train_df.empty:
        raise ValueError('No training rows remain after filtering invalid feature rows')

    X = train_df[feature_columns]
    y = train_df['target']

    model = make_model(cfg, model_type=model_type)
    model.fit(X, y)
    return model, feature_columns


def _regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        'mae': float(mean_absolute_error(y_true_arr, y_pred_arr)),
        'rmse': float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5),
        'r2': float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else None,
    }


def evaluate_predictions(
    df: pd.DataFrame,
    split_masks,
    model,
    feature_columns: list[str],
    split_name: str = 'test',
):
    requested_eval_df = df.loc[split_masks[split_name]].copy()
    requested_eval_df = requested_eval_df[requested_eval_df['target'].notna()].copy()
    if requested_eval_df.empty:
        raise ValueError(f'No evaluation rows available for split: {split_name}')

    valid_mask = _feature_valid_mask(requested_eval_df, feature_columns)
    if int(valid_mask.sum()) != len(requested_eval_df):
        failed_formulas = requested_eval_df.loc[~valid_mask, 'formula'].astype(str).head(5).tolist()
        raise ValueError(
            f'Feature set cannot evaluate all {split_name} rows; '
            f'invalid formulas include: {failed_formulas}'
        )

    eval_df = requested_eval_df.loc[valid_mask].copy()
    X = eval_df[feature_columns]
    y = eval_df['target']
    pred = model.predict(X)

    metrics = _regression_metrics(y, pred)

    prediction_df = eval_df[['formula', 'target']].copy()
    prediction_df['prediction'] = pred
    prediction_df['abs_error'] = (prediction_df['target'] - prediction_df['prediction']).abs()
    return metrics, prediction_df


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

    if default_feature_set not in eligible_feature_sets:
        default_feature_set = eligible_feature_sets[0]
    screening_default_feature_set = (
        default_feature_set
        if default_feature_set in screening_candidate_feature_sets
        else screening_candidate_feature_sets[0]
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
        'screening_selected_model_type': default_model_type,
        'screening_selected_feature_family': get_feature_family(screening_default_feature_set),
        'screening_selected_feature_count': int(
            summarize_feature_table(
                feature_tables[screening_default_feature_set],
                feature_set=screening_default_feature_set,
            )['n_features']
        ),
        'screening_selection_matches_overall': bool(
            screening_default_feature_set == default_feature_set
            and default_model_type == default_model_type
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


def build_candidate_prediction_ensemble(
    candidate_df: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    split_masks,
    cfg: dict,
    candidate_feature_sets: list[str] | None = None,
) -> pd.DataFrame:
    candidate_feature_sets = candidate_feature_sets or [
        value for value in get_candidate_screening_feature_sets(cfg) if value in feature_tables
    ]
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
        for model_type in candidate_model_types:
            model, _ = train_baseline_model(
                df=train_feature_df,
                split_masks=split_masks,
                cfg=cfg,
                model_type=model_type,
                include_validation=True,
            )
            prediction_frames.append(pd.DataFrame({
                'formula': candidate_feature_df['formula'].astype(str),
                'feature_set': feature_set,
                'model_type': model_type,
                'prediction': model.predict(candidate_feature_df[feature_columns]),
            }))

    if not prediction_frames:
        raise ValueError('No candidate feature/model combination was available for uncertainty estimation')

    prediction_df = pd.concat(prediction_frames, ignore_index=True)
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
    out = pd.DataFrame({
        formula_col: candidate_df[formula_col].astype(str).reset_index(drop=True),
    })
    out['bn_analog_evidence_enabled'] = bool(evidence_metadata['enabled'])
    out['bn_analog_evidence_aggregation'] = evidence_metadata['aggregation']
    out['bn_analog_evidence_reference_split'] = evidence_metadata['reference_split']
    out['bn_analog_evidence_exfoliation_reference'] = evidence_metadata['exfoliation_reference']
    out['bn_analog_evidence_note'] = evidence_metadata['note']
    out['bn_analog_reference_formula_count'] = 0
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

    nearest_formulas = candidate_df['bn_support_nearest_formula'].astype(str).reset_index(drop=True)
    neighbor_formula_strings = candidate_df['bn_support_neighbor_formulas'].astype(str).reset_index(drop=True)
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

        neighbor_band_gap_mean, _ = _lookup_metric(neighbor_formulas, 'target')
        neighbor_energy_mean, _ = _lookup_metric(neighbor_formulas, 'energy_per_atom')
        neighbor_exfoliation_mean, neighbor_exfoliation_count = _lookup_metric(
            neighbor_formulas,
            'exfoliation_energy_per_atom',
        )
        neighbor_abs_mag_mean, _ = _lookup_metric(neighbor_formulas, 'abs_total_magnetization')
        out.at[idx, 'bn_analog_neighbor_band_gap_mean'] = neighbor_band_gap_mean
        out.at[idx, 'bn_analog_neighbor_energy_per_atom_mean'] = neighbor_energy_mean
        out.at[idx, 'bn_analog_neighbor_exfoliation_energy_per_atom_mean'] = neighbor_exfoliation_mean
        out.at[idx, 'bn_analog_neighbor_abs_total_magnetization_mean'] = neighbor_abs_mag_mean
        out.at[idx, 'bn_analog_neighbor_exfoliation_available_formula_count'] = int(
            neighbor_exfoliation_count
        )

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

    ranking_config_metadata = get_screening_ranking_metadata(cfg)
    uncertainty_penalty = float(ranking_config_metadata['ranking_uncertainty_penalty'])
    use_model_disagreement = bool(cfg['screening'].get('use_model_disagreement', False))
    if use_model_disagreement and ensemble_prediction_df is not None:
        out['ranking_score'] = (
            out['ensemble_predicted_band_gap_mean']
            - uncertainty_penalty * out['ensemble_predicted_band_gap_std']
        )
    else:
        out['ranking_score'] = out['predicted_band_gap']
    out['ranking_score_before_domain_support_penalty'] = out['ranking_score']
    out['ranking_score_before_bn_support_penalty'] = out['ranking_score']

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

    ranking_metadata = get_screening_ranking_metadata(
        cfg,
        domain_support_penalty_applied=bool(out['domain_support_penalty'].fillna(0.0).gt(0.0).any()),
        bn_support_penalty_applied=bool(out['bn_support_penalty'].fillna(0.0).gt(0.0).any()),
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
    if 'bn_analog_evidence_enabled' in out.columns and bool(out['bn_analog_evidence_enabled'].fillna(False).any()):
        out['ranking_note'] = out['ranking_note'] + ' ' + BN_ANALOG_EVIDENCE_RANKING_NOTE
    if 'candidate_novelty_bucket' in out.columns:
        out['ranking_note'] = out['ranking_note'] + ' ' + NOVELTY_ANNOTATION_RANKING_NOTE
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
    out['screening_selection_decision'] = 'not_selected_top_k'
    out.loc[out['screening_selected_for_top_k'], 'screening_selection_decision'] = 'selected_top_k'
    if chemical_plausibility_enabled:
        out.loc[
            ~out['screening_selected_for_top_k'] & ~out['chemical_plausibility_pass'],
            'screening_selection_decision',
        ] = 'failed_chemical_plausibility'

    return out
