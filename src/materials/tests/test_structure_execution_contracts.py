from __future__ import annotations

from collections import Counter

import numpy as np
from pymatgen.core import Structure

from materials.constants import STRUCTURE_AWARE_FEATURE_SET
from materials.structure_helpers import (
    _apply_variant_plan,
    _build_variant_plans,
    _predict_structure_band_gap_proxy,
    _structure_to_atoms,
)


def _reference_structure() -> Structure:
    return Structure(
        np.eye(3) * 4.0,
        ['B', 'B', 'N', 'N'],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
    )


def _composition_counts(structure: Structure) -> Counter[str]:
    return Counter(site.specie.symbol for site in structure)


def test_variant_plan_relabels_one_donor_site_without_mutating_reference():
    reference = _reference_structure()

    plans, error = _build_variant_plans(
        reference,
        current_counts={'B': 2, 'N': 2},
        target_counts={'B': 1, 'C': 1, 'N': 2},
        max_variants=2,
    )

    assert error is None
    assert len(plans) == 2
    assert all(plan['plan_type'] == 'edited_structure' for plan in plans)
    assert all(len(plan['relabel_indices']) == 1 for plan in plans)
    assert all(plan['relabel_targets'] == ('C',) for plan in plans)
    assert all(plan['remove_indices'] == () for plan in plans)

    edited = _apply_variant_plan(
        reference,
        relabel_indices=plans[0]['relabel_indices'],
        relabel_targets=plans[0]['relabel_targets'],
        remove_indices=plans[0]['remove_indices'],
    )
    assert _composition_counts(edited) == Counter({'N': 2, 'B': 1, 'C': 1})
    assert _composition_counts(reference) == Counter({'B': 2, 'N': 2})


def test_variant_plan_removes_one_donor_site_for_vacancy_edit():
    reference = _reference_structure()

    plans, error = _build_variant_plans(
        reference,
        current_counts={'B': 2, 'N': 2},
        target_counts={'B': 1, 'N': 2},
        max_variants=2,
    )

    assert error is None
    assert len(plans) == 2
    assert all(plan['relabel_indices'] == () for plan in plans)
    assert all(len(plan['remove_indices']) == 1 for plan in plans)

    edited = _apply_variant_plan(
        reference,
        relabel_indices=plans[0]['relabel_indices'],
        relabel_targets=plans[0]['relabel_targets'],
        remove_indices=plans[0]['remove_indices'],
    )
    assert _composition_counts(edited) == Counter({'N': 2, 'B': 1})
    assert len(edited) == 3


def test_variant_plan_reports_unsupported_insertion_and_multiple_donor_edits():
    reference = _reference_structure()

    insertion_plans, insertion_error = _build_variant_plans(
        reference,
        current_counts={'B': 2, 'N': 2},
        target_counts={'Al': 1, 'B': 2, 'N': 2},
        max_variants=2,
    )
    donor_plans, donor_error = _build_variant_plans(
        reference,
        current_counts={'B': 2, 'N': 2},
        target_counts={'B': 1, 'C': 1, 'N': 1},
        max_variants=2,
    )

    assert insertion_plans == []
    assert insertion_error == 'requires_atom_insertion'
    assert donor_plans == []
    assert donor_error == 'multiple_donor_species_not_supported'


def test_structure_proxy_uses_only_declared_structure_aware_features():
    class CapturingModel:
        seen_columns = None

        def predict(self, feature_df):
            self.seen_columns = feature_df.columns.tolist()
            return np.asarray([4.25])

    model = CapturingModel()
    atoms = _structure_to_atoms(_reference_structure())

    prediction, error = _predict_structure_band_gap_proxy(
        candidate_formula='B2N2',
        atoms=atoms,
        structure_model=model,
        structure_feature_columns=['structure_n_sites'],
        structure_feature_set=STRUCTURE_AWARE_FEATURE_SET,
    )

    assert prediction == 4.25
    assert error is None
    assert model.seen_columns == ['structure_n_sites']


def test_structure_proxy_reports_missing_feature_columns_without_predicting():
    class PredictionMustNotRun:
        def predict(self, _feature_df):
            raise AssertionError('prediction must not run with missing structure features')

    prediction, error = _predict_structure_band_gap_proxy(
        candidate_formula='B2N2',
        atoms=_structure_to_atoms(_reference_structure()),
        structure_model=PredictionMustNotRun(),
        structure_feature_columns=['not_a_structure_feature'],
        structure_feature_set=STRUCTURE_AWARE_FEATURE_SET,
    )

    assert prediction is None
    assert error == 'missing_structure_feature_columns'
