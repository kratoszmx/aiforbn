from __future__ import annotations

import os
from pathlib import Path
import re

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ai_for_bn_mplconfig')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runtime.io_utils import make_json_safe, write_json_file
from materials.data import load_cached_raw_record_lookup
from materials.constants import *
from materials.candidate_space import *
from materials.candidate_space import _formula_amount_map, _structure_seed_edit_metadata
from materials.feature_building import *
from materials.benchmarking import *
from materials.common import *

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
    return make_json_safe(value)


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

