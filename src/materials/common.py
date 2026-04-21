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
from materials.candidate_space import (
    _bn_family_benchmark_config,
    _bn_slice_benchmark_config,
    _bn_stratified_error_config,
    _extrapolation_shortlist_config,
    _formula_amount_map,
    _proposal_shortlist_config,
    _structure_generation_seed_config,
    _structure_seed_edit_metadata,
    get_screening_ranking_metadata,
)
from materials.constants import (
    BN_ANALOG_EVIDENCE_RANKING_NOTE,
    BN_BAND_GAP_ALIGNMENT_RANKING_NOTE,
    BN_SUPPORT_RANKING_NOTE,
    DOMAIN_SUPPORT_RANKING_NOTE,
    GROUPED_ROBUSTNESS_UNCERTAINTY_RANKING_NOTE,
    NOVELTY_ANNOTATION_RANKING_NOTE,
    NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION,
    NOVELTY_BUCKET_HELD_OUT_KNOWN_FORMULA,
    NOVELTY_BUCKET_TRAIN_PLUS_VAL_REDISCOVERY,
)
from materials.feature_building import (
    feature_set_supports_formula_only_screening,
    get_feature_family,
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
BN_FAMILY_BENCHMARK_COLUMNS = [
    'feature_set',
    'feature_family',
    'model_type',
    'benchmark_role',
    'benchmark_status',
    'bn_family_benchmark_method',
    'bn_family_grouping_method',
    'bn_family_train_scope',
    'bn_family_count',
    'bn_formula_count',
    'bn_row_count',
    'completed_family_holds',
    'completed_formula_holds',
    'k_neighbors',
    'mae',
    'rmse',
    'r2',
]
BN_STRATIFIED_ERROR_COLUMNS = [
    'feature_set',
    'feature_family',
    'model_type',
    'benchmark_role',
    'benchmark_status',
    'bn_stratified_error_method',
    'bn_stratified_group_column',
    'requested_folds',
    'actual_folds',
    'completed_folds',
    'bn_formula_count',
    'non_bn_formula_count',
    'bn_mae',
    'bn_rmse',
    'bn_r2',
    'non_bn_mae',
    'non_bn_rmse',
    'non_bn_r2',
    'bn_to_non_bn_mae_ratio',
]

STRUCTURE_GENERATION_JOB_PLAN_LABEL = 'prototype_substitution_enumeration_job_plan'
STRUCTURE_GENERATION_JOB_PLAN_METHOD = 'candidate_seed_to_workflow_plan'
STRUCTURE_GENERATION_JOB_PLAN_NOTE = (
    'Converts each candidate/seed pairing into a deterministic downstream workflow plan '
    'for prototype substitution, enumeration, and relaxation. This is a planning artifact, '
    'not a generated-structure or validated-structure claim.'
)
STRUCTURE_GENERATION_FIRST_PASS_QUEUE_LABEL = 'prototype_edit_recipe_first_pass_queue'
STRUCTURE_GENERATION_FIRST_PASS_QUEUE_METHOD = 'job_plan_complexity_ranked_queue'
STRUCTURE_GENERATION_FIRST_PASS_QUEUE_NOTE = (
    'Builds a deterministic first-pass queue for downstream prototype work by combining '
    'candidate ranking context, seed rank, edit complexity, and reference readiness. '
    'This is still a planning artifact, not generated-structure output.'
)
STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_LABEL = 'prototype_grounded_followup_shortlist'
STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_METHOD = 'first_pass_queue_candidate_aggregation'
STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_NOTE = (
    'Aggregates the structure-generation first-pass queue back to the candidate level so '
    'advisor-facing follow-up can prioritize formulas that are not only high-ranked, but '
    'also better supported by prototype references and lower-complexity edit paths. This '
    'remains a planning artifact, not a validated structure result.'
)
STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_LABEL = (
    'novelty_aware_prototype_followup_shortlist'
)
STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_METHOD = (
    'structure_followup_filtered_by_formula_level_extrapolation'
)
STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_NOTE = (
    'Filters the prototype-grounded follow-up view down to formula-level extrapolation '
    'candidates so downstream structure work is not dominated by rediscovery or replay of '
    'already-observed BN formulas. This remains a planning artifact, not a discovery proof.'
)


def _structure_followup_shortlist_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    shortlist_cfg = screening_cfg.get('structure_followup_shortlist', {})
    out = {
        'enabled': bool(shortlist_cfg.get('enabled', True)),
        'label': str(shortlist_cfg.get('label', STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_LABEL)),
        'method': str(shortlist_cfg.get('method', STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_METHOD)),
        'shortlist_size': int(shortlist_cfg.get('shortlist_size', 5)),
        'note': str(shortlist_cfg.get('note', STRUCTURE_GENERATION_FOLLOWUP_SHORTLIST_NOTE)),
    }
    if out['shortlist_size'] <= 0:
        raise ValueError('structure_followup_shortlist.shortlist_size must be positive')
    return out


def _structure_followup_extrapolation_shortlist_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    shortlist_cfg = screening_cfg.get('structure_followup_extrapolation_shortlist', {})
    out = {
        'enabled': bool(shortlist_cfg.get('enabled', True)),
        'label': str(
            shortlist_cfg.get(
                'label', STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_LABEL
            )
        ),
        'method': str(
            shortlist_cfg.get(
                'method', STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_METHOD
            )
        ),
        'shortlist_size': int(shortlist_cfg.get('shortlist_size', 4)),
        'required_novelty_bucket': str(
            shortlist_cfg.get(
                'required_novelty_bucket', NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION
            )
        ),
        'note': str(
            shortlist_cfg.get(
                'note', STRUCTURE_GENERATION_FOLLOWUP_EXTRAPOLATION_SHORTLIST_NOTE
            )
        ),
    }
    if out['shortlist_size'] <= 0:
        raise ValueError('structure_followup_extrapolation_shortlist.shortlist_size must be positive')
    return out


def _ranking_stability_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    stability_cfg = screening_cfg.get('ranking_stability', {})
    top_k_values = [int(value) for value in stability_cfg.get('top_k_values', [3, 5, 10])]
    top_k_values = sorted({value for value in top_k_values if value > 0})
    if not top_k_values:
        raise ValueError('ranking_stability.top_k_values must contain at least one positive integer')
    lower_quantile = float(stability_cfg.get('prediction_interval_lower_quantile', 0.1))
    upper_quantile = float(stability_cfg.get('prediction_interval_upper_quantile', 0.9))
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError('ranking_stability prediction interval quantiles must satisfy 0 <= lower < upper <= 1')
    return {
        'enabled': bool(stability_cfg.get('enabled', True)),
        'top_k_values': top_k_values,
        'prediction_interval_lower_quantile': lower_quantile,
        'prediction_interval_upper_quantile': upper_quantile,
        'note': str(
            stability_cfg.get(
                'note',
                'Summarizes prediction and rank stability across candidate-compatible full-fit '
                'models plus grouped-fold ranking views. This is a ranking-honesty layer, not a '
                'calibrated discovery-confidence estimate.'
            )
        ),
    }



def _decision_policy_config(cfg: dict | None = None) -> dict[str, object]:
    screening_cfg = {} if cfg is None else cfg.get('screening', {})
    policy_cfg = screening_cfg.get('decision_policy', {})
    return {
        'enabled': bool(policy_cfg.get('enabled', True)),
        'global_support_abstain_below_percentile': float(
            policy_cfg.get('global_support_abstain_below_percentile', 25.0)
        ),
        'bn_support_abstain_below_percentile': float(
            policy_cfg.get('bn_support_abstain_below_percentile', 25.0)
        ),
        'prediction_std_above_quantile': float(
            policy_cfg.get('prediction_std_above_quantile', 0.75)
        ),
        'rank_std_above_quantile': float(policy_cfg.get('rank_std_above_quantile', 0.75)),
        'minimum_top_10_selection_frequency': float(
            policy_cfg.get('minimum_top_10_selection_frequency', 0.5)
        ),
        'note': str(
            policy_cfg.get(
                'note',
                'Turns candidate ranking into a lightweight decision policy by combining '
                'chemical plausibility, domain support, BN-local support, prediction/rank '
                'instability, and prototype readiness into abstention flags plus action labels. '
                'This remains heuristic and should not be interpreted as validated discovery '
                'confidence.'
            )
        ),
    }

