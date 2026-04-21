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
from materials.candidate_space import _structure_generation_seed_config
from materials.feature_building import *
from materials.benchmarking import *
from materials.common import *
from materials.common import (
    _structure_followup_extrapolation_shortlist_config,
    _structure_followup_shortlist_config,
)
from materials.ranking_tables import *
from materials.ranking_tables import (
    _build_bn_candidate_compatible_evaluation_table,
    _build_bn_evaluation_matrix_table,
    _build_bn_model_role_comparison_table,
    _candidate_ranking_comparison_payload,
    _candidate_ranking_uncertainty_table,
)
from materials.structure_artifacts import *
from materials.structure_artifacts import (
    _build_structure_generation_first_pass_queue_payload,
    _build_structure_generation_followup_extrapolation_shortlist_df,
    _build_structure_generation_followup_shortlist_df,
    _build_structure_generation_handoff_payload,
    _build_structure_generation_job_plan_payload,
    _build_structure_generation_reference_record_payload,
)
from materials.summary import *

def save_metrics_and_predictions(
    metrics,
    prediction_df,
    bn_df,
    screened_df,
    benchmark_df,
    robustness_df,
    bn_slice_benchmark_df,
    bn_slice_prediction_df,
    bn_centered_screened_df,
    structure_generation_seed_df,
    experiment_summary,
    manifest,
    cfg,
    candidate_prediction_member_df=None,
    candidate_grouped_robustness_member_df=None,
    bn_centered_grouped_robustness_member_df=None,
    structure_first_pass_execution_variant_df=None,
    structure_first_pass_execution_summary_df=None,
    structure_first_pass_execution_payload=None,
    bn_family_benchmark_df=None,
    bn_family_prediction_df=None,
    bn_stratified_error_df=None,
):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    formula_col = ((cfg.get('data') or {}).get('formula_column') or 'formula')
    bn_family_benchmark_df = (
        pd.DataFrame() if bn_family_benchmark_df is None else bn_family_benchmark_df.copy()
    )
    bn_family_prediction_df = (
        pd.DataFrame() if bn_family_prediction_df is None else bn_family_prediction_df.copy()
    )
    bn_stratified_error_df = (
        pd.DataFrame() if bn_stratified_error_df is None else bn_stratified_error_df.copy()
    )
    bn_centered_screened_df = (
        pd.DataFrame() if bn_centered_screened_df is None else bn_centered_screened_df.copy()
    )
    structure_first_pass_execution_variant_df = (
        pd.DataFrame()
        if structure_first_pass_execution_variant_df is None
        else structure_first_pass_execution_variant_df.copy()
    )
    structure_first_pass_execution_summary_df = (
        pd.DataFrame()
        if structure_first_pass_execution_summary_df is None
        else structure_first_pass_execution_summary_df.copy()
    )
    structure_first_pass_execution_payload = dict(structure_first_pass_execution_payload or {})
    write_json_file(metrics, artifact_dir / 'metrics.json', indent=2)
    prediction_df.to_csv(artifact_dir / 'predictions.csv', index=False)
    bn_df.to_csv(artifact_dir / 'bn_slice.csv', index=False)
    screened_df.to_csv(artifact_dir / 'demo_candidate_ranking.csv', index=False)
    candidate_uncertainty_path = artifact_dir / 'demo_candidate_ranking_uncertainty.csv'
    bn_candidate_compatible_evaluation_path = artifact_dir / 'bn_candidate_compatible_evaluation.csv'
    bn_family_benchmark_path = artifact_dir / 'bn_family_benchmark_results.csv'
    bn_family_prediction_path = artifact_dir / 'bn_family_predictions.csv'
    bn_stratified_error_path = artifact_dir / 'bn_stratified_error_results.csv'
    bn_evaluation_matrix_path = artifact_dir / 'bn_evaluation_matrix.csv'
    bn_model_role_comparison_path = artifact_dir / 'bn_model_role_comparison.csv'
    bn_centered_ranking_path = artifact_dir / 'demo_candidate_bn_centered_ranking.csv'
    candidate_rank_stability_summary_path = (
        artifact_dir / 'demo_candidate_rank_stability_summary.csv'
    )
    demo_candidate_structure_followup_report_path = (
        artifact_dir / 'demo_candidate_structure_followup_report.csv'
    )
    if bn_centered_screened_df is not None and not bn_centered_screened_df.empty:
        bn_centered_screened_df.to_csv(bn_centered_ranking_path, index=False)
    elif bn_centered_ranking_path.exists():
        bn_centered_ranking_path.unlink()
    structure_generation_seed_path = artifact_dir / 'demo_candidate_structure_generation_seeds.csv'
    structure_generation_handoff_path = artifact_dir / 'demo_candidate_structure_generation_handoff.json'
    structure_generation_reference_records_path = (
        artifact_dir / 'demo_candidate_structure_generation_reference_records.json'
    )
    structure_generation_job_plan_path = (
        artifact_dir / 'demo_candidate_structure_generation_job_plan.json'
    )
    structure_generation_first_pass_queue_path = (
        artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json'
    )
    structure_generation_followup_shortlist_path = (
        artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv'
    )
    structure_generation_followup_extrapolation_shortlist_path = (
        artifact_dir / 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'
    )
    structure_first_pass_execution_path = None
    structure_first_pass_execution_summary_path = None
    structure_first_pass_execution_variants_path = None
    structure_first_pass_execution_structure_dir = None
    if structure_first_pass_execution_payload:
        structure_first_pass_execution_path = artifact_dir / str(
            structure_first_pass_execution_payload.get('artifact')
        )
        structure_first_pass_execution_summary_path = artifact_dir / str(
            structure_first_pass_execution_payload.get('summary_artifact')
        )
        structure_first_pass_execution_variants_path = artifact_dir / str(
            structure_first_pass_execution_payload.get('variants_artifact')
        )
        structure_first_pass_execution_structure_dir = artifact_dir / str(
            structure_first_pass_execution_payload.get('structure_dir')
        )
    selected_followup_df = pd.DataFrame()
    if structure_generation_seed_df is not None and not structure_generation_seed_df.empty:
        structure_generation_seed_df.to_csv(structure_generation_seed_path, index=False)
        structure_generation_seed_cfg = _structure_generation_seed_config(cfg)
        structure_generation_handoff = _build_structure_generation_handoff_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        write_json_file(
            structure_generation_handoff,
            structure_generation_handoff_path,
            indent=2,
            ensure_ascii=False,
        )
        structure_generation_reference_records = _build_structure_generation_reference_record_payload(
            structure_generation_seed_df,
            cfg=cfg,
        )
        write_json_file(
            structure_generation_reference_records,
            structure_generation_reference_records_path,
            indent=2,
            ensure_ascii=False,
        )
        structure_generation_job_plan = _build_structure_generation_job_plan_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        write_json_file(
            structure_generation_job_plan,
            structure_generation_job_plan_path,
            indent=2,
            ensure_ascii=False,
        )
        structure_generation_first_pass_queue = _build_structure_generation_first_pass_queue_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        write_json_file(
            structure_generation_first_pass_queue,
            structure_generation_first_pass_queue_path,
            indent=2,
            ensure_ascii=False,
        )
        structure_followup_shortlist_cfg = _structure_followup_shortlist_config(cfg)
        structure_followup_shortlist_df = _build_structure_generation_followup_shortlist_df(
            structure_generation_first_pass_queue,
            formula_col=formula_col,
            cfg_defaults=structure_followup_shortlist_cfg,
        )
        selected_followup_df = (
            structure_followup_shortlist_df.loc[
                structure_followup_shortlist_df['structure_followup_shortlist_selected']
                .fillna(False)
                .astype(bool)
            ].copy()
            if not structure_followup_shortlist_df.empty
            else pd.DataFrame()
        )
        if not selected_followup_df.empty:
            if 'structure_followup_shortlist_rank' in selected_followup_df.columns:
                selected_followup_df = selected_followup_df.sort_values(
                    'structure_followup_shortlist_rank', ascending=True
                )
            selected_followup_df.to_csv(structure_generation_followup_shortlist_path, index=False)
        elif structure_generation_followup_shortlist_path.exists():
            structure_generation_followup_shortlist_path.unlink()
        structure_followup_extrapolation_shortlist_cfg = (
            _structure_followup_extrapolation_shortlist_config(cfg)
        )
        structure_followup_extrapolation_shortlist_df = (
            _build_structure_generation_followup_extrapolation_shortlist_df(
                structure_followup_shortlist_df,
                formula_col=formula_col,
                cfg_defaults=structure_followup_extrapolation_shortlist_cfg,
            )
        )
        selected_followup_extrapolation_df = (
            structure_followup_extrapolation_shortlist_df.loc[
                structure_followup_extrapolation_shortlist_df[
                    'structure_followup_extrapolation_shortlist_selected'
                ].fillna(False).astype(bool)
            ].copy()
            if not structure_followup_extrapolation_shortlist_df.empty
            else pd.DataFrame()
        )
        if not selected_followup_extrapolation_df.empty:
            if 'structure_followup_extrapolation_shortlist_rank' in selected_followup_extrapolation_df.columns:
                selected_followup_extrapolation_df = selected_followup_extrapolation_df.sort_values(
                    'structure_followup_extrapolation_shortlist_rank', ascending=True
                )
            selected_followup_extrapolation_df.to_csv(
                structure_generation_followup_extrapolation_shortlist_path, index=False
            )
        elif structure_generation_followup_extrapolation_shortlist_path.exists():
            structure_generation_followup_extrapolation_shortlist_path.unlink()
    else:
        if structure_generation_seed_path.exists():
            structure_generation_seed_path.unlink()
        if structure_generation_handoff_path.exists():
            structure_generation_handoff_path.unlink()
        if structure_generation_reference_records_path.exists():
            structure_generation_reference_records_path.unlink()
        if structure_generation_job_plan_path.exists():
            structure_generation_job_plan_path.unlink()
        if structure_generation_first_pass_queue_path.exists():
            structure_generation_first_pass_queue_path.unlink()
        if structure_generation_followup_shortlist_path.exists():
            structure_generation_followup_shortlist_path.unlink()
        if structure_generation_followup_extrapolation_shortlist_path.exists():
            structure_generation_followup_extrapolation_shortlist_path.unlink()
    if (
        structure_first_pass_execution_payload
        and structure_first_pass_execution_summary_path is not None
        and structure_first_pass_execution_variants_path is not None
        and structure_first_pass_execution_path is not None
        and not structure_first_pass_execution_summary_df.empty
    ):
        structure_first_pass_execution_summary_df.to_csv(
            structure_first_pass_execution_summary_path,
            index=False,
        )
        structure_followup_report_df = structure_first_pass_execution_summary_df.copy()
        if 'formula' not in structure_followup_report_df.columns:
            structure_followup_report_df['formula'] = (
                structure_followup_report_df[formula_col]
                if formula_col in structure_followup_report_df.columns
                else pd.NA
            )
        for column in (
            'structure_followup_shortlist_rank',
            'structure_followup_best_action_label',
            'structure_followup_best_seed_reference_formula',
            'structure_followup_best_seed_reference_record_id',
            'first_pass_execution_variant_count',
            'first_pass_execution_geometry_pass_variant_count',
            'first_pass_execution_selected_variant_id',
            'first_pass_execution_selected_cif_path',
            'first_pass_execution_selected_band_gap_proxy',
            'first_pass_execution_selected_min_distance_ratio',
            'first_pass_execution_selected_relaxation_status',
            'first_pass_execution_selected_final_status',
        ):
            if column not in structure_followup_report_df.columns:
                structure_followup_report_df[column] = pd.NA
        structure_followup_report_df.to_csv(
            demo_candidate_structure_followup_report_path,
            index=False,
        )
        structure_first_pass_execution_variant_df.to_csv(
            structure_first_pass_execution_variants_path,
            index=False,
        )
        if structure_first_pass_execution_structure_dir is not None:
            structure_first_pass_execution_structure_dir.mkdir(parents=True, exist_ok=True)
            for existing_cif_path in structure_first_pass_execution_structure_dir.glob('*.cif'):
                existing_cif_path.unlink()
        sanitized_candidates = []
        for candidate_payload in structure_first_pass_execution_payload.get('candidates', []):
            sanitized_candidate = {
                key: value
                for key, value in candidate_payload.items()
                if key != 'variants'
            }
            sanitized_variants = []
            for variant_payload in candidate_payload.get('variants', []):
                cif_text = variant_payload.get('_cif_text')
                cif_relative_path = variant_payload.get('generated_structure_cif_path')
                if (
                    cif_text
                    and cif_relative_path
                    and structure_first_pass_execution_structure_dir is not None
                ):
                    cif_output_path = artifact_dir / str(cif_relative_path)
                    cif_output_path.parent.mkdir(parents=True, exist_ok=True)
                    cif_output_path.write_text(str(cif_text), encoding='utf-8')
                sanitized_variants.append(
                    {
                        key: value
                        for key, value in variant_payload.items()
                        if key != '_cif_text'
                    }
                )
            sanitized_candidate['variants'] = sanitized_variants
            sanitized_candidates.append(sanitized_candidate)
        sanitized_payload = {
            **structure_first_pass_execution_payload,
            'candidates': sanitized_candidates,
        }
        write_json_file(
            sanitized_payload,
            structure_first_pass_execution_path,
            indent=2,
            ensure_ascii=False,
        )
    else:
        if demo_candidate_structure_followup_report_path.exists():
            demo_candidate_structure_followup_report_path.unlink()
        for cleanup_path in (
            structure_first_pass_execution_summary_path,
            structure_first_pass_execution_variants_path,
            structure_first_pass_execution_path,
        ):
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink()
        if structure_first_pass_execution_structure_dir is not None and structure_first_pass_execution_structure_dir.exists():
            for existing_cif_path in structure_first_pass_execution_structure_dir.glob('*.cif'):
                existing_cif_path.unlink()
    for selected_column, rank_column, artifact_name in (
        (
            'proposal_shortlist_selected',
            'proposal_shortlist_rank',
            'demo_candidate_proposal_shortlist.csv',
        ),
        (
            'extrapolation_shortlist_selected',
            'extrapolation_shortlist_rank',
            'demo_candidate_extrapolation_shortlist.csv',
        ),
    ):
        shortlist_path = artifact_dir / artifact_name
        if selected_column in screened_df.columns:
            shortlist_df = screened_df.loc[
                screened_df[selected_column].fillna(False).astype(bool)
            ].copy()
            if rank_column in shortlist_df.columns:
                shortlist_df = shortlist_df.sort_values(rank_column, ascending=True)
            shortlist_df.to_csv(shortlist_path, index=False)
        elif shortlist_path.exists():
            shortlist_path.unlink()
    bn_candidate_compatible_evaluation_df = _build_bn_candidate_compatible_evaluation_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df=bn_family_benchmark_df,
        bn_stratified_error_df=bn_stratified_error_df,
    )
    if not bn_candidate_compatible_evaluation_df.empty:
        bn_candidate_compatible_evaluation_df.to_csv(
            bn_candidate_compatible_evaluation_path,
            index=False,
        )
    elif bn_candidate_compatible_evaluation_path.exists():
        bn_candidate_compatible_evaluation_path.unlink()

    if bn_family_benchmark_df is not None and not bn_family_benchmark_df.empty:
        bn_family_benchmark_df.to_csv(bn_family_benchmark_path, index=False)
    elif bn_family_benchmark_path.exists():
        bn_family_benchmark_path.unlink()
    if bn_family_prediction_df is not None and not bn_family_prediction_df.empty:
        bn_family_prediction_df.to_csv(bn_family_prediction_path, index=False)
    elif bn_family_prediction_path.exists():
        bn_family_prediction_path.unlink()
    if bn_stratified_error_df is not None and not bn_stratified_error_df.empty:
        bn_stratified_error_df.to_csv(bn_stratified_error_path, index=False)
    elif bn_stratified_error_path.exists():
        bn_stratified_error_path.unlink()

    bn_evaluation_matrix_df = _build_bn_evaluation_matrix_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df,
        bn_stratified_error_df,
    )
    if not bn_evaluation_matrix_df.empty:
        bn_evaluation_matrix_df.to_csv(bn_evaluation_matrix_path, index=False)
    elif bn_evaluation_matrix_path.exists():
        bn_evaluation_matrix_path.unlink()

    candidate_ranking_uncertainty_df, _ = _candidate_ranking_uncertainty_table(
        screened_df,
        formula_col=((cfg.get('data') or {}).get('formula_column') or 'formula'),
        cfg=cfg,
        candidate_prediction_member_df=candidate_prediction_member_df,
        candidate_grouped_robustness_member_df=candidate_grouped_robustness_member_df,
        bn_centered_grouped_robustness_member_df=bn_centered_grouped_robustness_member_df,
        bn_centered_candidate_df=bn_centered_screened_df,
        structure_followup_shortlist_df=selected_followup_df,
    )
    if not candidate_ranking_uncertainty_df.empty:
        candidate_ranking_uncertainty_df.to_csv(candidate_uncertainty_path, index=False)
    elif candidate_uncertainty_path.exists():
        candidate_uncertainty_path.unlink()

    bn_slice_benchmark_for_model_role_df = bn_slice_benchmark_df.copy()
    if 'selected_by_validation' not in bn_slice_benchmark_for_model_role_df.columns:
        bn_slice_benchmark_for_model_role_df['selected_by_validation'] = pd.NA
    bn_model_role_comparison_df = _build_bn_model_role_comparison_table(
        bn_slice_benchmark_for_model_role_df,
        bn_family_benchmark_df=bn_family_benchmark_df,
        bn_stratified_error_df=bn_stratified_error_df,
    )
    if not bn_model_role_comparison_df.empty:
        bn_model_role_comparison_df.to_csv(
            bn_model_role_comparison_path,
            index=False,
        )
    elif bn_model_role_comparison_path.exists():
        bn_model_role_comparison_path.unlink()

    candidate_rank_stability_summary_df = pd.DataFrame(
        [
            _candidate_ranking_comparison_payload(
                screened_df,
                bn_centered_screened_df,
                formula_col=formula_col,
                top_k=top_k,
            )
            for top_k in [3, 5, 10, 20]
        ]
    )
    if not candidate_rank_stability_summary_df.empty:
        candidate_rank_stability_summary_df.to_csv(
            candidate_rank_stability_summary_path,
            index=False,
        )
    elif candidate_rank_stability_summary_path.exists():
        candidate_rank_stability_summary_path.unlink()

    benchmark_df.to_csv(artifact_dir / 'benchmark_results.csv', index=False)
    robustness_path = artifact_dir / 'robustness_results.csv'
    if robustness_df is not None and not robustness_df.empty:
        robustness_df.to_csv(robustness_path, index=False)
    elif robustness_path.exists():
        robustness_path.unlink()
    bn_slice_benchmark_path = artifact_dir / 'bn_slice_benchmark_results.csv'
    if bn_slice_benchmark_df is not None and not bn_slice_benchmark_df.empty:
        bn_slice_benchmark_df.to_csv(bn_slice_benchmark_path, index=False)
    elif bn_slice_benchmark_path.exists():
        bn_slice_benchmark_path.unlink()
    bn_slice_prediction_path = artifact_dir / 'bn_slice_predictions.csv'
    if bn_slice_prediction_df is not None and not bn_slice_prediction_df.empty:
        bn_slice_prediction_df.to_csv(bn_slice_prediction_path, index=False)
    elif bn_slice_prediction_path.exists():
        bn_slice_prediction_path.unlink()
    write_json_file(
        experiment_summary,
        artifact_dir / 'experiment_summary.json',
        indent=2,
        ensure_ascii=False,
    )
    write_json_file(manifest, artifact_dir / 'manifest.json', indent=2)
    legacy_screen_path = artifact_dir / 'screened_candidates.csv'
    if legacy_screen_path.exists():
        legacy_screen_path.unlink()
