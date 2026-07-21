from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np
import pandas as pd

from runtime.io_utils import (
    build_artifact_provenance,
    make_json_safe,
    validate_json_payload,
    validate_runtime_output_path,
    write_json_file,
)
from runtime.schema import (
    FIXED_REPORT_ARTIFACT_NAMES as _RESERVED_REPORT_ARTIFACT_NAMES,
    STRUCTURE_EXECUTION_OUTPUT_ROLES,
)
from materials.data import STRUCTURE_SUMMARY_COLUMNS, load_cached_raw_record_lookup
from materials.constants import *
from materials.candidate_space import *
from materials.candidate_space import (
    _extrapolation_shortlist_config,
    _proposal_shortlist_config,
    _structure_generation_seed_config,
)
from materials.feature_building import *
from materials.benchmarking import *
from materials.common import *
from materials.common import (
    _decision_policy_config,
    _ranking_stability_config,
    _resolve_artifact_path,
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
from materials.structure_helpers import (
    _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH,
    _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS,
    _STRUCTURE_EXECUTION_VARIANT_SELECTION_FIELDS,
    _STRUCTURE_EXECUTION_ZERO_VARIANT_STATUSES,
    _select_structure_execution_variant,
    _structure_first_pass_execution_config,
    _structure_execution_variant_expected_state,
)
from materials.summary import *


def _write_csv_file(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = validate_runtime_output_path(
        path,
        reject_leaf_symlink=True,
        expected_output_kind='file',
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='',
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            frame.to_csv(temporary_file, index=False)
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _canonical_path_parts(path: Path) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize('NFC', part).casefold()
        for part in path.parts
    )


def _same_or_descendant_path(path: Path, parent: Path) -> bool:
    path_parts = _canonical_path_parts(path)
    parent_parts = _canonical_path_parts(parent)
    return (
        len(path_parts) >= len(parent_parts)
        and path_parts[:len(parent_parts)] == parent_parts
    )


def _same_existing_path(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _reject_hardlinked_output(field_name: str, output_path: Path) -> None:
    try:
        has_multiple_links = output_path.is_file() and output_path.stat().st_nlink > 1
    except OSError:
        has_multiple_links = False
    if has_multiple_links:
        raise ValueError(
            f'{field_name} must not target a file with multiple hard links'
        )


def _resolve_and_validate_artifact_output_path(
    artifact_dir: Path,
    value: object,
    *,
    field_name: str,
    expected_output_kind: str,
    required_parent_path: Path | None = None,
) -> Path:
    resolved_path = _resolve_artifact_path(
        artifact_dir,
        value,
        field_name=field_name,
    )
    declared_path = artifact_dir / Path(str(value).strip())
    validate_runtime_output_path(
        declared_path,
        required_parent_path=(
            artifact_dir if required_parent_path is None else required_parent_path
        ),
        expected_output_kind=expected_output_kind,
    )
    return resolved_path


def _validate_structure_execution_output_paths(
    artifact_dir: Path,
    output_paths: dict[str, Path],
) -> None:
    expected_suffixes = {
        config_field: suffix
        for _artifact_key, _summary_field, config_field, suffix, _default_path
        in STRUCTURE_EXECUTION_OUTPUT_ROLES
    }
    for field_name, expected_suffix in expected_suffixes.items():
        if output_paths[field_name].suffix.lower() != expected_suffix:
            raise ValueError(
                f'structure_first_pass_execution.{field_name} must use a '
                f'{expected_suffix} file path'
            )

    file_fields = tuple(expected_suffixes)
    for index, field_name in enumerate(file_fields):
        file_path = output_paths[field_name]
        _reject_hardlinked_output(
            f'structure_first_pass_execution.{field_name}',
            file_path,
        )
        for other_field_name in file_fields[index + 1:]:
            other_path = output_paths[other_field_name]
            if (
                _same_or_descendant_path(file_path, other_path)
                or _same_or_descendant_path(other_path, file_path)
                or _same_existing_path(file_path, other_path)
            ):
                raise ValueError(
                    'structure_first_pass_execution output file paths must not collide or '
                    'contain one another'
                )

    structure_dir = output_paths['structure_dir']
    for field_name in file_fields:
        file_path = output_paths[field_name]
        if _same_or_descendant_path(structure_dir, file_path):
            raise ValueError(
                'structure_first_pass_execution.structure_dir must not equal or be nested '
                'beneath a configured output file path'
            )

    artifact_root = artifact_dir.resolve(strict=False)
    reserved_paths = {
        (artifact_root / artifact_name).resolve(strict=False)
        for artifact_name in _RESERVED_REPORT_ARTIFACT_NAMES
    }
    role_default_paths = {
        config_field: (artifact_root / default_path).resolve(strict=False)
        for _artifact_key, _summary_field, config_field, _suffix, default_path
        in STRUCTURE_EXECUTION_OUTPUT_ROLES
    }
    for field_name, output_path in output_paths.items():
        if field_name in role_default_paths:
            for other_field_name, other_default_path in role_default_paths.items():
                if (
                    other_field_name != field_name
                    and (
                        _canonical_path_parts(output_path)
                        == _canonical_path_parts(other_default_path)
                        or _same_existing_path(output_path, other_default_path)
                    )
                ):
                    raise ValueError(
                        f'structure_first_pass_execution.{field_name} collides with '
                        f'the canonical {other_field_name} output role'
                    )
        for reserved_path in reserved_paths:
            collides_with_reserved_file = (
                _same_or_descendant_path(output_path, reserved_path)
                or _same_existing_path(output_path, reserved_path)
                or (
                    field_name != 'structure_dir'
                    and _same_or_descendant_path(reserved_path, output_path)
                )
            )
            if collides_with_reserved_file:
                raise ValueError(
                    f'structure_first_pass_execution.{field_name} collides with a reserved '
                    'report artifact path'
                )


def _validate_experiment_summary_output_contract(
    artifact_dir: Path,
    experiment_summary: object,
    configured_execution_paths: dict[str, Path],
    *,
    execution_outputs_will_publish: bool,
) -> None:
    if not isinstance(experiment_summary, dict):
        raise ValueError('experiment_summary must be an object')
    screening = experiment_summary.get('screening')
    if screening is None:
        return
    if not isinstance(screening, dict):
        raise ValueError('experiment_summary.screening must be an object or null')
    bridge = screening.get('structure_generation_bridge')
    if bridge is None:
        return
    if not isinstance(bridge, dict):
        raise ValueError(
            'experiment_summary.screening.structure_generation_bridge '
            'must be an object or null'
        )

    for _artifact_key, summary_field, path_field, expected_suffix, _default_path in (
        STRUCTURE_EXECUTION_OUTPUT_ROLES
    ):
        if summary_field not in bridge:
            continue
        if not execution_outputs_will_publish:
            raise ValueError(
                f'experiment_summary {summary_field} may be declared only when '
                'structure execution outputs will be published'
            )
        declared_value = bridge[summary_field]
        if not isinstance(declared_value, (str, Path)) or not str(declared_value).strip():
            raise ValueError(
                f'experiment_summary {summary_field} must be a non-empty path string'
            )
        try:
            declared_path = _resolve_and_validate_artifact_output_path(
                artifact_dir,
                declared_value,
                field_name=f'experiment_summary.{summary_field}',
                expected_output_kind='file',
            )
        except ValueError as exc:
            raise ValueError(
                f'experiment_summary {summary_field} is invalid: {exc}'
            ) from exc
        configured_path = configured_execution_paths[path_field]
        if not (
            declared_path == configured_path
            or _same_existing_path(declared_path, configured_path)
        ):
            raise ValueError(
                f'experiment_summary {summary_field} must identify the configured '
                f'structure_first_pass_execution.{path_field} path'
            )
        if declared_path.suffix.casefold() != expected_suffix:
            raise ValueError(
                f'experiment_summary {summary_field} must use the configured file type'
            )


def _validate_structure_execution_frame_roles(
    summary_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    payload: dict[str, object],
    *,
    formula_col: str,
    geometry_min_distance_ratio_pass_threshold: float,
    geometry_min_distance_ratio_overlap_threshold: float,
) -> None:
    def normalized(value):
        return make_json_safe(value)

    def reject(field_name, detail='must match the canonical summary and variants frames'):
        raise ValueError(f'structure_first_pass_execution {field_name} {detail}')

    def text(value, field_name, *, nullable=False):
        value = normalized(value)
        if nullable and value is None:
            return None
        if not isinstance(value, str) or not value:
            reject(field_name, 'must be a non-empty string')
        return value

    def count(value, field_name):
        value = normalized(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            reject(field_name, 'must be a non-negative integer')
        return value

    def number(value, field_name, *, nullable=False):
        value = normalized(value)
        if nullable and value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reject(field_name, 'must be numeric')
        return value

    selected_summary_fields = tuple(
        summary_field
        for summary_field, _variant_field
        in _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
    )
    selected_variant_fields = tuple(
        variant_field
        for _summary_field, variant_field
        in _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
    )
    variant_fields = tuple(dict.fromkeys((
        formula_col,
        'execution_status',
        'execution_message',
        'relabeled_site_count',
        'removed_site_count',
        'geometry_mean_distance',
        'geometry_overlap_pair_count',
        'geometry_sanity_pass',
        *selected_variant_fields,
        *_STRUCTURE_EXECUTION_VARIANT_SELECTION_FIELDS,
        *STRUCTURE_SUMMARY_COLUMNS,
    )))
    summary_columns = {
        formula_col,
        'first_pass_execution_variant_count',
        'first_pass_execution_successful_variant_count',
        'first_pass_execution_geometry_pass_variant_count',
        'first_pass_execution_status',
        *selected_summary_fields,
    }
    variant_columns = set(variant_fields)
    for role_name, frame, required in (
        ('structure_first_pass_execution_summary_df', summary_df, summary_columns),
        ('structure_first_pass_execution_variant_df', variant_df, variant_columns),
    ):
        missing = required - set(frame.columns)
        if not frame.empty and missing:
            raise ValueError(
                f'{role_name} must contain canonical structure-execution columns: '
                f'{sorted(missing)}'
            )

    payload_fields = {
        'enabled', 'candidate_count', 'variant_count', 'successful_variant_count',
        'status_counts', 'executed_formulas', 'candidates',
    }
    missing = payload_fields - set(payload)
    if missing:
        reject('payload', f'is missing canonical relational fields: {sorted(missing)}')
    if normalized(payload['enabled']) is not True:
        reject('enabled state')

    def index_records(records, key, label):
        indexed = {}
        for record in records:
            if not isinstance(record, dict):
                reject(label, 'entries must be objects')
            identity = text(record.get(key), f'{label} {key}')
            if identity in indexed:
                reject(label, f'{key} values must be unique')
            indexed[identity] = record
        return indexed

    summary_records = summary_df.to_dict(orient='records')
    variant_records = variant_df.to_dict(orient='records')
    summaries = index_records(summary_records, formula_col, 'summary')
    variants = index_records(variant_records, 'execution_variant_id', 'variant')
    for row in variants.values():
        formula = text(row.get(formula_col), f'variant {formula_col}')
        if formula not in summaries:
            reject('variant candidate membership')
        text(row.get('execution_status'), 'variant execution_status')
        text(row.get('final_status'), 'variant final_status')
        if not isinstance(normalized(row.get('geometry_sanity_pass')), bool):
            reject('variant geometry_sanity_pass', 'must be boolean')
        if not isinstance(normalized(row.get('formula_matches_candidate')), bool):
            reject('variant formula_matches_candidate', 'must be boolean')
        if count(row.get('execution_variant_rank'), 'variant rank') < 1:
            reject('variant rank', 'must be a positive integer')
        number(row.get('execution_variant_selection_score'), 'variant selection score')
        number(
            row.get('structure_band_gap_proxy'),
            'variant band-gap proxy',
            nullable=True,
        )

    candidate_rows = payload['candidates']
    if not isinstance(candidate_rows, list):
        reject('candidates', 'must be a list')
    candidates = index_records(candidate_rows, formula_col, 'candidate')
    payload_variants = {}
    for formula, candidate in candidates.items():
        rows = candidate.get('variants')
        if not isinstance(rows, list):
            reject(f'{formula} variants', 'must be a list')
        for variant_id, row in index_records(
            rows, 'execution_variant_id', f'{formula} payload variants'
        ).items():
            missing_variant_fields = variant_columns - set(row)
            if missing_variant_fields:
                reject(
                    f'{formula} payload variants',
                    f'are missing canonical fields: {sorted(missing_variant_fields)}',
                )
            if text(row.get(formula_col), f'payload variant {formula_col}') != formula:
                reject('payload variant candidate membership')
            if variant_id in payload_variants:
                reject('payload variants', 'execution_variant_id values must be unique')
            payload_variants[variant_id] = row

    if set(candidates) != set(summaries):
        reject('candidate membership')
    if set(payload_variants) != set(variants):
        reject('variant membership')

    statuses = [text(row.get('first_pass_execution_status'), 'summary status')
                for row in summary_records]
    executed_formulas = [
        text(row.get(formula_col), f'summary {formula_col}')
        for row, status in zip(summary_records, statuses)
        if status == _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH['executed']
    ]
    aggregate_relations = {
        'candidate_count': len(summary_records),
        'variant_count': len(variant_records),
        'successful_variant_count': sum(
            normalized(row.get('execution_status')) == 'ok' for row in variant_records
        ),
    }
    for field_name, expected in aggregate_relations.items():
        if count(payload[field_name], field_name) != expected:
            reject(field_name)
    if normalized(payload['status_counts']) != dict(Counter(statuses)):
        reject('status_counts')
    if normalized(payload['executed_formulas']) != executed_formulas:
        reject('executed_formulas')

    for variant_id, row in variants.items():
        if any(
            normalized(row.get(field)) != normalized(payload_variants[variant_id].get(field))
            for field in variant_fields
        ):
            reject(f'payload variant {variant_id}')
        expected_relaxation_status, expected_final_status = (
            _structure_execution_variant_expected_state(
                candidate_formula=row.get(formula_col),
                execution_status=row.get('execution_status'),
                execution_message=row.get('execution_message'),
                atoms=payload_variants[variant_id].get('atoms'),
                generated_formula=row.get('generated_formula'),
                formula_matches_candidate=row.get('formula_matches_candidate'),
                geometry_min_distance=row.get('geometry_min_distance'),
                geometry_mean_distance=row.get('geometry_mean_distance'),
                geometry_min_distance_ratio=row.get('geometry_min_distance_ratio'),
                geometry_overlap_pair_count=row.get('geometry_overlap_pair_count'),
                geometry_sanity_pass=row.get('geometry_sanity_pass'),
                geometry_min_distance_ratio_pass_threshold=(
                    geometry_min_distance_ratio_pass_threshold
                ),
                geometry_min_distance_ratio_overlap_threshold=(
                    geometry_min_distance_ratio_overlap_threshold
                ),
                relabeled_site_count=row.get('relabeled_site_count'),
                removed_site_count=row.get('removed_site_count'),
                generated_structure_n_sites=row.get('generated_structure_n_sites'),
                structure_summary={
                    field_name: row.get(field_name)
                    for field_name in STRUCTURE_SUMMARY_COLUMNS
                },
                cif_text=payload_variants[variant_id].get('_cif_text'),
            )
        )
        if normalized(row.get('relaxation_status')) != expected_relaxation_status:
            reject(f'variant {variant_id} relaxation_status')
        if normalized(row.get('final_status')) != expected_final_status:
            reject(f'variant {variant_id} final_status')

    for formula, summary in summaries.items():
        candidate = candidates[formula]
        candidate_variants = [
            row for row in variants.values()
            if normalized(row.get(formula_col)) == formula
        ]
        successful = [
            row for row in candidate_variants
            if normalized(row.get('execution_status')) == 'ok'
        ]
        expected_counts = {
            'first_pass_execution_variant_count': len(candidate_variants),
            'first_pass_execution_successful_variant_count': len(successful),
            'first_pass_execution_geometry_pass_variant_count': sum(
                normalized(row.get('geometry_sanity_pass')) is True for row in successful
            ),
        }
        for field_name, expected in expected_counts.items():
            if count(summary.get(field_name), field_name) != expected:
                reject(f'{formula} {field_name}')

        status = text(summary.get('first_pass_execution_status'), f'{formula} status')
        if text(candidate.get('candidate_status'), f'{formula} candidate status') != status:
            reject(f'{formula} candidate status')
        if (
            not candidate_variants
            and status not in _STRUCTURE_EXECUTION_ZERO_VARIANT_STATUSES
        ):
            reject(f'{formula} zero-variant status')
        if (
            successful
            and status
            != _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH['executed']
        ):
            reject(f'{formula} executed status')
        if (
            candidate_variants
            and not successful
            and status
            != _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH[
                'no_successful_variant'
            ]
        ):
            reject(f'{formula} unsuccessful status')

        selected_id = text(
            summary.get('first_pass_execution_selected_variant_id'),
            f'{formula} selected variant',
            nullable=True,
        )
        if text(
            candidate.get('selected_variant_id'),
            f'{formula} payload selected variant',
            nullable=True,
        ) != selected_id:
            reject(f'{formula} selected variant')
        canonical_selected = _select_structure_execution_variant(
            pd.DataFrame(candidate_variants)
        )
        if canonical_selected is not None:
            canonical_selected_id = text(
                canonical_selected.get('execution_variant_id'),
                f'{formula} canonical selected variant',
            )
            if selected_id != canonical_selected_id:
                reject(f'{formula} canonical selected variant')
            for summary_field, variant_field in (
                _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
            ):
                if normalized(summary.get(summary_field)) != normalized(
                    canonical_selected.get(variant_field)
                ):
                    reject(f'{formula} {summary_field}')
        else:
            for summary_field, variant_field in (
                _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
            ):
                selected_value = normalized(summary.get(summary_field))
                if variant_field == 'final_status':
                    if (
                        not candidate_variants
                        and status
                        == _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH[
                            'invalid_reference'
                        ]
                    ):
                        valid_unattempted_status = (
                            isinstance(selected_value, str)
                            and ': ' in selected_value
                        )
                    else:
                        valid_unattempted_status = selected_value == 'not_executed'
                    if not valid_unattempted_status:
                        reject(f'{formula} selected final status')
                elif selected_value is not None:
                    reject(
                        f'{formula} '
                        + (
                            'absent selected variant'
                            if variant_field == 'execution_variant_id'
                            else 'absent selected projection'
                        )
                    )


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
    include_parity_plot=False,
):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    artifact_dir = validate_runtime_output_path(
        artifact_dir,
        expected_output_kind='directory',
    )
    formula_col = ((cfg.get('data') or {}).get('formula_column') or 'formula')
    structure_generation_seed_cfg = _structure_generation_seed_config(cfg)
    artifact_provenance_path = artifact_dir / 'artifact_provenance.json'
    published_output_paths: set[Path] = set()

    def _publish_csv(frame: pd.DataFrame, path: str | Path) -> None:
        output_path = validate_runtime_output_path(
            path,
            reject_leaf_symlink=True,
            expected_output_kind='file',
        )
        _write_csv_file(frame, output_path)
        published_output_paths.add(output_path)

    def _publish_json(payload, path: str | Path, **kwargs) -> None:
        output_path = validate_runtime_output_path(
            path,
            reject_leaf_symlink=True,
            expected_output_kind='file',
        )
        write_json_file(payload, output_path, **kwargs)
        published_output_paths.add(output_path)
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
    structure_first_pass_execution_cfg = _structure_first_pass_execution_config(cfg)
    structure_first_pass_execution_paths = {}
    execution_path_fields = tuple(
        config_field
        for _artifact_key, _summary_field, config_field, _suffix, _default_path
        in STRUCTURE_EXECUTION_OUTPUT_ROLES
    ) + ('structure_dir',)
    for path_field in execution_path_fields:
        output_kind = 'directory' if path_field == 'structure_dir' else 'file'
        configured_path = _resolve_and_validate_artifact_output_path(
            artifact_dir,
            structure_first_pass_execution_cfg[path_field],
            field_name=f'structure_first_pass_execution.{path_field}',
            expected_output_kind=output_kind,
        )
        structure_first_pass_execution_paths[path_field] = configured_path
        if structure_first_pass_execution_payload:
            payload_path = _resolve_and_validate_artifact_output_path(
                artifact_dir,
                structure_first_pass_execution_payload.get(path_field),
                field_name=f'structure_first_pass_execution.{path_field}',
                expected_output_kind=output_kind,
            )
            if not (
                payload_path == configured_path
                or _same_existing_path(payload_path, configured_path)
            ):
                raise ValueError(
                    f'structure_first_pass_execution.{path_field} must match the configured '
                    'artifact path'
                )
    structure_first_pass_execution_path = structure_first_pass_execution_paths['artifact']
    structure_first_pass_execution_summary_path = structure_first_pass_execution_paths[
        'summary_artifact'
    ]
    structure_first_pass_execution_variants_path = structure_first_pass_execution_paths[
        'variants_artifact'
    ]
    structure_first_pass_execution_structure_dir = structure_first_pass_execution_paths[
        'structure_dir'
    ]
    _validate_structure_execution_output_paths(
        artifact_dir,
        structure_first_pass_execution_paths,
    )
    seen_cif_output_keys: set[tuple[str, ...]] = set()
    for candidate_index, candidate_payload in enumerate(
        structure_first_pass_execution_payload.get('candidates', [])
    ):
        if not isinstance(candidate_payload, dict):
            raise ValueError(
                'structure_first_pass_execution.candidates entries must be objects; '
                f'entry {candidate_index} is invalid'
            )
        variants = candidate_payload.get('variants', [])
        if not isinstance(variants, list):
            raise ValueError(
                'structure_first_pass_execution candidate variants must be a list; '
                f'entry {candidate_index} is invalid'
            )
        for variant_index, variant_payload in enumerate(variants):
            if not isinstance(variant_payload, dict):
                raise ValueError(
                    'structure_first_pass_execution variants entries must be objects; '
                    f'candidate {candidate_index}, variant {variant_index} is invalid'
                )
            cif_text = variant_payload.get('_cif_text')
            cif_relative_path = variant_payload.get('generated_structure_cif_path')
            if not cif_text or not cif_relative_path:
                continue
            cif_output_path = _resolve_and_validate_artifact_output_path(
                artifact_dir,
                cif_relative_path,
                field_name='generated_structure_cif_path',
                expected_output_kind='file',
                required_parent_path=structure_first_pass_execution_structure_dir,
            )
            _reject_hardlinked_output(
                'generated_structure_cif_path',
                cif_output_path,
            )
            cif_output_key = _canonical_path_parts(cif_output_path)
            if cif_output_key in seen_cif_output_keys:
                raise ValueError(
                    'generated_structure_cif_path values must be unique across all '
                    'structure execution variants'
                )
            seen_cif_output_keys.add(cif_output_key)
            if (
                cif_output_path.parent != structure_first_pass_execution_structure_dir
                or cif_output_path.suffix.lower() != '.cif'
            ):
                raise ValueError(
                    'generated_structure_cif_path must be a .cif file directly under the '
                    'configured structure_first_pass_execution.structure_dir'
                )

    existing_cif_paths: tuple[Path, ...] = ()
    if structure_first_pass_execution_structure_dir.exists():
        existing_cif_paths = tuple(
            path
            for path in structure_first_pass_execution_structure_dir.iterdir()
            if path.suffix.casefold() == '.cif'
        )
        for existing_cif_path in existing_cif_paths:
            validate_runtime_output_path(
                existing_cif_path,
                required_parent_path=structure_first_pass_execution_structure_dir,
                expected_output_kind='file',
            )

    preflight_artifact_names = _RESERVED_REPORT_ARTIFACT_NAMES
    if not include_parity_plot:
        preflight_artifact_names = preflight_artifact_names - {'parity_plot.png'}
    for artifact_name in preflight_artifact_names:
        validate_runtime_output_path(
            artifact_dir / artifact_name,
            required_parent_path=artifact_dir,
            expected_output_kind='file',
        )
    for path_field, structure_output_path in structure_first_pass_execution_paths.items():
        validate_runtime_output_path(
            structure_output_path,
            required_parent_path=artifact_dir,
            expected_output_kind=(
                'directory' if path_field == 'structure_dir' else 'file'
            ),
        )
    validate_json_payload(metrics, indent=2)
    validate_json_payload(experiment_summary, ensure_ascii=False, indent=2)
    validate_json_payload(manifest, indent=2)
    execution_outputs_will_publish = bool(
        structure_first_pass_execution_payload
        and not structure_first_pass_execution_summary_df.empty
    )
    _validate_experiment_summary_output_contract(
        artifact_dir,
        experiment_summary,
        structure_first_pass_execution_paths,
        execution_outputs_will_publish=execution_outputs_will_publish,
    )
    if execution_outputs_will_publish:
        _validate_structure_execution_frame_roles(
            structure_first_pass_execution_summary_df,
            structure_first_pass_execution_variant_df,
            structure_first_pass_execution_payload,
            formula_col=formula_col,
            geometry_min_distance_ratio_pass_threshold=(
                structure_first_pass_execution_cfg[
                    'geometry_min_distance_ratio_pass_threshold'
                ]
            ),
            geometry_min_distance_ratio_overlap_threshold=(
                structure_first_pass_execution_cfg[
                    'geometry_min_distance_ratio_overlap_threshold'
                ]
            ),
        )
    if include_parity_plot:
        missing_plot_columns = {'target', 'prediction'} - set(prediction_df.columns)
        if missing_plot_columns:
            raise ValueError(
                'Parity plot publication requires target and prediction columns'
            )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if artifact_provenance_path.exists():
        artifact_provenance_path.unlink()

    if bn_centered_screened_df is not None and not bn_centered_screened_df.empty:
        _publish_csv(bn_centered_screened_df, bn_centered_ranking_path)
    elif bn_centered_ranking_path.exists():
        bn_centered_ranking_path.unlink()
    _publish_json(metrics, artifact_dir / 'metrics.json', indent=2)
    _publish_csv(prediction_df, artifact_dir / 'predictions.csv')
    _publish_csv(bn_df, artifact_dir / 'bn_slice.csv')
    _publish_csv(screened_df, artifact_dir / 'demo_candidate_ranking.csv')
    selected_followup_df = pd.DataFrame()
    if (
        bool(structure_generation_seed_cfg['enabled'])
        and structure_generation_seed_df is not None
        and not structure_generation_seed_df.empty
    ):
        _publish_csv(structure_generation_seed_df, structure_generation_seed_path)
        structure_generation_handoff = _build_structure_generation_handoff_payload(
            structure_generation_seed_df,
            formula_col=formula_col,
            cfg_defaults=structure_generation_seed_cfg,
        )
        _publish_json(
            structure_generation_handoff,
            structure_generation_handoff_path,
            indent=2,
            ensure_ascii=False,
        )
        structure_generation_reference_records = _build_structure_generation_reference_record_payload(
            structure_generation_seed_df,
            cfg=cfg,
        )
        _publish_json(
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
        _publish_json(
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
        _publish_json(
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
            _publish_csv(
                selected_followup_df,
                structure_generation_followup_shortlist_path,
            )
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
            _publish_csv(
                selected_followup_extrapolation_df,
                structure_generation_followup_extrapolation_shortlist_path,
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
        for structure_output_path in (
            structure_first_pass_execution_path,
            structure_first_pass_execution_summary_path,
            structure_first_pass_execution_variants_path,
        ):
            structure_output_path.parent.mkdir(parents=True, exist_ok=True)
        _publish_csv(
            structure_first_pass_execution_summary_df,
            structure_first_pass_execution_summary_path,
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
        _publish_csv(
            structure_followup_report_df,
            demo_candidate_structure_followup_report_path,
        )
        _publish_csv(
            structure_first_pass_execution_variant_df,
            structure_first_pass_execution_variants_path,
        )
        if structure_first_pass_execution_structure_dir is not None:
            structure_first_pass_execution_structure_dir.mkdir(parents=True, exist_ok=True)
            for existing_cif_path in existing_cif_paths:
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
                    cif_output_path = _resolve_artifact_path(
                        artifact_dir,
                        cif_relative_path,
                        field_name='generated_structure_cif_path',
                    )
                    cif_output_path.parent.mkdir(parents=True, exist_ok=True)
                    cif_output_path.write_text(str(cif_text), encoding='utf-8')
                    published_output_paths.add(cif_output_path)
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
        _publish_json(
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
        if structure_first_pass_execution_structure_dir is not None:
            for existing_cif_path in existing_cif_paths:
                existing_cif_path.unlink()
    for enabled, selected_column, rank_column, artifact_name in (
        (
            bool(_proposal_shortlist_config(cfg)['enabled']),
            'proposal_shortlist_selected',
            'proposal_shortlist_rank',
            'demo_candidate_proposal_shortlist.csv',
        ),
        (
            bool(_extrapolation_shortlist_config(cfg)['enabled']),
            'extrapolation_shortlist_selected',
            'extrapolation_shortlist_rank',
            'demo_candidate_extrapolation_shortlist.csv',
        ),
    ):
        shortlist_path = artifact_dir / artifact_name
        if enabled and selected_column in screened_df.columns:
            shortlist_df = screened_df.loc[
                screened_df[selected_column].fillna(False).astype(bool)
            ].copy()
            if rank_column in shortlist_df.columns:
                shortlist_df = shortlist_df.sort_values(rank_column, ascending=True)
            _publish_csv(shortlist_df, shortlist_path)
        elif shortlist_path.exists():
            shortlist_path.unlink()
    bn_candidate_compatible_evaluation_df = _build_bn_candidate_compatible_evaluation_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df=bn_family_benchmark_df,
        bn_stratified_error_df=bn_stratified_error_df,
    )
    if not bn_candidate_compatible_evaluation_df.empty:
        _publish_csv(
            bn_candidate_compatible_evaluation_df,
            bn_candidate_compatible_evaluation_path,
        )
    elif bn_candidate_compatible_evaluation_path.exists():
        bn_candidate_compatible_evaluation_path.unlink()

    if bn_family_benchmark_df is not None and not bn_family_benchmark_df.empty:
        _publish_csv(bn_family_benchmark_df, bn_family_benchmark_path)
    elif bn_family_benchmark_path.exists():
        bn_family_benchmark_path.unlink()
    if bn_family_prediction_df is not None and not bn_family_prediction_df.empty:
        _publish_csv(bn_family_prediction_df, bn_family_prediction_path)
    elif bn_family_prediction_path.exists():
        bn_family_prediction_path.unlink()
    if bn_stratified_error_df is not None and not bn_stratified_error_df.empty:
        _publish_csv(bn_stratified_error_df, bn_stratified_error_path)
    elif bn_stratified_error_path.exists():
        bn_stratified_error_path.unlink()

    bn_evaluation_matrix_df = _build_bn_evaluation_matrix_table(
        bn_slice_benchmark_df,
        bn_family_benchmark_df,
        bn_stratified_error_df,
    )
    if not bn_evaluation_matrix_df.empty:
        _publish_csv(bn_evaluation_matrix_df, bn_evaluation_matrix_path)
    elif bn_evaluation_matrix_path.exists():
        bn_evaluation_matrix_path.unlink()

    candidate_ranking_uncertainty_df = pd.DataFrame()
    if (
        bool(_ranking_stability_config(cfg)['enabled'])
        or bool(_decision_policy_config(cfg)['enabled'])
    ):
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
        _publish_csv(candidate_ranking_uncertainty_df, candidate_uncertainty_path)
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
        _publish_csv(
            bn_model_role_comparison_df,
            bn_model_role_comparison_path,
        )
    elif bn_model_role_comparison_path.exists():
        bn_model_role_comparison_path.unlink()

    candidate_rank_stability_summary_df = pd.DataFrame()
    if bool(_ranking_stability_config(cfg)['enabled']):
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
        _publish_csv(
            candidate_rank_stability_summary_df,
            candidate_rank_stability_summary_path,
        )
    elif candidate_rank_stability_summary_path.exists():
        candidate_rank_stability_summary_path.unlink()

    _publish_csv(benchmark_df, artifact_dir / 'benchmark_results.csv')
    robustness_path = artifact_dir / 'robustness_results.csv'
    if robustness_df is not None and not robustness_df.empty:
        _publish_csv(robustness_df, robustness_path)
    elif robustness_path.exists():
        robustness_path.unlink()
    bn_slice_benchmark_path = artifact_dir / 'bn_slice_benchmark_results.csv'
    if bn_slice_benchmark_df is not None and not bn_slice_benchmark_df.empty:
        _publish_csv(bn_slice_benchmark_df, bn_slice_benchmark_path)
    elif bn_slice_benchmark_path.exists():
        bn_slice_benchmark_path.unlink()
    bn_slice_prediction_path = artifact_dir / 'bn_slice_predictions.csv'
    if bn_slice_prediction_df is not None and not bn_slice_prediction_df.empty:
        _publish_csv(bn_slice_prediction_df, bn_slice_prediction_path)
    elif bn_slice_prediction_path.exists():
        bn_slice_prediction_path.unlink()
    _publish_json(
        experiment_summary,
        artifact_dir / 'experiment_summary.json',
        indent=2,
        ensure_ascii=False,
    )
    _publish_json(manifest, artifact_dir / 'manifest.json', indent=2)
    legacy_screen_path = artifact_dir / 'screened_candidates.csv'
    if legacy_screen_path.exists():
        legacy_screen_path.unlink()
    if include_parity_plot:
        from materials.plots import save_basic_plots

        published_output_paths.add(save_basic_plots(prediction_df, cfg))
    try:
        artifact_provenance = build_artifact_provenance(
            cfg,
            manifest,
            published_output_paths=tuple(
                sorted(published_output_paths, key=lambda path: path.as_posix())
            ),
        )
        write_json_file(
            artifact_provenance,
            artifact_provenance_path,
            indent=2,
        )
    except BaseException:
        artifact_provenance_path.unlink(missing_ok=True)
        raise
