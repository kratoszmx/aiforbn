from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime.io_utils import (
    assess_artifact_provenance,
    load_config,
    read_json_file,
    validate_runtime_output_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(PROJECT_ROOT / 'src' / 'config.py')
DEFAULT_ARTIFACT_ROOT = Path('artifacts')
ARTIFACT_PATHS = {
    'provenance': Path('artifacts/artifact_provenance.json'),
    'manifest': Path('artifacts/manifest.json'),
    'metrics': Path('artifacts/metrics.json'),
    'summary': Path('artifacts/experiment_summary.json'),
    'benchmark': Path('artifacts/benchmark_results.csv'),
    'robustness': Path('artifacts/robustness_results.csv'),
    'bn_slice_benchmark': Path('artifacts/bn_slice_benchmark_results.csv'),
    'bn_slice_prediction': Path('artifacts/bn_slice_predictions.csv'),
    'bn_candidate_eval': Path('artifacts/bn_candidate_compatible_evaluation.csv'),
    'bn_family_benchmark': Path('artifacts/bn_family_benchmark_results.csv'),
    'bn_family_prediction': Path('artifacts/bn_family_predictions.csv'),
    'bn_stratified_error': Path('artifacts/bn_stratified_error_results.csv'),
    'bn_evaluation_matrix': Path('artifacts/bn_evaluation_matrix.csv'),
    'bn_model_role_comparison': Path('artifacts/bn_model_role_comparison.csv'),
    'predictions': Path('artifacts/predictions.csv'),
    'candidate_ranking': Path('artifacts/demo_candidate_ranking.csv'),
    'candidate_uncertainty': Path('artifacts/demo_candidate_ranking_uncertainty.csv'),
    'bn_centered_ranking': Path('artifacts/demo_candidate_bn_centered_ranking.csv'),
    'candidate_rank_stability_summary': Path(
        'artifacts/demo_candidate_rank_stability_summary.csv'
    ),
    'structure_generation_seed': Path('artifacts/demo_candidate_structure_generation_seeds.csv'),
    'structure_generation_handoff': Path('artifacts/demo_candidate_structure_generation_handoff.json'),
    'structure_generation_reference_records': Path('artifacts/demo_candidate_structure_generation_reference_records.json'),
    'structure_generation_job_plan': Path('artifacts/demo_candidate_structure_generation_job_plan.json'),
    'structure_generation_first_pass_queue': Path('artifacts/demo_candidate_structure_generation_first_pass_queue.json'),
    'structure_generation_followup_shortlist': Path('artifacts/demo_candidate_structure_generation_followup_shortlist.csv'),
    'structure_generation_followup_extrapolation_shortlist': Path('artifacts/demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'),
    'structure_generation_first_pass_execution': Path('artifacts/demo_candidate_structure_generation_first_pass_execution.json'),
    'structure_generation_first_pass_execution_summary': Path('artifacts/demo_candidate_structure_generation_first_pass_execution_summary.csv'),
    'structure_generation_first_pass_execution_variants': Path('artifacts/demo_candidate_structure_generation_first_pass_execution_variants.csv'),
    'candidate_structure_followup_report': Path(
        'artifacts/demo_candidate_structure_followup_report.csv'
    ),
    'proposal_shortlist': Path('artifacts/demo_candidate_proposal_shortlist.csv'),
    'extrapolation_shortlist': Path('artifacts/demo_candidate_extrapolation_shortlist.csv'),
}
_SUMMARY_EXECUTION_PATH_FIELDS = {
    'structure_generation_first_pass_execution': 'first_pass_execution_artifact',
    'structure_generation_first_pass_execution_summary': (
        'first_pass_execution_summary_artifact'
    ),
    'structure_generation_first_pass_execution_variants': (
        'first_pass_execution_variants_artifact'
    ),
}
_REQUIRED_COMPLETE_BUNDLE_KEYS = (
    'metrics',
    'summary',
    'benchmark',
    'predictions',
    'candidate_ranking',
    'manifest',
)


def _artifact_file_path(artifact_root: Path, value: object) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    try:
        return validate_runtime_output_path(
            artifact_root / Path(str(value).strip()),
            required_parent_path=artifact_root,
            expected_output_kind='file',
        )
    except ValueError:
        return None


def _build_artifact_paths(
    cfg: dict,
    experiment_summary: dict | None = None,
) -> dict[str, Path | None]:
    artifact_root = Path(cfg['project']['artifact_dir'])
    paths = {
        key: _artifact_file_path(
            artifact_root,
            default_path.relative_to(DEFAULT_ARTIFACT_ROOT),
        )
        for key, default_path in ARTIFACT_PATHS.items()
    }
    bridge = (
        ((experiment_summary or {}).get('screening') or {}).get(
            'structure_generation_bridge'
        )
        or {}
    )
    for key, field_name in _SUMMARY_EXECUTION_PATH_FIELDS.items():
        configured_value = bridge.get(field_name)
        if configured_value:
            paths[key] = _artifact_file_path(artifact_root, configured_value)
    return paths

CSV_SECTIONS = [
    ('Benchmark results', 'benchmark'),
    ('Grouped robustness results', 'robustness'),
    ('BN-focused benchmark results', 'bn_slice_benchmark'),
    ('BN-focused benchmark predictions', 'bn_slice_prediction'),
    ('BN candidate-compatible evaluation', 'bn_candidate_eval'),
    ('BN family holdout benchmark results', 'bn_family_benchmark'),
    ('BN family holdout predictions', 'bn_family_prediction'),
    ('BN vs non-BN stratified errors', 'bn_stratified_error'),
    ('BN evaluation matrix', 'bn_evaluation_matrix'),
    ('BN model role comparison evidence', 'bn_model_role_comparison'),
    ('Prediction samples', 'predictions'),
    ('Top demo candidate ranking', 'candidate_ranking'),
    ('BN-centered alternative candidate ranking', 'bn_centered_ranking'),
    ('Candidate ranking uncertainty and decision policy', 'candidate_uncertainty'),
    ('Default vs BN-centered rank-stability evidence', 'candidate_rank_stability_summary'),
    ('Structure-generation seed bridge', 'structure_generation_seed'),
    ('Structure-grounded follow-up shortlist', 'structure_generation_followup_shortlist'),
    ('Novelty-aware structure follow-up shortlist', 'structure_generation_followup_extrapolation_shortlist'),
    ('Structure first-pass execution summary', 'structure_generation_first_pass_execution_summary'),
    ('Structure first-pass execution variants', 'structure_generation_first_pass_execution_variants'),
    ('Structure follow-up handoff (unrelaxed evidence)', 'candidate_structure_followup_report'),
    ('Proposal shortlist', 'proposal_shortlist'),
    ('Formula-level extrapolation shortlist', 'extrapolation_shortlist'),
]
HEAD_LIMITED_KEYS = {
    'predictions',
    'candidate_ranking',
    'bn_centered_ranking',
    'candidate_uncertainty',
    'structure_generation_seed',
}
JSON_SECTIONS = [
    ('Structure-generation handoff JSON', 'structure_generation_handoff'),
    ('Structure-generation reference records JSON', 'structure_generation_reference_records'),
    ('Structure-generation job-plan JSON', 'structure_generation_job_plan'),
    ('Structure-generation first-pass queue JSON', 'structure_generation_first_pass_queue'),
    ('Structure first-pass execution JSON', 'structure_generation_first_pass_execution'),
]


def render_streamlit_app() -> None:
    st.set_page_config(page_title='AI-Powered Boron Nitride Material Exploration', layout='wide')
    st.title('AI-Powered Boron Nitride Material Exploration')
    st.write(
        'Uncertainty-aware Boron Nitride formula-level exploration '
        'pipeline with transparent candidate prioritization and structure follow-up outputs.'
    )

    base_paths = _build_artifact_paths(CONFIG)
    summary_payload = None
    summary_unreadable = False
    summary_path = base_paths['summary']
    if summary_path is not None and summary_path.exists():
        try:
            summary_payload = read_json_file(summary_path)
        except (OSError, TypeError, ValueError):
            summary_unreadable = True
        if not isinstance(summary_payload, dict):
            summary_payload = None
            summary_unreadable = True
        if summary_unreadable:
            st.warning(
                'Experiment summary is unreadable; artifact provenance cannot be current.'
            )
    artifact_paths = _build_artifact_paths(CONFIG, summary_payload)
    artifact_root = validate_runtime_output_path(
        CONFIG['project']['artifact_dir'],
        expected_output_kind='directory',
    )
    committed_output_paths: set[Path] | None = None
    committed_outputs_verified = False
    uncommitted_artifact_keys: list[str] = []
    has_artifacts = any(
        path is not None and path.exists()
        for key, path in artifact_paths.items()
        if key != 'provenance'
    )
    provenance_path = artifact_paths['provenance']
    manifest_path = artifact_paths['manifest']
    missing_bundle_keys = [
        key
        for key in _REQUIRED_COMPLETE_BUNDLE_KEYS
        if artifact_paths.get(key) is None or not artifact_paths[key].exists()
    ]
    if has_artifacts:
        if provenance_path is None or not provenance_path.exists():
            st.warning(
                'Artifact provenance is unavailable; this bundle cannot be attributed to '
                'the current source and configuration.'
            )
        else:
            try:
                provenance_payload = read_json_file(provenance_path)
            except (OSError, TypeError, ValueError):
                provenance_payload = None
                provenance_assessment = {
                    'status': 'unverified',
                    'reason': 'artifact_provenance_unreadable',
                }
            else:
                published_outputs = (
                    provenance_payload.get('published_outputs')
                    if isinstance(provenance_payload, dict)
                    else None
                )
                if isinstance(published_outputs, dict) and published_outputs:
                    resolved_commitments = {
                        _artifact_file_path(artifact_root, relative_path)
                        for relative_path in published_outputs
                    }
                    if None not in resolved_commitments:
                        committed_output_paths = resolved_commitments
                        uncommitted_artifact_keys = [
                            key
                            for key, path in artifact_paths.items()
                            if (
                                key != 'provenance'
                                and path is not None
                                and path.exists()
                                and path not in committed_output_paths
                            )
                        ]
                        missing_bundle_keys = [
                            key
                            for key in _REQUIRED_COMPLETE_BUNDLE_KEYS
                            if (
                                artifact_paths.get(key) is None
                                or not artifact_paths[key].exists()
                                or artifact_paths[key] not in committed_output_paths
                            )
                        ]
                manifest_payload = None
                if manifest_path is not None and manifest_path.exists():
                    try:
                        manifest_payload = read_json_file(manifest_path)
                    except (OSError, TypeError, ValueError):
                        manifest_payload = {}
                provenance_assessment = assess_artifact_provenance(
                    provenance_payload,
                    CONFIG,
                    manifest_payload,
                )
                if provenance_assessment['status'] == 'current' and (
                    missing_bundle_keys
                    or summary_unreadable
                    or uncommitted_artifact_keys
                ):
                    provenance_assessment = {
                        'status': 'unverified',
                        'reason': 'artifact_bundle_incomplete_unreadable_or_uncommitted',
                    }
                committed_outputs_verified = (
                    provenance_assessment['status'] == 'current'
                    and committed_output_paths is not None
                )
            st.subheader('Artifact bundle provenance')
            provenance_display = (
                provenance_payload
                if isinstance(provenance_payload, dict)
                else {'stored_provenance': provenance_payload}
            )
            st.json(
                {
                    **provenance_display,
                    'assessment': provenance_assessment,
                    'missing_required_artifacts': missing_bundle_keys,
                    'uncommitted_known_artifacts': uncommitted_artifact_keys,
                }
            )
            if provenance_assessment['status'] == 'current':
                st.success(
                    'Artifact provenance matches the current source, configuration, '
                    'dataset, and published output contents.'
                )
            else:
                st.warning(
                    'Artifact provenance is '
                    f"{provenance_assessment['status']}: "
                    f"{provenance_assessment['reason']}."
                )

    def _is_renderable(path: Path | None) -> bool:
        return (
            path is not None
            and path.exists()
            and committed_outputs_verified
            and committed_output_paths is not None
            and path in committed_output_paths
        )

    metrics_path = artifact_paths['metrics']
    if _is_renderable(metrics_path):
        st.subheader('Metrics')
        try:
            st.json(read_json_file(metrics_path))
        except (OSError, TypeError, ValueError):
            st.warning('Metrics artifact exists but is unreadable.')
    else:
        st.info('Run `python main.py` first to generate artifacts.')

    if summary_payload is not None and _is_renderable(summary_path):
        st.subheader('Experiment summary')
        st.json(summary_payload)

    for title, key in CSV_SECTIONS:
        path = artifact_paths.get(key)
        if not _is_renderable(path):
            continue
        st.subheader(title)
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            st.warning(f'{title} exists but has no readable CSV schema.')
            continue
        except (OSError, ValueError):
            st.warning(f'{title} exists but has no readable CSV content.')
            continue
        if key in HEAD_LIMITED_KEYS:
            df = df.head(30)
        st.dataframe(df, width='stretch')

    for title, key in JSON_SECTIONS:
        path = artifact_paths.get(key)
        if not _is_renderable(path):
            continue
        st.subheader(title)
        try:
            st.json(read_json_file(path))
        except (OSError, TypeError, ValueError):
            st.warning(f'{title} exists but is unreadable.')


if __name__ == '__main__':
    render_streamlit_app()
