from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

_MYUTILS_ROOT = Path(__file__).resolve().parents[3] / 'myutils'
_MYUTILS_FILE_UTILS_DIR = _MYUTILS_ROOT / 'file_utils'
if str(_MYUTILS_FILE_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_MYUTILS_FILE_UTILS_DIR))

from json_io import read_json_file


ARTIFACT_PATHS = {
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
    'predictions': Path('artifacts/predictions.csv'),
    'candidate_ranking': Path('artifacts/demo_candidate_ranking.csv'),
    'candidate_uncertainty': Path('artifacts/demo_candidate_ranking_uncertainty.csv'),
    'bn_centered_ranking': Path('artifacts/demo_candidate_bn_centered_ranking.csv'),
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
    'proposal_shortlist': Path('artifacts/demo_candidate_proposal_shortlist.csv'),
    'extrapolation_shortlist': Path('artifacts/demo_candidate_extrapolation_shortlist.csv'),
}

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
    ('Prediction samples', 'predictions'),
    ('Top demo candidate ranking', 'candidate_ranking'),
    ('BN-centered alternative candidate ranking', 'bn_centered_ranking'),
    ('Candidate ranking uncertainty and decision policy', 'candidate_uncertainty'),
    ('Structure-generation seed bridge', 'structure_generation_seed'),
    ('Structure-grounded follow-up shortlist', 'structure_generation_followup_shortlist'),
    ('Novelty-aware structure follow-up shortlist', 'structure_generation_followup_extrapolation_shortlist'),
    ('Structure first-pass execution summary', 'structure_generation_first_pass_execution_summary'),
    ('Structure first-pass execution variants', 'structure_generation_first_pass_execution_variants'),
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
    st.set_page_config(page_title='BN Explorer', layout='wide')
    st.title('BN Explorer')
    st.write('Minimal PoC UI for BN property prediction, grouped evaluation, and demo candidate ranking.')

    if ARTIFACT_PATHS['metrics'].exists():
        st.subheader('Metrics')
        st.json(read_json_file(ARTIFACT_PATHS['metrics']))
    else:
        st.info('Run `python main.py` first to generate artifacts.')

    if ARTIFACT_PATHS['summary'].exists():
        st.subheader('Experiment summary')
        st.json(read_json_file(ARTIFACT_PATHS['summary']))

    for title, key in CSV_SECTIONS:
        path = ARTIFACT_PATHS[key]
        if not path.exists():
            continue
        st.subheader(title)
        df = pd.read_csv(path)
        if key in HEAD_LIMITED_KEYS:
            df = df.head(30)
        st.dataframe(df, use_container_width=True)

    for title, key in JSON_SECTIONS:
        path = ARTIFACT_PATHS[key]
        if not path.exists():
            continue
        st.subheader(title)
        st.json(read_json_file(path))


if __name__ == '__main__':
    render_streamlit_app()
