from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys
import types


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.calls: list[tuple[str, object]] = []
        self.dataframe_kwargs: list[dict[str, object]] = []

    def set_page_config(self, **kwargs):
        self.calls.append(('set_page_config', kwargs))

    def title(self, value):
        self.calls.append(('title', value))

    def write(self, value):
        self.calls.append(('write', value))

    def subheader(self, value):
        self.calls.append(('subheader', value))

    def json(self, value):
        self.calls.append(('json', value))

    def info(self, value):
        self.calls.append(('info', value))

    def warning(self, value):
        self.calls.append(('warning', value))

    def success(self, value):
        self.calls.append(('success', value))

    def dataframe(self, value, **kwargs):
        self.calls.append(('dataframe', getattr(value, 'shape', None)))
        self.dataframe_kwargs.append(kwargs)


def test_streamlit_app_reads_generated_artifacts(tmp_path, monkeypatch):
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    (artifact_dir / 'metrics.json').write_text(json.dumps({'mae': 1.0}), encoding='utf-8')
    (artifact_dir / 'experiment_summary.json').write_text(json.dumps({'dataset': {'rows': 1}}), encoding='utf-8')
    (artifact_dir / 'benchmark_results.csv').write_text('model_type,mae\nlinear_regression,1.0\n', encoding='utf-8')
    (artifact_dir / 'robustness_results.csv').write_text('model_type,mae_mean\nlinear_regression,1.1\n', encoding='utf-8')
    (artifact_dir / 'bn_slice_benchmark_results.csv').write_text('model_type,mae\nlinear_regression,0.9\n', encoding='utf-8')
    (artifact_dir / 'bn_slice_predictions.csv').write_text('formula,target,prediction\nBN,5.0,4.8\n', encoding='utf-8')
    (artifact_dir / 'bn_candidate_compatible_evaluation.csv').write_text('model_type,mae\nlinear_regression,0.9\n', encoding='utf-8')
    (artifact_dir / 'bn_family_benchmark_results.csv').write_text('model_type,mae\nlinear_regression,1.1\n', encoding='utf-8')
    (artifact_dir / 'bn_family_predictions.csv').write_text('formula,target,prediction\nBN,5.0,4.6\n', encoding='utf-8')
    (artifact_dir / 'bn_stratified_error_results.csv').write_text('model_type,bn_mae,non_bn_mae\nlinear_regression,1.2,0.8\n', encoding='utf-8')
    (artifact_dir / 'bn_evaluation_matrix.csv').write_text('model_type,formula_holdout_mae,family_holdout_mae\nlinear_regression,0.9,1.1\n', encoding='utf-8')
    (artifact_dir / 'bn_model_role_comparison.csv').write_text(
        'benchmark_role,bn_slice_mae\nselected_model,0.9\n',
        encoding='utf-8',
    )
    (artifact_dir / 'predictions.csv').write_text('formula,target,prediction\nBN,5.0,4.8\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_ranking.csv').write_text('formula,predicted_band_gap\nBN,4.8\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_ranking_uncertainty.csv').write_text('formula,rank_mean\nBN,1\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_bn_centered_ranking.csv').write_text('formula,predicted_band_gap\nAlBN,4.2\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_rank_stability_summary.csv').write_text(
        'top_k,top_k_overlap_count\n3,2\n',
        encoding='utf-8',
    )
    (artifact_dir / 'demo_candidate_structure_generation_seeds.csv').write_text('formula,seed_reference_formula\nBN,BN\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_handoff.json').write_text(json.dumps({'candidate_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_reference_records.json').write_text(json.dumps({'record_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_job_plan.json').write_text(json.dumps({'job_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json').write_text(json.dumps({'queue_entry_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv').write_text('formula,structure_followup_shortlist_rank\nBN,1\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv').write_text('formula,structure_followup_extrapolation_shortlist_rank\nBCN2,1\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution.json').write_text(json.dumps({'candidate_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_summary.csv').write_text('formula,first_pass_execution_status\nBN,executed\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_variants.csv').write_text('formula,execution_variant_id\nBN,bn__variant_01\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_followup_report.csv').write_text(
        'formula,first_pass_execution_selected_final_status\nBN,executed\n',
        encoding='utf-8',
    )
    (artifact_dir / 'demo_candidate_proposal_shortlist.csv').write_text('formula,proposal_shortlist_rank\nBN,1\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_extrapolation_shortlist.csv').write_text('formula,extrapolation_shortlist_rank\nBCN2,1\n', encoding='utf-8')

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake_streamlit)
    monkeypatch.chdir(tmp_path)

    ROOT = Path(__file__).resolve().parents[3]
    app_path = ROOT / 'src' / 'ui' / 'streamlit_app.py'
    spec = spec_from_file_location('streamlit_app_test', app_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    from runtime.io_utils import read_json_file as shared_read_json_file

    assert module.read_json_file is shared_read_json_file
    module.render_streamlit_app()

    assert ('title', 'AI-Powered Boron Nitride Material Exploration') in fake_streamlit.calls
    assert ('subheader', 'Metrics') in fake_streamlit.calls
    assert ('subheader', 'Experiment summary') in fake_streamlit.calls
    assert ('subheader', 'Benchmark results') in fake_streamlit.calls
    assert ('subheader', 'Grouped robustness results') in fake_streamlit.calls
    assert ('subheader', 'BN-focused benchmark results') in fake_streamlit.calls
    assert ('subheader', 'BN-focused benchmark predictions') in fake_streamlit.calls
    assert ('subheader', 'BN candidate-compatible evaluation') in fake_streamlit.calls
    assert ('subheader', 'BN family holdout benchmark results') in fake_streamlit.calls
    assert ('subheader', 'BN family holdout predictions') in fake_streamlit.calls
    assert ('subheader', 'BN vs non-BN stratified errors') in fake_streamlit.calls
    assert ('subheader', 'BN evaluation matrix') in fake_streamlit.calls
    assert ('subheader', 'BN model role comparison evidence') in fake_streamlit.calls
    assert ('subheader', 'Prediction samples') in fake_streamlit.calls
    assert ('subheader', 'Top demo candidate ranking') in fake_streamlit.calls
    assert ('subheader', 'BN-centered alternative candidate ranking') in fake_streamlit.calls
    assert ('subheader', 'Candidate ranking uncertainty and decision policy') in fake_streamlit.calls
    assert ('subheader', 'Default vs BN-centered rank-stability evidence') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation seed bridge') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation handoff JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation reference records JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation job-plan JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation first-pass queue JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-grounded follow-up shortlist') in fake_streamlit.calls
    assert ('subheader', 'Novelty-aware structure follow-up shortlist') in fake_streamlit.calls
    assert ('subheader', 'Structure first-pass execution summary') in fake_streamlit.calls
    assert ('subheader', 'Structure first-pass execution variants') in fake_streamlit.calls
    assert ('subheader', 'Structure first-pass execution JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure follow-up handoff (unrelaxed evidence)') in fake_streamlit.calls
    assert ('subheader', 'Proposal shortlist') in fake_streamlit.calls
    assert ('subheader', 'Formula-level extrapolation shortlist') in fake_streamlit.calls
    assert any(
        call_name == 'warning' and 'provenance' in str(value).lower()
        for call_name, value in fake_streamlit.calls
    )
    assert fake_streamlit.dataframe_kwargs
    assert all(kwargs == {'width': 'stretch'} for kwargs in fake_streamlit.dataframe_kwargs)


def test_streamlit_app_uses_configured_artifact_root_and_summary_paths(
    tmp_path,
    monkeypatch,
):
    configured_artifact_dir = tmp_path / 'current-output'
    configured_artifact_dir.mkdir()
    stale_artifact_dir = tmp_path / 'artifacts'
    stale_artifact_dir.mkdir()
    (stale_artifact_dir / 'metrics.json').write_text(
        json.dumps({'mae': 99.0}),
        encoding='utf-8',
    )
    (configured_artifact_dir / 'metrics.json').write_text(
        json.dumps({'mae': 1.0}),
        encoding='utf-8',
    )
    summary_payload = {
        'screening': {
            'structure_generation_bridge': {
                'first_pass_execution_artifact': 'nested/execution.json',
                'first_pass_execution_summary_artifact': 'nested/summary.csv',
                'first_pass_execution_variants_artifact': 'nested/variants.csv',
            },
        },
    }
    (configured_artifact_dir / 'experiment_summary.json').write_text(
        json.dumps(summary_payload),
        encoding='utf-8',
    )
    nested_dir = configured_artifact_dir / 'nested'
    nested_dir.mkdir()
    (nested_dir / 'execution.json').write_text(
        json.dumps({'candidate_count': 2}),
        encoding='utf-8',
    )
    (nested_dir / 'summary.csv').write_text(
        'formula,first_pass_execution_status\nXBN,executed\n',
        encoding='utf-8',
    )
    (nested_dir / 'variants.csv').write_text('', encoding='utf-8')

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake_streamlit)
    monkeypatch.chdir(tmp_path)
    from runtime import io_utils

    monkeypatch.setattr(
        io_utils,
        'load_config',
        lambda _path: {
            'project': {'artifact_dir': str(configured_artifact_dir)},
            'screening': {},
        },
    )

    root = Path(__file__).resolve().parents[3]
    app_path = root / 'src' / 'ui' / 'streamlit_app.py'
    spec = spec_from_file_location('streamlit_app_configured_paths_test', app_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.render_streamlit_app()

    json_values = [value for call_name, value in fake_streamlit.calls if call_name == 'json']
    assert {'mae': 1.0} in json_values
    assert {'mae': 99.0} not in json_values
    assert ('subheader', 'Structure first-pass execution summary') in fake_streamlit.calls
    assert ('subheader', 'Structure first-pass execution variants') in fake_streamlit.calls
    assert ('subheader', 'Structure first-pass execution JSON') in fake_streamlit.calls
    assert any(
        call_name == 'warning' and 'no readable CSV schema' in str(value)
        for call_name, value in fake_streamlit.calls
    )


def test_streamlit_app_runs_through_real_streamlit_renderer(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[3]
    app = AppTest.from_file(str(root / 'src' / 'ui' / 'streamlit_app.py')).run(timeout=10)

    assert len(app.exception) == 0
    assert [node.value for node in app.info] == [
        'Run `python main.py` first to generate artifacts.'
    ]
