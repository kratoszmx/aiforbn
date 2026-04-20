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

    def dataframe(self, value, **kwargs):
        self.calls.append(('dataframe', getattr(value, 'shape', None)))


def test_streamlit_app_reads_generated_artifacts(tmp_path, monkeypatch):
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    (artifact_dir / 'metrics.json').write_text(json.dumps({'mae': 1.0}), encoding='utf-8')
    (artifact_dir / 'experiment_summary.json').write_text(json.dumps({'dataset': {'rows': 1}}), encoding='utf-8')
    (artifact_dir / 'benchmark_results.csv').write_text('model_type,mae\nlinear_regression,1.0\n', encoding='utf-8')
    (artifact_dir / 'robustness_results.csv').write_text('model_type,mae_mean\nlinear_regression,1.1\n', encoding='utf-8')
    (artifact_dir / 'bn_slice_benchmark_results.csv').write_text('model_type,mae\nlinear_regression,0.9\n', encoding='utf-8')
    (artifact_dir / 'bn_slice_predictions.csv').write_text('formula,target,prediction\nBN,5.0,4.8\n', encoding='utf-8')
    (artifact_dir / 'predictions.csv').write_text('formula,target,prediction\nBN,5.0,4.8\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_ranking.csv').write_text('formula,predicted_band_gap\nBN,4.8\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_bn_centered_ranking.csv').write_text('formula,predicted_band_gap\nAlBN,4.2\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_seeds.csv').write_text('formula,seed_reference_formula\nBN,BN\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_handoff.json').write_text(json.dumps({'candidate_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_reference_records.json').write_text(json.dumps({'record_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_job_plan.json').write_text(json.dumps({'job_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json').write_text(json.dumps({'queue_entry_count': 1}), encoding='utf-8')
    (artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv').write_text('formula,structure_followup_shortlist_rank\nBN,1\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_proposal_shortlist.csv').write_text('formula,proposal_shortlist_rank\nBN,1\n', encoding='utf-8')
    (artifact_dir / 'demo_candidate_extrapolation_shortlist.csv').write_text('formula,extrapolation_shortlist_rank\nBCN2,1\n', encoding='utf-8')

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake_streamlit)
    monkeypatch.chdir(tmp_path)

    app_path = Path(__file__).resolve().parents[1] / 'apps' / 'streamlit_app.py'
    spec = spec_from_file_location('streamlit_app_test', app_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    assert ('title', 'BN Explorer') in fake_streamlit.calls
    assert ('subheader', 'Metrics') in fake_streamlit.calls
    assert ('subheader', 'Experiment summary') in fake_streamlit.calls
    assert ('subheader', 'Benchmark results') in fake_streamlit.calls
    assert ('subheader', 'Grouped robustness results') in fake_streamlit.calls
    assert ('subheader', 'BN-focused benchmark results') in fake_streamlit.calls
    assert ('subheader', 'BN-focused benchmark predictions') in fake_streamlit.calls
    assert ('subheader', 'Prediction samples') in fake_streamlit.calls
    assert ('subheader', 'Top demo candidate ranking') in fake_streamlit.calls
    assert ('subheader', 'BN-centered alternative candidate ranking') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation seed bridge') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation handoff JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation reference records JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation job-plan JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-generation first-pass queue JSON') in fake_streamlit.calls
    assert ('subheader', 'Structure-grounded follow-up shortlist') in fake_streamlit.calls
    assert ('subheader', 'Proposal shortlist') in fake_streamlit.calls
    assert ('subheader', 'Formula-level extrapolation shortlist') in fake_streamlit.calls
