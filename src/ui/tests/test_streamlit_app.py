from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import shutil
import sys
import types

import pandas as pd
import pytest

from materials.artifacts import save_metrics_and_predictions


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.calls: list[tuple[str, object]] = []
        self.dataframes: list[pd.DataFrame] = []
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
        self.dataframes.append(value.copy())
        self.dataframe_kwargs.append(kwargs)


def _save_bn_prediction_bundle(
    artifact_dir: Path,
    *,
    slice_nonempty: bool,
    family_nonempty: bool,
):
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
        },
    }
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    empty_df = pd.DataFrame()
    slice_prediction_df = (
        pd.DataFrame([{'formula': 'BN', 'target': 5.0, 'prediction': 4.8}])
        if slice_nonempty
        else empty_df
    )
    family_prediction_df = (
        pd.DataFrame([{'formula': 'BCN', 'target': 4.0, 'prediction': 3.6}])
        if family_nonempty
        else empty_df
    )

    def _benchmark_frame(status):
        metric = 0.5 if status == 'ok' else None
        return pd.DataFrame([{
            'benchmark_role': 'selected_model',
            'benchmark_status': status,
            'feature_set': 'basic_formula_composition',
            'feature_family': 'composition_only',
            'model_type': 'linear_regression',
            'candidate_compatible': True,
            'selected_by_validation': True,
            'mae': metric,
            'rmse': metric,
            'r2': metric,
        }])

    slice_benchmark_df = _benchmark_frame(
        'ok' if slice_nonempty else 'insufficient_bn_formulas'
    )
    family_benchmark_df = _benchmark_frame(
        'ok' if family_nonempty else 'insufficient_bn_families'
    )
    summary = {
        'bn_slice_benchmark': {
            'benchmark_artifact': 'bn_slice_benchmark_results.csv',
            'prediction_artifact': (
                'bn_slice_predictions.csv' if slice_nonempty else None
            ),
            'family_benchmark_artifact': 'bn_family_benchmark_results.csv',
            'family_prediction_artifact': (
                'bn_family_predictions.csv' if family_nonempty else None
            ),
        },
    }
    save_metrics_and_predictions(
        {'mae': 1.0},
        pd.DataFrame([{'formula': 'BN', 'prediction': 5.0}]),
        empty_df,
        pd.DataFrame([{'formula': 'BN', 'ranking_rank': 1}]),
        pd.DataFrame([{'model_type': 'linear', 'mae': 1.0}]),
        empty_df,
        slice_benchmark_df,
        slice_prediction_df,
        empty_df,
        empty_df,
        summary,
        manifest,
        cfg,
        bn_family_benchmark_df=family_benchmark_df,
        bn_family_prediction_df=family_prediction_df,
    )
    return cfg


def _load_fake_streamlit_app(monkeypatch, tmp_path, cfg, module_name):
    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake_streamlit)
    monkeypatch.chdir(tmp_path)
    from runtime import io_utils

    monkeypatch.setattr(io_utils, 'load_config', lambda _path: cfg)
    root = Path(__file__).resolve().parents[3]
    spec = spec_from_file_location(
        module_name,
        root / 'src' / 'ui' / 'streamlit_app.py',
    )
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.render_streamlit_app()
    return fake_streamlit, module


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

    root = Path(__file__).resolve().parents[3]
    from runtime import io_utils

    cfg = io_utils.load_config(root / 'src' / 'config.py')
    cfg['project']['artifact_dir'] = str(artifact_dir)
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(io_utils, 'load_config', lambda _path: cfg)
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    provenance = io_utils.build_artifact_provenance(
        cfg,
        manifest,
        published_output_paths=tuple(
            path for path in artifact_dir.rglob('*') if path.is_file()
        ),
    )
    (artifact_dir / 'artifact_provenance.json').write_text(
        json.dumps(provenance),
        encoding='utf-8',
    )

    fake_streamlit, module = _load_fake_streamlit_app(
        monkeypatch,
        tmp_path,
        cfg,
        'streamlit_app_test',
    )
    from runtime.io_utils import read_json_file as shared_read_json_file

    assert module.read_json_file is shared_read_json_file

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
    assert any(call_name == 'success' for call_name, _value in fake_streamlit.calls)
    assert not any(
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
    (configured_artifact_dir / 'benchmark_results.csv').write_text(
        'model_type,mae\nlinear,1.0\n',
        encoding='utf-8',
    )
    (configured_artifact_dir / 'predictions.csv').write_text(
        'formula,prediction\nBN,5.0\n',
        encoding='utf-8',
    )
    (configured_artifact_dir / 'demo_candidate_ranking.csv').write_text(
        'formula,ranking_rank\nBN,1\n',
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

    from runtime import io_utils

    cfg = {
        'project': {'artifact_dir': str(configured_artifact_dir)},
        'screening': {},
    }
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    (configured_artifact_dir / 'manifest.json').write_text(
        json.dumps(manifest),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    provenance = io_utils.build_artifact_provenance(
        cfg,
        manifest,
        published_output_paths=tuple(
            path for path in configured_artifact_dir.rglob('*') if path.is_file()
        ),
    )
    (configured_artifact_dir / 'artifact_provenance.json').write_text(
        json.dumps(provenance),
        encoding='utf-8',
    )

    fake_streamlit, _module = _load_fake_streamlit_app(
        monkeypatch,
        tmp_path,
        cfg,
        'streamlit_app_configured_paths_test',
    )

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


@pytest.mark.parametrize(
    ('slice_nonempty', 'family_nonempty'),
    [(True, False), (False, True), (False, False), (True, True)],
    ids=['slice-only', 'family-only', 'neither', 'both'],
)
def test_streamlit_current_v2_renders_exact_bn_prediction_sections_and_rows(
    tmp_path,
    monkeypatch,
    slice_nonempty,
    family_nonempty,
):
    from runtime import io_utils

    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg = _save_bn_prediction_bundle(
        tmp_path / 'artifacts',
        slice_nonempty=slice_nonempty,
        family_nonempty=family_nonempty,
    )
    fake_streamlit, _module = _load_fake_streamlit_app(
        monkeypatch,
        tmp_path,
        cfg,
        f'streamlit_app_bn_matrix_{slice_nonempty}_{family_nonempty}',
    )

    rendered_subheaders = {
        value for name, value in fake_streamlit.calls if name == 'subheader'
    }
    assert 'BN-focused benchmark results' in rendered_subheaders
    assert 'BN family holdout benchmark results' in rendered_subheaders
    assert ('BN-focused benchmark predictions' in rendered_subheaders) is (
        slice_nonempty
    )
    assert ('BN family holdout predictions' in rendered_subheaders) is (
        family_nonempty
    )
    rendered_prediction_rows = {
        (str(row['formula']), float(row['prediction']))
        for frame in fake_streamlit.dataframes
        if {'formula', 'prediction'}.issubset(frame.columns)
        for _index, row in frame.iterrows()
    }
    assert (('BN', 4.8) in rendered_prediction_rows) is slice_nonempty
    assert (('BCN', 3.6) in rendered_prediction_rows) is family_nonempty


@pytest.mark.parametrize('prediction_kind', ['slice', 'family'])
@pytest.mark.parametrize(
    'provenance_case',
    [
        'current-v2',
        'byte-mismatch',
        'missing-committed',
        'malformed-marker',
        'malformed-manifest',
        'legacy-v1',
        'relocated-root',
    ],
)
def test_streamlit_asymmetric_bn_predictions_require_current_v2_provenance(
    tmp_path,
    monkeypatch,
    prediction_kind,
    provenance_case,
):
    from runtime import io_utils
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    artifact_dir = tmp_path / 'artifacts'
    cfg = _save_bn_prediction_bundle(
        artifact_dir,
        slice_nonempty=prediction_kind == 'slice',
        family_nonempty=prediction_kind == 'family',
    )
    prediction_name = f'bn_{prediction_kind}_predictions.csv'
    prediction_path = artifact_dir / prediction_name
    provenance_path = artifact_dir / 'artifact_provenance.json'
    manifest_path = artifact_dir / 'manifest.json'
    if provenance_case == 'byte-mismatch':
        prediction_path.write_text(
            'formula,target,prediction\nolder,0.0,99.0\n',
            encoding='utf-8',
        )
    elif provenance_case == 'missing-committed':
        prediction_path.unlink()
    elif provenance_case == 'malformed-marker':
        provenance_path.write_text('{', encoding='utf-8')
    elif provenance_case == 'malformed-manifest':
        manifest_path.write_text('{', encoding='utf-8')
    elif provenance_case == 'legacy-v1':
        provenance = io_utils.read_json_file(provenance_path)
        provenance.pop('published_outputs')
        provenance['schema'] = 'aiforbn.artifact_provenance.v1'
        provenance_path.write_text(json.dumps(provenance), encoding='utf-8')
    elif provenance_case == 'relocated-root':
        relocated_dir = tmp_path / 'relocated-artifacts'
        shutil.copytree(artifact_dir, relocated_dir)
        artifact_dir = relocated_dir
        cfg['project']['artifact_dir'] = str(artifact_dir)

    wrapper_path = tmp_path / f'{prediction_kind}_{provenance_case}.py'
    wrapper_path.write_text(
        'from runtime import io_utils\n'
        '_original_source_reader = io_utils._read_local_source_state\n'
        "io_utils._read_local_source_state = lambda _root: "
        "{'revision': 'abc123', 'dirty': False}\n"
        'from ui import streamlit_app\n'
        '_original_config = streamlit_app.CONFIG\n'
        f'streamlit_app.CONFIG = {cfg!r}\n'
        'try:\n'
        '    streamlit_app.render_streamlit_app()\n'
        'finally:\n'
        '    streamlit_app.CONFIG = _original_config\n'
        '    io_utils._read_local_source_state = _original_source_reader\n',
        encoding='utf-8',
    )
    app = AppTest.from_file(str(wrapper_path)).run(timeout=10)

    expect_current = provenance_case == 'current-v2'
    assert len(app.exception) == 0
    assert (len(app.success) == 1) is expect_current
    if not expect_current:
        assert len(app.warning) >= 1
    rendered_subheaders = {subheader.value for subheader in app.subheader}
    assert ('Metrics' in rendered_subheaders) is expect_current
    assert ('Benchmark results' in rendered_subheaders) is expect_current
    expected_title = (
        'BN-focused benchmark predictions'
        if prediction_kind == 'slice'
        else 'BN family holdout predictions'
    )
    opposite_title = (
        'BN family holdout predictions'
        if prediction_kind == 'slice'
        else 'BN-focused benchmark predictions'
    )
    assert (expected_title in rendered_subheaders) is expect_current
    assert opposite_title not in rendered_subheaders


@pytest.mark.parametrize(
    ('case', 'expect_current'),
    [
        ('current', True),
        ('missing-manifest', False),
        ('malformed-manifest', False),
        ('manifest-mismatch', False),
        ('malformed-provenance', False),
        ('non-object-provenance', False),
        ('legacy-provenance', False),
        ('non-object-summary', False),
        ('missing-summary', False),
        ('marker-only', False),
    ],
)
def test_streamlit_provenance_never_marks_incomplete_or_malformed_bundle_current(
    tmp_path,
    monkeypatch,
    case,
    expect_current,
):
    artifact_dir = tmp_path / 'configured-artifacts'
    artifact_dir.mkdir()
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'screening': {},
    }
    valid_manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    if case != 'marker-only':
        for name, contents in {
            'metrics.json': json.dumps({'mae': 1.0}),
            'benchmark_results.csv': 'model_type,mae\nlinear_regression,1.0\n',
            'predictions.csv': 'formula,prediction\nBN,5.0\n',
            'demo_candidate_ranking.csv': 'formula,ranking_rank\nBN,1\n',
        }.items():
            (artifact_dir / name).write_text(contents, encoding='utf-8')
    if case not in {'missing-summary', 'marker-only'}:
        summary_contents = '[]' if case == 'non-object-summary' else '{}'
        (artifact_dir / 'experiment_summary.json').write_text(
            summary_contents,
            encoding='utf-8',
        )

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake_streamlit)
    monkeypatch.chdir(tmp_path)
    from runtime import io_utils

    monkeypatch.setattr(io_utils, 'load_config', lambda _path: cfg)
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    manifest_path = artifact_dir / 'manifest.json'
    if case == 'malformed-manifest':
        manifest_path.write_text('{', encoding='utf-8')
    elif case == 'manifest-mismatch':
        manifest_path.write_text(
            json.dumps({**valid_manifest, 'name': 'different'}),
            encoding='utf-8',
        )
    elif case != 'missing-manifest':
        manifest_path.write_text(json.dumps(valid_manifest), encoding='utf-8')

    marker_manifest = None if case == 'missing-manifest' else valid_manifest
    published_output_paths = tuple(
        path for path in artifact_dir.rglob('*') if path.is_file()
    )
    if not published_output_paths:
        marker_anchor = artifact_dir / 'marker-anchor.json'
        marker_anchor.write_text('{}', encoding='utf-8')
        published_output_paths = (marker_anchor,)
    else:
        marker_anchor = None
    provenance = io_utils.build_artifact_provenance(
        cfg,
        marker_manifest,
        published_output_paths=published_output_paths,
    )
    if marker_anchor is not None:
        marker_anchor.unlink()
    provenance_path = artifact_dir / 'artifact_provenance.json'
    if case == 'malformed-provenance':
        provenance_path.write_text('{', encoding='utf-8')
    elif case == 'non-object-provenance':
        provenance_path.write_text('[]', encoding='utf-8')
    else:
        if case == 'legacy-provenance':
            provenance.pop('published_outputs')
            provenance['schema'] = 'aiforbn.artifact_provenance.v1'
        provenance_path.write_text(json.dumps(provenance), encoding='utf-8')

    root = Path(__file__).resolve().parents[3]
    app_path = root / 'src' / 'ui' / 'streamlit_app.py'
    spec = spec_from_file_location(f'streamlit_app_provenance_{case}', app_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.render_streamlit_app()

    success_calls = [value for name, value in fake_streamlit.calls if name == 'success']
    if expect_current:
        assert success_calls == [
            'Artifact provenance matches the current source, configuration, '
            'dataset, and published output contents.'
        ]
    else:
        assert success_calls == []
        assert any(name == 'warning' for name, _value in fake_streamlit.calls)
        rendered_subheaders = [
            value for name, value in fake_streamlit.calls if name == 'subheader'
        ]
        assert 'Metrics' not in rendered_subheaders
        assert 'Benchmark results' not in rendered_subheaders
        assert 'Prediction samples' not in rendered_subheaders
        assert 'Top demo candidate ranking' not in rendered_subheaders


@pytest.mark.parametrize(
    ('case', 'expected_assessor_current', 'expected_viewer_current'),
    [
        ('baseline', True, True),
        ('missing-json', False, False),
        ('missing-csv', False, False),
        ('malformed-json', False, False),
        ('malformed-csv', False, False),
        ('different-json', False, False),
        ('different-csv', False, False),
        ('changed-summary', False, False),
        ('changed-manifest', False, False),
        ('changed-optional', False, False),
        ('relocated-root', False, False),
        ('uncommitted-known-optional', True, False),
        ('unrelated-extra', True, True),
    ],
)
def test_streamlit_real_renderer_never_marks_content_mixed_bundle_current(
    tmp_path,
    monkeypatch,
    case,
    expected_assessor_current,
    expected_viewer_current,
):
    from runtime import io_utils
    from streamlit.testing.v1 import AppTest

    artifact_dir = tmp_path / 'custom-root' / 'nested-artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
        },
    }
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    empty_df = pd.DataFrame()
    save_metrics_and_predictions(
        {'mae': 1.0},
        pd.DataFrame([{'formula': 'BN', 'prediction': 5.0}]),
        empty_df,
        pd.DataFrame([{'formula': 'BN', 'ranking_rank': 1}]),
        pd.DataFrame([{'model_type': 'linear', 'mae': 1.0}]),
        pd.DataFrame([{'model_type': 'linear', 'mae_mean': 1.1}]),
        empty_df,
        empty_df,
        empty_df,
        empty_df,
        {'dataset': {'rows': 1}},
        manifest,
        cfg,
    )

    mutations = {
        'baseline': lambda: None,
        'missing-json': lambda: (artifact_dir / 'metrics.json').unlink(),
        'missing-csv': lambda: (artifact_dir / 'demo_candidate_ranking.csv').unlink(),
        'malformed-json': lambda: (artifact_dir / 'metrics.json').write_text(
            '{', encoding='utf-8'
        ),
        'malformed-csv': lambda: (artifact_dir / 'benchmark_results.csv').write_bytes(
            b'\xff\xfe'
        ),
        'different-json': lambda: (artifact_dir / 'metrics.json').write_text(
            '{"mae": 99.0}\n', encoding='utf-8'
        ),
        'different-csv': lambda: (artifact_dir / 'benchmark_results.csv').write_text(
            'model_type,mae\nolder,99.0\n', encoding='utf-8'
        ),
        'changed-summary': lambda: (artifact_dir / 'experiment_summary.json').write_text(
            '{"dataset":{"rows":999}}\n', encoding='utf-8'
        ),
        'changed-manifest': lambda: (artifact_dir / 'manifest.json').write_text(
            json.dumps({**manifest, 'name': 'older_dataset'}), encoding='utf-8'
        ),
        'changed-optional': lambda: (artifact_dir / 'robustness_results.csv').write_text(
            'model_type,mae_mean\nolder,99.0\n', encoding='utf-8'
        ),
        'uncommitted-known-optional': lambda: (
            artifact_dir / 'bn_family_predictions.csv'
        ).write_text(
            'formula,target,prediction\nBN,5.0,4.0\n',
            encoding='utf-8',
        ),
        'unrelated-extra': lambda: (
            (artifact_dir / 'cache').mkdir(),
            (artifact_dir / 'cache' / 'scratch.bin').write_bytes(b'unrelated'),
        ),
    }
    if case == 'relocated-root':
        relocated_artifact_dir = tmp_path / 'relocated-artifacts'
        shutil.copytree(artifact_dir, relocated_artifact_dir)
        artifact_dir = relocated_artifact_dir
        cfg['project']['artifact_dir'] = str(artifact_dir)
    else:
        mutations[case]()

    provenance = io_utils.read_json_file(artifact_dir / 'artifact_provenance.json')
    current_manifest = io_utils.read_json_file(artifact_dir / 'manifest.json')
    assessment = io_utils.assess_artifact_provenance(
        provenance,
        cfg,
        current_manifest,
        project_root_path=tmp_path,
    )
    assert (
        assessment['status'] == 'current'
    ) is expected_assessor_current

    wrapper_path = tmp_path / f'streamlit_{case}.py'
    wrapper_path.write_text(
        'from runtime import io_utils\n'
        '_original_source_reader = io_utils._read_local_source_state\n'
        "io_utils._read_local_source_state = lambda _root: "
        "{'revision': 'abc123', 'dirty': False}\n"
        'from ui import streamlit_app\n'
        '_original_config = streamlit_app.CONFIG\n'
        f'streamlit_app.CONFIG = {cfg!r}\n'
        'try:\n'
        '    streamlit_app.render_streamlit_app()\n'
        'finally:\n'
        '    streamlit_app.CONFIG = _original_config\n'
        '    io_utils._read_local_source_state = _original_source_reader\n',
        encoding='utf-8',
    )
    app = AppTest.from_file(str(wrapper_path)).run(timeout=10)

    assert len(app.exception) == 0
    assert (len(app.success) == 1) is expected_viewer_current
    if not expected_viewer_current:
        assert len(app.warning) >= 1
    rendered_subheaders = [subheader.value for subheader in app.subheader]
    if not expected_assessor_current:
        assert 'Metrics' not in rendered_subheaders
        assert 'Benchmark results' not in rendered_subheaders
    if case == 'uncommitted-known-optional':
        assert 'Metrics' not in rendered_subheaders
        assert 'BN family holdout predictions' not in rendered_subheaders


def test_streamlit_app_runs_through_real_streamlit_renderer(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[3]
    app = AppTest.from_file(str(root / 'src' / 'ui' / 'streamlit_app.py')).run(timeout=10)

    assert len(app.exception) == 0
    assert [node.value for node in app.info] == [
        'Run `python main.py` first to generate artifacts.'
    ]
