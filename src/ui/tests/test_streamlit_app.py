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


def _save_structure_execution_bundle(
    artifact_dir: Path,
    *,
    execution_paths: dict[str, str] | None,
    execution_active: bool,
    summary_execution_overrides: dict[str, object] | None = None,
    summary_mutator=None,
):
    from pymatgen.core import Lattice, Structure

    from runtime import io_utils
    from materials.data import _structure_summary_from_atoms
    from materials.summary import build_experiment_summary
    from materials.structure_helpers import (
        _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS,
        _pair_distance_statistics,
        _structure_first_pass_execution_config,
        _structure_to_atoms,
    )

    root = Path(__file__).resolve().parents[3]
    cfg = io_utils.load_config(root / 'src' / 'config.py')
    cfg['project']['artifact_dir'] = str(artifact_dir)
    for gate in (
        'ranking_stability',
        'decision_policy',
        'proposal_shortlist',
        'extrapolation_shortlist',
    ):
        cfg['screening'][gate]['enabled'] = False
    if execution_paths is not None:
        cfg['screening']['structure_first_pass_execution'].update(execution_paths)
    cfg['screening']['structure_first_pass_execution'][
        'max_variants_per_candidate'
    ] = 1

    dataset_df = pd.DataFrame([{
        'formula': 'BN', 'target': 5.0, 'band_gap': 5.0,
    }])
    candidate_df = pd.DataFrame([{
        'formula': 'AlBN', 'ranking_rank': 1, 'ranking_score': 1.0,
        'predicted_band_gap': 5.0,
    }])
    structure_generation_seed_df = pd.DataFrame([{
        'formula': 'AlBN', 'structure_generation_seed_status': 'ok',
        'seed_reference_formula': 'B2N', 'seed_reference_record_id': 'jid-1',
    }])
    selection_summary = {
        'selected_feature_set': cfg['features']['feature_set'],
        'selected_model_type': cfg['model']['type'],
        'selected_feature_family': 'composition_only',
        'screening_selected_feature_set': cfg['features']['feature_set'],
        'screening_selected_model_type': cfg['model']['type'],
        'screening_selected_feature_family': 'composition_only',
    }
    execution_cfg = _structure_first_pass_execution_config(cfg)
    structure = Structure(
        Lattice.tetragonal(4.0, 20.0),
        ['Al', 'B', 'N'],
        [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    )
    atoms = _structure_to_atoms(structure)
    reference_structure = Structure(
        structure.lattice,
        ['B', 'B', 'N'],
        structure.frac_coords,
    )
    raw_dir = artifact_dir.parent / f'{artifact_dir.name}-raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg['data'].update({
        'dataset': 'twod_matpd',
        'raw_dir': str(raw_dir),
    })
    (raw_dir / 'twod_matpd.json').write_text(
        json.dumps([{
            'jid': 'jid-1',
            'formula': 'B2N',
            'atoms': _structure_to_atoms(reference_structure),
        }]),
        encoding='utf-8',
    )
    structure_summary = _structure_summary_from_atoms(atoms)
    (
        min_distance,
        min_distance_ratio,
        overlap_pair_count,
        mean_distance,
    ) = _pair_distance_statistics(
        structure,
        overlap_threshold=execution_cfg[
            'geometry_min_distance_ratio_overlap_threshold'
        ],
    )
    variant_row = {
        'formula': 'AlBN',
        'execution_variant_id': 'albn__variant_01',
        'execution_variant_rank': 1,
        'execution_status': 'ok',
        'execution_message': None,
        'seed_reference_formula': 'B2N',
        'seed_reference_record_id': 'jid-1',
        'execution_plan_type': 'edited_structure',
        'relabel_site_indices': '0',
        'relabel_target_elements': 'Al',
        'removed_site_indices': '',
        'relabeled_site_count': 1,
        'removed_site_count': 0,
        'formula_matches_candidate': True,
        'geometry_sanity_pass': True,
        'execution_variant_selection_score': 0.0,
        'generated_structure_cif_path': (
            f"{execution_cfg['structure_dir']}/albn__variant_01.cif"
        ),
        'generated_formula': 'AlBN',
        'generated_structure_n_sites': len(structure),
        'geometry_min_distance': min_distance,
        'geometry_mean_distance': mean_distance,
        'geometry_min_distance_ratio': min_distance_ratio,
        'geometry_overlap_pair_count': overlap_pair_count,
        'structure_band_gap_proxy': None,
        'relaxation_status': 'not_run_unrelaxed_species_edit',
        'final_status': 'ready_for_external_relaxation',
        **structure_summary,
    }
    structure_payload = {
        field: execution_cfg[field]
        for field in (
            'enabled', 'artifact', 'summary_artifact', 'variants_artifact', 'structure_dir',
        )
    }
    structure_payload.update({
        'candidate_count': int(execution_active),
        'variant_count': int(execution_active),
        'successful_variant_count': int(execution_active),
        'status_counts': {'executed': 1} if execution_active else {},
        'executed_formulas': ['AlBN'] if execution_active else [],
        'candidates': ([{
            'formula': 'AlBN',
            'seed_reference_formula': 'B2N',
            'seed_reference_record_id': 'jid-1',
            'candidate_status': 'executed',
            'selected_variant_id': 'albn__variant_01',
            'variants': [{
                **variant_row,
                'atoms': atoms,
                '_cif_text': structure.to(fmt='cif'),
            }],
        }] if execution_active else []),
    })
    structure_summary_df = (
        pd.DataFrame([{
            'formula': 'AlBN',
            'first_pass_execution_variant_count': 1,
            'first_pass_execution_successful_variant_count': 1,
            'first_pass_execution_geometry_pass_variant_count': 1,
            'first_pass_execution_status': 'executed',
            'structure_followup_best_seed_reference_formula': 'B2N',
            'structure_followup_best_seed_reference_record_id': 'jid-1',
            **{
                summary_field: variant_row[variant_field]
                for summary_field, variant_field in (
                    _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
                )
            },
        }])
        if execution_active
        else pd.DataFrame()
    )
    structure_variant_df = (
        pd.DataFrame([variant_row])
        if execution_active
        else pd.DataFrame()
    )
    summary = build_experiment_summary(
        dataset_df,
        dataset_df,
        candidate_df,
        {
            'train': [True],
            'val': [False],
            'test': [False],
            'metadata': {},
        },
        selection_summary,
        cfg,
        structure_generation_seed_df=structure_generation_seed_df,
        structure_first_pass_execution_summary_df=structure_summary_df,
        structure_first_pass_execution_payload=structure_payload,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    save_metrics_and_predictions(
        {'mae': 0.0},
        dataset_df.assign(prediction=5.0),
        dataset_df,
        candidate_df,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        structure_generation_seed_df,
        summary,
        manifest,
        cfg,
        structure_first_pass_execution_variant_df=structure_variant_df,
        structure_first_pass_execution_summary_df=structure_summary_df,
        structure_first_pass_execution_payload=structure_payload,
    )
    if summary_execution_overrides or summary_mutator is not None:
        provenance_path = artifact_dir / 'artifact_provenance.json'
        published_relative_paths = tuple(
            io_utils.read_json_file(provenance_path)['published_outputs']
        )
        provenance_path.unlink()
        if summary_execution_overrides:
            summary['screening']['structure_generation_bridge'].update(
                summary_execution_overrides
            )
        if summary_mutator is not None:
            summary_mutator(summary)
        io_utils.write_json_file(
            summary,
            artifact_dir / 'experiment_summary.json',
            ensure_ascii=False,
            indent=2,
        )
        io_utils.write_json_file(
            io_utils.build_artifact_provenance(
                cfg,
                manifest,
                published_output_paths=tuple(
                    artifact_dir / relative_path
                    for relative_path in published_relative_paths
                ),
            ),
            provenance_path,
            indent=2,
        )
    return cfg, summary, execution_cfg


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


def _run_real_streamlit_app(tmp_path: Path, cfg: dict, module_name: str):
    from streamlit.testing.v1 import AppTest

    wrapper_path = tmp_path / f'{module_name}.py'
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
    return AppTest.from_file(str(wrapper_path)).run(timeout=10)


def _stub_current_source(monkeypatch):
    from runtime import io_utils

    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    return io_utils


def _assert_saved_bundle_current(io_utils, artifact_dir, cfg, project_root):
    provenance = io_utils.read_json_file(
        artifact_dir / 'artifact_provenance.json'
    )
    manifest = io_utils.read_json_file(artifact_dir / 'manifest.json')
    assert io_utils.assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=project_root,
    )['status'] == 'current'


def _assert_only_provenance_rendered(app):
    assert len(app.exception) == 0
    assert len(app.success) == 0
    assert {node.value for node in app.subheader} == {
        'Artifact bundle provenance'
    }


def test_structure_execution_output_role_contract_is_nonvacuous():
    from materials.structure_execution import (
        build_structure_first_pass_execution_artifacts,
    )
    from materials.structure_helpers import _structure_first_pass_execution_config
    from runtime.schema import STRUCTURE_EXECUTION_OUTPUT_ROLES
    from ui import streamlit_app

    execution_cfg = _structure_first_pass_execution_config({})
    _variant_df, _summary_df, builder_payload = (
        build_structure_first_pass_execution_artifacts(pd.DataFrame(), cfg={})
    )
    assert len(STRUCTURE_EXECUTION_OUTPUT_ROLES) == 3
    assert all(len(role) == 5 for role in STRUCTURE_EXECUTION_OUTPUT_ROLES)
    assert len({role[0] for role in STRUCTURE_EXECUTION_OUTPUT_ROLES}) == 3
    assert len({role[1] for role in STRUCTURE_EXECUTION_OUTPUT_ROLES}) == 3
    assert len({role[2] for role in STRUCTURE_EXECUTION_OUTPUT_ROLES}) == 3
    assert streamlit_app._SUMMARY_EXECUTION_PATH_FIELDS == {
        artifact_key: summary_field
        for artifact_key, summary_field, _config_field, _suffix, _default_path
        in STRUCTURE_EXECUTION_OUTPUT_ROLES
    }
    assert all(
        artifact_key in streamlit_app.ARTIFACT_PATHS
        and streamlit_app.ARTIFACT_PATHS[artifact_key].suffix.casefold() == suffix
        and streamlit_app.ARTIFACT_PATHS[artifact_key].relative_to(
            streamlit_app.DEFAULT_ARTIFACT_ROOT
        ).as_posix() == execution_cfg[config_field] == default_path
        for artifact_key, _summary_field, config_field, suffix, default_path
        in STRUCTURE_EXECUTION_OUTPUT_ROLES
    )
    assert all(
        builder_payload[config_field]
        == execution_cfg[config_field]
        == default_path
        for _artifact_key, _summary_field, config_field, _suffix, default_path
        in STRUCTURE_EXECUTION_OUTPUT_ROLES
    )


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

    current_app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        'all_artifact_sections',
    )
    uncommitted_provenance = {
        **provenance,
        'published_outputs': dict(provenance['published_outputs']),
    }
    uncommitted_provenance['published_outputs'].pop(
        'demo_candidate_structure_generation_first_pass_execution_variants.csv'
    )
    (artifact_dir / 'artifact_provenance.json').write_text(
        json.dumps(uncommitted_provenance),
        encoding='utf-8',
    )
    downgraded_app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        'all_artifact_sections_downgraded',
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

    csv_keys = [key for _title, key in module.CSV_SECTIONS]
    json_keys = [key for _title, key in module.JSON_SECTIONS]
    assert csv_keys and len(csv_keys) == len(set(csv_keys))
    assert json_keys and len(json_keys) == len(set(json_keys))
    report_keys = {'metrics', 'summary', *csv_keys, *json_keys}
    assert set(module.ARTIFACT_PATHS) - {'provenance', 'manifest'} == report_keys
    expected_report_titles = {
        'Metrics',
        'Experiment summary',
        *(title for title, _key in module.CSV_SECTIONS),
        *(title for title, _key in module.JSON_SECTIONS),
    }
    assert len(current_app.exception) == 0
    assert {
        node.value for node in current_app.subheader
    } - {'Artifact bundle provenance'} == expected_report_titles
    assert len(downgraded_app.exception) == 0
    assert {node.value for node in downgraded_app.subheader} == {
        'Artifact bundle provenance'
    }

    assert ('title', 'AI-Powered Boron Nitride Material Exploration') in fake_streamlit.calls
    assert expected_report_titles <= {
        value for call_name, value in fake_streamlit.calls if call_name == 'subheader'
    }
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
        'screening': {
            'structure_first_pass_execution': {
                'artifact': 'nested/execution.json',
                'summary_artifact': 'nested/summary.csv',
                'variants_artifact': 'nested/variants.csv',
            },
        },
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


def test_streamlit_empty_custom_execution_ignores_stale_default_paths(
    tmp_path,
    monkeypatch,
):
    from runtime import io_utils
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    artifact_dir = tmp_path / 'artifacts'
    _default_cfg, _default_summary, default_execution_cfg = (
        _save_structure_execution_bundle(
            artifact_dir,
            execution_paths=None,
            execution_active=True,
        )
    )
    custom_execution_paths = {
        'artifact': 'custom/execution.json',
        'summary_artifact': 'custom/execution-summary.csv',
        'variants_artifact': 'custom/execution-variants.csv',
        'structure_dir': 'custom/cifs',
    }
    _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=custom_execution_paths,
        execution_active=True,
    )
    cfg, summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=custom_execution_paths,
        execution_active=False,
    )

    bridge = summary['screening']['structure_generation_bridge']
    assert bridge.get('first_pass_execution_artifact') is None
    assert all(
        (artifact_dir / default_execution_cfg[field]).exists()
        for field in ('artifact', 'summary_artifact', 'variants_artifact')
    )
    provenance = io_utils.read_json_file(
        artifact_dir / 'artifact_provenance.json'
    )
    assert not any(
        path in provenance['published_outputs']
        for path in (
            default_execution_cfg['artifact'],
            default_execution_cfg['summary_artifact'],
            default_execution_cfg['variants_artifact'],
        )
    )

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        'empty_custom_execution',
    )

    assert len(app.exception) == 0
    assert len(app.success) == 1
    rendered_subheaders = {subheader.value for subheader in app.subheader}
    assert 'Metrics' in rendered_subheaders
    assert 'Benchmark results' in rendered_subheaders
    assert 'Structure first-pass execution summary' not in rendered_subheaders
    assert 'Structure first-pass execution variants' not in rendered_subheaders
    assert 'Structure first-pass execution JSON' not in rendered_subheaders


@pytest.mark.parametrize(
    'override_case',
    [
        'blank',
        'traversal',
        'absolute',
        'non-string',
        'directory',
        'wrong-suffix',
        'fixed-alias',
        'missing',
    ],
)
def test_streamlit_rejects_invalid_dynamic_summary_path_contract(
    tmp_path,
    monkeypatch,
    override_case,
):
    from runtime import io_utils

    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    artifact_dir = tmp_path / 'artifacts'
    override_values = {
        'blank': ' ',
        'traversal': '../escape.json',
        'absolute': str(tmp_path / 'outside.json'),
        'non-string': 7,
        'directory': 'nested/execution.json',
        'wrong-suffix': 'nested/execution.csv',
        'fixed-alias': 'metrics.json',
        'missing': 'nested/missing.json',
    }
    if override_case == 'directory':
        (artifact_dir / 'nested' / 'execution.json').mkdir(parents=True)
    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=False,
        summary_execution_overrides={
            'first_pass_execution_artifact': override_values[override_case],
        },
    )
    provenance = io_utils.read_json_file(
        artifact_dir / 'artifact_provenance.json'
    )
    manifest = io_utils.read_json_file(artifact_dir / 'manifest.json')
    assert io_utils.assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['status'] == 'current'

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        f'invalid_dynamic_{override_case}',
    )

    assert len(app.exception) == 0
    assert len(app.success) == 0
    assert {node.value for node in app.subheader} == {
        'Artifact bundle provenance'
    }


@pytest.mark.parametrize(
    'summary_execution_overrides',
    [
        pytest.param(
            {'first_pass_execution_summary_artifact': 'bn_slice.csv'},
            id='summary-relabels-unviewed-csv',
        ),
        pytest.param(
            {'first_pass_execution_variants_artifact': 'bn_slice.csv'},
            id='variants-relabels-unviewed-csv',
        ),
        pytest.param(
            {
                'first_pass_execution_summary_artifact': (
                    'demo_candidate_structure_generation_first_pass_execution_variants.csv'
                ),
                'first_pass_execution_variants_artifact': (
                    'demo_candidate_structure_generation_first_pass_execution_summary.csv'
                ),
            },
            id='summary-and-variants-swap-identities',
        ),
    ],
)
def test_streamlit_rejects_relabelled_committed_dynamic_output_identity(
    tmp_path,
    monkeypatch,
    summary_execution_overrides,
):
    io_utils = _stub_current_source(monkeypatch)
    artifact_dir = tmp_path / 'artifacts'
    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=True,
        summary_execution_overrides=summary_execution_overrides,
    )
    _assert_saved_bundle_current(io_utils, artifact_dir, cfg, tmp_path)

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        'relabelled_dynamic_output_identity',
    )

    _assert_only_provenance_rendered(app)


@pytest.mark.parametrize(
    'execution_config_overrides',
    [
        pytest.param(
            {'artifact': 'metrics.json'},
            id='json-role-relabels-fixed-json',
        ),
        pytest.param(
            {'summary_artifact': 'bn_slice.csv'},
            id='summary-role-relabels-fixed-csv',
        ),
        pytest.param(
            {'summary_artifact': 'BN_SLICE.CSV'},
            id='summary-role-casefolds-to-fixed-csv',
        ),
        pytest.param(
            {
                'summary_artifact': 'bn_slice.csv',
                'variants_artifact': 'bn_slice.csv',
            },
            id='two-roles-identify-the-same-file',
        ),
        pytest.param(
            {'artifact': 'nested/execution.csv'},
            id='configured-role-has-wrong-suffix',
        ),
    ],
)
def test_streamlit_rejects_invalid_configured_dynamic_output_identity_without_summary_declarations(
    tmp_path,
    monkeypatch,
    execution_config_overrides,
):
    from runtime.schema import STRUCTURE_EXECUTION_OUTPUT_ROLES

    io_utils = _stub_current_source(monkeypatch)
    artifact_dir = tmp_path / 'artifacts'

    def remove_execution_declarations(summary):
        summary['screening']['structure_generation_bridge'] = {}

    cfg, _summary, execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=True,
        summary_mutator=remove_execution_declarations,
    )
    cfg['screening']['structure_first_pass_execution'].update(
        execution_config_overrides
    )
    provenance_path = artifact_dir / 'artifact_provenance.json'
    superseded_role_paths = {
        execution_cfg[config_field]
        for _artifact_key, _summary_field, config_field, _suffix, _default_path
        in STRUCTURE_EXECUTION_OUTPUT_ROLES
        if config_field in execution_config_overrides
    }
    published_relative_paths = tuple(
        relative_path
        for relative_path in io_utils.read_json_file(provenance_path)[
            'published_outputs'
        ]
        if relative_path not in superseded_role_paths
    )
    provenance_path.unlink()
    manifest = io_utils.read_json_file(artifact_dir / 'manifest.json')
    io_utils.write_json_file(
        io_utils.build_artifact_provenance(
            cfg,
            manifest,
            published_output_paths=tuple(
                artifact_dir / relative_path
                for relative_path in published_relative_paths
            ),
        ),
        provenance_path,
        indent=2,
    )
    _assert_saved_bundle_current(io_utils, artifact_dir, cfg, tmp_path)

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        'invalid_configured_dynamic_output_identity',
    )

    _assert_only_provenance_rendered(app)


def test_streamlit_accepts_normalized_equivalent_dynamic_output_identity(
    tmp_path,
    monkeypatch,
):
    _stub_current_source(monkeypatch)
    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        tmp_path / 'artifacts',
        execution_paths={
            'artifact': 'nested/a/../execution.json',
            'summary_artifact': 'nested/a/../execution-summary.csv',
            'variants_artifact': 'nested/a/../execution-variants.csv',
            'structure_dir': 'nested/a/../cifs',
        },
        execution_active=True,
    )

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        'normalized_equivalent_dynamic_output_identity',
    )

    assert len(app.exception) == 0
    assert len(app.success) == 1
    rendered_subheaders = {node.value for node in app.subheader}
    assert 'Structure first-pass execution JSON' in rendered_subheaders
    assert 'Structure first-pass execution summary' in rendered_subheaders
    assert 'Structure first-pass execution variants' in rendered_subheaders


def test_streamlit_accepts_case_only_dynamic_identity_when_samefile(
    tmp_path,
    monkeypatch,
):
    _stub_current_source(monkeypatch)
    artifact_dir = tmp_path / 'artifacts'
    configured_name = (
        'demo_candidate_structure_generation_first_pass_execution.json'
    )
    summary_name = configured_name.upper()
    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=True,
        summary_execution_overrides={
            'first_pass_execution_artifact': summary_name,
        },
    )
    configured_path = artifact_dir / configured_name
    summary_path = artifact_dir / summary_name
    if not summary_path.exists():
        pytest.skip('local filesystem treats case-only names as distinct files')
    assert configured_path.samefile(summary_path)

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        'case_only_samefile_dynamic_output_identity',
    )

    assert len(app.exception) == 0
    assert len(app.success) == 1
    assert 'Structure first-pass execution JSON' in {
        node.value for node in app.subheader
    }


@pytest.mark.parametrize(
    'shape_value',
    [
        pytest.param('', id='empty-string'),
        pytest.param('wrong-shape', id='nonempty-string'),
        pytest.param([], id='empty-list'),
        pytest.param([1], id='nonempty-list'),
        pytest.param(0, id='zero'),
        pytest.param(1, id='nonzero-number'),
        pytest.param(False, id='false'),
        pytest.param(True, id='true'),
    ],
)
@pytest.mark.parametrize('container_name', ['screening', 'structure-generation-bridge'])
def test_streamlit_rejects_wrong_shaped_nested_summary_containers(
    tmp_path,
    monkeypatch,
    container_name,
    shape_value,
):
    io_utils = _stub_current_source(monkeypatch)

    def mutate_summary(summary):
        if container_name == 'screening':
            summary['screening'] = shape_value
        else:
            summary['screening']['structure_generation_bridge'] = shape_value

    artifact_dir = tmp_path / 'artifacts'
    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=False,
        summary_mutator=mutate_summary,
    )
    _assert_saved_bundle_current(io_utils, artifact_dir, cfg, tmp_path)

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        f'wrong_shape_{container_name}_{type(shape_value).__name__}',
    )

    _assert_only_provenance_rendered(app)


@pytest.mark.parametrize('empty_state', ['absent', 'null', 'empty-mapping'])
@pytest.mark.parametrize('container_name', ['screening', 'structure-generation-bridge'])
@pytest.mark.parametrize('execution_active', [False, True], ids=['inactive', 'active'])
def test_streamlit_accepts_semantically_empty_nested_summary_containers(
    tmp_path,
    monkeypatch,
    container_name,
    empty_state,
    execution_active,
):
    io_utils = _stub_current_source(monkeypatch)

    def mutate_summary(summary):
        parent = summary
        key = 'screening'
        if container_name == 'structure-generation-bridge':
            parent = summary['screening']
            key = 'structure_generation_bridge'
        if empty_state == 'absent':
            parent.pop(key, None)
        elif empty_state == 'null':
            parent[key] = None
        else:
            parent[key] = {}

    artifact_dir = tmp_path / 'artifacts'
    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=execution_active,
        summary_mutator=mutate_summary,
    )
    _assert_saved_bundle_current(io_utils, artifact_dir, cfg, tmp_path)

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        f'empty_shape_{container_name}_{empty_state}',
    )

    assert len(app.exception) == 0
    assert len(app.success) == 1
    rendered_subheaders = {node.value for node in app.subheader}
    assert 'Metrics' in rendered_subheaders
    for subheader in (
        'Structure first-pass execution JSON',
        'Structure first-pass execution summary',
        'Structure first-pass execution variants',
    ):
        assert (subheader in rendered_subheaders) is execution_active


@pytest.mark.parametrize('container_name', ['screening', 'structure-generation-bridge'])
def test_streamlit_same_root_wrong_shape_transition_recovers_cleanly(
    tmp_path,
    monkeypatch,
    container_name,
):
    _stub_current_source(monkeypatch)
    artifact_dir = tmp_path / 'artifacts'
    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=True,
    )
    initial_marker = (artifact_dir / 'artifact_provenance.json').read_bytes()
    initial_app = _run_real_streamlit_app(tmp_path, cfg, 'shape_transition_initial')
    assert len(initial_app.success) == 1

    def corrupt_summary(summary):
        if container_name == 'screening':
            summary['screening'] = []
        else:
            summary['screening']['structure_generation_bridge'] = []

    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=True,
        summary_mutator=corrupt_summary,
    )
    invalid_marker = (artifact_dir / 'artifact_provenance.json').read_bytes()
    assert invalid_marker != initial_marker
    invalid_app = _run_real_streamlit_app(tmp_path, cfg, 'shape_transition_invalid')
    _assert_only_provenance_rendered(invalid_app)

    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=True,
    )
    assert (artifact_dir / 'artifact_provenance.json').read_bytes() == initial_marker
    recovered_app = _run_real_streamlit_app(tmp_path, cfg, 'shape_transition_recovered')
    assert len(recovered_app.exception) == 0
    assert len(recovered_app.success) == 1
    recovered_subheaders = {node.value for node in recovered_app.subheader}
    assert 'Structure first-pass execution JSON' in recovered_subheaders
    assert 'Structure first-pass execution summary' in recovered_subheaders
    assert 'Structure first-pass execution variants' in recovered_subheaders


@pytest.mark.parametrize('empty_state', ['absent', 'null', 'empty-mapping'])
def test_streamlit_active_to_empty_bridge_transition_removes_dynamic_outputs(
    tmp_path,
    monkeypatch,
    empty_state,
):
    io_utils = _stub_current_source(monkeypatch)
    artifact_dir = tmp_path / 'artifacts'
    _cfg, _summary, execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=True,
    )

    def empty_bridge(summary):
        bridge_parent = summary['screening']
        if empty_state == 'absent':
            bridge_parent.pop('structure_generation_bridge', None)
        elif empty_state == 'null':
            bridge_parent['structure_generation_bridge'] = None
        else:
            bridge_parent['structure_generation_bridge'] = {}

    cfg, _summary, _execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=None,
        execution_active=False,
        summary_mutator=empty_bridge,
    )
    assert all(
        not (artifact_dir / execution_cfg[field]).exists()
        for field in ('artifact', 'summary_artifact', 'variants_artifact')
    )
    _assert_saved_bundle_current(io_utils, artifact_dir, cfg, tmp_path)
    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        f'active_to_empty_bridge_{empty_state}',
    )
    assert len(app.exception) == 0
    assert len(app.success) == 1
    rendered_subheaders = {node.value for node in app.subheader}
    assert 'Metrics' in rendered_subheaders
    assert 'Structure first-pass execution JSON' not in rendered_subheaders
    assert 'Structure first-pass execution summary' not in rendered_subheaders
    assert 'Structure first-pass execution variants' not in rendered_subheaders


@pytest.mark.parametrize(
    'path_field',
    ['artifact', 'summary_artifact', 'variants_artifact'],
)
@pytest.mark.parametrize('mutation', ['mismatch', 'missing'])
def test_streamlit_dynamic_outputs_require_matching_v2_content(
    tmp_path,
    monkeypatch,
    path_field,
    mutation,
):
    from runtime import io_utils

    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    custom_execution_paths = {
        'artifact': 'nested/execution.json',
        'summary_artifact': 'nested/execution-summary.csv',
        'variants_artifact': 'nested/execution-variants.csv',
        'structure_dir': 'nested/cifs',
    }
    artifact_dir = tmp_path / mutation / 'artifacts'
    cfg, _summary, execution_cfg = _save_structure_execution_bundle(
        artifact_dir,
        execution_paths=custom_execution_paths,
        execution_active=True,
    )
    output_path = artifact_dir / execution_cfg[path_field]
    if mutation == 'mismatch':
        output_path.write_bytes(b'changed-after-publication')
    else:
        output_path.unlink()
    provenance = io_utils.read_json_file(
        artifact_dir / 'artifact_provenance.json'
    )
    manifest = io_utils.read_json_file(artifact_dir / 'manifest.json')
    assert io_utils.assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['status'] != 'current'

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        f'dynamic_{path_field}_{mutation}',
    )

    assert len(app.exception) == 0
    assert len(app.success) == 0
    assert {node.value for node in app.subheader} == {
        'Artifact bundle provenance'
    }


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

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        f'{prediction_kind}_{provenance_case}',
    )

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
        ('malformed-summary', False, False),
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
        'malformed-summary': lambda: (
            artifact_dir / 'experiment_summary.json'
        ).write_text('{', encoding='utf-8'),
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

    app = _run_real_streamlit_app(
        tmp_path,
        cfg,
        f'streamlit_{case}',
    )

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
