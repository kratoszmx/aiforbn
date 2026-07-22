from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from runtime import io_utils
from materials import artifacts as artifacts_module
from materials import plots as plots_module
from materials import structure_execution as structure_execution_module
from materials.artifacts import save_metrics_and_predictions
from materials.benchmarking import benchmark_bn_family_holdout, benchmark_bn_slice
from materials.constants import NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION
from materials.data import (
    REFERENCE_PROPERTY_COLUMNS,
    STRUCTURE_SUMMARY_COLUMNS,
    _normalize,
    _structure_summary_from_atoms,
)
from materials.plots import save_basic_plots
from materials.ranking_tables import (
    _build_bn_model_role_comparison_table,
    _candidate_ranking_uncertainty_table,
)
from materials.screening import build_candidate_structure_generation_seeds
from materials.summary import build_experiment_summary
from materials.structure_execution import build_structure_first_pass_execution_artifacts
from materials.structure_helpers import (
    _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH,
    _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS,
    _STRUCTURE_EXECUTION_VARIANT_STATUS_BY_BRANCH,
    _STRUCTURE_EXECUTION_ZERO_VARIANT_STATUSES,
    _pair_distance_statistics,
    _select_structure_execution_variant,
    _structure_first_pass_execution_config,
    _structure_to_atoms,
)


def _save_minimal_report_bundle(
    cfg,
    *,
    bn_df=None,
    screened_df=None,
    structure_generation_seed_df=None,
    experiment_summary=None,
    structure_payload=None,
    structure_summary_df=None,
    structure_variant_df=None,
    prediction_df=None,
    include_parity_plot=False,
    manifest=None,
):
    empty_df = pd.DataFrame()
    return save_metrics_and_predictions(
        {},
        empty_df if prediction_df is None else prediction_df,
        empty_df if bn_df is None else bn_df,
        (
            pd.DataFrame(columns=['formula', 'ranking_rank'])
            if screened_df is None
            else screened_df
        ),
        empty_df,
        empty_df,
        empty_df,
        empty_df,
        empty_df,
        (
            empty_df
            if structure_generation_seed_df is None
            else structure_generation_seed_df
        ),
        {} if experiment_summary is None else experiment_summary,
        {} if manifest is None else manifest,
        cfg,
        structure_first_pass_execution_variant_df=structure_variant_df,
        structure_first_pass_execution_summary_df=structure_summary_df,
        structure_first_pass_execution_payload=structure_payload,
        include_parity_plot=include_parity_plot,
    )


def test_plot_module_rejects_human_docs_mpl_cache_before_import(tmp_path):
    project_root = tmp_path / 'synthetic-project'
    human_docs_cache = project_root / 'human_docs' / 'mpl'
    env = dict(os.environ)
    env.update({
        'AIFORBN_SYNTHETIC_PROJECT_ROOT': str(project_root),
        'MPLCONFIGDIR': str(human_docs_cache),
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
    })
    script = (
        'import os\n'
        'from pathlib import Path\n'
        'from runtime import io_utils\n'
        "io_utils.PROJECT_ROOT = Path(os.environ['AIFORBN_SYNTHETIC_PROJECT_ROOT'])\n"
        'try:\n'
        '    import materials.plots\n'
        'except ValueError as exc:\n'
        "    assert 'user-owned human_docs' in str(exc)\n"
        'else:\n'
        "    raise AssertionError('materials.plots import must reject protected MPLCONFIGDIR')\n"
    )

    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (project_root / 'human_docs').exists()


def test_public_artifact_and_plot_writers_reject_human_docs_output(tmp_path, monkeypatch):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    cfg = {'project': {'artifact_dir': str(tmp_path / 'human_docs' / 'artifacts')}}
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='user-owned human_docs'):
        save_basic_plots(empty_df, cfg)

    with pytest.raises(ValueError, match='user-owned human_docs'):
        save_metrics_and_predictions(
            {},
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            {},
            {},
            cfg,
        )

    assert not (tmp_path / 'human_docs').exists()


@pytest.mark.parametrize('writer_name', ['artifacts', 'plot'])
@pytest.mark.parametrize('alias_kind', ['broken_symlink', 'hardlink'])
def test_public_artifact_writers_reject_leaf_aliases_into_human_docs(
    tmp_path,
    monkeypatch,
    writer_name,
    alias_kind,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    output_name = 'predictions.csv' if writer_name == 'artifacts' else 'parity_plot.png'
    human_docs_file = tmp_path / 'human_docs' / output_name
    human_docs_file.parent.mkdir()
    output_alias = artifact_dir / output_name
    if alias_kind == 'broken_symlink':
        output_alias.symlink_to(human_docs_file)
    else:
        human_docs_file.write_text('user-owned', encoding='utf-8')
        output_alias.hardlink_to(human_docs_file)
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='human_docs|multiple hard links'):
        if writer_name == 'plot':
            save_basic_plots(empty_df, cfg)
        else:
            save_metrics_and_predictions(
                {},
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                {},
                {},
                cfg,
            )

    if alias_kind == 'broken_symlink':
        assert not human_docs_file.exists()
    else:
        assert human_docs_file.read_text(encoding='utf-8') == 'user-owned'


@pytest.mark.parametrize('writer_name', ['artifacts', 'plot'])
@pytest.mark.parametrize(
    'invalid_leaf_kind',
    ['external_symlink', 'in_root_symlink', 'directory'],
)
def test_public_artifact_writers_reject_invalid_file_leaves_before_effects(
    tmp_path,
    monkeypatch,
    writer_name,
    invalid_leaf_kind,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    output_name = 'predictions.csv' if writer_name == 'artifacts' else 'parity_plot.png'
    output_path = artifact_dir / output_name
    external_target = tmp_path / 'outside' / output_name
    if invalid_leaf_kind == 'external_symlink':
        output_path.symlink_to(external_target)
    elif invalid_leaf_kind == 'in_root_symlink':
        output_path.symlink_to(artifact_dir / 'metrics.json')
    else:
        output_path.mkdir()
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='configured output root|symbolic-link|regular-file'):
        if writer_name == 'plot':
            save_basic_plots(empty_df, cfg)
        else:
            save_metrics_and_predictions(
                {},
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                empty_df,
                {},
                {},
                cfg,
            )

    assert not external_target.exists()
    assert not (artifact_dir / 'metrics.json').exists()


def test_reporting_rejects_non_directory_structure_parent_before_fixed_outputs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    nested_parent = artifact_dir / 'nested'
    nested_parent.write_text('keep', encoding='utf-8')
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'structure_first_pass_execution': {
                'artifact': 'nested/execution.json',
                'summary_artifact': 'nested/summary.csv',
                'variants_artifact': 'nested/variants.csv',
                'structure_dir': 'nested/structures',
            },
        },
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='parent paths'):
        save_metrics_and_predictions(
            {},
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            empty_df,
            {},
            {},
            cfg,
        )

    assert nested_parent.read_text(encoding='utf-8') == 'keep'
    assert not (artifact_dir / 'metrics.json').exists()
    assert not (artifact_dir / 'predictions.csv').exists()


def test_reserved_artifact_collision_contract_covers_writer_literals():
    source_path = Path(artifacts_module.__file__)
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    writer_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'save_metrics_and_predictions'
    )
    literal_artifact_names = {
        node.value
        for node in ast.walk(writer_node)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.endswith(('.csv', '.json', '.png'))
    }

    assert literal_artifact_names <= artifacts_module._RESERVED_REPORT_ARTIFACT_NAMES


def test_decision_policy_holds_candidates_outside_application_target_windows():
    cfg = {
        'screening': {
            'top_k': 10,
            'decision_policy': {
                'enabled': True,
                'global_support_abstain_below_percentile': 25.0,
                'bn_support_abstain_below_percentile': 25.0,
                'prediction_std_above_quantile': 0.75,
                'rank_std_above_quantile': 0.75,
                'minimum_top_10_selection_frequency': 0.5,
                'application_tracks': [
                    {
                        'label': 'uv_wide_band_gap',
                        'target_window_eV': [4.5, 6.5],
                        'note': 'UV/WBG formula-stage proxy.',
                    },
                    {
                        'label': 'dielectric_2d_support',
                        'target_window_eV': [4.5, 8.0],
                        'note': 'Dielectric support formula-stage proxy.',
                    },
                ],
            },
        },
    }
    candidate_df = pd.DataFrame({
        'formula': ['XBN'],
        'ranking_rank': [1],
        'ranking_score': [10.0],
        'predicted_band_gap': [8.6],
        'ensemble_predicted_band_gap_mean': [8.6],
        'ensemble_predicted_band_gap_std': [0.0],
        'domain_support_percentile': [100.0],
        'domain_support_mean_k_distance': [0.1],
        'bn_support_percentile': [100.0],
        'bn_support_mean_k_distance': [0.1],
        'chemical_plausibility_pass': [True],
        'candidate_novelty_bucket': [NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION],
        'proposal_shortlist_selected': [False],
        'proposal_shortlist_rank': [pd.NA],
        'extrapolation_shortlist_selected': [True],
        'extrapolation_shortlist_rank': [1],
        'bn_band_gap_alignment_label': ['within_local_bn_analog_band_gap_window'],
    })
    structure_followup_shortlist_df = pd.DataFrame({
        'formula': ['XBN'],
        'structure_followup_shortlist_selected': [True],
        'structure_followup_shortlist_rank': [1],
        'structure_followup_priority_score': [1.0],
        'structure_followup_best_queue_rank': [1],
        'structure_followup_best_action_label': ['prototype_transfer'],
        'structure_followup_readiness_label': ['ready'],
    })

    uncertainty_df, summary = _candidate_ranking_uncertainty_table(
        candidate_df,
        formula_col='formula',
        cfg=cfg,
        structure_followup_shortlist_df=structure_followup_shortlist_df,
    )

    row = uncertainty_df.iloc[0]
    assert row['application_track_primary'] == 'dielectric_2d_support'
    assert row['application_track_target_window_eV'] == '4.5-8'
    assert bool(row['abstain_flag']) is True
    assert row['final_action_label'] == 'hold'
    assert row['recommended_action_label'] == 'hold'
    assert 'application_target_window_above' in row['reason_for_abstention']
    assert summary['final_action_counts'] == {'hold': 1}


def test_disabled_decision_policy_leaves_ranking_uncertainty_policy_neutral():
    cfg = {
        'screening': {
            'top_k': 10,
            'decision_policy': {
                'enabled': False,
                'global_support_abstain_below_percentile': 25.0,
                'bn_support_abstain_below_percentile': 25.0,
                'prediction_std_above_quantile': 0.75,
                'rank_std_above_quantile': 0.75,
                'minimum_top_10_selection_frequency': 0.5,
                'application_tracks': [
                    {
                        'label': 'uv_wide_band_gap',
                        'target_window_eV': [4.5, 6.5],
                        'note': 'UV/WBG formula-stage proxy.',
                    },
                ],
            },
        },
    }
    candidate_df = pd.DataFrame({
        'formula': ['XBN'],
        'ranking_rank': [1],
        'ranking_score': [10.0],
        'predicted_band_gap': [8.6],
        'ensemble_predicted_band_gap_mean': [8.6],
        'ensemble_predicted_band_gap_std': [0.5],
        'domain_support_percentile': [0.0],
        'domain_support_mean_k_distance': [10.0],
        'bn_support_percentile': [0.0],
        'bn_support_mean_k_distance': [10.0],
        'chemical_plausibility_pass': [True],
        'candidate_novelty_bucket': [NOVELTY_BUCKET_FORMULA_LEVEL_EXTRAPOLATION],
        'proposal_shortlist_selected': [False],
        'proposal_shortlist_rank': [pd.NA],
        'extrapolation_shortlist_selected': [True],
        'extrapolation_shortlist_rank': [1],
        'bn_band_gap_alignment_label': ['above_local_bn_analog_band_gap_window'],
    })
    structure_followup_shortlist_df = pd.DataFrame({
        'formula': ['XBN'],
        'structure_followup_shortlist_selected': [True],
        'structure_followup_shortlist_rank': [1],
        'structure_followup_priority_score': [1.0],
        'structure_followup_best_queue_rank': [1],
        'structure_followup_best_action_label': ['prototype_transfer'],
        'structure_followup_readiness_label': ['ready'],
    })

    uncertainty_df, summary = _candidate_ranking_uncertainty_table(
        candidate_df,
        formula_col='formula',
        cfg=cfg,
        structure_followup_shortlist_df=structure_followup_shortlist_df,
    )

    row = uncertainty_df.iloc[0]
    assert row['predicted_band_gap_mean'] == 8.6
    assert bool(row['abstain_flag']) is False
    assert row['reason_for_abstention'] == ''
    assert pd.isna(row['final_action_label'])
    assert pd.isna(row['recommended_action_label'])
    assert pd.isna(row['application_track_primary'])
    assert summary['prediction_std_abstain_threshold'] is None
    assert summary['rank_std_abstain_threshold'] is None
    assert summary['abstained_candidate_count'] == 0
    assert summary['final_action_counts'] == {}


def test_ranking_uncertainty_deduplicates_one_canonical_prediction_source():
    cfg = {
        'screening': {
            'top_k': 1,
            'ranking_stability': {'enabled': True, 'top_k_values': [1]},
            'decision_policy': {'enabled': False},
        },
    }
    candidate_df = pd.DataFrame({
        'formula': ['XBN', 'YBN'],
        'ranking_rank': [1, 2],
        'ranking_score': [2.0, 1.0],
        'predicted_band_gap': [6.0, 5.0],
    })
    grouped_member_df = pd.DataFrame({
        'formula': ['XBN', 'YBN'],
        'prediction_source': [
            'group_kfold__basic_formula_composition__linear_regression__fold_1',
        ] * 2,
        'prediction_source_family': ['group_kfold_candidate_model'] * 2,
        'feature_set': ['basic_formula_composition'] * 2,
        'model_type': ['linear_regression'] * 2,
        'prediction': [6.0, 5.0],
    })

    baseline_df, baseline_summary = _candidate_ranking_uncertainty_table(
        candidate_df,
        formula_col='formula',
        cfg=cfg,
        candidate_grouped_robustness_member_df=grouped_member_df,
    )
    duplicated_df, duplicated_summary = _candidate_ranking_uncertainty_table(
        candidate_df,
        formula_col='formula',
        cfg=cfg,
        candidate_grouped_robustness_member_df=grouped_member_df,
        bn_centered_grouped_robustness_member_df=grouped_member_df.copy(),
    )

    columns = ['formula', 'ranking_source_count', 'rank_mean', 'top_1_selection_frequency']
    pd.testing.assert_frame_equal(
        duplicated_df[columns].reset_index(drop=True),
        baseline_df[columns].reset_index(drop=True),
    )
    assert duplicated_summary['source_count'] == baseline_summary['source_count'] == 1


def test_bn_model_role_comparison_preserves_one_model_identity_across_scopes():
    identity_columns = {
        'benchmark_role': 'candidate_model',
        'feature_family': 'composition',
        'candidate_compatible': True,
        'selected_by_validation': False,
    }
    slice_df = pd.DataFrame([
        {
            **identity_columns,
            'feature_set': 'basic_formula_composition',
            'model_type': 'linear_regression',
            'mae': 1.0,
            'r2': 0.1,
        },
        {
            **identity_columns,
            'feature_set': 'fractional_composition_vector',
            'model_type': 'torch_mlp',
            'mae': 2.0,
            'r2': 0.2,
        },
    ])
    family_df = slice_df.assign(mae=[4.0, 0.5], r2=[0.4, 0.8])
    stratified_df = slice_df.assign(
        bn_mae=[3.0, 0.25],
        non_bn_mae=[1.5, 0.5],
        bn_to_non_bn_mae_ratio=[2.0, 0.5],
    )

    comparison_df = _build_bn_model_role_comparison_table(
        slice_df,
        bn_family_benchmark_df=family_df,
        bn_stratified_error_df=stratified_df,
    )

    row = comparison_df.iloc[0]
    assert row['feature_set'] == 'basic_formula_composition'
    assert row['model_type'] == 'linear_regression'
    assert row['bn_slice_mae'] == 1.0
    assert row['bn_family_mae'] == 4.0
    assert row['bn_mae'] == 3.0
    assert row['non_bn_mae'] == 1.5
    assert row['bn_to_non_bn_mae_ratio'] == 2.0


@pytest.mark.parametrize(
    'artifact_field',
    ['artifact', 'summary_artifact', 'variants_artifact', 'structure_dir'],
)
def test_structure_first_pass_config_rejects_artifact_path_escape(
    tmp_path,
    artifact_field,
):
    for unsafe_value in ('../outside', str(tmp_path / 'absolute-outside')):
        cfg = {
            'screening': {
                'structure_first_pass_execution': {
                    artifact_field: unsafe_value,
                },
            },
        }
        with pytest.raises(ValueError, match='artifact directory'):
            _structure_first_pass_execution_config(cfg)


def test_structure_first_pass_config_preserves_contained_relative_paths():
    cfg = {
        'screening': {
            'structure_first_pass_execution': {
                'artifact': 'nested/execution.json',
                'summary_artifact': 'nested/summary.csv',
                'variants_artifact': 'nested/variants.csv',
                'structure_dir': 'nested/structures',
            },
        },
    }

    execution_cfg = _structure_first_pass_execution_config(cfg)

    assert execution_cfg['artifact'] == 'nested/execution.json'
    assert execution_cfg['summary_artifact'] == 'nested/summary.csv'
    assert execution_cfg['variants_artifact'] == 'nested/variants.csv'
    assert execution_cfg['structure_dir'] == 'nested/structures'


@pytest.mark.parametrize(
    'execution_overrides',
    [
        {'artifact': 'metrics.json'},
        {'artifact': 'METRICS.JSON'},
        {'summary_artifact': 'predictions.csv'},
        {'variants_artifact': 'benchmark_results.csv'},
        {'structure_dir': 'manifest.json/structures'},
        {
            'summary_artifact': 'nested/shared.csv',
            'variants_artifact': 'nested/shared.csv',
        },
        {
            'summary_artifact': 'nested/shared.csv',
            'variants_artifact': 'nested/SHARED.CSV',
        },
        {
            'summary_artifact': 'nested/caf\u00e9.csv',
            'variants_artifact': 'nested/cafe\u0301.csv',
        },
        {
            'artifact': 'nested/execution.json',
            'summary_artifact': 'nested/execution.json/summary.csv',
        },
        {
            'artifact': 'nested/execution.json',
            'structure_dir': 'nested/execution.json/structures',
        },
        {
            'summary_artifact': (
                'demo_candidate_structure_generation_first_pass_execution_variants.csv'
            ),
            'variants_artifact': (
                'demo_candidate_structure_generation_first_pass_execution_summary.csv'
            ),
        },
        {
            'summary_artifact': (
                'DEMO_CANDIDATE_STRUCTURE_GENERATION_FIRST_PASS_EXECUTION_VARIANTS.CSV'
            ),
            'variants_artifact': 'nested/custom-variants.csv',
        },
    ],
    ids=[
        'core-json-collision',
        'casefolded-core-json-collision',
        'core-summary-collision',
        'core-variants-collision',
        'structure-dir-beneath-core-file',
        'pairwise-file-collision',
        'casefolded-pairwise-file-collision',
        'unicode-normalized-pairwise-file-collision',
        'file-path-contains-file-path',
        'structure-dir-beneath-configured-file',
        'cross-role-canonical-default-swap',
        'casefolded-cross-role-canonical-default',
    ],
)
def test_reporting_rejects_structure_execution_output_path_collisions(
    tmp_path,
    execution_overrides,
):
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    bn_centered_sentinel_path = artifact_dir / 'demo_candidate_bn_centered_ranking.csv'
    bn_centered_sentinel_path.write_text('sentinel\n', encoding='utf-8')
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {'structure_first_pass_execution': execution_overrides},
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='collide|contain|nested beneath'):
        save_metrics_and_predictions(
            metrics={'must': 'survive'},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=pd.DataFrame(columns=['formula', 'ranking_rank']),
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
            structure_first_pass_execution_payload={},
        )

    assert not (artifact_dir / 'metrics.json').exists()
    assert not (artifact_dir / 'predictions.csv').exists()
    assert bn_centered_sentinel_path.read_text(encoding='utf-8') == 'sentinel\n'


@pytest.mark.parametrize(
    ('artifact_field', 'invalid_path'),
    [
        ('artifact', 'nested/execution.csv'),
        ('summary_artifact', 'nested/summary.json'),
        ('variants_artifact', 'nested/variants.json'),
    ],
)
def test_reporting_rejects_structure_execution_output_extension_mismatch(
    tmp_path,
    artifact_field,
    invalid_path,
):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'structure_first_pass_execution': {artifact_field: invalid_path},
        },
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='file path'):
        save_metrics_and_predictions(
            metrics={},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=pd.DataFrame(columns=['formula', 'ranking_rank']),
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
        )

    assert not artifact_dir.exists()


def test_reporting_rejects_hardlink_alias_to_reserved_artifact(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    metrics_path = artifact_dir / 'metrics.json'
    metrics_path.write_text('{"sentinel": true}\n', encoding='utf-8')
    execution_alias_path = artifact_dir / 'custom_execution.json'
    execution_alias_path.hardlink_to(metrics_path)
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'structure_first_pass_execution': {
                'artifact': execution_alias_path.name,
            },
        },
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='hard links'):
        save_metrics_and_predictions(
            metrics={'replacement': True},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=pd.DataFrame(columns=['formula', 'ranking_rank']),
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
            structure_first_pass_execution_payload={},
        )

    assert metrics_path.read_text(encoding='utf-8') == '{"sentinel": true}\n'
    assert execution_alias_path.read_text(encoding='utf-8') == '{"sentinel": true}\n'


def test_reporting_rejects_configured_output_symlink_before_cleanup(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    nested_dir = artifact_dir / 'nested'
    nested_dir.mkdir(parents=True)
    sentinel_path = artifact_dir / 'unrelated_execution.json'
    sentinel_path.write_text('{"sentinel": true}\n', encoding='utf-8')
    execution_alias_path = nested_dir / 'execution.json'
    execution_alias_path.symlink_to(sentinel_path)
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'structure_first_pass_execution': {
                'artifact': 'nested/execution.json',
            },
        },
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='symbolic-link'):
        save_metrics_and_predictions(
            metrics={},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=pd.DataFrame(columns=['formula', 'ranking_rank']),
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
            structure_first_pass_execution_payload={},
        )

    assert sentinel_path.read_text(encoding='utf-8') == '{"sentinel": true}\n'
    assert execution_alias_path.is_symlink()
    assert not (artifact_dir / 'metrics.json').exists()


@pytest.mark.parametrize(
    ('first_name', 'second_name'),
    [
        ('variant.cif', 'variant.cif'),
        ('variant.cif', 'VARIANT.CIF'),
        ('caf\u00e9.cif', 'cafe\u0301.cif'),
    ],
    ids=['exact', 'casefolded', 'unicode-normalized'],
)
def test_reporting_rejects_duplicate_cif_output_aliases(
    tmp_path,
    first_name,
    second_name,
):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
    }
    execution_cfg = _structure_first_pass_execution_config(cfg)
    structure_dir = execution_cfg['structure_dir']
    structure_payload = {
        **{
            field: execution_cfg[field]
            for field in ('artifact', 'summary_artifact', 'variants_artifact', 'structure_dir')
        },
        'candidates': [
            {
                'formula': 'XBN',
                'variants': [
                    {
                        'generated_structure_cif_path': f'{structure_dir}/{first_name}',
                        '_cif_text': 'first',
                    },
                    {
                        'generated_structure_cif_path': f'{structure_dir}/{second_name}',
                        '_cif_text': 'second',
                    },
                ],
            },
        ],
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='must be unique'):
        save_metrics_and_predictions(
            metrics={},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=pd.DataFrame(columns=['formula', 'ranking_rank']),
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
            structure_first_pass_execution_variant_df=pd.DataFrame([
                {'formula': 'XBN', 'variant': 1},
                {'formula': 'XBN', 'variant': 2},
            ]),
            structure_first_pass_execution_summary_df=pd.DataFrame([
                {'formula': 'XBN'},
            ]),
            structure_first_pass_execution_payload=structure_payload,
        )

    assert not (artifact_dir / 'metrics.json').exists()


def test_reporting_rejects_non_file_cif_leaf_before_fixed_outputs(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
    }
    execution_cfg = _structure_first_pass_execution_config(cfg)
    cif_relative_path = f"{execution_cfg['structure_dir']}/variant.cif"
    (artifact_dir / cif_relative_path).mkdir(parents=True)
    structure_payload = {
        **{
            field: execution_cfg[field]
            for field in ('artifact', 'summary_artifact', 'variants_artifact', 'structure_dir')
        },
        'candidates': [
            {
                'formula': 'XBN',
                'variants': [
                    {
                        'generated_structure_cif_path': cif_relative_path,
                        '_cif_text': 'data_XBN\n',
                    },
                ],
            },
        ],
    }
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match='regular-file'):
        save_metrics_and_predictions(
            metrics={},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=pd.DataFrame(columns=['formula', 'ranking_rank']),
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
            structure_first_pass_execution_variant_df=pd.DataFrame([
                {'formula': 'XBN', 'variant': 1},
            ]),
            structure_first_pass_execution_summary_df=pd.DataFrame([
                {'formula': 'XBN'},
            ]),
            structure_first_pass_execution_payload=structure_payload,
        )

    assert (artifact_dir / cif_relative_path).is_dir()
    assert not (artifact_dir / 'metrics.json').exists()


@pytest.mark.parametrize('invalid_leaf_kind', ['symlink', 'hardlink', 'directory'])
def test_reporting_rejects_invalid_stale_cif_leaf_before_fixed_outputs(
    tmp_path,
    monkeypatch,
    invalid_leaf_kind,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
    }
    execution_cfg = _structure_first_pass_execution_config(cfg)
    structure_dir = artifact_dir / execution_cfg['structure_dir']
    structure_dir.mkdir(parents=True)
    stale_cif_path = structure_dir / 'stale.cif'
    human_docs_cif = tmp_path / 'human_docs' / 'stale.cif'
    if invalid_leaf_kind == 'directory':
        stale_cif_path.mkdir()
    else:
        human_docs_cif.parent.mkdir()
        human_docs_cif.write_text('user-owned', encoding='utf-8')
        if invalid_leaf_kind == 'symlink':
            stale_cif_path.symlink_to(human_docs_cif)
        else:
            stale_cif_path.hardlink_to(human_docs_cif)
    empty_df = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match='user-owned human_docs|multiple hard links|regular-file',
    ):
        save_metrics_and_predictions(
            metrics={},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=pd.DataFrame(columns=['formula', 'ranking_rank']),
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
            structure_first_pass_execution_payload={},
        )

    if invalid_leaf_kind == 'directory':
        assert stale_cif_path.is_dir()
    else:
        assert stale_cif_path.exists()
        assert human_docs_cif.read_text(encoding='utf-8') == 'user-owned'
    assert not (artifact_dir / 'metrics.json').exists()


def test_reporting_rejects_structure_artifact_write_or_cleanup_escape(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    outside_dir = tmp_path / 'outside'
    artifact_dir.mkdir()
    outside_dir.mkdir()
    outside_artifact = outside_dir / 'execution.json'
    outside_artifact.write_text('keep', encoding='utf-8')
    (artifact_dir / 'escape').symlink_to(outside_dir, target_is_directory=True)
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
    }
    empty_df = pd.DataFrame()
    screened_df = pd.DataFrame(columns=['formula', 'ranking_rank'])

    def save_structure_payload(structure_payload, *, summary_df=None, variant_df=None):
        save_metrics_and_predictions(
            metrics={},
            prediction_df=empty_df,
            bn_df=empty_df,
            screened_df=screened_df,
            benchmark_df=empty_df,
            robustness_df=empty_df,
            bn_slice_benchmark_df=empty_df,
            bn_slice_prediction_df=empty_df,
            bn_centered_screened_df=empty_df,
            structure_generation_seed_df=empty_df,
            experiment_summary={},
            manifest={},
            cfg=cfg,
            structure_first_pass_execution_variant_df=variant_df,
            structure_first_pass_execution_summary_df=summary_df,
            structure_first_pass_execution_payload=structure_payload,
        )

    for artifact_value in ('../outside/execution.json', 'escape/execution.json'):
        structure_payload = {
            'artifact': artifact_value,
            'summary_artifact': 'nested/summary.csv',
            'variants_artifact': 'nested/variants.csv',
            'structure_dir': 'nested/structures',
            'candidates': [],
        }
        with pytest.raises(ValueError, match='artifact directory'):
            save_structure_payload(structure_payload)
        assert outside_artifact.read_text(encoding='utf-8') == 'keep'

    outside_cif = outside_dir / 'variant.cif'
    outside_cif.write_text('keep-cif', encoding='utf-8')
    cfg['screening'] = {
        'structure_first_pass_execution': {
            'artifact': 'nested/execution.json',
            'summary_artifact': 'nested/summary.csv',
            'variants_artifact': 'nested/variants.csv',
            'structure_dir': 'nested/structures',
        },
    }
    structure_payload = {
        'artifact': 'nested/execution.json',
        'summary_artifact': 'nested/summary.csv',
        'variants_artifact': 'nested/variants.csv',
        'structure_dir': 'nested/structures',
        'candidates': [
            {
                'formula': 'XBN',
                'variants': [
                    {
                        'generated_structure_cif_path': 'escape/variant.cif',
                        '_cif_text': 'do-not-write',
                    },
                ],
            },
        ],
    }
    with pytest.raises(ValueError, match='artifact directory'):
        save_structure_payload(
            structure_payload,
            summary_df=pd.DataFrame([{'formula': 'XBN'}]),
            variant_df=pd.DataFrame([{'formula': 'XBN'}]),
        )
    assert outside_cif.read_text(encoding='utf-8') == 'keep-cif'
    assert not (artifact_dir / 'metrics.json').exists()
    assert not (artifact_dir / 'nested/summary.csv').exists()
    assert not (artifact_dir / 'nested/variants.csv').exists()

    structure_payload['candidates'][0]['variants'][0][
        'generated_structure_cif_path'
    ] = 'metrics.json'
    with pytest.raises(ValueError, match='configured output root|directly under'):
        save_structure_payload(
            structure_payload,
            summary_df=pd.DataFrame([{'formula': 'XBN'}]),
            variant_df=pd.DataFrame([{'formula': 'XBN'}]),
        )
    assert not (artifact_dir / 'metrics.json').exists()


def test_reporting_clears_stale_optional_artifacts_on_disabled_second_run(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': True},
            'decision_policy': {'enabled': True},
            'proposal_shortlist': {'enabled': True},
            'extrapolation_shortlist': {'enabled': True},
        },
    }
    screened_df = pd.DataFrame({
        'formula': ['XBN'],
        'ranking_rank': [1],
        'ranking_score': [1.0],
        'predicted_band_gap': [5.0],
        'proposal_shortlist_selected': [True],
        'proposal_shortlist_rank': [1],
        'extrapolation_shortlist_selected': [True],
        'extrapolation_shortlist_rank': [1],
    })
    execution_cfg = _structure_first_pass_execution_config(cfg)
    cif_relative_path = f"{execution_cfg['structure_dir']}/xbn__variant_1.CIF"
    writer_kwargs, _execution_cfg = _structure_execution_writer_kwargs(cfg)
    writer_kwargs['structure_payload']['candidates'][0]['variants'][0].update({
        'generated_structure_cif_path': cif_relative_path,
    })
    writer_kwargs['structure_variant_df'].loc[
        0, 'generated_structure_cif_path'
    ] = cif_relative_path
    writer_kwargs['structure_summary_df'].loc[
        0, 'first_pass_execution_selected_cif_path'
    ] = cif_relative_path

    _save_minimal_report_bundle(
        cfg,
        screened_df=screened_df,
        **writer_kwargs,
    )
    stale_paths = [
        artifact_dir / execution_cfg['artifact'],
        artifact_dir / execution_cfg['summary_artifact'],
        artifact_dir / execution_cfg['variants_artifact'],
        artifact_dir / cif_relative_path,
        artifact_dir / 'demo_candidate_structure_followup_report.csv',
        artifact_dir / 'demo_candidate_rank_stability_summary.csv',
        artifact_dir / 'demo_candidate_proposal_shortlist.csv',
        artifact_dir / 'demo_candidate_extrapolation_shortlist.csv',
        artifact_dir / 'demo_candidate_ranking_uncertainty.csv',
    ]
    assert all(path.exists() for path in stale_paths)

    cfg['screening']['ranking_stability']['enabled'] = False
    cfg['screening']['decision_policy']['enabled'] = False
    cfg['screening']['proposal_shortlist']['enabled'] = False
    cfg['screening']['extrapolation_shortlist']['enabled'] = False
    _save_minimal_report_bundle(cfg, screened_df=screened_df, structure_payload={})

    assert all(not path.exists() for path in stale_paths)


def test_disabled_structure_generation_seed_stage_emits_no_bundle(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula', 'target_column': 'target'},
        'screening': {
            'structure_generation_seeds': {'enabled': False},
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
        },
    }
    candidate_df = pd.DataFrame([{
        'formula': 'XBN',
        'ranking_rank': 1,
        'proposal_shortlist_selected': True,
        'proposal_shortlist_rank': 1,
        'extrapolation_shortlist_selected': False,
    }])
    seed_df = build_candidate_structure_generation_seeds(
        candidate_df,
        pd.DataFrame(columns=['formula', 'target']),
        {
            'train': pd.Series(dtype=bool),
            'val': pd.Series(dtype=bool),
        },
        cfg,
    )

    assert seed_df.empty
    _save_minimal_report_bundle(
        cfg,
        screened_df=candidate_df,
        structure_generation_seed_df=seed_df,
    )

    for artifact_name in (
        'demo_candidate_structure_generation_seeds.csv',
        'demo_candidate_structure_generation_handoff.json',
        'demo_candidate_structure_generation_reference_records.json',
        'demo_candidate_structure_generation_job_plan.json',
        'demo_candidate_structure_generation_first_pass_queue.json',
        'demo_candidate_structure_generation_followup_shortlist.csv',
        'demo_candidate_structure_generation_first_pass_execution.json',
        'demo_candidate_structure_generation_first_pass_execution_summary.csv',
        'demo_candidate_structure_generation_first_pass_execution_variants.csv',
        'demo_candidate_structure_followup_report.csv',
    ):
        assert not (artifact_dir / artifact_name).exists()


def test_empty_structure_generation_bridge_does_not_advertise_absent_artifacts():
    cfg = io_utils.load_config(Path(__file__).resolve().parents[2] / 'config.py')
    candidate_df = pd.DataFrame({
        'formula': ['XBN'],
        'ranking_rank': [1],
        'ranking_score': [1.0],
        'predicted_band_gap': [5.0],
    })
    dataset_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0]})
    selection_summary = {
        'selected_feature_set': cfg['features']['feature_set'],
        'selected_model_type': cfg['model']['type'],
        'selected_feature_family': 'composition_only',
        'screening_selected_feature_set': cfg['features']['feature_set'],
        'screening_selected_model_type': cfg['model']['type'],
        'screening_selected_feature_family': 'composition_only',
    }

    summary = build_experiment_summary(
        dataset_df,
        dataset_df,
        candidate_df,
        {'train': [True], 'val': [False], 'test': [False], 'metadata': {}},
        selection_summary,
        cfg,
        structure_generation_seed_df=pd.DataFrame(),
    )
    bridge = summary['screening']['structure_generation_bridge']

    assert bridge['enabled'] is True
    for artifact_field in (
        'artifact',
        'handoff_artifact',
        'reference_record_payload_artifact',
        'job_plan_artifact',
        'first_pass_queue_artifact',
        'followup_shortlist_artifact',
        'followup_extrapolation_shortlist_artifact',
        'first_pass_execution_artifact',
        'first_pass_execution_summary_artifact',
        'first_pass_execution_variants_artifact',
    ):
        assert bridge.get(artifact_field) is None

    summary_without_execution_payload = build_experiment_summary(
        dataset_df,
        dataset_df,
        candidate_df,
        {'train': [True], 'val': [False], 'test': [False], 'metadata': {}},
        selection_summary,
        cfg,
        structure_generation_seed_df=pd.DataFrame(),
        structure_first_pass_execution_summary_df=pd.DataFrame([
            {'formula': 'XBN', 'first_pass_execution_status': 'executed'},
        ]),
    )
    assert (
        summary_without_execution_payload['screening'][
            'structure_generation_bridge'
        ]['first_pass_execution_followup_report_artifact']
        is None
    )


def test_summary_handles_insufficient_bn_diagnostics_without_advertising_empty_predictions():
    cfg = io_utils.load_config(Path(__file__).resolve().parents[2] / 'config.py')
    cfg['features']['feature_set'] = 'basic_formula_composition'
    cfg['features']['candidate_sets'] = ['basic_formula_composition']
    cfg['model']['type'] = 'linear_regression'
    cfg['model']['candidate_types'] = ['linear_regression']
    dataset_df = pd.DataFrame({
        'formula': ['BN', 'BN', 'AlN'],
        'band_gap': [5.0, 5.1, 3.0],
        'target': [5.0, 5.1, 3.0],
    })
    feature_tables = {
        'basic_formula_composition': pd.DataFrame({
            'formula': ['BN', 'BN', 'AlN'],
            'target': [5.0, 5.1, 3.0],
            'feature_1': [1.0, 2.0, 3.0],
            'feature_generation_failed': [False, False, False],
            'feature_generation_error': [None, None, None],
            'feature_set': ['basic_formula_composition'] * 3,
        }),
    }
    diagnostic_kwargs = {
        'selected_feature_set': 'basic_formula_composition',
        'selected_model_type': 'linear_regression',
        'screening_feature_set': 'basic_formula_composition',
        'screening_model_type': 'linear_regression',
    }
    bn_slice_df, bn_slice_prediction_df = benchmark_bn_slice(
        dataset_df,
        feature_tables,
        cfg,
        **diagnostic_kwargs,
    )
    bn_family_df, bn_family_prediction_df = benchmark_bn_family_holdout(
        dataset_df,
        feature_tables,
        cfg,
        **diagnostic_kwargs,
    )

    summary = build_experiment_summary(
        dataset_df,
        dataset_df.loc[dataset_df['formula'].eq('BN')],
        pd.DataFrame({
            'formula': ['XBN'],
            'ranking_rank': [1],
            'ranking_score': [1.0],
            'predicted_band_gap': [5.0],
        }),
        {
            'train': [True, True, False],
            'val': [False, False, False],
            'test': [False, False, True],
            'metadata': {},
        },
        {
            'selected_feature_set': 'basic_formula_composition',
            'selected_model_type': 'linear_regression',
            'selected_feature_family': 'composition_only',
            'screening_selected_feature_set': 'basic_formula_composition',
            'screening_selected_model_type': 'linear_regression',
            'screening_selected_feature_family': 'composition_only',
        },
        cfg,
        bn_slice_benchmark_df=bn_slice_df,
        bn_slice_prediction_df=bn_slice_prediction_df,
        bn_family_benchmark_df=bn_family_df,
        bn_family_prediction_df=bn_family_prediction_df,
    )

    diagnostic_summary = summary['bn_slice_benchmark']
    assert set(bn_slice_df['benchmark_status']) == {'insufficient_bn_formulas'}
    assert set(bn_family_df['benchmark_status']) == {'insufficient_bn_families'}
    assert bn_slice_prediction_df.empty and bn_family_prediction_df.empty
    assert diagnostic_summary['prediction_artifact'] is None
    assert diagnostic_summary['family_prediction_artifact'] is None
    assert diagnostic_summary['selected_model_beats_global_dummy'] is None
    assert diagnostic_summary['family_selected_model_beats_global_dummy'] is None


@pytest.mark.parametrize(
    'diagnostic_states',
    [
        ((True, False),),
        ((False, True),),
        ((False, False),),
        ((True, True),),
        ((True, False), (False, True)),
        ((False, True), (True, False)),
        ((True, True), (False, False)),
    ],
    ids=[
        'slice-only',
        'family-only',
        'neither',
        'both',
        'slice-to-family',
        'family-to-slice',
        'both-to-neither',
    ],
)
def test_bn_prediction_summary_writer_and_provenance_align_across_repeat_runs(
    tmp_path,
    monkeypatch,
    diagnostic_states,
):
    cfg = io_utils.load_config(Path(__file__).resolve().parents[2] / 'config.py')
    artifact_dir = tmp_path / 'artifacts'
    cfg['project']['artifact_dir'] = str(artifact_dir)
    for gate in (
        'ranking_stability',
        'decision_policy',
        'proposal_shortlist',
        'extrapolation_shortlist',
        'structure_generation_seeds',
    ):
        cfg['screening'][gate]['enabled'] = False
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
    dataset_df = pd.DataFrame({
        'formula': ['BN'],
        'target': [5.0],
        'band_gap': [5.0],
    })
    candidate_df = pd.DataFrame({
        'formula': ['XBN'],
        'ranking_rank': [1],
        'ranking_score': [1.0],
        'predicted_band_gap': [5.0],
    })
    split_masks = {
        'train': [True],
        'val': [False],
        'test': [False],
        'metadata': {'method': 'group_by_formula'},
    }
    selection_summary = {
        'selected_feature_set': 'basic_formula_composition',
        'selected_model_type': 'linear_regression',
        'selected_feature_family': 'composition_only',
        'screening_selected_feature_set': 'basic_formula_composition',
        'screening_selected_model_type': 'linear_regression',
        'screening_selected_feature_family': 'composition_only',
    }

    def benchmark_frame(kind, has_predictions):
        status = 'ok' if has_predictions else f'insufficient_bn_{kind}'
        metric_values = [0.5, 0.9] if has_predictions else [None, None]
        frame = pd.DataFrame({
            'feature_set': ['basic_formula_composition', 'feature_agnostic_dummy'],
            'feature_family': ['composition_only', 'baseline'],
            'model_type': ['linear_regression', 'dummy_mean'],
            'benchmark_role': ['selected_model', 'global_dummy_mean_baseline'],
            'benchmark_status': [status, status],
            'candidate_compatible': [True, False],
            'selected_by_validation': [True, False],
            'mae': metric_values,
            'rmse': metric_values,
            'r2': metric_values,
            'k_neighbors': [None, None],
        })
        if kind == 'formulas':
            return frame.assign(
                bn_slice_method='leave_one_bn_formula_out',
                bn_slice_train_scope='full_dataset_minus_held_out_bn_formula',
                bn_formula_count=1,
                bn_row_count=1,
                completed_holds=1 if has_predictions else 0,
            )
        return frame.assign(
            bn_family_benchmark_method='leave_one_bn_family_out',
            bn_family_grouping_method='reduced_bn_chemical_system',
            bn_family_train_scope='full_dataset_minus_held_out_bn_family',
            bn_family_count=1,
            bn_formula_count=1,
            bn_row_count=1,
            completed_family_holds=1 if has_predictions else 0,
            completed_formula_holds=1 if has_predictions else 0,
        )

    def prediction_frame(kind, has_predictions, run_index):
        if not has_predictions:
            return pd.DataFrame()
        return pd.DataFrame([{
            'formula': 'BN',
            'target': 5.0,
            'prediction': 4.0 + run_index + (0.1 if kind == 'family' else 0.0),
            'benchmark_role': 'selected_model',
        }])

    for run_index, (slice_has_predictions, family_has_predictions) in enumerate(
        diagnostic_states
    ):
        bn_slice_benchmark_df = benchmark_frame('formulas', slice_has_predictions)
        bn_family_benchmark_df = benchmark_frame('families', family_has_predictions)
        bn_slice_prediction_df = prediction_frame(
            'slice', slice_has_predictions, run_index
        )
        bn_family_prediction_df = prediction_frame(
            'family', family_has_predictions, run_index
        )
        summary = build_experiment_summary(
            dataset_df,
            dataset_df,
            candidate_df,
            split_masks,
            selection_summary,
            cfg,
            bn_slice_benchmark_df=bn_slice_benchmark_df,
            bn_slice_prediction_df=bn_slice_prediction_df,
            bn_family_benchmark_df=bn_family_benchmark_df,
            bn_family_prediction_df=bn_family_prediction_df,
        )
        save_metrics_and_predictions(
            {'mae': 0.1},
            dataset_df.assign(prediction=5.0),
            dataset_df,
            candidate_df,
            pd.DataFrame([{
                'feature_set': 'basic_formula_composition',
                'model_type': 'linear_regression',
                'mae': 0.1,
            }]),
            pd.DataFrame(),
            bn_slice_benchmark_df,
            bn_slice_prediction_df,
            pd.DataFrame(),
            pd.DataFrame(),
            summary,
            manifest,
            cfg,
            bn_family_benchmark_df=bn_family_benchmark_df,
            bn_family_prediction_df=bn_family_prediction_df,
        )

        diagnostic_summary = summary['bn_slice_benchmark']
        expected_slice_path = (
            'bn_slice_predictions.csv' if slice_has_predictions else None
        )
        expected_family_path = (
            'bn_family_predictions.csv' if family_has_predictions else None
        )
        assert diagnostic_summary['benchmark_artifact'] == (
            'bn_slice_benchmark_results.csv'
        )
        assert diagnostic_summary['family_benchmark_artifact'] == (
            'bn_family_benchmark_results.csv'
        )
        assert diagnostic_summary['prediction_artifact'] == expected_slice_path
        assert diagnostic_summary['family_prediction_artifact'] == expected_family_path
        assert diagnostic_summary['selected_model_metrics']['benchmark_status'] == (
            'ok' if slice_has_predictions else 'insufficient_bn_formulas'
        )
        assert diagnostic_summary['family_selected_model_metrics']['benchmark_status'] == (
            'ok' if family_has_predictions else 'insufficient_bn_families'
        )
        assert diagnostic_summary['selected_model_beats_global_dummy'] is (
            True if slice_has_predictions else None
        )
        assert diagnostic_summary['family_selected_model_beats_global_dummy'] is (
            True if family_has_predictions else None
        )

        provenance = io_utils.read_json_file(
            artifact_dir / 'artifact_provenance.json'
        )
        for has_predictions, artifact_name in (
            (slice_has_predictions, 'bn_slice_predictions.csv'),
            (family_has_predictions, 'bn_family_predictions.csv'),
        ):
            output_path = artifact_dir / artifact_name
            assert output_path.exists() is has_predictions
            assert (artifact_name in provenance['published_outputs']) is has_predictions
            if has_predictions:
                assert provenance['published_outputs'][artifact_name] == hashlib.sha256(
                    output_path.read_bytes()
                ).hexdigest()
        for artifact_name in (
            'bn_slice_benchmark_results.csv',
            'bn_family_benchmark_results.csv',
        ):
            output_path = artifact_dir / artifact_name
            assert output_path.exists()
            assert provenance['published_outputs'][artifact_name] == hashlib.sha256(
                output_path.read_bytes()
            ).hexdigest()
        assert io_utils.read_json_file(
            artifact_dir / 'experiment_summary.json'
        )['bn_slice_benchmark'] == diagnostic_summary
        assert io_utils.assess_artifact_provenance(
            provenance,
            cfg,
            manifest,
            project_root_path=tmp_path,
        ) == {
            'status': 'current',
            'reason': 'source_config_dataset_and_outputs_match',
        }


def test_reporting_preflights_summary_before_mutating_existing_bundle(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
        },
    }
    _save_minimal_report_bundle(cfg)
    before = {
        path.relative_to(artifact_dir): path.read_bytes()
        for path in artifact_dir.rglob('*')
        if path.is_file()
    }
    assert Path('artifact_provenance.json') in before

    with pytest.raises(ValueError, match='not JSON-serializable'):
        _save_minimal_report_bundle(
            cfg,
            experiment_summary={'invalid': object()},
        )

    after = {
        path.relative_to(artifact_dir): path.read_bytes()
        for path in artifact_dir.rglob('*')
        if path.is_file()
    }
    assert after == before


def _report_bundle_snapshot(artifact_dir: Path):
    if not artifact_dir.exists():
        return None
    snapshot = {}
    for path in (artifact_dir, *artifact_dir.rglob('*')):
        relative_path = path.relative_to(artifact_dir).as_posix() or '.'
        if path.is_symlink():
            snapshot[relative_path] = ('symlink', os.readlink(path))
        elif path.is_dir():
            snapshot[relative_path] = ('directory',)
        else:
            snapshot[relative_path] = ('file', path.read_bytes())
    return snapshot


def _structure_execution_writer_kwargs(cfg):
    execution_cfg = _structure_first_pass_execution_config(cfg)
    structure = Structure(
        Lattice.tetragonal(3.0, 20.0),
        ['B', 'N'],
        [[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]],
    )
    atoms = _structure_to_atoms(structure)
    artifact_dir = Path(cfg['project']['artifact_dir'])
    raw_dir = artifact_dir.parent / f'{artifact_dir.name}-raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.setdefault('data', {})
    data_cfg.update({
        'dataset': 'twod_matpd',
        'raw_dir': str(raw_dir),
        'formula_column': data_cfg.get('formula_column', 'formula'),
    })
    (raw_dir / 'twod_matpd.json').write_text(
        json.dumps([{'jid': 'jid-1', 'formula': 'BN', 'atoms': atoms}]),
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
        'formula': 'BN',
        'execution_variant_id': 'xbn__variant_01',
        'execution_variant_rank': 1,
        'execution_status': 'ok',
        'execution_message': None,
        'seed_reference_formula': 'BN',
        'seed_reference_record_id': 'jid-1',
        'execution_plan_type': 'reference_reuse',
        'relabel_site_indices': '',
        'relabel_target_elements': '',
        'removed_site_indices': '',
        'relabeled_site_count': 0,
        'removed_site_count': 0,
        'formula_matches_candidate': True,
        'geometry_sanity_pass': True,
        'execution_variant_selection_score': 0.0,
        'generated_structure_cif_path': (
            f"{execution_cfg['structure_dir']}/xbn__variant_01.cif"
        ),
        'generated_formula': 'BN',
        'generated_structure_n_sites': len(structure),
        'geometry_min_distance': min_distance,
        'geometry_mean_distance': mean_distance,
        'geometry_min_distance_ratio': min_distance_ratio,
        'geometry_overlap_pair_count': overlap_pair_count,
        'structure_band_gap_proxy': None,
        'relaxation_status': 'not_run_reference_geometry_reused',
        'final_status': 'reference_control_ready',
        **structure_summary,
    }
    return {
        'structure_generation_seed_df': pd.DataFrame([{
            'formula': 'BN',
            'ranking_rank': 1,
            'structure_generation_seed_rank': 1,
            'structure_generation_seed_status': 'ok',
            'seed_reference_formula': 'BN',
            'seed_reference_record_id': 'jid-1',
        }]),
        'structure_payload': {
            **execution_cfg,
            'candidate_count': 1,
            'variant_count': 1,
            'successful_variant_count': 1,
            'status_counts': {'executed': 1},
            'executed_formulas': ['BN'],
            'candidates': [{
                'formula': 'BN',
                'seed_reference_formula': 'BN',
                'seed_reference_record_id': 'jid-1',
                'candidate_status': 'executed',
                'selected_variant_id': 'xbn__variant_01',
                'variants': [{
                    **variant_row,
                    'atoms': atoms,
                    '_cif_text': structure.to(fmt='cif'),
                }],
            }],
        },
        'structure_summary_df': pd.DataFrame([
            {
                'formula': 'BN',
                'first_pass_execution_variant_count': 1,
                'first_pass_execution_successful_variant_count': 1,
                'first_pass_execution_geometry_pass_variant_count': 1,
                'first_pass_execution_status': 'executed',
                'structure_followup_best_seed_reference_formula': 'BN',
                'structure_followup_best_seed_reference_record_id': 'jid-1',
                'first_pass_execution_selected_variant_id': 'xbn__variant_01',
                'first_pass_execution_selected_variant_rank': 1,
                'first_pass_execution_selected_cif_path': (
                    f"{execution_cfg['structure_dir']}/xbn__variant_01.cif"
                ),
                'first_pass_execution_selected_generated_formula': 'BN',
                'first_pass_execution_selected_structure_n_sites': len(structure),
                'first_pass_execution_selected_min_distance': min_distance,
                'first_pass_execution_selected_min_distance_ratio': (
                    min_distance_ratio
                ),
                'first_pass_execution_selected_band_gap_proxy': None,
                'first_pass_execution_selected_relaxation_status': (
                    'not_run_reference_geometry_reused'
                ),
                'first_pass_execution_selected_final_status': (
                    'reference_control_ready'
                ),
            },
        ]),
        'structure_variant_df': pd.DataFrame([variant_row]),
    }, execution_cfg


def _canonical_structure_execution_writer_inputs(
    tmp_path: Path,
    *,
    baseline_case: str = 'full',
    formula_col: str = 'formula',
):
    artifact_dir = tmp_path / 'artifacts'
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    atoms = {
        'elements': ['B', 'N'],
        'coords': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        'lattice_mat': [
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 20.0],
        ],
        'abc': [2.5, 2.5, 20.0],
        'angles': [90.0, 90.0, 120.0],
        'cartesian': False,
    }
    raw_formula = 'BN'
    if baseline_case in {'multiple-success', 'vacancy'}:
        raw_formula = 'B2N'
        atoms = {
            'elements': ['B', 'B', 'N'],
            'coords': [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            'lattice_mat': [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            'abc': [10.0, 10.0, 10.0],
            'angles': [90.0, 90.0, 90.0],
            'cartesian': True,
        }
    if baseline_case == 'source-formula-structure-mismatch':
        raw_formula = 'BN'
        atoms = {
            'elements': ['B', 'N', 'C'],
            'coords': [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [5.0, 0.0, 0.0]],
            'lattice_mat': [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            'abc': [10.0, 10.0, 10.0],
            'angles': [90.0, 90.0, 90.0],
            'cartesian': True,
        }
    if baseline_case == 'invalid-reference':
        atoms = {'elements': ['B', 'N']}
    (raw_dir / 'twod_matpd.json').write_text(
        json.dumps([
            {
                'jid': 'jid-1',
                'formula': raw_formula,
                'band_gap': 5.8,
                'atoms': atoms,
            },
        ]),
        encoding='utf-8',
    )
    execution_overrides = {
        'enabled': baseline_case != 'inactive',
        'max_candidates': 2,
        'max_variants_per_candidate': 2,
    }
    if baseline_case in {'custom-paths', 'error-custom-paths'}:
        execution_overrides.update({
            'artifact': 'nested/execution.json',
            'summary_artifact': 'nested/execution-summary.csv',
            'variants_artifact': 'nested/execution-variants.csv',
            'structure_dir': 'nested/cifs',
        })
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'formula_column': formula_col,
            'target_column': 'band_gap',
        },
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
            'structure_first_pass_execution': execution_overrides,
        },
    }
    if baseline_case == 'empty':
        seed_df = pd.DataFrame()
    else:
        seed_rows = {
            'error': [('BN', 'missing-jid', 'BN')],
            'error-custom-paths': [('BN', 'missing-jid', 'BN')],
            'invalid-reference': [('BN', 'jid-1', 'BN')],
            'unresolved-scale': [('BN', 'jid-1', 'C')],
            'formula-scale-mismatch': [('B0.5N0.5', 'jid-1', 'BN')],
            'multiple-donor': [('C', 'jid-1', 'BN')],
            'no-plan': [('AlBN', 'jid-1', 'BN')],
            'multiple-success': [('AlBN', 'jid-1', 'B2N')],
            'vacancy': [('BN', 'jid-1', 'B2N')],
        }.get(baseline_case, [('BN', 'jid-1', 'BN')])
        if baseline_case == 'partial':
            seed_rows.append(('AlBN', 'missing-jid', 'BN'))
        seed_df = pd.DataFrame({
            formula_col: [formula for formula, _record_id, _seed_formula in seed_rows],
            'ranking_rank': list(range(1, len(seed_rows) + 1)),
            'ranking_score': [4.8 - index for index in range(len(seed_rows))],
            'candidate_family': ['bn_binary_anchor'] * len(seed_rows),
            'candidate_novelty_bucket': ['train_plus_val_rediscovery'] * len(seed_rows),
            'chemical_plausibility_pass': [True] * len(seed_rows),
            'proposal_shortlist_selected': [True] * len(seed_rows),
            'proposal_shortlist_rank': list(range(1, len(seed_rows) + 1)),
            'extrapolation_shortlist_selected': [False] * len(seed_rows),
            'extrapolation_shortlist_rank': [None] * len(seed_rows),
            'structure_generation_candidate_priority_reason': [
                'proposal_shortlist'
            ] * len(seed_rows),
            'structure_generation_seed_rank': [1] * len(seed_rows),
            'structure_generation_seed_status': ['ok'] * len(seed_rows),
            'seed_reference_formula': [
                seed_formula for _formula, _record_id, seed_formula in seed_rows
            ],
            'seed_reference_record_id': [
                record_id for _formula, record_id, _seed_formula in seed_rows
            ],
        })
    variant_df, summary_df, payload = build_structure_first_pass_execution_artifacts(
        seed_df,
        cfg=cfg,
        formula_col=formula_col,
    )
    return cfg, {
        'structure_generation_seed_df': seed_df,
        'structure_payload': payload,
        'structure_summary_df': summary_df,
        'structure_variant_df': variant_df,
    }


def _real_builder_seed_evidence_writer_inputs(
    tmp_path: Path,
    *,
    formula_col: str = 'formula',
    evidence_case: str = 'complete',
    execution_enabled: bool = True,
    additional_formula_band_gaps: tuple[float, ...] = (),
):
    cfg, _unused_manual_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        formula_col=formula_col,
    )
    raw_path = Path(cfg['data']['raw_dir']) / 'twod_matpd.json'
    raw_records = json.loads(raw_path.read_text(encoding='utf-8'))
    raw_record = raw_records[0]
    raw_record.update({
        'energy_per_atom': -2.5,
        'exfoliation_energy_per_atom': 0.12,
        'total_magnetization': -0.25,
    })
    if evidence_case == 'missing_optional':
        for field_name in (
            'band_gap',
            'energy_per_atom',
            'exfoliation_energy_per_atom',
            'total_magnetization',
        ):
            raw_record.pop(field_name)
    elif evidence_case == 'missing_structure':
        raw_record['atoms'] = {'elements': ['B', 'N']}
    elif evidence_case != 'complete':  # pragma: no cover - helper contract
        raise AssertionError(evidence_case)
    for offset, band_gap in enumerate(additional_formula_band_gaps, start=2):
        extra_record = copy.deepcopy(raw_record)
        extra_record['jid'] = f'jid-{offset}'
        extra_record['band_gap'] = band_gap
        raw_records.append(extra_record)
    raw_path.write_text(
        json.dumps(raw_records),
        encoding='utf-8',
    )
    cfg['screening']['structure_generation_seeds'] = {
        'enabled': True,
        'per_candidate_seed_limit': 1,
        'bn_centered_top_n': 1,
    }
    cfg['screening']['structure_followup_shortlist'] = {
        'enabled': True,
        'shortlist_size': 1,
    }
    cfg['screening']['structure_first_pass_execution']['enabled'] = execution_enabled
    cfg['screening']['structure_first_pass_execution']['max_candidates'] = 1
    cfg['screening']['structure_first_pass_execution']['max_variants_per_candidate'] = 1
    dataset_df = _normalize(raw_records, 'band_gap')
    if formula_col != 'formula':
        dataset_df[formula_col] = dataset_df['formula']
    candidate_df = pd.DataFrame([{
        formula_col: 'BN',
        'candidate_family': 'bn_binary_anchor',
        'candidate_template': 'B1N1',
        'candidate_novelty_bucket': 'train_plus_val_rediscovery',
        'chemical_plausibility_pass': True,
        'ranking_rank': 1,
        'proposal_shortlist_selected': True,
        'proposal_shortlist_rank': 1,
        'extrapolation_shortlist_selected': False,
        'extrapolation_shortlist_rank': pd.NA,
        'bn_analog_neighbor_formulas': 'BN',
        'bn_analog_nearest_formula': 'BN',
        'bn_support_neighbor_formulas': 'BN',
    }])
    seed_df = build_candidate_structure_generation_seeds(
        candidate_df,
        dataset_df,
        {
            'train': [True] * len(dataset_df),
            'val': [False] * len(dataset_df),
            'test': [False] * len(dataset_df),
        },
        cfg,
        formula_col=formula_col,
    )
    variant_df, summary_df, payload = build_structure_first_pass_execution_artifacts(
        seed_df,
        cfg=cfg,
        formula_col=formula_col,
    )
    return cfg, {
        'bn_df': dataset_df,
        'structure_generation_seed_df': seed_df,
        'structure_payload': payload,
        'structure_summary_df': summary_df,
        'structure_variant_df': variant_df,
    }


def _mutate_seed_reference_evidence(
    cfg,
    canonical_kwargs,
    field_name,
    invalid_value,
):
    invalid_kwargs = copy.deepcopy(canonical_kwargs)
    seed_df = invalid_kwargs['structure_generation_seed_df']
    seed_df.loc[seed_df.index[0], field_name] = invalid_value
    formula_col = cfg['data']['formula_column']
    variant_df, summary_df, payload = build_structure_first_pass_execution_artifacts(
        seed_df,
        cfg=cfg,
        formula_col=formula_col,
    )
    invalid_kwargs.update({
        'structure_payload': payload,
        'structure_summary_df': summary_df,
        'structure_variant_df': variant_df,
    })
    return invalid_kwargs


_SEED_REFERENCE_EVIDENCE_MUTATIONS = (
    ('seed_reference_source', 'forged_source'),
    ('seed_reference_band_gap', 99.0),
    *((f'seed_reference_{field_name}', 99.0) for field_name in REFERENCE_PROPERTY_COLUMNS),
    ('seed_reference_has_structure_summary', False),
    *((f'seed_reference_{field_name}', 99.0) for field_name in STRUCTURE_SUMMARY_COLUMNS),
)


_SELECTED_PROJECTION_INVALID_VALUES = {
    'execution_variant_rank': 99,
    'generated_structure_cif_path': (
        'demo_candidate_structure_generation_first_pass_structures/'
        'forged__variant_99.cif'
    ),
    'generated_formula': 'AlBN',
    'generated_structure_n_sites': 99,
    'geometry_min_distance': 99.0,
    'geometry_min_distance_ratio': 99.0,
    'structure_band_gap_proxy': 99.0,
    'relaxation_status': 'forged_relaxation_status',
}
_SELECTED_PROJECTION_MUTATIONS = tuple(
    (summary_field, variant_field, _SELECTED_PROJECTION_INVALID_VALUES[variant_field])
    for summary_field, variant_field
    in _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
    if variant_field in _SELECTED_PROJECTION_INVALID_VALUES
)


_STRUCTURE_EXECUTION_RELATION_MUTATIONS = {
    'payload-enabled': ('payload', 'enabled', False),
    'payload-candidate-count': ('payload', 'candidate_count', 2),
    'payload-variant-count': ('payload', 'variant_count', 2),
    'payload-successful-variant-count': ('payload', 'successful_variant_count', 0),
    'payload-status-counts': ('payload', 'status_counts', {'missing_reference_record': 1}),
    'payload-executed-formulas': ('payload', 'executed_formulas', ['AlBN']),
    'payload-candidate-formula': ('candidate', 'formula', 'AlBN'),
    'payload-candidate-status': ('candidate', 'candidate_status', 'missing_reference_record'),
    'payload-selected-variant-id': ('candidate', 'selected_variant_id', 'wrong__variant_99'),
    'payload-variant-id': ('payload-variant', 'execution_variant_id', 'wrong__variant_99'),
    'payload-variant-status': ('payload-variant', 'execution_status', 'error'),
    'payload-variant-formula-match': (
        'payload-variant', 'formula_matches_candidate', False,
    ),
    'payload-variant-selection-score': (
        'payload-variant', 'execution_variant_selection_score', 99.0,
    ),
    'summary-variant-count': ('summary', 'first_pass_execution_variant_count', 2),
    'summary-successful-variant-count': (
        'summary', 'first_pass_execution_successful_variant_count', 0,
    ),
    'summary-geometry-pass-count': (
        'summary', 'first_pass_execution_geometry_pass_variant_count', 0,
    ),
    'summary-status': ('summary', 'first_pass_execution_status', 'no_successful_variant'),
    'summary-selected-variant-id': (
        'summary', 'first_pass_execution_selected_variant_id', 'wrong__variant_99',
    ),
    'summary-selected-final-status': (
        'summary', 'first_pass_execution_selected_final_status', 'execution_error',
    ),
    'variant-formula': ('variant', 'formula', 'AlBN'),
    'variant-id': ('variant', 'execution_variant_id', 'wrong__variant_99'),
    'variant-status': ('variant', 'execution_status', 'error'),
    'variant-geometry-status': ('variant', 'geometry_sanity_pass', False),
    'variant-formula-match': ('variant', 'formula_matches_candidate', False),
    'variant-selection-score': (
        'variant', 'execution_variant_selection_score', 99.0,
    ),
    'variant-final-status': ('variant', 'final_status', 'execution_error'),
    **{
        f'summary-selected-{variant_field}': ('summary', summary_field, invalid_value)
        for summary_field, variant_field, invalid_value
        in _SELECTED_PROJECTION_MUTATIONS
    },
    **{
        f'payload-selected-{variant_field}': ('payload-variant', variant_field, invalid_value)
        for _summary_field, variant_field, invalid_value
        in _SELECTED_PROJECTION_MUTATIONS
    },
    **{
        f'variant-selected-{variant_field}': ('variant', variant_field, invalid_value)
        for _summary_field, variant_field, invalid_value
        in _SELECTED_PROJECTION_MUTATIONS
    },
}


def _mutate_structure_execution_relation(writer_kwargs, mismatch_case):
    payload = writer_kwargs['structure_payload']
    target_name, field_name, value = _STRUCTURE_EXECUTION_RELATION_MUTATIONS[mismatch_case]
    targets = {
        'payload': payload,
        'candidate': payload['candidates'][0],
        'payload-variant': payload['candidates'][0]['variants'][0],
    }
    if target_name in targets:
        targets[target_name][field_name] = copy.deepcopy(value)
    else:
        frame = writer_kwargs[f'structure_{target_name}_df']
        frame.loc[0, field_name] = value


def _assert_structure_execution_rejection_is_atomic(
    tmp_path,
    cfg,
    canonical_kwargs,
    invalid_kwargs,
    manifest,
    expected_message='structure_first_pass_execution',
    invalid_cfg=None,
):
    artifact_dir = Path(cfg['project']['artifact_dir'])
    invalid_cfg = cfg if invalid_cfg is None else invalid_cfg
    with pytest.raises(ValueError, match=expected_message):
        _save_minimal_report_bundle(
            invalid_cfg, manifest=manifest, **invalid_kwargs
        )
    assert _report_bundle_snapshot(artifact_dir) is None

    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    valid_snapshot = _report_bundle_snapshot(artifact_dir)
    with pytest.raises(ValueError, match=expected_message):
        _save_minimal_report_bundle(
            invalid_cfg, manifest=manifest, **invalid_kwargs
        )
    assert _report_bundle_snapshot(artifact_dir) == valid_snapshot

    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(artifact_dir / 'artifact_provenance.json'),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert assessment['status'] == 'current'


def _canonical_structure_execution_publication_case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _real_builder_seed_evidence_writer_inputs(tmp_path)
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-22T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    return cfg, canonical_kwargs, manifest


@pytest.mark.parametrize(
    ('field_name', 'invalid_value'),
    [
        ('label', 'experimentally_synthesized_material'),
        ('method', 'thermodynamic_stability_proof'),
        ('note', 'This execution proves synthesis, stability, and discovery.'),
        ('model_available', True),
    ],
)
def test_reporting_rejects_structure_execution_metadata_mismatch_atomically(
    tmp_path,
    monkeypatch,
    field_name,
    invalid_value,
):
    cfg, canonical_kwargs, manifest = (
        _canonical_structure_execution_publication_case(tmp_path, monkeypatch)
    )
    invalid_kwargs = copy.deepcopy(canonical_kwargs)
    invalid_kwargs['structure_payload'][field_name] = invalid_value
    if field_name == 'model_available':
        invalid_kwargs['structure_payload'].update({
            'model_feature_set': 'fabricated_structure_stability_features',
            'model_type': 'synthesis_proof_model',
        })
    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
        expected_message='structure_first_pass_execution',
    )


def test_reporting_rejects_disabled_config_with_execution_outputs_atomically(
    tmp_path,
    monkeypatch,
):
    cfg, canonical_kwargs, manifest = (
        _canonical_structure_execution_publication_case(tmp_path, monkeypatch)
    )
    invalid_cfg = copy.deepcopy(cfg)
    invalid_cfg['screening']['structure_first_pass_execution']['enabled'] = False

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        canonical_kwargs,
        manifest,
        expected_message='structure_first_pass_execution enabled',
        invalid_cfg=invalid_cfg,
    )


@pytest.mark.parametrize(
    ('summary_field', 'invalid_value'),
    [
        ('first_pass_execution_method', 'experimental_synthesis_validation'),
        ('first_pass_execution_note', 'This proves stability and discovery.'),
        ('first_pass_execution_candidate_count', 99),
        ('first_pass_execution_model_available', True),
    ],
)
def test_reporting_rejects_structure_execution_summary_projection_mismatch(
    tmp_path,
    monkeypatch,
    summary_field,
    invalid_value,
):
    cfg, canonical_kwargs, manifest = (
        _canonical_structure_execution_publication_case(tmp_path, monkeypatch)
    )
    payload = canonical_kwargs['structure_payload']
    bridge = {
        f'first_pass_execution_{field_name}': payload[field_name]
        for field_name in (
            'method',
            'note',
            'candidate_count',
            'variant_count',
            'successful_variant_count',
            'status_counts',
            'executed_formulas',
            'model_feature_set',
            'model_type',
            'model_available',
        )
    }
    canonical_kwargs['experiment_summary'] = {
        'screening': {'structure_generation_bridge': bridge},
    }
    invalid_kwargs = copy.deepcopy(canonical_kwargs)
    invalid_kwargs['experiment_summary']['screening'][
        'structure_generation_bridge'
    ][summary_field] = invalid_value
    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
        expected_message='experiment_summary',
    )


@pytest.mark.parametrize(
    ('field_name', 'invalid_value'),
    _SEED_REFERENCE_EVIDENCE_MUTATIONS,
)
def test_reporting_rejects_builder_seed_reference_evidence_mismatch(
    tmp_path,
    monkeypatch,
    field_name,
    invalid_value,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _real_builder_seed_evidence_writer_inputs(tmp_path)
    invalid_kwargs = _mutate_seed_reference_evidence(
        cfg,
        canonical_kwargs,
        field_name,
        invalid_value,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-22T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    with pytest.raises(ValueError, match='seed reference evidence'):
        _save_minimal_report_bundle(cfg, manifest=manifest, **invalid_kwargs)
    assert _report_bundle_snapshot(Path(cfg['project']['artifact_dir'])) is None


@pytest.mark.parametrize(
    ('field_name', 'invalid_value', 'formula_col', 'execution_enabled'),
    [
        ('seed_reference_band_gap', 99.0, 'formula', True),
        ('seed_reference_structure_lattice_a', 99.0, 'composition', True),
        ('seed_reference_source', 'forged_source', 'formula', False),
    ],
)
def test_reporting_seed_reference_evidence_rejection_is_atomic_and_recovers(
    tmp_path,
    monkeypatch,
    field_name,
    invalid_value,
    formula_col,
    execution_enabled,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _real_builder_seed_evidence_writer_inputs(
        tmp_path,
        formula_col=formula_col,
        execution_enabled=execution_enabled,
    )
    invalid_kwargs = _mutate_seed_reference_evidence(
        cfg,
        canonical_kwargs,
        field_name,
        invalid_value,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-22T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
        expected_message='seed reference evidence',
    )


@pytest.mark.parametrize(
    ('field_name', 'invalid_value', 'execution_enabled'),
    [
        ('seed_reference_formula_row_count', 999, False),
        ('seed_reference_formula_mean_band_gap', 99.0, True),
    ],
)
def test_reporting_rejects_grouped_formula_seed_aggregate_mismatch_atomically(
    tmp_path,
    monkeypatch,
    field_name,
    invalid_value,
    execution_enabled,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _real_builder_seed_evidence_writer_inputs(
        tmp_path,
        additional_formula_band_gaps=(6.2,),
        execution_enabled=execution_enabled,
    )
    cfg['split'] = {
        'method': 'group_by_formula',
        'group_column': 'formula',
    }
    invalid_kwargs = _mutate_seed_reference_evidence(
        cfg,
        canonical_kwargs,
        field_name,
        invalid_value,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-22T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
        expected_message='seed reference evidence',
    )


@pytest.mark.parametrize(
    ('evidence_case', 'formula_col'),
    [('missing_optional', 'formula'), ('missing_structure', 'composition')],
)
def test_reporting_accepts_builder_normalized_missing_seed_reference_evidence(
    tmp_path,
    monkeypatch,
    evidence_case,
    formula_col,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _real_builder_seed_evidence_writer_inputs(
        tmp_path,
        formula_col=formula_col,
        evidence_case=evidence_case,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-22T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    artifact_dir = Path(cfg['project']['artifact_dir'])
    assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(artifact_dir / 'artifact_provenance.json'),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert assessment['status'] == 'current'


_EDIT_PLAN_IDENTITY_CASES = (
    ('reference-reuse-false-relabel', 'full', 'formula'),
    ('relabel-site-index', 'multiple-success', 'formula'),
    ('relabel-target-element', 'multiple-success', 'composition'),
    ('removed-site-index', 'vacancy', 'formula'),
)


def _mutate_structure_execution_edit_plan_story(
    canonical_kwargs,
    mutation_kind,
):
    invalid_kwargs = copy.deepcopy(canonical_kwargs)
    variant_df = invalid_kwargs['structure_variant_df']
    variant_index = variant_df.index[0]
    variant_id = variant_df.loc[variant_index, 'execution_variant_id']
    payload_variant = next(
        variant
        for candidate in invalid_kwargs['structure_payload']['candidates']
        for variant in candidate['variants']
        if variant['execution_variant_id'] == variant_id
    )

    if mutation_kind == 'reference-reuse-false-relabel':
        updates = {
            'execution_plan_type': 'edited_structure',
            'relabel_site_indices': '0',
            'relabel_target_elements': 'Al',
            'relabeled_site_count': 1,
            'relaxation_status': 'not_run_unrelaxed_species_edit',
            'final_status': 'ready_for_external_relaxation',
        }
        summary_updates = {
            'first_pass_execution_selected_relaxation_status': (
                'not_run_unrelaxed_species_edit'
            ),
            'first_pass_execution_selected_final_status': (
                'ready_for_external_relaxation'
            ),
        }
    elif mutation_kind == 'relabel-site-index':
        current_index = int(variant_df.loc[variant_index, 'relabel_site_indices'])
        updates = {'relabel_site_indices': str(1 - current_index)}
        summary_updates = {}
    elif mutation_kind == 'relabel-target-element':
        updates = {'relabel_target_elements': 'C'}
        summary_updates = {}
    elif mutation_kind == 'removed-site-index':
        current_index = int(variant_df.loc[variant_index, 'removed_site_indices'])
        updates = {'removed_site_indices': str(1 - current_index)}
        summary_updates = {}
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(mutation_kind)

    for field_name, value in updates.items():
        variant_df.loc[variant_index, field_name] = value
        payload_variant[field_name] = value
    for field_name, value in summary_updates.items():
        invalid_kwargs['structure_summary_df'].loc[0, field_name] = value
    return invalid_kwargs


@pytest.mark.parametrize(
    ('mutation_kind', 'baseline_case', 'formula_col'),
    _EDIT_PLAN_IDENTITY_CASES,
)
def test_reporting_rejects_noncanonical_edit_plan_identity_atomically(
    tmp_path,
    monkeypatch,
    mutation_kind,
    baseline_case,
    formula_col,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case=baseline_case,
        formula_col=formula_col,
    )
    invalid_kwargs = _mutate_structure_execution_edit_plan_story(
        canonical_kwargs,
        mutation_kind,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
    )


@pytest.mark.parametrize(
    ('baseline_case', 'formula_col'),
    [('full', 'formula'), ('multiple-success', 'composition')],
)
def test_reporting_rejects_seed_record_relabel_atomically(
    tmp_path,
    monkeypatch,
    baseline_case,
    formula_col,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case=baseline_case,
        formula_col=formula_col,
    )
    raw_path = Path(cfg['data']['raw_dir']) / 'twod_matpd.json'
    raw_records = json.loads(raw_path.read_text(encoding='utf-8'))
    second_record = copy.deepcopy(raw_records[0])
    second_record['jid'] = 'jid-2'
    raw_records.append(second_record)
    raw_path.write_text(json.dumps(raw_records), encoding='utf-8')
    second_seed = canonical_kwargs['structure_generation_seed_df'].iloc[[0]].copy()
    second_seed.loc[:, 'structure_generation_seed_rank'] = 2
    second_seed.loc[:, 'seed_reference_record_id'] = 'jid-2'
    canonical_kwargs['structure_generation_seed_df'] = pd.concat(
        [canonical_kwargs['structure_generation_seed_df'], second_seed],
        ignore_index=True,
    )

    invalid_kwargs = copy.deepcopy(canonical_kwargs)
    invalid_kwargs['structure_payload']['candidates'][0][
        'seed_reference_record_id'
    ] = 'jid-2'
    for variant in invalid_kwargs['structure_payload']['candidates'][0]['variants']:
        variant['seed_reference_record_id'] = 'jid-2'
    invalid_kwargs['structure_summary_df'].loc[
        0, 'structure_followup_best_seed_reference_record_id'
    ] = 'jid-2'
    invalid_kwargs['structure_variant_df'].loc[
        :, 'seed_reference_record_id'
    ] = 'jid-2'
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
    )


@pytest.mark.parametrize(
    (
        'baseline_case',
        'formula_col',
        'invalid_raw_formula',
        'equivalent_raw_formula',
    ),
    [
        ('full', 'formula', 'AlBN', 'NB'),
        ('vacancy', 'composition', 'BN', 'NB2'),
    ],
)
def test_reporting_rejects_cached_raw_formula_mismatch_atomically(
    tmp_path,
    monkeypatch,
    baseline_case,
    formula_col,
    invalid_raw_formula,
    equivalent_raw_formula,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case=baseline_case,
        formula_col=formula_col,
    )
    raw_path = Path(cfg['data']['raw_dir']) / 'twod_matpd.json'
    valid_raw_bytes = raw_path.read_bytes()
    invalid_raw = json.loads(valid_raw_bytes)
    invalid_raw[0]['formula'] = invalid_raw_formula
    invalid_raw_bytes = json.dumps(invalid_raw).encode()
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    artifact_dir = Path(cfg['project']['artifact_dir'])

    raw_path.write_bytes(invalid_raw_bytes)
    with pytest.raises(ValueError, match='structure_first_pass_execution'):
        _save_minimal_report_bundle(
            cfg,
            manifest=manifest,
            **canonical_kwargs,
        )
    assert _report_bundle_snapshot(artifact_dir) is None

    raw_path.write_bytes(valid_raw_bytes)
    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    valid_snapshot = _report_bundle_snapshot(artifact_dir)

    raw_path.write_bytes(invalid_raw_bytes)
    with pytest.raises(ValueError, match='structure_first_pass_execution'):
        _save_minimal_report_bundle(
            cfg,
            manifest=manifest,
            **canonical_kwargs,
        )
    assert _report_bundle_snapshot(artifact_dir) == valid_snapshot

    raw_path.write_bytes(valid_raw_bytes)
    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(artifact_dir / 'artifact_provenance.json'),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert assessment['status'] == 'current'

    raw_records = json.loads(valid_raw_bytes)
    raw_records[0]['formula'] = equivalent_raw_formula
    raw_path.write_text(json.dumps(raw_records), encoding='utf-8')
    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    equivalent_assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(
            Path(cfg['project']['artifact_dir']) / 'artifact_provenance.json'
        ),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert equivalent_assessment['status'] == 'current'


@pytest.mark.parametrize('formula_col', ['formula', 'composition'])
def test_builder_rejects_seed_formula_with_unclaimed_source_elements(
    tmp_path,
    formula_col,
):
    cfg, writer_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case='source-formula-structure-mismatch',
        formula_col=formula_col,
    )

    assert writer_kwargs['structure_variant_df'].empty
    assert writer_kwargs['structure_payload']['variant_count'] == 0
    assert writer_kwargs['structure_payload']['candidates'][0][
        'candidate_status'
    ] == 'unresolved_reference_scale_factor'
    assert writer_kwargs['structure_summary_df'].loc[
        0, 'first_pass_execution_status'
    ] == 'unresolved_reference_scale_factor'
    assert not Path(cfg['project']['artifact_dir']).exists()


@pytest.mark.parametrize(
    ('invalid_role', 'expected_message'),
    [
        ('summary', 'structure_first_pass_execution_summary_df must contain'),
        ('variants', 'structure_first_pass_execution_variant_df must contain'),
    ],
)
def test_reporting_rejects_mislabeled_structure_execution_frames_before_mutation(
    tmp_path,
    invalid_role,
    expected_message,
):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
        },
    }
    writer_kwargs, _execution_cfg = _structure_execution_writer_kwargs(cfg)
    _save_minimal_report_bundle(cfg, **writer_kwargs)
    before = _report_bundle_snapshot(artifact_dir)

    invalid_writer_kwargs = dict(writer_kwargs)
    invalid_writer_kwargs[
        'structure_summary_df' if invalid_role == 'summary' else 'structure_variant_df'
    ] = writer_kwargs[
        'structure_variant_df' if invalid_role == 'summary' else 'structure_summary_df'
    ]
    with pytest.raises(ValueError, match=expected_message):
        _save_minimal_report_bundle(cfg, **invalid_writer_kwargs)

    assert _report_bundle_snapshot(artifact_dir) == before


_CANONICAL_STRUCTURE_EXECUTION_CASES = (
    ('inactive', 'formula', ()),
    ('empty', 'formula', ()),
    ('error', 'formula', ('missing_reference_record',)),
    ('invalid-reference', 'formula', ('invalid_reference_structure',)),
    ('unresolved-scale', 'formula', ('unresolved_reference_scale_factor',)),
    (
        'formula-scale-mismatch',
        'formula',
        ('candidate_formula_does_not_scale_to_reference_cell',),
    ),
    ('multiple-donor', 'formula', ('multiple_donor_species_not_supported',)),
    ('no-plan', 'formula', ('requires_atom_insertion',)),
    ('partial', 'formula', ('executed', 'missing_reference_record')),
    ('full', 'formula', ('executed',)),
    ('custom-paths', 'formula', ('executed',)),
    ('custom-paths', 'composition', ('executed',)),
    ('error', 'composition', ('missing_reference_record',)),
    ('invalid-reference', 'composition', ('invalid_reference_structure',)),
    ('unresolved-scale', 'composition', ('unresolved_reference_scale_factor',)),
    (
        'formula-scale-mismatch',
        'composition',
        ('candidate_formula_does_not_scale_to_reference_cell',),
    ),
    ('multiple-donor', 'composition', ('multiple_donor_species_not_supported',)),
    ('no-plan', 'composition', ('requires_atom_insertion',)),
    ('error-custom-paths', 'composition', ('missing_reference_record',)),
)


@pytest.mark.parametrize(
    ('baseline_case', 'formula_col', 'expected_statuses'),
    _CANONICAL_STRUCTURE_EXECUTION_CASES,
)
def test_reporting_accepts_canonical_structure_execution_builder_states(
    tmp_path,
    monkeypatch,
    baseline_case,
    formula_col,
    expected_statuses,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, writer_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case=baseline_case,
        formula_col=formula_col,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _save_minimal_report_bundle(cfg, manifest=manifest, **writer_kwargs)

    artifact_dir = Path(cfg['project']['artifact_dir'])
    assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(artifact_dir / 'artifact_provenance.json'),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert assessment['status'] == 'current'
    execution_cfg = _structure_first_pass_execution_config(cfg)
    should_publish_execution = baseline_case not in {'inactive', 'empty'}
    assert (artifact_dir / execution_cfg['artifact']).exists() is should_publish_execution
    summary_statuses = tuple(
        writer_kwargs['structure_summary_df'].get(
            'first_pass_execution_status', pd.Series(dtype='object')
        ).astype(str)
    )
    candidate_statuses = tuple(
        candidate['candidate_status']
        for candidate in writer_kwargs['structure_payload']['candidates']
    )
    assert summary_statuses == expected_statuses
    assert candidate_statuses == expected_statuses
    assert writer_kwargs['structure_payload']['enabled'] is (baseline_case != 'inactive')
    assert writer_kwargs['structure_payload']['candidate_count'] == len(expected_statuses)


_ZERO_VARIANT_BRANCH_CASES = tuple(
    (baseline_case, formula_col)
    for baseline_case, formula_col, expected_statuses
    in _CANONICAL_STRUCTURE_EXECUTION_CASES
    if expected_statuses
    and not {'executed', 'no_successful_variant'}.intersection(expected_statuses)
)


def _coordinated_zero_variant_status_story(
    writer_kwargs,
    *,
    formula_col,
    false_status,
):
    mutated = copy.deepcopy(writer_kwargs)
    payload = mutated['structure_payload']
    summary = mutated['structure_summary_df']
    formula = str(summary.loc[0, formula_col])
    payload['successful_variant_count'] = 0
    payload['status_counts'] = {false_status: 1}
    payload['executed_formulas'] = [formula] if false_status == 'executed' else []
    payload['candidates'][0]['candidate_status'] = false_status
    payload['candidates'][0]['selected_variant_id'] = None
    summary.loc[0, 'first_pass_execution_status'] = false_status
    summary.loc[0, 'first_pass_execution_variant_count'] = 0
    summary.loc[0, 'first_pass_execution_successful_variant_count'] = 0
    summary.loc[0, 'first_pass_execution_geometry_pass_variant_count'] = 0
    summary.loc[0, 'first_pass_execution_selected_variant_id'] = None
    summary.loc[0, 'first_pass_execution_selected_final_status'] = 'not_executed'
    return mutated


def _coordinated_variant_state_story(
    writer_kwargs,
    *,
    row_updates=None,
    clear_selection=False,
    remove_cif_text=False,
):
    mutated = copy.deepcopy(writer_kwargs)
    payload = mutated['structure_payload']
    summary = mutated['structure_summary_df']
    variant_df = mutated['structure_variant_df']
    payload_candidate = payload['candidates'][0]
    payload_variant = payload_candidate['variants'][0]
    row_updates = {} if row_updates is None else row_updates
    for field_name, value in row_updates.items():
        variant_df.loc[0, field_name] = value
        payload_variant[field_name] = value
    projection_fields = dict(
        (variant_field, summary_field)
        for summary_field, variant_field
        in _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
    )
    for field_name, value in row_updates.items():
        if field_name in projection_fields and not clear_selection:
            summary.loc[0, projection_fields[field_name]] = value
    if 'geometry_sanity_pass' in row_updates and not clear_selection:
        summary.loc[
            0, 'first_pass_execution_geometry_pass_variant_count'
        ] = int(row_updates['geometry_sanity_pass'] is True)
    if remove_cif_text:
        payload_variant['_cif_text'] = None
    if clear_selection:
        payload['successful_variant_count'] = 0
        payload['status_counts'] = {'no_successful_variant': 1}
        payload['executed_formulas'] = []
        payload_candidate['candidate_status'] = 'no_successful_variant'
        payload_candidate['selected_variant_id'] = None
        summary.loc[0, 'first_pass_execution_successful_variant_count'] = 0
        summary.loc[0, 'first_pass_execution_geometry_pass_variant_count'] = 0
        summary.loc[0, 'first_pass_execution_status'] = 'no_successful_variant'
        for column in summary.columns:
            if column.startswith('first_pass_execution_selected_'):
                summary.loc[0, column] = (
                    'not_executed' if column.endswith('final_status') else None
                )
    return mutated


_INVALID_VARIANT_STATE_STORIES = (
    (
        'unknown-execution-and-final-status',
        {'execution_status': 'validated', 'final_status': 'stable'},
        True,
        False,
    ),
    (
        'claim-like-final-status',
        {'final_status': 'experimentally_confirmed'},
        False,
        False,
    ),
    (
        'reference-relaxation-with-edit-final-status',
        {'final_status': 'ready_for_external_relaxation'},
        False,
        False,
    ),
    (
        'matching-formula-labelled-as-mismatch',
        {
            'formula_matches_candidate': False,
            'final_status': 'formula_mismatch_after_edit',
        },
        False,
        False,
    ),
    (
        'unsupported-relaxation-status',
        {'relaxation_status': 'stable'},
        False,
        False,
    ),
    (
        'geometry-ratio-below-pass-threshold',
        {'geometry_min_distance_ratio': 0.1},
        False,
        False,
    ),
    (
        'geometry-overlap-with-pass-status',
        {'geometry_overlap_pair_count': 1},
        False,
        False,
    ),
    (
        'execution-error-with-success-evidence',
        {
            'execution_status': 'error',
            'execution_message': 'RuntimeError: synthetic detail',
            'relaxation_status': 'not_run_due_to_execution_error',
            'final_status': 'execution_error',
        },
        True,
        False,
    ),
    ('successful-execution-without-cif-bytes', {}, False, True),
)


@pytest.mark.parametrize(
    ('baseline_case', 'formula_col'),
    [('full', 'formula'), ('custom-paths', 'composition')],
)
@pytest.mark.parametrize(
    ('case_name', 'row_updates', 'clear_selection', 'remove_cif_text'),
    _INVALID_VARIANT_STATE_STORIES,
    ids=[entry[0] for entry in _INVALID_VARIANT_STATE_STORIES],
)
def test_reporting_rejects_noncanonical_variant_states_atomically(
    tmp_path,
    monkeypatch,
    baseline_case,
    formula_col,
    case_name,
    row_updates,
    clear_selection,
    remove_cif_text,
):
    del case_name
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case=baseline_case,
        formula_col=formula_col,
    )
    invalid_kwargs = _coordinated_variant_state_story(
        canonical_kwargs,
        row_updates=row_updates,
        clear_selection=clear_selection,
        remove_cif_text=remove_cif_text,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
    )


_STRUCTURE_IDENTITY_MUTATIONS = (
    'atoms-species',
    'atoms-coordinates',
    'cif-coordinates',
    'coordinated-site-count',
    'coordinated-lattice-metadata',
    'coordinated-geometry-metadata',
)


def _mutate_structure_identity_story(writer_kwargs, mutation_kind):
    mutated = copy.deepcopy(writer_kwargs)
    payload_variant = mutated['structure_payload']['candidates'][0]['variants'][0]
    variant_df = mutated['structure_variant_df']
    summary_df = mutated['structure_summary_df']
    if mutation_kind == 'atoms-species':
        payload_variant['atoms']['elements'][0] = 'C'
    elif mutation_kind == 'atoms-coordinates':
        payload_variant['atoms']['coords'][1] = [0.1, 0.1, 0.2]
    elif mutation_kind == 'cif-coordinates':
        cif_structure = Structure.from_str(payload_variant['_cif_text'], fmt='cif')
        cif_structure.translate_sites(
            [1],
            [0.125, 0.0, 0.0],
            frac_coords=True,
            to_unit_cell=True,
        )
        payload_variant['_cif_text'] = cif_structure.to(fmt='cif')
    elif mutation_kind == 'coordinated-site-count':
        payload_variant['generated_structure_n_sites'] = 3
        payload_variant['structure_n_sites'] = 3.0
        variant_df.loc[0, 'generated_structure_n_sites'] = 3
        variant_df.loc[0, 'structure_n_sites'] = 3.0
        summary_df.loc[
            0, 'first_pass_execution_selected_structure_n_sites'
        ] = 3
    elif mutation_kind == 'coordinated-lattice-metadata':
        payload_variant['atoms']['abc'][0] = 99.0
        payload_variant['structure_lattice_a'] = 99.0
        variant_df.loc[0, 'structure_lattice_a'] = 99.0
    elif mutation_kind == 'coordinated-geometry-metadata':
        for field_name in (
            'geometry_min_distance',
            'geometry_mean_distance',
            'geometry_min_distance_ratio',
        ):
            payload_variant[field_name] = 9.0
            variant_df.loc[0, field_name] = 9.0
        summary_df.loc[0, 'first_pass_execution_selected_min_distance'] = 9.0
        summary_df.loc[
            0, 'first_pass_execution_selected_min_distance_ratio'
        ] = 9.0
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(mutation_kind)
    return mutated


@pytest.mark.parametrize(
    ('baseline_case', 'formula_col'),
    [('full', 'formula'), ('custom-paths', 'composition')],
)
@pytest.mark.parametrize('mutation_kind', _STRUCTURE_IDENTITY_MUTATIONS)
def test_reporting_rejects_noncanonical_structure_identity_atomically(
    tmp_path,
    monkeypatch,
    baseline_case,
    formula_col,
    mutation_kind,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case=baseline_case,
        formula_col=formula_col,
    )
    invalid_kwargs = _mutate_structure_identity_story(
        canonical_kwargs,
        mutation_kind,
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
    )


@pytest.mark.parametrize(('baseline_case', 'formula_col'), _ZERO_VARIANT_BRANCH_CASES)
@pytest.mark.parametrize(
    'mutation_kind',
    ['executed', 'no_successful_variant', 'selected-final-status'],
)
def test_reporting_rejects_coordinated_impossible_zero_variant_statuses_atomically(
    tmp_path,
    monkeypatch,
    baseline_case,
    formula_col,
    mutation_kind,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case=baseline_case,
        formula_col=formula_col,
    )
    artifact_dir = Path(cfg['project']['artifact_dir'])
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    assert canonical_kwargs['structure_variant_df'].empty
    assert len(canonical_kwargs['structure_summary_df']) == 1
    assert canonical_kwargs['structure_payload']['enabled'] is True
    assert canonical_kwargs['structure_payload']['candidate_count'] == 1
    if mutation_kind == 'selected-final-status':
        invalid_kwargs = copy.deepcopy(canonical_kwargs)
        invalid_kwargs['structure_summary_df'].loc[
            0, 'first_pass_execution_selected_final_status'
        ] = 'ready_for_external_relaxation'
        expected_message = 'selected final status'
    else:
        invalid_kwargs = _coordinated_zero_variant_status_story(
            canonical_kwargs,
            formula_col=formula_col,
            false_status=mutation_kind,
        )
        expected_message = 'zero-variant status'

    with pytest.raises(ValueError, match=expected_message):
        _save_minimal_report_bundle(cfg, manifest=manifest, **invalid_kwargs)
    assert _report_bundle_snapshot(artifact_dir) is None

    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    valid_snapshot = _report_bundle_snapshot(artifact_dir)
    with pytest.raises(ValueError, match=expected_message):
        _save_minimal_report_bundle(cfg, manifest=manifest, **invalid_kwargs)
    assert _report_bundle_snapshot(artifact_dir) == valid_snapshot

    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(artifact_dir / 'artifact_provenance.json'),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert assessment['status'] == 'current'


_NONCANONICAL_ZERO_VARIANT_STATUSES = (
    ('pending_manual_review', 'zero-variant status'),
    ('execution_complete', 'zero-variant status'),
    ('validated', 'zero-variant status'),
    ('stable', 'zero-variant status'),
    ('synthesized', 'zero-variant status'),
    ('ready', 'zero-variant status'),
    ('Missing_Reference_Record', 'zero-variant status'),
    (' missing_reference_record ', 'zero-variant status'),
    ('', 'non-empty string'),
    ('   ', 'zero-variant status'),
    (None, 'non-empty string'),
    (17, 'non-empty string'),
)


def test_structure_execution_candidate_status_authority_is_finite_and_complete():
    assert _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH == {
        'missing_reference': 'missing_reference_record',
        'invalid_reference': 'invalid_reference_structure',
        'unresolved_reference_scale': 'unresolved_reference_scale_factor',
        'unscalable_candidate_formula': (
            'candidate_formula_does_not_scale_to_reference_cell'
        ),
        'requires_atom_insertion': 'requires_atom_insertion',
        'multiple_donor_species': 'multiple_donor_species_not_supported',
        'no_donor_species': 'no_donor_species_found',
        'invalid_edit_counts': 'invalid_edit_counts',
        'insufficient_donor_sites': 'insufficient_donor_sites',
        'no_variant_plan': 'no_variant_plan_generated',
        'executed': 'executed',
        'no_successful_variant': 'no_successful_variant',
    }
    assert len(set(_STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH.values())) == 12
    assert _STRUCTURE_EXECUTION_ZERO_VARIANT_STATUSES == frozenset(
        status
        for branch, status
        in _STRUCTURE_EXECUTION_CANDIDATE_STATUS_BY_BRANCH.items()
        if branch not in {'executed', 'no_successful_variant'}
    )


def test_structure_execution_variant_status_authority_is_finite_and_complete():
    assert _STRUCTURE_EXECUTION_VARIANT_STATUS_BY_BRANCH == {
        'execution_ok': 'ok',
        'execution_error': 'error',
        'relaxation_reference_geometry_reused': (
            'not_run_reference_geometry_reused'
        ),
        'relaxation_unrelaxed_species_edit': (
            'not_run_unrelaxed_species_edit'
        ),
        'relaxation_execution_error': 'not_run_due_to_execution_error',
        'final_formula_mismatch': 'formula_mismatch_after_edit',
        'final_geometry_failure': 'geometry_sanity_failed',
        'final_reference_control': 'reference_control_ready',
        'final_external_relaxation': 'ready_for_external_relaxation',
        'final_execution_error': 'execution_error',
    }
    assert len(set(_STRUCTURE_EXECUTION_VARIANT_STATUS_BY_BRANCH.values())) == 10


@pytest.mark.parametrize(('baseline_case', 'formula_col'), _ZERO_VARIANT_BRANCH_CASES)
def test_reporting_rejects_noncanonical_zero_variant_status_vocabulary(
    tmp_path,
    baseline_case,
    formula_col,
):
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    for index, (false_status, expected_message) in enumerate(
        _NONCANONICAL_ZERO_VARIANT_STATUSES
    ):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
            case_root,
            baseline_case=baseline_case,
            formula_col=formula_col,
        )
        invalid_kwargs = _coordinated_zero_variant_status_story(
            canonical_kwargs,
            formula_col=formula_col,
            false_status=false_status,
        )

        with pytest.raises(ValueError, match=expected_message):
            _save_minimal_report_bundle(cfg, manifest=manifest, **invalid_kwargs)
        assert _report_bundle_snapshot(Path(cfg['project']['artifact_dir'])) is None


def test_reporting_rejects_claim_like_zero_variant_status_atomically_and_recovers(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case='error-custom-paths',
        formula_col='composition',
    )
    invalid_kwargs = _coordinated_zero_variant_status_story(
        canonical_kwargs,
        formula_col='composition',
        false_status='experimentally_synthesized',
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
    )


@pytest.mark.parametrize('formula_col', ['formula', 'composition'])
def test_reporting_accepts_canonical_failed_variant_no_success_control(
    tmp_path,
    monkeypatch,
    formula_col,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    monkeypatch.setattr(
        structure_execution_module,
        '_apply_variant_plan',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('synthetic failure')),
    )
    cfg, writer_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        formula_col=formula_col,
    )
    assert len(writer_kwargs['structure_variant_df']) == 1
    assert writer_kwargs['structure_variant_df']['execution_status'].tolist() == ['error']
    assert writer_kwargs['structure_variant_df'][
        [
            'execution_message',
            'formula_matches_candidate',
            'geometry_sanity_pass',
            'relaxation_status',
            'final_status',
        ]
    ].to_dict(orient='records') == [{
        'execution_message': 'RuntimeError: synthetic failure',
        'formula_matches_candidate': False,
        'geometry_sanity_pass': False,
        'relaxation_status': 'not_run_due_to_execution_error',
        'final_status': 'execution_error',
    }]
    assert writer_kwargs['structure_payload']['candidates'][0]['variants'][0][
        '_cif_text'
    ] is None
    assert writer_kwargs['structure_summary_df'][
        'first_pass_execution_status'
    ].tolist() == ['no_successful_variant']

    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    _save_minimal_report_bundle(cfg, manifest=manifest, **writer_kwargs)
    artifact_dir = Path(cfg['project']['artifact_dir'])
    assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(artifact_dir / 'artifact_provenance.json'),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert assessment['status'] == 'current'


@pytest.mark.parametrize('formula_col', ['formula', 'composition'])
def test_reporting_rejects_nonapplied_builder_edit_plan(
    tmp_path,
    monkeypatch,
    formula_col,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    monkeypatch.setattr(
        structure_execution_module,
        '_apply_variant_plan',
        lambda structure, **_kwargs: structure.copy(),
    )
    cfg, writer_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case='multiple-success',
        formula_col=formula_col,
    )
    variant_df = writer_kwargs['structure_variant_df']
    assert variant_df['execution_status'].eq('ok').all()
    assert not variant_df['formula_matches_candidate'].any()
    assert variant_df['relaxation_status'].eq(
        'not_run_unrelaxed_species_edit'
    ).all()
    assert variant_df['final_status'].eq('formula_mismatch_after_edit').all()
    assert all(
        variant['_cif_text']
        for variant
        in writer_kwargs['structure_payload']['candidates'][0]['variants']
    )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    with pytest.raises(ValueError, match='edit plan disagrees'):
        _save_minimal_report_bundle(cfg, manifest=manifest, **writer_kwargs)
    assert _report_bundle_snapshot(Path(cfg['project']['artifact_dir'])) is None


def test_reporting_accepts_builder_mixed_variant_outcomes_and_diagnostics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    original_apply_variant_plan = structure_execution_module._apply_variant_plan
    call_count = 0

    def mixed_variant_outcome(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError('variant-specific diagnostic detail')
        return original_apply_variant_plan(*args, **kwargs)

    monkeypatch.setattr(
        structure_execution_module,
        '_apply_variant_plan',
        mixed_variant_outcome,
    )
    cfg, writer_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case='multiple-success',
    )
    variant_df = writer_kwargs['structure_variant_df']
    assert variant_df[
        ['execution_status', 'execution_message', 'final_status']
    ].to_dict(orient='records') == [
        {
            'execution_status': 'ok',
            'execution_message': None,
            'final_status': 'geometry_sanity_failed',
        },
        {
            'execution_status': 'error',
            'execution_message': (
                'RuntimeError: variant-specific diagnostic detail'
            ),
            'final_status': 'execution_error',
        },
    ]
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    _save_minimal_report_bundle(cfg, manifest=manifest, **writer_kwargs)
    assessment = io_utils.assess_artifact_provenance(
        io_utils.read_json_file(
            Path(cfg['project']['artifact_dir']) / 'artifact_provenance.json'
        ),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert assessment['status'] == 'current'


_NO_SELECTION_SELECTED_MUTATIONS = (
    ('first_pass_execution_selected_variant_id', 'ghost__variant_01'),
    *(entry[::2] for entry in _SELECTED_PROJECTION_MUTATIONS),
    (
        'first_pass_execution_selected_final_status',
        'ready_for_external_relaxation',
    ),
)
_NO_SELECTION_BRANCH_CASES = (
    *_ZERO_VARIANT_BRANCH_CASES,
    ('failed-variant', 'formula'),
    ('failed-variant', 'composition'),
)


@pytest.mark.parametrize(('baseline_case', 'formula_col'), _NO_SELECTION_BRANCH_CASES)
def test_reporting_rejects_nonnull_selected_projections_without_selection(
    tmp_path,
    monkeypatch,
    baseline_case,
    formula_col,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    if baseline_case == 'failed-variant':
        monkeypatch.setattr(
            structure_execution_module,
            '_apply_variant_plan',
            lambda *_args, **_kwargs: (
                _ for _ in ()
            ).throw(RuntimeError('synthetic failure')),
        )
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    for index, (summary_field, invalid_value) in enumerate(
        _NO_SELECTION_SELECTED_MUTATIONS
    ):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
            case_root,
            baseline_case=(
                'full' if baseline_case == 'failed-variant' else baseline_case
            ),
            formula_col=formula_col,
        )
        invalid_kwargs = copy.deepcopy(canonical_kwargs)
        if (
            baseline_case == 'invalid-reference'
            and summary_field == 'first_pass_execution_selected_final_status'
        ):
            invalid_value = 'not_executed'
        if summary_field == 'first_pass_execution_selected_variant_id':
            invalid_kwargs['structure_payload']['candidates'][0][
                'selected_variant_id'
            ] = invalid_value
        invalid_kwargs['structure_summary_df'].loc[
            0, summary_field
        ] = invalid_value
        _assert_structure_execution_rejection_is_atomic(
            case_root,
            cfg,
            canonical_kwargs,
            invalid_kwargs,
            manifest,
        )


@pytest.mark.parametrize('formula_col', ['formula', 'composition'])
def test_reporting_rejects_noncanonical_successful_variant_selection_atomically(
    tmp_path,
    monkeypatch,
    formula_col,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(
        tmp_path,
        baseline_case='multiple-success',
        formula_col=formula_col,
    )
    variant_df = canonical_kwargs['structure_variant_df']
    summary_df = canonical_kwargs['structure_summary_df']
    assert variant_df['execution_status'].tolist() == ['ok', 'ok']
    assert variant_df[
        ['geometry_sanity_pass', 'relaxation_status', 'final_status']
    ].to_dict(orient='records') == [
        {
            'geometry_sanity_pass': False,
            'relaxation_status': 'not_run_unrelaxed_species_edit',
            'final_status': 'geometry_sanity_failed',
        },
        {
            'geometry_sanity_pass': True,
            'relaxation_status': 'not_run_unrelaxed_species_edit',
            'final_status': 'ready_for_external_relaxation',
        },
    ]
    assert summary_df['first_pass_execution_selected_variant_rank'].tolist() == [2]
    invalid_kwargs = copy.deepcopy(canonical_kwargs)
    noncanonical_variant = variant_df.iloc[0]
    invalid_kwargs['structure_payload']['candidates'][0][
        'selected_variant_id'
    ] = noncanonical_variant['execution_variant_id']
    for summary_field, variant_field in (
        (
            'first_pass_execution_selected_variant_id',
            'execution_variant_id',
        ),
        *((summary_field, variant_field)
          for summary_field, variant_field, _invalid_value
          in _SELECTED_PROJECTION_MUTATIONS),
        (
            'first_pass_execution_selected_final_status',
            'final_status',
        ),
    ):
        invalid_kwargs['structure_summary_df'].loc[
            0, summary_field
        ] = noncanonical_variant[variant_field]
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
    )


@pytest.mark.parametrize(
    'mismatch_case',
    _STRUCTURE_EXECUTION_RELATION_MUTATIONS,
)
def test_reporting_rejects_structure_execution_relational_mismatches_atomically(
    tmp_path,
    monkeypatch,
    mismatch_case,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(tmp_path)
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    invalid_kwargs = copy.deepcopy(canonical_kwargs)
    _mutate_structure_execution_relation(invalid_kwargs, mismatch_case)
    _assert_structure_execution_rejection_is_atomic(
        tmp_path,
        cfg,
        canonical_kwargs,
        invalid_kwargs,
        manifest,
    )


def test_structure_execution_selected_projection_contract_is_nonvacuous_and_atomic(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    projection_fields = _STRUCTURE_EXECUTION_SELECTED_PROJECTION_FIELDS
    assert projection_fields == (
        ('first_pass_execution_selected_variant_id', 'execution_variant_id'),
        ('first_pass_execution_selected_variant_rank', 'execution_variant_rank'),
        ('first_pass_execution_selected_cif_path', 'generated_structure_cif_path'),
        ('first_pass_execution_selected_generated_formula', 'generated_formula'),
        (
            'first_pass_execution_selected_structure_n_sites',
            'generated_structure_n_sites',
        ),
        ('first_pass_execution_selected_min_distance', 'geometry_min_distance'),
        (
            'first_pass_execution_selected_min_distance_ratio',
            'geometry_min_distance_ratio',
        ),
        ('first_pass_execution_selected_band_gap_proxy', 'structure_band_gap_proxy'),
        ('first_pass_execution_selected_relaxation_status', 'relaxation_status'),
        ('first_pass_execution_selected_final_status', 'final_status'),
    )
    assert len({summary_field for summary_field, _variant_field in projection_fields}) == 10
    assert len({variant_field for _summary_field, variant_field in projection_fields}) == 10

    cfg, canonical_kwargs = _canonical_structure_execution_writer_inputs(tmp_path)
    summary = canonical_kwargs['structure_summary_df'].iloc[0]
    selected_variant = _select_structure_execution_variant(
        canonical_kwargs['structure_variant_df']
    )
    assert selected_variant is not None
    for summary_field, variant_field in projection_fields:
        assert io_utils.make_json_safe(summary[summary_field]) == io_utils.make_json_safe(
            selected_variant[variant_field]
        )

    artifact_dir = Path(cfg['project']['artifact_dir'])
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    missing_field_cases = (
        *(('structure_summary_df', summary_field)
          for summary_field, _variant_field in projection_fields),
        *(('structure_variant_df', variant_field)
          for _summary_field, variant_field in projection_fields),
        *(('structure_payload', variant_field)
          for _summary_field, variant_field in projection_fields),
    )
    def without_field(role_name, field_name):
        invalid = copy.deepcopy(canonical_kwargs)
        if role_name == 'structure_payload':
            invalid[role_name]['candidates'][0]['variants'][0].pop(field_name)
        else:
            invalid[role_name] = invalid[role_name].drop(columns=[field_name])
        return invalid

    for role_name, missing_field in missing_field_cases:
        with pytest.raises(ValueError, match='structure_first_pass_execution'):
            _save_minimal_report_bundle(
                cfg,
                manifest=manifest,
                **without_field(role_name, missing_field),
            )
        assert _report_bundle_snapshot(artifact_dir) is None

    _save_minimal_report_bundle(cfg, manifest=manifest, **canonical_kwargs)
    valid_snapshot = _report_bundle_snapshot(artifact_dir)
    for role_name, missing_field in missing_field_cases:
        with pytest.raises(ValueError, match='structure_first_pass_execution'):
            _save_minimal_report_bundle(
                cfg,
                manifest=manifest,
                **without_field(role_name, missing_field),
            )
        assert _report_bundle_snapshot(artifact_dir) == valid_snapshot
    assert io_utils.assess_artifact_provenance(
        io_utils.read_json_file(artifact_dir / 'artifact_provenance.json'),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['status'] == 'current'


def _assert_summary_preflight_rejection_is_atomic(
    tmp_path,
    invalid_summary,
    *,
    execution_active,
    prepare_artifact_root=None,
):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
        },
    }
    writer_kwargs = {}
    if execution_active:
        writer_kwargs, _execution_cfg = _structure_execution_writer_kwargs(cfg)
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    if prepare_artifact_root is not None:
        prepare_artifact_root(artifact_dir)

    before = _report_bundle_snapshot(artifact_dir)
    with pytest.raises(ValueError, match='experiment_summary'):
        _save_minimal_report_bundle(
            cfg,
            experiment_summary=invalid_summary,
            manifest=manifest,
            **writer_kwargs,
        )
    assert _report_bundle_snapshot(artifact_dir) == before

    _save_minimal_report_bundle(cfg, manifest=manifest, **writer_kwargs)
    provenance_path = artifact_dir / 'artifact_provenance.json'
    provenance = io_utils.read_json_file(provenance_path)
    assert io_utils.assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['status'] == 'current'
    valid_snapshot = _report_bundle_snapshot(artifact_dir)

    with pytest.raises(ValueError, match='experiment_summary'):
        _save_minimal_report_bundle(
            cfg,
            experiment_summary=invalid_summary,
            manifest=manifest,
            **writer_kwargs,
        )
    assert _report_bundle_snapshot(artifact_dir) == valid_snapshot
    _save_minimal_report_bundle(cfg, manifest=manifest, **writer_kwargs)
    assert io_utils.assess_artifact_provenance(
        io_utils.read_json_file(provenance_path),
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['status'] == 'current'


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
def test_reporting_rejects_wrong_shaped_summary_before_any_bundle_mutation(
    tmp_path,
    monkeypatch,
    container_name,
    shape_value,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    invalid_summary = {'screening': shape_value}
    if container_name == 'structure-generation-bridge':
        invalid_summary = {
            'screening': {'structure_generation_bridge': shape_value},
        }
    _assert_summary_preflight_rejection_is_atomic(
        tmp_path,
        invalid_summary,
        execution_active=False,
    )


@pytest.mark.parametrize(
    'override_case',
    [
        'blank',
        'traversal',
        'absolute',
        'non-string-number',
        'non-string-list',
        'non-string-mapping',
        'non-string-bool',
        'null',
        'directory',
        'wrong-suffix',
        'fixed-json-alias',
        'summary-bn-slice-alias',
        'variants-bn-slice-alias',
        'missing',
        'roles-swapped',
        'inactive-declaration',
    ],
)
def test_reporting_rejects_invalid_dynamic_summary_declaration_before_mutation(
    tmp_path,
    monkeypatch,
    override_case,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg = {
        'project': {'artifact_dir': str(tmp_path / 'artifacts')},
        'screening': {},
    }
    execution_cfg = _structure_first_pass_execution_config(cfg)
    bridge = {
        f'first_pass_execution_{field}': execution_cfg[field]
        for field in ('artifact', 'summary_artifact', 'variants_artifact')
    }
    values = {
        'blank': ' ',
        'traversal': '../escape.json',
        'absolute': str(tmp_path / 'outside.json'),
        'non-string-number': 7,
        'non-string-list': ['execution.json'],
        'non-string-mapping': {'path': 'execution.json'},
        'non-string-bool': False,
        'null': None,
        'directory': 'declared-directory.json',
        'wrong-suffix': 'nested/execution.csv',
        'fixed-json-alias': 'metrics.json',
        'summary-bn-slice-alias': 'bn_slice.csv',
        'variants-bn-slice-alias': 'bn_slice.csv',
        'missing': 'nested/missing.json',
    }
    field_by_case = {
        'summary-bn-slice-alias': 'first_pass_execution_summary_artifact',
        'variants-bn-slice-alias': 'first_pass_execution_variants_artifact',
    }
    if override_case == 'roles-swapped':
        bridge.update({
            'first_pass_execution_summary_artifact': execution_cfg['variants_artifact'],
            'first_pass_execution_variants_artifact': execution_cfg['summary_artifact'],
        })
    elif override_case != 'inactive-declaration':
        bridge[field_by_case.get(override_case, 'first_pass_execution_artifact')] = (
            values[override_case]
        )

    prepare_artifact_root = None
    if override_case == 'directory':
        def prepare_artifact_root(artifact_dir):
            (artifact_dir / 'declared-directory.json').mkdir(parents=True)

    _assert_summary_preflight_rejection_is_atomic(
        tmp_path,
        {'screening': {'structure_generation_bridge': bridge}},
        execution_active=override_case != 'inactive-declaration',
        prepare_artifact_root=prepare_artifact_root,
    )


@pytest.mark.parametrize(
    'valid_case',
    [
        'screening-absent',
        'screening-null',
        'screening-empty',
        'bridge-absent',
        'bridge-null',
        'bridge-empty',
        'exact-paths',
        'normalized-paths',
        'case-only-samefile',
    ],
)
def test_reporting_accepts_valid_summary_fallback_and_role_contracts(
    tmp_path,
    monkeypatch,
    valid_case,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    artifact_dir = tmp_path / 'artifacts'
    execution_overrides = {}
    if valid_case == 'normalized-paths':
        execution_overrides = {
            'artifact': 'nested/a/../execution.json',
            'summary_artifact': 'nested/a/../execution-summary.csv',
            'variants_artifact': 'nested/a/../execution-variants.csv',
            'structure_dir': 'nested/a/../cifs',
        }
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
            'structure_first_pass_execution': execution_overrides,
        },
    }
    writer_kwargs, execution_cfg = _structure_execution_writer_kwargs(cfg)
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    if valid_case.startswith('screening-'):
        state = valid_case.removeprefix('screening-')
        summary = {} if state == 'absent' else {
            'screening': None if state == 'null' else {},
        }
    elif valid_case.startswith('bridge-'):
        state = valid_case.removeprefix('bridge-')
        bridge = None if state == 'null' else {}
        summary = {'screening': {}}
        if state != 'absent':
            summary['screening']['structure_generation_bridge'] = bridge
    else:
        summary_paths = execution_cfg
        if valid_case == 'normalized-paths':
            summary_paths = {
                'artifact': 'nested/execution.json',
                'summary_artifact': 'nested/execution-summary.csv',
                'variants_artifact': 'nested/execution-variants.csv',
            }
        if valid_case == 'case-only-samefile':
            _save_minimal_report_bundle(cfg, manifest=manifest, **writer_kwargs)
            case_only_value = execution_cfg['artifact'].upper()
            case_only_path = artifact_dir / case_only_value
            if not case_only_path.exists():
                pytest.skip('local filesystem treats case-only names as distinct files')
            assert (artifact_dir / execution_cfg['artifact']).samefile(case_only_path)
            writer_kwargs['structure_payload'] = {
                **writer_kwargs['structure_payload'],
                'artifact': case_only_value,
            }
            summary_paths = {**execution_cfg, 'artifact': case_only_value}
        summary = {
            'screening': {
                'structure_generation_bridge': {
                    f'first_pass_execution_{field}': summary_paths[field]
                    for field in ('artifact', 'summary_artifact', 'variants_artifact')
                },
            },
        }

    _save_minimal_report_bundle(
        cfg,
        experiment_summary=summary,
        manifest=manifest,
        **writer_kwargs,
    )
    provenance = io_utils.read_json_file(
        artifact_dir / 'artifact_provenance.json'
    )
    assert io_utils.assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['status'] == 'current'
    assert all(
        (artifact_dir / execution_cfg[field]).resolve().relative_to(
            artifact_dir.resolve()
        ).as_posix() in provenance['published_outputs']
        for field in ('artifact', 'summary_artifact', 'variants_artifact')
    )


@pytest.mark.parametrize('prior_bundle', [False, True], ids=['no-prior', 'known-good-prior'])
@pytest.mark.parametrize(
    'failure_target',
    [
        'metrics.json',
        'nested/execution-summary.csv',
        'experiment_summary.json',
        'artifact_provenance.json',
    ],
    ids=['early-json', 'middle-nested-csv', 'late-summary', 'completion-marker'],
)
def test_reporting_failure_after_publication_begins_leaves_no_completion_marker(
    tmp_path,
    monkeypatch,
    prior_bundle,
    failure_target,
):
    artifact_dir = tmp_path / 'custom-output' / 'artifacts'
    execution_cfg = {
        'artifact': 'nested/execution.json',
        'summary_artifact': 'nested/execution-summary.csv',
        'variants_artifact': 'nested/execution-variants.csv',
        'structure_dir': 'nested/cifs',
    }
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_first_pass_execution': execution_cfg,
        },
    }
    writer_kwargs, _execution_cfg = _structure_execution_writer_kwargs(cfg)
    completion_marker = artifact_dir / 'artifact_provenance.json'
    if prior_bundle:
        _save_minimal_report_bundle(cfg, **writer_kwargs)
        assert completion_marker.exists()

    original_json_writer = artifacts_module.write_json_file
    original_csv_writer = artifacts_module._write_csv_file

    def fail_json_after_write(payload, path, **kwargs):
        result = original_json_writer(payload, path, **kwargs)
        if Path(path).as_posix().endswith(failure_target):
            exception_type = (
                KeyboardInterrupt
                if failure_target == 'artifact_provenance.json'
                else RuntimeError
            )
            raise exception_type(f'synthetic failure after {failure_target}')
        return result

    def fail_csv_after_write(frame, path):
        result = original_csv_writer(frame, path)
        if Path(path).as_posix().endswith(failure_target):
            raise RuntimeError(f'synthetic failure after {failure_target}')
        return result

    monkeypatch.setattr(artifacts_module, 'write_json_file', fail_json_after_write)
    monkeypatch.setattr(artifacts_module, '_write_csv_file', fail_csv_after_write)

    expected_exception = (
        KeyboardInterrupt
        if failure_target == 'artifact_provenance.json'
        else RuntimeError
    )
    with pytest.raises(expected_exception, match='synthetic failure'):
        _save_minimal_report_bundle(cfg, **writer_kwargs)

    assert not completion_marker.exists()


def test_completion_marker_commits_exact_successful_bundle_outputs(
    tmp_path,
    monkeypatch,
):
    artifact_dir = tmp_path / 'custom-output' / 'artifacts'
    execution_cfg = {
        'artifact': 'nested/execution.json',
        'summary_artifact': 'nested/execution-summary.csv',
        'variants_artifact': 'nested/execution-variants.csv',
        'structure_dir': 'nested/cifs',
    }
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
            'structure_first_pass_execution': execution_cfg,
        },
    }
    writer_kwargs, _execution_cfg = _structure_execution_writer_kwargs(cfg)
    structure_payload = writer_kwargs['structure_payload']
    structure_payload['candidates'][0]['variants'][0].update({
        'generated_structure_cif_path': 'nested/cifs/xbn__variant_01.cif',
    })
    writer_kwargs['structure_variant_df'].loc[
        0, 'generated_structure_cif_path'
    ] = 'nested/cifs/xbn__variant_01.cif'
    writer_kwargs['structure_summary_df'].loc[
        0, 'first_pass_execution_selected_cif_path'
    ] = 'nested/cifs/xbn__variant_01.cif'
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
        pd.DataFrame([{'formula': 'BN', 'target': 5.0, 'prediction': 5.0}]),
        empty_df,
        pd.DataFrame([{'formula': 'BN', 'ranking_rank': 1}]),
        pd.DataFrame([{'model_type': 'linear', 'mae': 1.0}]),
        pd.DataFrame([{'model_type': 'linear', 'mae_mean': 1.1}]),
        empty_df,
        empty_df,
        empty_df,
        writer_kwargs['structure_generation_seed_df'],
        {'dataset': {'rows': 1}},
        manifest,
        cfg,
        structure_first_pass_execution_variant_df=writer_kwargs['structure_variant_df'],
        structure_first_pass_execution_summary_df=writer_kwargs['structure_summary_df'],
        structure_first_pass_execution_payload=structure_payload,
        include_parity_plot=True,
    )

    provenance_path = artifact_dir / 'artifact_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    actual_bundle_files = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob('*')
        if path.is_file() and path != provenance_path
    }

    assert provenance['schema'] == 'aiforbn.artifact_provenance.v2'
    assert set(provenance['published_outputs']) == actual_bundle_files
    assert 'bn_slice.csv' in provenance['published_outputs']
    assert 'robustness_results.csv' in provenance['published_outputs']
    assert 'nested/execution.json' in provenance['published_outputs']
    assert 'nested/cifs/xbn__variant_01.cif' in provenance['published_outputs']
    assert 'parity_plot.png' in provenance['published_outputs']


def test_parity_plot_precedes_and_is_committed_by_final_completion_marker(
    tmp_path,
    monkeypatch,
):
    artifact_dir = tmp_path / 'artifacts'
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {'formula_column': 'formula'},
        'screening': {
            'ranking_stability': {'enabled': False},
            'decision_policy': {'enabled': False},
            'structure_generation_seeds': {'enabled': False},
        },
    }
    prediction_df = pd.DataFrame([{
        'formula': 'BN',
        'target': 5.0,
        'prediction': 4.8,
    }])
    _save_minimal_report_bundle(cfg)
    completion_marker = artifact_dir / 'artifact_provenance.json'
    assert completion_marker.exists()
    original_marker = completion_marker.read_bytes()

    with pytest.raises(ValueError, match='target and prediction columns'):
        _save_minimal_report_bundle(
            cfg,
            prediction_df=pd.DataFrame([{'target': 5.0}]),
            include_parity_plot=True,
        )

    assert completion_marker.read_bytes() == original_marker

    def fail_plot_after_write(_figure, path, *args, **kwargs):
        Path(path).write_bytes(b'partial-plot')
        raise RuntimeError('synthetic parity plotting failure')

    with monkeypatch.context() as plot_patch:
        plot_patch.setattr(plots_module.plt.Figure, 'savefig', fail_plot_after_write)
        with pytest.raises(RuntimeError, match='synthetic parity plotting failure'):
            _save_minimal_report_bundle(
                cfg,
                prediction_df=prediction_df,
                include_parity_plot=True,
            )

    assert not completion_marker.exists()
    publication_events: list[str] = []
    original_savefig = plots_module.plt.Figure.savefig
    original_json_writer = artifacts_module.write_json_file

    def record_plot(_figure, path, *args, **kwargs):
        result = original_savefig(_figure, path, *args, **kwargs)
        publication_events.append('parity_plot.png')
        return result

    def record_marker(payload, path, **kwargs):
        result = original_json_writer(payload, path, **kwargs)
        if Path(path).name == 'artifact_provenance.json':
            publication_events.append('artifact_provenance.json')
        return result

    with monkeypatch.context() as order_patch:
        order_patch.setattr(plots_module.plt.Figure, 'savefig', record_plot)
        order_patch.setattr(artifacts_module, 'write_json_file', record_marker)
        _save_minimal_report_bundle(
            cfg,
            prediction_df=prediction_df,
            include_parity_plot=True,
        )

    assert publication_events == ['parity_plot.png', 'artifact_provenance.json']

    parity_plot_path = artifact_dir / 'parity_plot.png'
    provenance = json.loads(completion_marker.read_text(encoding='utf-8'))
    assert 'parity_plot.png' in provenance['published_outputs']
    assert provenance['published_outputs']['parity_plot.png'] == hashlib.sha256(
        parity_plot_path.read_bytes()
    ).hexdigest()

    no_plot_dir = tmp_path / 'no-plot-artifacts'
    cfg['project']['artifact_dir'] = str(no_plot_dir)
    _save_minimal_report_bundle(cfg)
    no_plot_provenance = json.loads(
        (no_plot_dir / 'artifact_provenance.json').read_text(encoding='utf-8')
    )
    assert 'parity_plot.png' not in no_plot_provenance['published_outputs']


def test_csv_report_replace_preserves_previous_file_on_serialization_failure(tmp_path):
    output_path = tmp_path / 'report.csv'
    output_path.write_text('old,value\n1,keep\n', encoding='utf-8')

    class BrokenCsvValue:
        def __str__(self):
            raise RuntimeError('cannot serialize CSV value')

    with pytest.raises(RuntimeError, match='cannot serialize CSV value'):
        artifacts_module._write_csv_file(
            pd.DataFrame({'value': [BrokenCsvValue()]}),
            output_path,
        )

    assert output_path.read_text(encoding='utf-8') == 'old,value\n1,keep\n'
    assert list(tmp_path.glob(f'.{output_path.name}.*.tmp')) == []


def test_reporting_writes_expected_artifacts(tmp_path):
    artifact_dir = tmp_path / 'artifacts'
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    (raw_dir / 'twod_matpd.json').write_text(
        json.dumps(
            [
                {
                    'jid': 'jid-1',
                    'formula': 'BN',
                    'band_gap': 5.8,
                    'atoms': {
                        'elements': ['B', 'N'],
                        'coords': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
                        'lattice_mat': [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 20.0]],
                        'abc': [2.5, 2.5, 20.0],
                        'angles': [90.0, 90.0, 120.0],
                        'cartesian': False,
                    },
                },
                {
                    'jid': 'jid-2',
                    'formula': 'B2N',
                    'band_gap': 4.2,
                    'atoms': {
                        'elements': ['B', 'B', 'N'],
                        'coords': [[0.0, 0.0, 0.0], [0.33, 0.33, 0.0], [0.66, 0.66, 0.0]],
                        'lattice_mat': [[2.8, 0.0, 0.0], [0.0, 2.8, 0.0], [0.0, 0.0, 20.0]],
                        'abc': [2.8, 2.8, 20.0],
                        'angles': [90.0, 90.0, 120.0],
                        'cartesian': False,
                    },
                },
            ]
        ),
        encoding='utf-8',
    )
    cfg = {
        'project': {'artifact_dir': str(artifact_dir)},
        'data': {
            'dataset': 'twod_matpd',
            'raw_dir': str(raw_dir),
            'formula_column': 'formula',
            'target_column': 'band_gap',
        },
        'features': {
            'feature_set': 'basic_formula_composition',
            'candidate_sets': [
                'basic_formula_composition',
                'matminer_composition',
                'matminer_composition_plus_structure_summary',
            ],
            'feature_family': 'mixed_formula_and_structure',
        },
        'model': {
            'type': 'hist_gradient_boosting',
            'candidate_types': ['linear_regression', 'hist_gradient_boosting'],
            'benchmark_baselines': ['dummy_mean'],
        },
        'robustness': {
            'enabled': True,
            'method': 'group_kfold_by_formula',
            'group_column': 'formula',
            'n_splits': 4,
            'note': 'demo grouped robustness note',
        },
        'bn_slice_benchmark': {
            'enabled': True,
            'method': 'leave_one_bn_formula_out',
            'k_neighbors': 2,
            'note': 'demo bn slice benchmark note',
        },
        'bn_family_benchmark': {
            'enabled': True,
            'method': 'leave_one_bn_family_out',
            'grouping_method': 'reduced_bn_chemical_system',
            'k_neighbors': 2,
            'note': 'demo bn family benchmark note',
        },
        'bn_stratified_error': {
            'enabled': True,
            'method': 'group_kfold_bn_vs_non_bn_formula_stratified_error',
            'group_column': 'formula',
            'n_splits': 3,
            'note': 'demo bn stratified error note',
        },
        'screening': {
            'objective_name': 'ai_powered_boron_nitride_material_exploration',
            'objective_target_property': 'band_gap',
            'objective_target_direction': 'target_window_proxy',
            'objective_decision_unit': 'formula_level_candidate',
            'objective_decision_consequence': 'low_confidence_prioritization_for_structure_followup',
            'objective_note': 'Uncertainty-aware BN-material candidate prioritization for downstream structure exploration, not direct discovery.',
            'candidate_generation_strategy': 'bn_anchored_formula_family_grid',
            'candidate_space_name': 'bn_anchored_formula_family_grid',
            'candidate_space_kind': 'bn_family_demo',
            'candidate_space_note': 'bn-anchored demo note',
            'top_k': 5,
            'use_model_disagreement': True,
            'uncertainty_method': 'small_feature_model_disagreement',
            'uncertainty_penalty': 0.5,
            'grouped_robustness_uncertainty': {
                'enabled': True,
                'method': 'selected_formula_only_group_kfold_candidate_prediction_std',
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.15,
                'note': 'demo grouped candidate robustness note',
            },
            'domain_support': {
                'enabled': True,
                'method': 'train_plus_val_knn_feature_space_support',
                'distance_metric': 'z_scored_euclidean_rms',
                'k_neighbors': 5,
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.15,
                'penalize_below_percentile': 25.0,
                'note': 'demo domain-support note',
            },
            'bn_support': {
                'enabled': True,
                'method': 'train_plus_val_bn_knn_feature_space_support',
                'distance_metric': 'z_scored_euclidean_rms',
                'k_neighbors': 3,
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.1,
                'penalize_below_percentile': 25.0,
                'note': 'demo bn-support note',
            },
            'bn_analog_evidence': {
                'enabled': True,
                'aggregation': 'mean_over_k_nearest_bn_formulas',
                'reference_split': 'train_plus_val_bn_unique_formulas',
                'exfoliation_reference': 'train_plus_val_bn_formula_median',
                'note': 'demo bn-analog evidence note',
            },
            'bn_band_gap_alignment': {
                'enabled': True,
                'method': 'predicted_band_gap_vs_local_bn_analog_window',
                'reference_split': 'train_plus_val_bn_unique_formulas',
                'window_expansion_iqr_factor': 0.5,
                'minimum_neighbor_formula_count_for_penalty': 2,
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.08,
                'note': 'demo bn-local band-gap alignment note',
            },
            'bn_analog_validation': {
                'enabled': True,
                'method': 'bn_analog_alignment_vote_fraction',
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.12,
                'note': 'demo bn-analog validation note',
            },
            'chemical_plausibility': {
                'enabled': True,
                'method': 'pymatgen_common_oxidation_state_balance',
                'selection_policy': 'annotate_and_prioritize_passing_candidates',
                'note': 'demo plausibility note',
            },
            'proposal_shortlist': {
                'enabled': True,
                'label': 'family_aware_proposal_shortlist',
                'method': 'ranked_family_cap',
                'shortlist_size': 2,
                'max_per_candidate_family': 1,
                'chemical_plausibility_priority': True,
                'note': 'demo proposal shortlist note',
            },
            'extrapolation_shortlist': {
                'enabled': True,
                'label': 'formula_level_extrapolation_shortlist',
                'method': 'novelty_bucket_ranked_family_cap',
                'shortlist_size': 1,
                'max_per_candidate_family': 1,
                'required_novelty_bucket': 'formula_level_extrapolation',
                'chemical_plausibility_priority': True,
                'note': 'demo extrapolation shortlist note',
            },
            'structure_first_pass_execution': {
                'enabled': True,
                'max_candidates': 2,
                'max_variants_per_candidate': 2,
            },
        },
    }

    metrics = {
        'mae': 1.0,
        'rmse': 2.0,
        'r2': 0.5,
        'selected_model_type': 'linear_regression',
        'selected_feature_set': 'matminer_composition',
        'selected_feature_family': 'composition_only',
        'screening_feature_set': 'matminer_composition',
        'screening_model_type': 'linear_regression',
        'screening_feature_family': 'composition_only',
    }
    prediction_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0], 'prediction': [4.8]})
    bn_df = pd.DataFrame({'formula': ['BN'], 'target': [5.0]})
    candidate_df = pd.DataFrame({
        'formula': ['BN', 'AlBN'],
        'candidate_space_name': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
        'candidate_space_kind': ['bn_family_demo', 'bn_family_demo'],
        'candidate_generation_strategy': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
        'candidate_family': ['bn_binary_anchor', 'group13_bn_111_family'],
        'candidate_template': ['B1N1', 'X1B1N1'],
        'candidate_family_note': ['BN anchor', 'Group-III BN ternary extension'],
        'ranking_rank': [1, 2],
        'ranking_score': [4.8, 1.2],
        'grouped_robustness_prediction_enabled': [True, True],
        'grouped_robustness_prediction_method': [
            'selected_formula_only_group_kfold_candidate_prediction_std',
            'selected_formula_only_group_kfold_candidate_prediction_std',
        ],
        'grouped_robustness_prediction_note': [
            'demo grouped candidate robustness note',
            'demo grouped candidate robustness note',
        ],
        'grouped_robustness_prediction_feature_set': ['matminer_composition', 'matminer_composition'],
        'grouped_robustness_prediction_model_type': ['linear_regression', 'linear_regression'],
        'grouped_robustness_prediction_fold_count': [4, 4],
        'grouped_robustness_predicted_band_gap_mean': [4.82, 1.24],
        'grouped_robustness_predicted_band_gap_std': [0.02, 0.30],
        'grouped_robustness_uncertainty_penalty': [0.003, 0.045],
        'ranking_score_before_grouped_robustness_penalty': [4.803, 1.245],
        'domain_support_reference_formula_count': [12, 12],
        'domain_support_k_neighbors': [5, 5],
        'domain_support_nearest_formula': ['BN', 'BN'],
        'domain_support_nearest_distance': [0.0, 0.8],
        'domain_support_mean_k_distance': [0.0, 1.1],
        'domain_support_percentile': [100.0, 10.0],
        'domain_support_penalty': [0.0, 0.09],
        'bn_support_reference_formula_count': [4, 4],
        'bn_support_k_neighbors': [3, 3],
        'bn_support_nearest_formula': ['BN', 'BN'],
        'bn_support_neighbor_formulas': ['BN', 'BN|Si2BN'],
        'bn_support_neighbor_formula_count': [1, 2],
        'bn_support_nearest_distance': [0.0, 0.4],
        'bn_support_mean_k_distance': [0.0, 0.6],
        'bn_support_percentile': [100.0, 0.0],
        'bn_support_penalty': [0.0, 0.1],
        'bn_analog_evidence_enabled': [True, True],
        'bn_analog_evidence_aggregation': ['mean_over_k_nearest_bn_formulas', 'mean_over_k_nearest_bn_formulas'],
        'bn_analog_reference_formula_count': [4, 4],
        'bn_analog_reference_band_gap_median': [3.6, 3.6],
        'bn_analog_reference_band_gap_iqr': [1.2, 1.2],
        'bn_analog_reference_exfoliation_energy_median': [0.07, 0.07],
        'bn_analog_reference_energy_per_atom_median': [-8.0, -8.0],
        'bn_analog_reference_abs_total_magnetization_median': [0.0, 0.0],
        'bn_analog_nearest_formula': ['BN', 'BN'],
        'bn_analog_neighbor_formulas': ['BN', 'BN|Si2BN'],
        'bn_analog_neighbor_formula_count': [1, 2],
        'bn_analog_nearest_band_gap': [4.8, 4.8],
        'bn_analog_nearest_energy_per_atom': [-8.3, -8.3],
        'bn_analog_nearest_exfoliation_energy_per_atom': [0.06, 0.06],
        'bn_analog_nearest_abs_total_magnetization': [0.0, 0.0],
        'bn_analog_neighbor_band_gap_mean': [4.8, 2.4],
        'bn_analog_neighbor_band_gap_min': [4.8, 0.0],
        'bn_analog_neighbor_band_gap_max': [4.8, 4.8],
        'bn_analog_neighbor_band_gap_std': [0.0, 2.4],
        'bn_analog_neighbor_energy_per_atom_mean': [-8.3, -7.3],
        'bn_analog_neighbor_exfoliation_energy_per_atom_mean': [0.06, 0.06],
        'bn_analog_neighbor_abs_total_magnetization_mean': [0.0, 0.0],
        'bn_analog_neighbor_exfoliation_available_formula_count': [1, 1],
        'bn_band_gap_alignment_enabled': [True, True],
        'bn_band_gap_alignment_method': [
            'predicted_band_gap_vs_local_bn_analog_window',
            'predicted_band_gap_vs_local_bn_analog_window',
        ],
        'bn_band_gap_alignment_reference_split': [
            'train_plus_val_bn_unique_formulas',
            'train_plus_val_bn_unique_formulas',
        ],
        'bn_band_gap_alignment_note': [
            'demo bn-local band-gap alignment note',
            'demo bn-local band-gap alignment note',
        ],
        'bn_band_gap_alignment_neighbor_available_formula_count': [1, 2],
        'bn_band_gap_alignment_window_lower': [4.2, -0.6],
        'bn_band_gap_alignment_window_upper': [5.4, 5.4],
        'bn_band_gap_alignment_distance_to_window': [0.0, 0.6],
        'bn_band_gap_alignment_relative_distance': [0.0, 0.5],
        'bn_band_gap_alignment_penalty_eligible': [False, True],
        'bn_band_gap_alignment_label': [
            'within_local_bn_analog_band_gap_window',
            'above_local_bn_analog_band_gap_window',
        ],
        'bn_band_gap_alignment_penalty': [0.0, 0.04],
        'bn_analog_exfoliation_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
        'bn_analog_energy_support_label': ['lower_or_equal_bn_reference_median', 'higher_than_bn_reference_median'],
        'bn_analog_abs_total_magnetization_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
        'bn_analog_support_vote_count': [3, 2],
        'bn_analog_support_available_metric_count': [3, 3],
        'bn_analog_validation_label': ['reference_like_on_available_metrics', 'mixed_reference_alignment'],
        'bn_analog_validation_support_fraction': [1.0, 2.0 / 3.0],
        'bn_analog_validation_penalty': [0.0, 0.04],
        'chemical_plausibility_pass': [True, False],
        'chemical_plausibility_guess_count': [1, 0],
        'chemical_plausibility_primary_oxidation_state_guess': ['B(+3), N(-3)', ''],
        'chemical_plausibility_note': ['pass', 'fail'],
        'seen_in_dataset': [True, False],
        'dataset_formula_row_count': [3, 0],
        'seen_in_train_plus_val': [True, False],
        'train_plus_val_formula_row_count': [2, 0],
        'candidate_is_seen_in_dataset': [True, False],
        'candidate_is_seen_in_train_plus_val': [True, False],
        'candidate_is_formula_level_extrapolation': [False, True],
        'candidate_novelty_bucket': ['train_plus_val_rediscovery', 'formula_level_extrapolation'],
        'candidate_novelty_priority': [1, 3],
        'candidate_novelty_note': ['rediscovery note', 'novel note'],
        'novelty_rank_within_bucket': [1, 1],
        'novel_formula_rank': [pd.NA, 1],
        'screening_selected_for_top_k': [True, False],
        'screening_selection_decision': ['selected_top_k', 'failed_chemical_plausibility'],
        'proposal_shortlist_enabled': [True, True],
        'proposal_shortlist_label': ['family_aware_proposal_shortlist', 'family_aware_proposal_shortlist'],
        'proposal_shortlist_method': ['ranked_family_cap', 'ranked_family_cap'],
        'proposal_shortlist_note': ['demo proposal shortlist note', 'demo proposal shortlist note'],
        'proposal_shortlist_size': [2, 2],
        'proposal_shortlist_family_cap': [1, 1],
        'proposal_shortlist_chemical_plausibility_priority': [True, True],
        'proposal_shortlist_family_count_before_selection': [0, 0],
        'proposal_shortlist_selected': [True, False],
        'proposal_shortlist_rank': [1, pd.NA],
        'proposal_shortlist_decision': [
            'selected_for_proposal_shortlist',
            'not_selected_failed_chemical_plausibility',
        ],
        'extrapolation_shortlist_enabled': [True, True],
        'extrapolation_shortlist_label': [
            'formula_level_extrapolation_shortlist',
            'formula_level_extrapolation_shortlist',
        ],
        'extrapolation_shortlist_method': [
            'novelty_bucket_ranked_family_cap',
            'novelty_bucket_ranked_family_cap',
        ],
        'extrapolation_shortlist_note': [
            'demo extrapolation shortlist note',
            'demo extrapolation shortlist note',
        ],
        'extrapolation_shortlist_size': [1, 1],
        'extrapolation_shortlist_family_cap': [1, 1],
        'extrapolation_shortlist_chemical_plausibility_priority': [True, True],
        'extrapolation_shortlist_target_novelty_bucket': [
            'formula_level_extrapolation',
            'formula_level_extrapolation',
        ],
        'extrapolation_shortlist_family_count_before_selection': [0, 0],
        'extrapolation_shortlist_selected': [False, False],
        'extrapolation_shortlist_rank': [pd.NA, pd.NA],
        'extrapolation_shortlist_decision': [
            'not_selected_novelty_bucket_mismatch',
            'not_selected_failed_chemical_plausibility',
        ],
    })
    screened_df = pd.DataFrame({
        'formula': ['BN', 'AlBN'],
        'predicted_band_gap': [4.8, 1.2],
        'screening_selected_for_top_k': [True, False],
        'screening_selection_decision': ['selected_top_k', 'failed_chemical_plausibility'],
        'proposal_shortlist_selected': [True, False],
        'proposal_shortlist_rank': [1, pd.NA],
        'proposal_shortlist_decision': [
            'selected_for_proposal_shortlist',
            'not_selected_failed_chemical_plausibility',
        ],
        'extrapolation_shortlist_selected': [False, False],
        'extrapolation_shortlist_rank': [pd.NA, pd.NA],
        'extrapolation_shortlist_decision': [
            'not_selected_novelty_bucket_mismatch',
            'not_selected_failed_chemical_plausibility',
        ],
    })
    benchmark_df = pd.DataFrame({
        'feature_set': ['matminer_composition', 'feature_agnostic_dummy'],
        'model_type': ['linear_regression', 'dummy_mean'],
        'mae': [1.0, 1.4],
    })
    robustness_df = pd.DataFrame({
        'feature_set': ['matminer_composition', 'basic_formula_composition', 'feature_agnostic_dummy'],
        'feature_family': ['composition_only', 'composition_only', 'baseline'],
        'candidate_compatible': [True, True, False],
        'n_features': [138, 5, 138],
        'model_type': ['linear_regression', 'hist_gradient_boosting', 'dummy_mean'],
        'benchmark_role': ['selected_model', 'candidate_model', 'dummy_baseline'],
        'selected_by_validation': [True, False, False],
        'robustness_method': ['group_kfold_by_formula'] * 3,
        'robustness_group_column': ['formula'] * 3,
        'requested_folds': [4, 4, 4],
        'actual_folds': [4, 4, 4],
        'completed_folds': [4, 4, 4],
        'robustness_status': ['ok', 'ok', 'ok'],
        'robustness_note': ['demo grouped robustness note', 'demo grouped robustness note', 'feature-agnostic dummy baseline'],
        'mae_mean': [1.1, 1.3, 1.6],
        'mae_std': [0.1, 0.2, 0.3],
        'rmse_mean': [1.4, 1.7, 2.0],
        'rmse_std': [0.1, 0.2, 0.3],
        'r2_mean': [0.6, 0.4, 0.1],
        'r2_std': [0.05, 0.08, 0.1],
    })
    bn_slice_benchmark_df = pd.DataFrame({
        'feature_set': [
            'matminer_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
            'matminer_composition',
            'matminer_composition',
        ],
        'feature_family': [
            'composition_only',
            'composition_only',
            'structure_aware',
            'composition_only',
            'composition_only',
        ],
        'model_type': [
            'linear_regression',
            'linear_regression',
            'hist_gradient_boosting',
            'dummy_mean',
            'bn_local_knn_mean',
        ],
        'benchmark_role': [
            'selected_model',
            'screening_model',
            'candidate_model',
            'global_dummy_mean_baseline',
            'bn_local_reference_baseline',
        ],
        'benchmark_status': ['ok', 'ok', 'ok', 'ok', 'ok'],
        'bn_slice_method': ['leave_one_bn_formula_out'] * 5,
        'bn_slice_train_scope': [
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'bn_only_reference_formulas',
        ],
        'bn_formula_count': [1, 1, 1, 1, 1],
        'bn_row_count': [1, 1, 1, 1, 1],
        'completed_holds': [3, 3, 3, 3, 3],
        'k_neighbors': [pd.NA, pd.NA, pd.NA, pd.NA, 2],
        'mae': [0.6, 0.6, 0.5, 0.9, 0.8],
        'rmse': [0.7, 0.7, 0.6, 1.0, 0.9],
        'r2': [0.5, 0.5, 0.6, 0.1, 0.2],
    })
    bn_slice_prediction_df = pd.DataFrame({
        'formula': ['BN', 'BN', 'BN', 'BN', 'BN'],
        'benchmark_role': [
            'selected_model',
            'screening_model',
            'candidate_model',
            'global_dummy_mean_baseline',
            'bn_local_reference_baseline',
        ],
        'feature_set': [
            'matminer_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
            'matminer_composition',
            'matminer_composition',
        ],
        'feature_family': [
            'composition_only',
            'composition_only',
            'structure_aware',
            'composition_only',
            'composition_only',
        ],
        'model_type': [
            'linear_regression',
            'linear_regression',
            'hist_gradient_boosting',
            'dummy_mean',
            'bn_local_knn_mean',
        ],
        'selected_by_validation': [True, False, False, False, False],
        'bn_slice_method': ['leave_one_bn_formula_out'] * 5,
        'bn_slice_train_scope': [
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'full_dataset_minus_held_out_bn_formula',
            'bn_only_reference_formulas',
        ],
        'target': [5.0, 5.0, 5.0, 5.0, 5.0],
        'prediction': [4.4, 4.3, 4.5, 4.1, 3.9],
        'absolute_error': [0.6, 0.7, 0.5, 0.9, 1.1],
    })
    bn_centered_candidate_df = pd.DataFrame({
        'formula': ['AlBN', 'BN'],
        'ranking_rank': [1, 2],
        'ranking_score': [1.5, 1.4],
        'ranking_basis': ['composition_only_selected_model_low_support_and_bn_support_and_grouped_robustness_and_bn_band_gap_alignment_and_bn_analog_validation_penalties'] * 2,
        'ranking_note': ['bn-centered alternative note'] * 2,
    })
    bn_centered_screening_selection = {
        'enabled': True,
        'selection_source_artifact': 'bn_slice_benchmark_results.csv',
        'selection_scope': 'bn_slice_candidate_compatible_best',
        'selection_note': 'bn-centered alternative ranking note',
        'ranking_artifact': 'demo_candidate_bn_centered_ranking.csv',
        'feature_set': 'matminer_composition',
        'feature_family': 'composition_only',
        'model_type': 'linear_regression',
        'benchmark_role': 'selected_model',
        'mae': 0.6,
        'rmse': 0.7,
        'r2': 0.5,
        'matches_general_screening_combo': True,
    }
    structure_generation_seed_df = pd.DataFrame({
        'formula': ['BN', 'AlBN'],
        'ranking_rank': [1, 2],
        'ranking_score': [4.8, 1.2],
        'bn_centered_ranking_rank': [2, 1],
        'candidate_family': ['bn_binary_anchor', 'group13_bn_111_family'],
        'candidate_novelty_bucket': ['train_plus_val_rediscovery', 'formula_level_extrapolation'],
        'chemical_plausibility_pass': [True, False],
        'proposal_shortlist_selected': [True, False],
        'proposal_shortlist_rank': [1, None],
        'extrapolation_shortlist_selected': [False, True],
        'extrapolation_shortlist_rank': [None, 1],
        'structure_generation_candidate_priority_reason': ['proposal_shortlist', 'extrapolation_shortlist'],
        'structure_generation_seed_rank': [1, 1],
        'structure_generation_seed_status': ['ok', 'ok'],
        'seed_reference_formula': ['BN', 'B2N'],
        'seed_reference_record_id': ['jid-1', 'jid-2'],
    })
    split_masks = {
        'train': [True],
        'val': [False],
        'test': [False],
        'metadata': {'method': 'group_by_formula'},
    }
    selection_summary = {
        'selected_feature_set': 'matminer_composition',
        'selected_feature_count': 138,
        'selected_model_type': 'linear_regression',
        'selected_feature_family': 'composition_only',
        'used_validation_selection': True,
        'candidate_feature_sets': [
            'basic_formula_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
        ],
        'candidate_model_types': ['linear_regression', 'hist_gradient_boosting'],
        'screening_selection_scope': 'candidate_compatible_formula_only',
        'screening_candidate_feature_sets': [
            'basic_formula_composition',
            'matminer_composition',
        ],
        'screening_selected_feature_set': 'matminer_composition',
        'screening_selected_feature_family': 'composition_only',
        'screening_selected_model_type': 'linear_regression',
        'screening_selected_feature_count': 138,
        'screening_selection_matches_overall': True,
        'screening_selection_note': 'Best overall validation combo is candidate-compatible, so screening reuses it.',
        'feature_set_results': [
            {'feature_set': 'basic_formula_composition', 'status': 'ok', 'candidate_compatible': True},
            {'feature_set': 'matminer_composition', 'status': 'ok', 'candidate_compatible': True},
            {
                'feature_set': 'matminer_composition_plus_structure_summary',
                'status': 'ok',
                'candidate_compatible': False,
            },
        ],
    }
    (
        structure_first_pass_variant_df,
        structure_first_pass_summary_df,
        structure_first_pass_payload,
    ) = build_structure_first_pass_execution_artifacts(
        structure_generation_seed_df,
        cfg=cfg,
        formula_col='formula',
    )
    experiment_summary = build_experiment_summary(
        dataset_df=prediction_df,
        bn_df=bn_df,
        candidate_df=candidate_df,
        split_masks=split_masks,
        selection_summary=selection_summary,
        cfg=cfg,
        robustness_df=robustness_df,
        bn_slice_benchmark_df=bn_slice_benchmark_df,
        bn_slice_prediction_df=bn_slice_prediction_df,
        bn_centered_candidate_df=bn_centered_candidate_df,
        bn_centered_screening_selection=bn_centered_screening_selection,
        structure_generation_seed_df=structure_generation_seed_df,
        structure_first_pass_execution_summary_df=structure_first_pass_summary_df,
        structure_first_pass_execution_payload=structure_first_pass_payload,
    )
    manifest = {'name': 'twod_matpd'}

    save_metrics_and_predictions(
        metrics,
        prediction_df,
        bn_df,
        screened_df,
        benchmark_df,
        robustness_df,
        bn_slice_benchmark_df,
        bn_slice_prediction_df,
        bn_centered_candidate_df,
        structure_generation_seed_df,
        experiment_summary,
        manifest,
        cfg,
        structure_first_pass_execution_variant_df=structure_first_pass_variant_df,
        structure_first_pass_execution_summary_df=structure_first_pass_summary_df,
        structure_first_pass_execution_payload=structure_first_pass_payload,
        include_parity_plot=True,
    )

    assert json.loads((artifact_dir / 'metrics.json').read_text()) == metrics
    assert json.loads((artifact_dir / 'manifest.json').read_text()) == manifest
    assert json.loads((artifact_dir / 'experiment_summary.json').read_text()) == experiment_summary
    artifact_provenance = json.loads(
        (artifact_dir / 'artifact_provenance.json').read_text(encoding='utf-8')
    )
    assert artifact_provenance['schema'] == 'aiforbn.artifact_provenance.v2'
    assert len(artifact_provenance['config_sha256']) == 64
    assert len(artifact_provenance['dataset_manifest_sha256']) == 64
    assert experiment_summary['features']['selected_feature_set'] == 'matminer_composition'
    assert experiment_summary['features']['selected_feature_family'] == 'composition_only'
    assert experiment_summary['feature_model_selection']['selected_model_type'] == 'linear_regression'
    assert experiment_summary['robustness']['enabled'] is True
    assert experiment_summary['robustness']['robustness_artifact'] == 'robustness_results.csv'
    assert experiment_summary['robustness']['method'] == 'group_kfold_by_formula'
    assert experiment_summary['robustness']['group_column'] == 'formula'
    assert experiment_summary['robustness']['requested_folds'] == 4
    assert experiment_summary['robustness']['result_row_count'] == 3
    assert experiment_summary['robustness']['successful_result_rows'] == 3
    assert experiment_summary['robustness']['failed_result_rows'] == 0
    assert experiment_summary['robustness']['selected_model_metrics']['mae_mean'] == 1.1
    assert experiment_summary['robustness']['screening_model_metrics']['model_type'] == 'linear_regression'
    assert experiment_summary['robustness']['dummy_baseline_metrics']['model_type'] == 'dummy_mean'
    assert experiment_summary['bn_slice_benchmark']['enabled'] is True
    assert experiment_summary['bn_slice_benchmark']['benchmark_artifact'] == 'bn_slice_benchmark_results.csv'
    assert experiment_summary['bn_slice_benchmark']['prediction_artifact'] == 'bn_slice_predictions.csv'
    assert experiment_summary['bn_slice_benchmark']['method'] == 'leave_one_bn_formula_out'
    assert experiment_summary['bn_slice_benchmark']['k_neighbors'] == 2
    assert experiment_summary['bn_slice_benchmark']['standard_split_bn_train_rows'] == 1
    assert experiment_summary['bn_slice_benchmark']['standard_split_bn_val_rows'] == 0
    assert experiment_summary['bn_slice_benchmark']['standard_split_bn_test_rows'] == 0
    assert experiment_summary['bn_slice_benchmark']['standard_split_has_bn_eval_rows'] is False
    assert experiment_summary['bn_slice_benchmark']['selected_model_metrics']['mae'] == 0.6
    assert experiment_summary['bn_slice_benchmark']['screening_model_metrics']['benchmark_role'] == 'screening_model'
    assert experiment_summary['bn_slice_benchmark']['bn_local_reference_metrics']['model_type'] == 'bn_local_knn_mean'
    assert experiment_summary['bn_slice_benchmark']['global_dummy_baseline_metrics']['model_type'] == 'dummy_mean'
    assert experiment_summary['bn_slice_benchmark']['best_candidate_model_metrics']['benchmark_role'] == 'candidate_model'
    assert experiment_summary['bn_slice_benchmark']['best_candidate_model_metrics']['feature_set'] == (
        'matminer_composition_plus_structure_summary'
    )
    assert experiment_summary['bn_slice_benchmark']['selected_model_beats_global_dummy'] is True
    assert experiment_summary['bn_slice_benchmark']['screening_model_beats_global_dummy'] is True
    assert experiment_summary['bn_slice_benchmark']['best_candidate_model_beats_global_dummy'] is True
    assert experiment_summary['bn_slice_benchmark']['selected_model_matches_best_candidate'] is False
    assert experiment_summary['bn_slice_benchmark']['model_role_comparison_artifact'] == 'bn_model_role_comparison.csv'
    assert experiment_summary['screening']['ranking_basis'] == (
        'composition_only_mean_band_gap_minus_model_disagreement_low_support_and_bn_support_and_grouped_robustness_and_bn_band_gap_alignment_and_bn_analog_validation_penalties'
    )
    assert experiment_summary['screening']['ranking_feature_family'] == 'composition_only'
    assert experiment_summary['screening']['bn_centered_alternative']['enabled'] is True
    assert (
        experiment_summary['screening']['bn_centered_alternative']['ranking_artifact']
        == 'demo_candidate_bn_centered_ranking.csv'
    )
    assert experiment_summary['screening']['bn_centered_alternative']['ranking_feature_set'] == 'matminer_composition'
    assert experiment_summary['screening']['bn_centered_alternative']['ranking_model_type'] == 'linear_regression'
    assert experiment_summary['screening']['bn_centered_alternative']['bn_slice_mae'] == 0.6
    assert experiment_summary['screening']['bn_centered_alternative']['top_k_overlap_count'] == 2
    assert experiment_summary['screening']['bn_centered_alternative']['top_k_overlap_formulas'] == ['BN', 'AlBN']
    assert experiment_summary['screening']['bn_centered_alternative']['general_top_k_formulas'] == ['BN', 'AlBN']
    assert experiment_summary['screening']['bn_centered_alternative']['bn_centered_top_k_formulas'] == ['AlBN', 'BN']
    assert experiment_summary['screening']['bn_centered_alternative']['mean_absolute_rank_shift'] == 1.0
    assert experiment_summary['screening']['bn_centered_alternative']['max_absolute_rank_shift'] == 1.0
    assert experiment_summary['screening']['bn_centered_alternative']['max_absolute_rank_shift_formula'] == 'BN'
    assert experiment_summary['screening']['structure_generation_bridge']['enabled'] is True
    assert (
        experiment_summary['screening']['structure_generation_bridge']['artifact']
        == 'demo_candidate_structure_generation_seeds.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['handoff_artifact']
        == 'demo_candidate_structure_generation_handoff.json'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['reference_record_payload_artifact']
        == 'demo_candidate_structure_generation_reference_records.json'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['job_plan_artifact']
        == 'demo_candidate_structure_generation_job_plan.json'
    )
    assert experiment_summary['screening']['structure_generation_bridge']['candidate_rows'] == 2
    assert experiment_summary['screening']['structure_generation_bridge']['seed_rows'] == 2
    assert experiment_summary['screening']['structure_generation_bridge']['job_count'] == 2
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_queue_artifact']
        == 'demo_candidate_structure_generation_first_pass_queue.json'
    )
    assert experiment_summary['screening']['structure_generation_bridge']['first_pass_queue_size'] == 2
    assert (
        experiment_summary['screening']['structure_generation_bridge']['direct_substitution_job_count']
        == 0
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['simple_relabeling_job_count']
        == 0
    )
    assert experiment_summary['screening']['structure_generation_bridge']['mean_edit_complexity_score'] == 1.5
    assert experiment_summary['screening']['structure_generation_bridge']['max_edit_complexity_score'] == 2.5
    assert experiment_summary['screening']['structure_generation_bridge']['job_action_counts'] == {
        'reference_reuse_control': 1,
        'element_insertion_enumeration': 1,
    }
    assert (
        experiment_summary['screening']['structure_generation_bridge']['followup_shortlist_artifact']
        == 'demo_candidate_structure_generation_followup_shortlist.csv'
    )
    assert experiment_summary['screening']['structure_generation_bridge']['followup_shortlist_size'] == 2
    assert experiment_summary['screening']['structure_generation_bridge']['followup_shortlist_formulas'] == ['BN', 'AlBN']
    assert experiment_summary['screening']['structure_generation_bridge']['followup_readiness_counts'] == {
        'reference_reuse_control_available': 1,
        'moderate_formula_edit_required': 1,
    }
    assert (
        experiment_summary['screening']['structure_generation_bridge'][
            'followup_extrapolation_shortlist_artifact'
        ]
        == 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['followup_extrapolation_shortlist_size']
        == 1
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['followup_extrapolation_shortlist_formulas']
        == ['AlBN']
    )
    assert experiment_summary['screening']['structure_generation_bridge']['unique_seed_reference_formulas'] == 2
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_artifact']
        == 'demo_candidate_structure_generation_first_pass_execution.json'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_summary_artifact']
        == 'demo_candidate_structure_generation_first_pass_execution_summary.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_variants_artifact']
        == 'demo_candidate_structure_generation_first_pass_execution_variants.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_structure_dir']
        == 'demo_candidate_structure_generation_first_pass_structures'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_method']
        == 'deterministic_unrelaxed_reference_reuse_species_edit'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_candidate_count']
        == 2
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_variant_count']
        == 3
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_successful_variant_count']
        == 3
    )
    assert experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_status_counts'] == {
        'executed': 2,
    }
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_executed_formulas']
        == ['BN', 'AlBN']
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_model_feature_set']
        is None
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_model_type']
        is None
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_model_available']
        is False
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_followup_report_artifact']
        == 'demo_candidate_structure_followup_report.csv'
    )
    assert experiment_summary['screening']['ranking_matches_best_overall_evaluation'] is True
    assert experiment_summary['screening']['best_overall_evaluation_feature_set'] == 'matminer_composition'
    assert experiment_summary['screening']['ranking_uncertainty_method'] == 'small_feature_model_disagreement'
    assert experiment_summary['screening']['domain_support_enabled'] is True
    assert experiment_summary['screening']['domain_support_method'] == 'train_plus_val_knn_feature_space_support'
    assert experiment_summary['screening']['domain_support_reference_formula_count'] == 12
    assert experiment_summary['screening']['domain_support_penalty_enabled'] is True
    assert experiment_summary['screening']['domain_support_penalized_rows'] == 1
    assert experiment_summary['screening']['domain_support_low_support_rows'] == 1
    assert experiment_summary['screening']['bn_support_enabled'] is True
    assert experiment_summary['screening']['bn_support_method'] == 'train_plus_val_bn_knn_feature_space_support'
    assert experiment_summary['screening']['bn_support_reference_formula_count'] == 4
    assert experiment_summary['screening']['bn_support_penalty_enabled'] is True
    assert experiment_summary['screening']['bn_support_penalized_rows'] == 1
    assert experiment_summary['screening']['bn_support_low_support_rows'] == 1
    assert experiment_summary['screening']['grouped_robustness_uncertainty_enabled'] is True
    assert (
        experiment_summary['screening']['grouped_robustness_uncertainty_method']
        == 'selected_formula_only_group_kfold_candidate_prediction_std'
    )
    assert experiment_summary['screening']['grouped_robustness_penalty_enabled'] is True
    assert experiment_summary['screening']['grouped_robustness_penalty_active'] is True
    assert experiment_summary['screening']['grouped_robustness_penalty_weight'] == 0.15
    assert experiment_summary['screening']['grouped_robustness_prediction_fold_count'] == 4
    assert experiment_summary['screening']['grouped_robustness_prediction_std_mean'] == 0.16
    assert experiment_summary['screening']['grouped_robustness_penalized_rows'] == 2
    assert experiment_summary['screening']['bn_analog_evidence_enabled'] is True
    assert experiment_summary['screening']['bn_analog_reference_formula_count'] == 4
    assert experiment_summary['screening']['bn_analog_reference_band_gap_median'] == 3.6
    assert experiment_summary['screening']['bn_analog_reference_band_gap_iqr'] == 1.2
    assert experiment_summary['screening']['bn_analog_reference_exfoliation_energy_median'] == 0.07
    assert experiment_summary['screening']['bn_analog_reference_energy_per_atom_median'] == -8.0
    assert experiment_summary['screening']['bn_analog_reference_abs_total_magnetization_median'] == 0.0
    assert experiment_summary['screening']['bn_analog_exfoliation_available_rows'] == 2
    assert experiment_summary['screening']['bn_analog_lower_or_equal_reference_rows'] == 2
    assert experiment_summary['screening']['bn_analog_higher_reference_rows'] == 0
    assert experiment_summary['screening']['bn_band_gap_alignment_enabled'] is True
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_method']
        == 'predicted_band_gap_vs_local_bn_analog_window'
    )
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_reference_split']
        == 'train_plus_val_bn_unique_formulas'
    )
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_window_expansion_iqr_factor']
        == 0.5
    )
    assert (
        experiment_summary['screening']['bn_band_gap_alignment_minimum_neighbor_formula_count_for_penalty']
        == 2
    )
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_enabled'] is True
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_active'] is True
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_weight'] == 0.08
    assert experiment_summary['screening']['bn_band_gap_alignment_penalty_eligible_rows'] == 1
    assert experiment_summary['screening']['bn_band_gap_alignment_within_window_rows'] == 1
    assert experiment_summary['screening']['bn_band_gap_alignment_below_window_rows'] == 0
    assert experiment_summary['screening']['bn_band_gap_alignment_above_window_rows'] == 1
    assert experiment_summary['screening']['bn_band_gap_alignment_penalized_rows'] == 1
    assert experiment_summary['screening']['bn_analog_reference_like_rows'] == 1
    assert experiment_summary['screening']['bn_analog_mixed_alignment_rows'] == 1
    assert experiment_summary['screening']['bn_analog_reference_divergent_rows'] == 0
    assert experiment_summary['screening']['bn_analog_validation_enabled'] is True
    assert experiment_summary['screening']['bn_analog_validation_method'] == 'bn_analog_alignment_vote_fraction'
    assert experiment_summary['screening']['bn_analog_validation_penalty_enabled'] is True
    assert experiment_summary['screening']['bn_analog_validation_penalty_active'] is True
    assert experiment_summary['screening']['bn_analog_validation_penalty_weight'] == 0.12
    assert experiment_summary['screening']['bn_analog_validation_penalized_rows'] == 1
    assert experiment_summary['screening']['chemical_plausibility_enabled'] is True
    assert experiment_summary['screening']['chemical_plausibility_passed_rows'] == 1
    assert experiment_summary['screening']['chemical_plausibility_failed_rows'] == 1
    assert experiment_summary['screening']['candidate_generation_strategy'] == 'bn_anchored_formula_family_grid'
    assert experiment_summary['screening']['candidate_family_counts'] == {
        'bn_binary_anchor': 1,
        'group13_bn_111_family': 1,
    }
    assert experiment_summary['screening']['proposal_shortlist_enabled'] is True
    assert experiment_summary['screening']['proposal_shortlist_artifact'] == (
        'demo_candidate_proposal_shortlist.csv'
    )
    assert experiment_summary['screening']['proposal_shortlist_label'] == (
        'family_aware_proposal_shortlist'
    )
    assert experiment_summary['screening']['proposal_shortlist_method'] == 'ranked_family_cap'
    assert experiment_summary['screening']['proposal_shortlist_note'] == 'demo proposal shortlist note'
    assert experiment_summary['screening']['proposal_shortlist_size'] == 2
    assert experiment_summary['screening']['proposal_shortlist_family_cap'] == 1
    assert experiment_summary['screening']['proposal_shortlist_selected_rows'] == 1
    assert experiment_summary['screening']['proposal_shortlist_selected_family_counts'] == {
        'bn_binary_anchor': 1,
    }
    assert experiment_summary['screening']['proposal_shortlist_novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 1,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 0,
    }
    assert experiment_summary['screening']['proposal_shortlist_formulas'] == [
        {
            'formula': 'BN',
            'proposal_shortlist_rank': 1,
            'ranking_rank': 1,
            'candidate_family': 'bn_binary_anchor',
            'ranking_score': 4.8,
        }
    ]
    assert experiment_summary['screening']['extrapolation_shortlist_enabled'] is True
    assert experiment_summary['screening']['extrapolation_shortlist_artifact'] == (
        'demo_candidate_extrapolation_shortlist.csv'
    )
    assert (
        experiment_summary['screening']['extrapolation_shortlist_label']
        == 'formula_level_extrapolation_shortlist'
    )
    assert (
        experiment_summary['screening']['extrapolation_shortlist_method']
        == 'novelty_bucket_ranked_family_cap'
    )
    assert (
        experiment_summary['screening']['extrapolation_shortlist_note']
        == 'demo extrapolation shortlist note'
    )
    assert experiment_summary['screening']['extrapolation_shortlist_size'] == 1
    assert experiment_summary['screening']['extrapolation_shortlist_family_cap'] == 1
    assert (
        experiment_summary['screening']['extrapolation_shortlist_target_novelty_bucket']
        == 'formula_level_extrapolation'
    )
    assert experiment_summary['screening']['extrapolation_shortlist_candidate_count'] == 1
    assert experiment_summary['screening']['extrapolation_shortlist_selected_rows'] == 0
    assert experiment_summary['screening']['extrapolation_shortlist_selected_family_counts'] == {}
    assert experiment_summary['screening']['extrapolation_shortlist_novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 0,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 0,
    }
    assert experiment_summary['screening']['extrapolation_shortlist_formulas'] == []
    assert experiment_summary['screening']['chemical_plausibility_failed_formulas'] == ['AlBN']
    assert experiment_summary['screening']['novelty_annotation_enabled'] is True
    assert experiment_summary['screening']['novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 1,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 1,
    }
    assert experiment_summary['screening']['standard_top_k_novelty_bucket_counts'] == {
        'train_plus_val_rediscovery': 1,
        'held_out_known_formula': 0,
        'formula_level_extrapolation': 0,
    }
    assert experiment_summary['screening']['formula_level_extrapolation_candidate_count'] == 1
    assert experiment_summary['screening']['formula_level_extrapolation_shortlist'] == [
        {
            'formula': 'AlBN',
            'ranking_rank': 2,
            'novel_formula_rank': 1,
            'ranking_score': 1.2,
            'chemical_plausibility_pass': False,
            'screening_selected_for_top_k': False,
            'screening_selection_decision': 'failed_chemical_plausibility',
            'extrapolation_shortlist_selected': False,
            'extrapolation_shortlist_decision': 'not_selected_failed_chemical_plausibility',
        }
    ]
    assert 'novelty should be interpreted separately' in (
        experiment_summary['screening']['novelty_interpretation_note']
    )
    assert 'Novelty is tracked only at the formula level' in experiment_summary['screening']['ranking_note']
    assert 'known BN slice' in experiment_summary['screening']['ranking_note']
    assert 'BN-local analog band-gap window' in experiment_summary['screening']['ranking_note']
    assert 'observed-property evidence from nearby BN-containing train+val formulas' in experiment_summary['screening']['ranking_note']
    assert 'BN analog-validation penalty' in experiment_summary['screening']['ranking_note']
    assert experiment_summary['bn_slice_benchmark']['candidate_compatible_evaluation_artifact'] == (
        'bn_candidate_compatible_evaluation.csv'
    )
    assert experiment_summary['bn_slice_benchmark']['candidate_compatible_result_row_count'] == 4
    assert experiment_summary['screening']['ranking_stability']['enabled'] is True
    assert experiment_summary['screening']['ranking_stability']['artifact'] == (
        'demo_candidate_ranking_uncertainty.csv'
    )
    assert experiment_summary['screening']['ranking_stability']['top_k_values'] == [3, 5, 10]
    assert experiment_summary['screening']['ranking_stability']['bn_centered_comparison_summary_artifact'] == (
        'demo_candidate_rank_stability_summary.csv'
    )
    assert experiment_summary['screening']['ranking_stability']['bn_centered_comparison_summary_top_k_values'] == [3, 5, 10, 20]
    assert experiment_summary['screening']['decision_policy']['enabled'] is True
    assert experiment_summary['screening']['decision_policy']['artifact'] == (
        'demo_candidate_ranking_uncertainty.csv'
    )
    assert experiment_summary['screening']['decision_policy']['application_tracks'][0] == {
        'label': 'uv_wide_band_gap',
        'target_window_eV': [4.5, 6.5],
        'note': (
            'Formula-stage proxy for the UV/wide-band-gap track. Direct-gap '
            'evidence is unavailable until structure-resolved follow-up.'
        ),
    }
    assert experiment_summary['screening']['decision_policy']['application_tracks'][1][
        'label'
    ] == 'dielectric_2d_support'
    assert experiment_summary['screening']['decision_policy']['abstained_candidate_count'] >= 0
    assert experiment_summary['screening']['candidate_annotations'] == [
        'candidate_family',
        'candidate_template',
        'candidate_family_note',
        'domain_support_reference_formula_count',
        'domain_support_k_neighbors',
        'domain_support_nearest_formula',
        'domain_support_nearest_distance',
        'domain_support_mean_k_distance',
        'domain_support_percentile',
        'domain_support_penalty',
        'bn_support_reference_formula_count',
        'bn_support_k_neighbors',
        'bn_support_nearest_formula',
        'bn_support_neighbor_formulas',
        'bn_support_neighbor_formula_count',
        'bn_support_nearest_distance',
        'bn_support_mean_k_distance',
        'bn_support_percentile',
        'bn_support_penalty',
        'bn_analog_nearest_formula',
        'bn_analog_neighbor_formulas',
        'bn_analog_neighbor_formula_count',
        'bn_analog_reference_band_gap_median',
        'bn_analog_reference_band_gap_iqr',
        'bn_analog_nearest_band_gap',
        'bn_analog_nearest_energy_per_atom',
        'bn_analog_nearest_exfoliation_energy_per_atom',
        'bn_analog_nearest_abs_total_magnetization',
        'bn_analog_neighbor_band_gap_mean',
        'bn_analog_neighbor_band_gap_min',
        'bn_analog_neighbor_band_gap_max',
        'bn_analog_neighbor_band_gap_std',
        'bn_analog_neighbor_energy_per_atom_mean',
        'bn_analog_neighbor_exfoliation_energy_per_atom_mean',
        'bn_analog_neighbor_abs_total_magnetization_mean',
        'bn_analog_neighbor_exfoliation_available_formula_count',
        'bn_band_gap_alignment_neighbor_available_formula_count',
        'bn_band_gap_alignment_window_lower',
        'bn_band_gap_alignment_window_upper',
        'bn_band_gap_alignment_distance_to_window',
        'bn_band_gap_alignment_relative_distance',
        'bn_band_gap_alignment_penalty_eligible',
        'bn_band_gap_alignment_label',
        'bn_band_gap_alignment_penalty',
        'bn_analog_exfoliation_support_label',
        'bn_analog_energy_support_label',
        'bn_analog_abs_total_magnetization_support_label',
        'bn_analog_support_vote_count',
        'bn_analog_support_available_metric_count',
        'bn_analog_validation_label',
        'bn_analog_validation_support_fraction',
        'bn_analog_validation_penalty',
        'chemical_plausibility_pass',
        'chemical_plausibility_guess_count',
        'chemical_plausibility_primary_oxidation_state_guess',
        'chemical_plausibility_note',
        'seen_in_dataset',
        'dataset_formula_row_count',
        'seen_in_train_plus_val',
        'train_plus_val_formula_row_count',
        'candidate_is_seen_in_dataset',
        'candidate_is_seen_in_train_plus_val',
        'candidate_is_formula_level_extrapolation',
        'candidate_novelty_bucket',
        'candidate_novelty_priority',
        'candidate_novelty_note',
        'novelty_rank_within_bucket',
        'novel_formula_rank',
        'screening_selected_for_top_k',
        'screening_selection_decision',
        'objective_name',
        'objective_target_property',
        'objective_target_direction',
        'objective_decision_unit',
        'objective_decision_consequence',
        'objective_note',
        'ranking_signal_property',
        'ranking_signal_direction',
        'ranking_signal_source',
        'ranking_signal_value',
        'ranking_signal_rank',
        'ranking_signal_selected_for_top_k',
        'ranking_uncertainty_penalty_component',
        'ranking_total_penalty',
        'ranking_score_formula',
        'ranking_active_penalty_terms',
        'ranking_main_penalty_driver',
        'ranking_penalty_rank_shift',
        'ranking_penalty_impact_label',
        'ranking_decision_summary',
        'proposal_shortlist_family_count_before_selection',
        'proposal_shortlist_selected',
        'proposal_shortlist_rank',
        'proposal_shortlist_decision',
        'extrapolation_shortlist_target_novelty_bucket',
        'extrapolation_shortlist_family_count_before_selection',
        'extrapolation_shortlist_selected',
        'extrapolation_shortlist_rank',
        'extrapolation_shortlist_decision',
        'ranking_source_count',
        'predicted_band_gap_mean',
        'predicted_band_gap_std',
        'predicted_band_gap_interval_lower',
        'predicted_band_gap_interval_upper',
        'rank_mean',
        'rank_std',
        'rank_min',
        'rank_max',
        'top_3_selection_frequency',
        'top_5_selection_frequency',
        'top_10_selection_frequency',
        'bn_centered_ranking_rank',
        'structure_followup_priority_score',
        'structure_followup_best_queue_rank',
        'structure_followup_best_action_label',
        'structure_followup_readiness_label',
        'structure_followup_shortlist_selected',
        'structure_followup_shortlist_rank',
        'abstain_flag',
        'reason_for_abstention',
        'final_action_label',
        'recommended_action_label',
        'application_track_primary',
        'application_track_secondary',
        'application_track_target_window_eV',
        'application_track_note',
    ]
    assert (artifact_dir / 'predictions.csv').exists()
    assert (artifact_dir / 'bn_slice.csv').exists()
    assert (artifact_dir / 'demo_candidate_ranking.csv').exists()
    assert (artifact_dir / 'demo_candidate_bn_centered_ranking.csv').exists()
    assert (artifact_dir / 'bn_candidate_compatible_evaluation.csv').exists()
    bn_candidate_compatible_df = pd.read_csv(artifact_dir / 'bn_candidate_compatible_evaluation.csv')
    assert 'family_holdout_mae' in bn_candidate_compatible_df.columns
    assert 'grouped_bn_to_non_bn_mae_ratio' in bn_candidate_compatible_df.columns
    assert (artifact_dir / 'bn_model_role_comparison.csv').exists()
    bn_model_role_comparison_df = pd.read_csv(artifact_dir / 'bn_model_role_comparison.csv')
    assert {
        'benchmark_role',
        'feature_set',
        'feature_family',
        'model_type',
        'candidate_compatible',
        'selected_by_validation',
        'bn_slice_mae',
        'bn_slice_r2',
        'bn_family_mae',
        'bn_family_r2',
        'bn_mae',
        'non_bn_mae',
        'bn_to_non_bn_mae_ratio',
    }.issubset(set(bn_model_role_comparison_df.columns))
    expected_roles = {
        'selected_model',
        'screening_model',
        'candidate_model',
        'global_dummy_mean_baseline',
        'bn_local_reference_baseline',
    }
    assert set(bn_model_role_comparison_df['benchmark_role']) == expected_roles
    assert len(bn_model_role_comparison_df) == len(expected_roles)
    assert (
        experiment_summary['bn_slice_benchmark']['model_role_comparison_row_count']
        == len(bn_model_role_comparison_df)
    )
    assert (artifact_dir / 'demo_candidate_rank_stability_summary.csv').exists()
    demo_candidate_rank_stability_summary_df = pd.read_csv(
        artifact_dir / 'demo_candidate_rank_stability_summary.csv'
    )
    assert len(demo_candidate_rank_stability_summary_df) == 4
    assert sorted(demo_candidate_rank_stability_summary_df['top_k'].astype(int).tolist()) == [3, 5, 10, 20]
    assert 'top_k_overlap_count' in demo_candidate_rank_stability_summary_df.columns
    assert (
        experiment_summary['screening']['ranking_stability']['bn_centered_comparison_summary_row_count']
        == len(demo_candidate_rank_stability_summary_df)
    )
    assert (artifact_dir / 'demo_candidate_structure_followup_report.csv').exists()
    followup_report_df = pd.read_csv(artifact_dir / 'demo_candidate_structure_followup_report.csv')
    assert {
        'formula',
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
    }.issubset(set(followup_report_df.columns))
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_followup_report_artifact']
        == 'demo_candidate_structure_followup_report.csv'
    )
    assert (
        experiment_summary['screening']['structure_generation_bridge']['first_pass_execution_followup_report_row_count']
        == len(followup_report_df)
    )
    assert (artifact_dir / 'demo_candidate_ranking_uncertainty.csv').exists()
    uncertainty_df = pd.read_csv(artifact_dir / 'demo_candidate_ranking_uncertainty.csv')
    assert {'control', 'priority', 'explore', 'hold'}.issuperset(
        set(uncertainty_df['final_action_label'].dropna().unique().tolist())
    )
    assert {'control', 'priority', 'explore', 'hold'}.issuperset(
        set(uncertainty_df['recommended_action_label'].dropna().unique().tolist())
    )
    assert uncertainty_df['final_action_label'].equals(
        uncertainty_df['recommended_action_label']
    )
    assert uncertainty_df['application_track_target_window_eV'].dropna().isin(
        {'4.5-6.5', '4.5-8'}
    ).all()
    assert {
        'application_track_primary',
        'application_track_secondary',
        'application_track_target_window_eV',
        'application_track_note',
        'recommended_action_label',
    }.issubset(set(uncertainty_df.columns))
    assert (artifact_dir / 'demo_candidate_structure_generation_seeds.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_handoff.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_reference_records.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_job_plan.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution.json').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_summary.csv').exists()
    assert (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_variants.csv').exists()
    handoff_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_handoff.json').read_text()
    )
    reference_record_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_reference_records.json').read_text()
    )
    job_plan_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_job_plan.json').read_text()
    )
    first_pass_queue_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_first_pass_queue.json').read_text()
    )
    first_pass_execution_payload = json.loads(
        (artifact_dir / 'demo_candidate_structure_generation_first_pass_execution.json').read_text()
    )
    followup_shortlist_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_followup_shortlist.csv'
    )
    followup_extrapolation_shortlist_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_followup_extrapolation_shortlist.csv'
    )
    first_pass_execution_summary_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_summary.csv'
    )
    first_pass_execution_variant_df = pd.read_csv(
        artifact_dir / 'demo_candidate_structure_generation_first_pass_execution_variants.csv'
    )
    assert handoff_payload['candidate_count'] == 2
    assert handoff_payload['seed_row_count'] == 2
    assert handoff_payload['candidates'][0]['formula'] == 'BN'
    assert handoff_payload['candidates'][0]['seeds'][0]['seed_reference_record_id'] == 'jid-1'
    assert handoff_payload['candidates'][0]['seeds'][0]['seed_formula_edit_strategy'] == (
        'same_reduced_formula_reference'
    )
    assert handoff_payload['candidates'][1]['seeds'][0]['seed_formula_candidate_only_elements'] == 'Al'
    assert reference_record_payload['record_count'] == 2
    assert reference_record_payload['reference_records'][0]['record_id'] == 'jid-1'
    assert reference_record_payload['reference_records'][0]['atoms']['elements'] == ['B', 'N']
    assert job_plan_payload['job_count'] == 2
    assert job_plan_payload['direct_substitution_job_count'] == 0
    assert job_plan_payload['simple_relabeling_job_count'] == 0
    assert job_plan_payload['job_action_counts'] == {
        'reference_reuse_control': 1,
        'element_insertion_enumeration': 1,
    }
    assert first_pass_execution_payload['candidate_count'] == 2
    assert first_pass_execution_payload['variant_count'] == 3
    assert first_pass_execution_payload['successful_variant_count'] == 3
    assert first_pass_execution_payload['status_counts'] == {'executed': 2}
    assert first_pass_execution_payload['executed_formulas'] == ['BN', 'AlBN']
    assert first_pass_execution_payload['model_available'] is False
    assert first_pass_execution_summary_df['first_pass_execution_status'].tolist() == [
        'executed',
        'executed',
    ]
    assert first_pass_execution_summary_df['first_pass_execution_selected_final_status'].tolist() == [
        'reference_control_ready',
        'geometry_sanity_failed',
    ]
    assert set(first_pass_execution_variant_df['formula']) == {'BN', 'AlBN'}
    assert first_pass_execution_variant_df['execution_status'].eq('ok').all()
    assert first_pass_execution_variant_df.groupby('formula')['geometry_sanity_pass'].agg(lambda values: list(values)).to_dict() == {
        'AlBN': [False, False],
        'BN': [True],
    }
    structure_dir = artifact_dir / 'demo_candidate_structure_generation_first_pass_structures'
    assert structure_dir.exists()
    assert len(list(structure_dir.glob('*.cif'))) == len(first_pass_execution_variant_df)
    assert job_plan_payload['candidates'][0]['jobs'][0]['job_action_label'] == 'reference_reuse_control'
    assert job_plan_payload['candidates'][0]['jobs'][0]['candidate_formula_element_counts'] == {
        'B': 1,
        'N': 1,
    }
    assert job_plan_payload['candidates'][1]['jobs'][0]['workflow_steps'][0] == 'load_reference_atoms'
    assert job_plan_payload['candidates'][1]['jobs'][0]['simple_element_relabeling_feasible'] is False
    assert job_plan_payload['candidates'][1]['jobs'][0]['element_count_deltas'] == {'Al': 1, 'B': -1}
    assert job_plan_payload['candidates'][1]['jobs'][0]['edit_operations'][0] == {
        'operation': 'increase_element_count',
        'element': 'Al',
        'delta': 1,
    }
    assert job_plan_payload['candidates'][1]['jobs'][0]['reference_record_payload_artifact'] == (
        'demo_candidate_structure_generation_reference_records.json'
    )
    assert first_pass_queue_payload['queue_entry_count'] == 2
    assert first_pass_queue_payload['simple_relabeling_job_count'] == 0
    assert first_pass_queue_payload['queue'][0]['job_id'] == 'bn__seed_1__jid_1'
    assert first_pass_queue_payload['queue'][1]['candidate_first_pass_rank'] == 1
    assert followup_shortlist_df['formula'].tolist() == ['BN', 'AlBN']
    assert followup_shortlist_df['structure_followup_shortlist_rank'].tolist() == [1, 2]
    assert followup_shortlist_df['structure_followup_best_action_label'].tolist() == [
        'reference_reuse_control',
        'element_insertion_enumeration',
    ]
    assert followup_shortlist_df['structure_followup_readiness_label'].tolist() == [
        'reference_reuse_control_available',
        'moderate_formula_edit_required',
    ]
    assert followup_extrapolation_shortlist_df['formula'].tolist() == ['AlBN']
    assert followup_extrapolation_shortlist_df[
        'structure_followup_extrapolation_shortlist_rank'
    ].tolist() == [1]
    assert (artifact_dir / 'demo_candidate_proposal_shortlist.csv').exists()
    assert (artifact_dir / 'demo_candidate_extrapolation_shortlist.csv').exists()
    assert (artifact_dir / 'benchmark_results.csv').exists()
    assert (artifact_dir / 'robustness_results.csv').exists()
    assert (artifact_dir / 'bn_slice_benchmark_results.csv').exists()
    assert (artifact_dir / 'bn_slice_predictions.csv').exists()
    assert (artifact_dir / 'parity_plot.png').exists()


def test_experiment_summary_explains_structure_aware_evaluation_vs_formula_only_screening():
    cfg = {
        'data': {
            'dataset': 'twod_matpd',
            'formula_column': 'formula',
            'target_column': 'band_gap',
        },
        'features': {
            'feature_set': 'basic_formula_composition',
            'candidate_sets': [
                'basic_formula_composition',
                'matminer_composition',
                'matminer_composition_plus_structure_summary',
            ],
            'feature_family': 'mixed_formula_and_structure',
        },
        'model': {
            'type': 'hist_gradient_boosting',
            'candidate_types': ['linear_regression', 'hist_gradient_boosting'],
            'benchmark_baselines': ['dummy_mean'],
        },
        'screening': {
            'candidate_generation_strategy': 'bn_anchored_formula_family_grid',
            'candidate_space_name': 'bn_anchored_formula_family_grid',
            'candidate_space_kind': 'bn_family_demo',
            'candidate_space_note': 'bn-anchored demo note',
            'top_k': 5,
            'use_model_disagreement': True,
            'uncertainty_method': 'small_feature_model_disagreement',
            'uncertainty_penalty': 0.5,
            'grouped_robustness_uncertainty': {
                'enabled': True,
                'method': 'selected_formula_only_group_kfold_candidate_prediction_std',
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.15,
                'note': 'demo grouped candidate robustness note',
            },
            'domain_support': {
                'enabled': True,
                'method': 'train_plus_val_knn_feature_space_support',
                'distance_metric': 'z_scored_euclidean_rms',
                'k_neighbors': 5,
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.15,
                'penalize_below_percentile': 25.0,
                'note': 'demo domain-support note',
            },
            'bn_support': {
                'enabled': True,
                'method': 'train_plus_val_bn_knn_feature_space_support',
                'distance_metric': 'z_scored_euclidean_rms',
                'k_neighbors': 3,
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.1,
                'penalize_below_percentile': 25.0,
                'note': 'demo bn-support note',
            },
            'bn_analog_evidence': {
                'enabled': True,
                'aggregation': 'mean_over_k_nearest_bn_formulas',
                'reference_split': 'train_plus_val_bn_unique_formulas',
                'exfoliation_reference': 'train_plus_val_bn_formula_median',
                'note': 'demo bn-analog evidence note',
            },
            'bn_band_gap_alignment': {
                'enabled': True,
                'method': 'predicted_band_gap_vs_local_bn_analog_window',
                'reference_split': 'train_plus_val_bn_unique_formulas',
                'window_expansion_iqr_factor': 0.5,
                'minimum_neighbor_formula_count_for_penalty': 2,
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.08,
                'note': 'demo bn-local band-gap alignment note',
            },
            'bn_analog_validation': {
                'enabled': True,
                'method': 'bn_analog_alignment_vote_fraction',
                'ranking_penalty_enabled': True,
                'ranking_penalty_weight': 0.12,
                'note': 'demo bn-analog validation note',
            },
            'chemical_plausibility': {
                'enabled': True,
                'method': 'pymatgen_common_oxidation_state_balance',
                'selection_policy': 'annotate_and_prioritize_passing_candidates',
                'note': 'demo plausibility note',
            },
            'proposal_shortlist': {
                'enabled': True,
                'label': 'family_aware_proposal_shortlist',
                'method': 'ranked_family_cap',
                'shortlist_size': 2,
                'max_per_candidate_family': 1,
                'chemical_plausibility_priority': True,
                'note': 'demo proposal shortlist note',
            },
            'extrapolation_shortlist': {
                'enabled': True,
                'label': 'formula_level_extrapolation_shortlist',
                'method': 'novelty_bucket_ranked_family_cap',
                'shortlist_size': 1,
                'max_per_candidate_family': 1,
                'required_novelty_bucket': 'formula_level_extrapolation',
                'chemical_plausibility_priority': True,
                'note': 'demo extrapolation shortlist note',
            },
        },
    }
    selection_summary = {
        'selected_feature_set': 'matminer_composition_plus_structure_summary',
        'selected_feature_family': 'structure_aware',
        'selected_feature_count': 30,
        'selected_model_type': 'hist_gradient_boosting',
        'screening_selection_scope': 'candidate_compatible_formula_only',
        'screening_candidate_feature_sets': ['basic_formula_composition', 'matminer_composition'],
        'screening_selected_feature_set': 'matminer_composition',
        'screening_selected_feature_family': 'composition_only',
        'screening_selected_feature_count': 19,
        'screening_selected_model_type': 'hist_gradient_boosting',
        'screening_selection_matches_overall': False,
        'screening_selection_note': (
            'Best overall validation combo requires structure-derived inputs, so formula-only '
            'candidate screening falls back to the best candidate-compatible validation combo.'
        ),
        'candidate_feature_sets': [
            'basic_formula_composition',
            'matminer_composition',
            'matminer_composition_plus_structure_summary',
        ],
        'candidate_model_types': ['linear_regression', 'hist_gradient_boosting'],
        'feature_set_results': [],
    }

    summary = build_experiment_summary(
        dataset_df=pd.DataFrame({'formula': ['BN'], 'target': [5.0]}),
        bn_df=pd.DataFrame({'formula': ['BN'], 'target': [5.0]}),
        candidate_df=pd.DataFrame({
            'formula': ['BN', 'AlBN'],
            'candidate_space_name': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
            'candidate_space_kind': ['bn_family_demo', 'bn_family_demo'],
            'candidate_generation_strategy': ['bn_anchored_formula_family_grid', 'bn_anchored_formula_family_grid'],
            'candidate_family': ['bn_binary_anchor', 'group13_bn_111_family'],
            'candidate_template': ['B1N1', 'X1B1N1'],
            'candidate_family_note': ['BN anchor', 'Group-III BN ternary extension'],
            'grouped_robustness_prediction_enabled': [True, True],
            'grouped_robustness_prediction_method': [
                'selected_formula_only_group_kfold_candidate_prediction_std',
                'selected_formula_only_group_kfold_candidate_prediction_std',
            ],
            'grouped_robustness_prediction_note': [
                'demo grouped candidate robustness note',
                'demo grouped candidate robustness note',
            ],
            'grouped_robustness_prediction_feature_set': ['matminer_composition', 'matminer_composition'],
            'grouped_robustness_prediction_model_type': ['hist_gradient_boosting', 'hist_gradient_boosting'],
            'grouped_robustness_prediction_fold_count': [4, 4],
            'grouped_robustness_predicted_band_gap_mean': [4.82, 1.24],
            'grouped_robustness_predicted_band_gap_std': [0.02, 0.30],
            'grouped_robustness_uncertainty_penalty': [0.003, 0.045],
            'domain_support_reference_formula_count': [12, 12],
            'domain_support_k_neighbors': [5, 5],
            'domain_support_nearest_formula': ['BN', 'BN'],
            'domain_support_nearest_distance': [0.0, 0.8],
            'domain_support_mean_k_distance': [0.0, 1.1],
            'domain_support_percentile': [100.0, 10.0],
            'domain_support_penalty': [0.0, 0.09],
            'bn_support_reference_formula_count': [4, 4],
            'bn_support_k_neighbors': [3, 3],
            'bn_support_nearest_formula': ['BN', 'BN'],
            'bn_support_neighbor_formulas': ['BN', 'BN|Si2BN'],
            'bn_support_neighbor_formula_count': [1, 2],
            'bn_support_nearest_distance': [0.0, 0.4],
            'bn_support_mean_k_distance': [0.0, 0.6],
            'bn_support_percentile': [100.0, 0.0],
            'bn_support_penalty': [0.0, 0.1],
            'bn_analog_evidence_enabled': [True, True],
            'bn_analog_evidence_aggregation': ['mean_over_k_nearest_bn_formulas', 'mean_over_k_nearest_bn_formulas'],
            'bn_analog_reference_formula_count': [4, 4],
            'bn_analog_reference_band_gap_median': [3.6, 3.6],
            'bn_analog_reference_band_gap_iqr': [1.2, 1.2],
            'bn_analog_reference_exfoliation_energy_median': [0.07, 0.07],
            'bn_analog_reference_energy_per_atom_median': [-8.0, -8.0],
            'bn_analog_reference_abs_total_magnetization_median': [0.0, 0.0],
            'bn_analog_nearest_formula': ['BN', 'BN'],
            'bn_analog_neighbor_formulas': ['BN', 'BN|Si2BN'],
            'bn_analog_neighbor_formula_count': [1, 2],
            'bn_analog_nearest_band_gap': [4.8, 4.8],
            'bn_analog_nearest_energy_per_atom': [-8.3, -8.3],
            'bn_analog_nearest_exfoliation_energy_per_atom': [0.06, 0.06],
            'bn_analog_nearest_abs_total_magnetization': [0.0, 0.0],
            'bn_analog_neighbor_band_gap_mean': [4.8, 2.4],
            'bn_analog_neighbor_band_gap_min': [4.8, 0.0],
            'bn_analog_neighbor_band_gap_max': [4.8, 4.8],
            'bn_analog_neighbor_band_gap_std': [0.0, 2.4],
            'bn_analog_neighbor_energy_per_atom_mean': [-8.3, -7.3],
            'bn_analog_neighbor_exfoliation_energy_per_atom_mean': [0.06, 0.06],
            'bn_analog_neighbor_abs_total_magnetization_mean': [0.0, 0.0],
            'bn_analog_neighbor_exfoliation_available_formula_count': [1, 1],
            'bn_band_gap_alignment_enabled': [True, True],
            'bn_band_gap_alignment_method': [
                'predicted_band_gap_vs_local_bn_analog_window',
                'predicted_band_gap_vs_local_bn_analog_window',
            ],
            'bn_band_gap_alignment_reference_split': [
                'train_plus_val_bn_unique_formulas',
                'train_plus_val_bn_unique_formulas',
            ],
            'bn_band_gap_alignment_note': [
                'demo bn-local band-gap alignment note',
                'demo bn-local band-gap alignment note',
            ],
            'bn_band_gap_alignment_neighbor_available_formula_count': [1, 2],
            'bn_band_gap_alignment_window_lower': [4.2, -0.6],
            'bn_band_gap_alignment_window_upper': [5.4, 5.4],
            'bn_band_gap_alignment_distance_to_window': [0.0, 0.6],
            'bn_band_gap_alignment_relative_distance': [0.0, 0.5],
            'bn_band_gap_alignment_penalty_eligible': [False, True],
            'bn_band_gap_alignment_label': [
                'within_local_bn_analog_band_gap_window',
                'above_local_bn_analog_band_gap_window',
            ],
            'bn_band_gap_alignment_penalty': [0.0, 0.04],
            'bn_analog_exfoliation_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
            'bn_analog_energy_support_label': ['lower_or_equal_bn_reference_median', 'higher_than_bn_reference_median'],
            'bn_analog_abs_total_magnetization_support_label': ['lower_or_equal_bn_reference_median', 'lower_or_equal_bn_reference_median'],
            'bn_analog_support_vote_count': [3, 2],
            'bn_analog_support_available_metric_count': [3, 3],
            'bn_analog_validation_label': ['reference_like_on_available_metrics', 'mixed_reference_alignment'],
            'bn_analog_validation_support_fraction': [1.0, 2.0 / 3.0],
            'bn_analog_validation_penalty': [0.0, 0.04],
            'chemical_plausibility_pass': [True, False],
            'chemical_plausibility_guess_count': [1, 0],
            'chemical_plausibility_primary_oxidation_state_guess': ['B(+3), N(-3)', ''],
            'chemical_plausibility_note': ['pass', 'fail'],
            'proposal_shortlist_enabled': [True, True],
            'proposal_shortlist_label': ['family_aware_proposal_shortlist', 'family_aware_proposal_shortlist'],
            'proposal_shortlist_method': ['ranked_family_cap', 'ranked_family_cap'],
            'proposal_shortlist_note': ['demo proposal shortlist note', 'demo proposal shortlist note'],
            'proposal_shortlist_size': [2, 2],
            'proposal_shortlist_family_cap': [1, 1],
            'proposal_shortlist_chemical_plausibility_priority': [True, True],
            'proposal_shortlist_family_count_before_selection': [0, 0],
            'proposal_shortlist_selected': [True, False],
            'proposal_shortlist_rank': [1, pd.NA],
            'proposal_shortlist_decision': [
                'selected_for_proposal_shortlist',
                'not_selected_failed_chemical_plausibility',
            ],
            'extrapolation_shortlist_enabled': [True, True],
            'extrapolation_shortlist_label': [
                'formula_level_extrapolation_shortlist',
                'formula_level_extrapolation_shortlist',
            ],
            'extrapolation_shortlist_method': [
                'novelty_bucket_ranked_family_cap',
                'novelty_bucket_ranked_family_cap',
            ],
            'extrapolation_shortlist_note': [
                'demo extrapolation shortlist note',
                'demo extrapolation shortlist note',
            ],
            'extrapolation_shortlist_size': [1, 1],
            'extrapolation_shortlist_family_cap': [1, 1],
            'extrapolation_shortlist_chemical_plausibility_priority': [True, True],
            'extrapolation_shortlist_target_novelty_bucket': [
                'formula_level_extrapolation',
                'formula_level_extrapolation',
            ],
            'extrapolation_shortlist_family_count_before_selection': [0, 0],
            'extrapolation_shortlist_selected': [False, False],
            'extrapolation_shortlist_rank': [pd.NA, pd.NA],
            'extrapolation_shortlist_decision': [
                'not_selected_novelty_bucket_mismatch',
                'not_selected_failed_chemical_plausibility',
            ],
        }),
        split_masks={'metadata': {'method': 'group_by_formula'}},
        selection_summary=selection_summary,
        cfg=cfg,
    )

    assert summary['features']['selected_feature_family'] == 'structure_aware'
    assert summary['screening']['ranking_feature_set'] == 'matminer_composition'
    assert summary['screening']['ranking_feature_family'] == 'composition_only'
    assert summary['screening']['ranking_matches_best_overall_evaluation'] is False
    assert summary['screening']['best_overall_evaluation_feature_set'] == (
        'matminer_composition_plus_structure_summary'
    )
    assert 'falls back to the best candidate-compatible combo' in summary['screening']['ranking_note']
    assert 'grouped-fold candidate robustness penalty' in summary['screening']['ranking_note']
    assert 'train+val feature-space domain-support layer' in summary['screening']['ranking_note']
    assert 'known BN slice' in summary['screening']['ranking_note']
    assert 'BN-local analog band-gap window' in summary['screening']['ranking_note']
    assert 'observed-property evidence from nearby BN-containing train+val formulas' in summary['screening']['ranking_note']
    assert 'BN analog-validation penalty' in summary['screening']['ranking_note']
    assert 'lightweight pymatgen oxidation-state plausibility screen' in summary['screening']['ranking_note']
    assert summary['screening']['candidate_generation_strategy'] == 'bn_anchored_formula_family_grid'
    assert summary['screening']['candidate_family_counts'] == {
        'bn_binary_anchor': 1,
        'group13_bn_111_family': 1,
    }
    assert summary['screening']['proposal_shortlist_enabled'] is True
    assert summary['screening']['proposal_shortlist_label'] == 'family_aware_proposal_shortlist'
    assert summary['screening']['proposal_shortlist_method'] == 'ranked_family_cap'
    assert summary['screening']['proposal_shortlist_note'] == 'demo proposal shortlist note'
    assert summary['screening']['proposal_shortlist_size'] == 2
    assert summary['screening']['proposal_shortlist_family_cap'] == 1
    assert summary['screening']['proposal_shortlist_selected_rows'] == 1
    assert summary['screening']['proposal_shortlist_selected_family_counts'] == {
        'bn_binary_anchor': 1,
    }
    assert summary['screening']['proposal_shortlist_formulas'] == [
        {
            'formula': 'BN',
            'proposal_shortlist_rank': 1,
            'candidate_family': 'bn_binary_anchor',
        }
    ]
    assert summary['screening']['extrapolation_shortlist_enabled'] is True
    assert summary['screening']['extrapolation_shortlist_label'] == (
        'formula_level_extrapolation_shortlist'
    )
    assert summary['screening']['extrapolation_shortlist_method'] == (
        'novelty_bucket_ranked_family_cap'
    )
    assert summary['screening']['extrapolation_shortlist_note'] == (
        'demo extrapolation shortlist note'
    )
    assert summary['screening']['extrapolation_shortlist_size'] == 1
    assert summary['screening']['extrapolation_shortlist_family_cap'] == 1
    assert summary['screening']['extrapolation_shortlist_target_novelty_bucket'] == (
        'formula_level_extrapolation'
    )
    assert summary['screening']['extrapolation_shortlist_selected_rows'] == 0
    assert summary['screening']['extrapolation_shortlist_selected_family_counts'] == {}
    assert summary['screening']['extrapolation_shortlist_formulas'] == []
    assert summary['screening']['bn_support_reference_formula_count'] == 4
    assert summary['screening']['bn_support_penalized_rows'] == 1
    assert summary['screening']['grouped_robustness_uncertainty_enabled'] is True
    assert summary['screening']['grouped_robustness_penalized_rows'] == 2
    assert summary['screening']['bn_analog_evidence_enabled'] is True
    assert summary['screening']['bn_analog_reference_formula_count'] == 4
    assert summary['screening']['bn_analog_reference_band_gap_median'] == 3.6
    assert summary['screening']['bn_analog_reference_band_gap_iqr'] == 1.2
    assert summary['screening']['bn_analog_reference_energy_per_atom_median'] == -8.0
    assert summary['screening']['bn_analog_reference_abs_total_magnetization_median'] == 0.0
    assert summary['screening']['bn_analog_exfoliation_available_rows'] == 2
    assert summary['screening']['bn_band_gap_alignment_enabled'] is True
    assert summary['screening']['bn_band_gap_alignment_penalty_eligible_rows'] == 1
    assert summary['screening']['bn_band_gap_alignment_within_window_rows'] == 1
    assert summary['screening']['bn_band_gap_alignment_above_window_rows'] == 1
    assert summary['screening']['bn_band_gap_alignment_penalized_rows'] == 1
    assert summary['screening']['bn_analog_reference_like_rows'] == 1
    assert summary['screening']['bn_analog_mixed_alignment_rows'] == 1
    assert summary['screening']['bn_analog_reference_divergent_rows'] == 0
    assert summary['screening']['bn_analog_validation_enabled'] is True
    assert summary['screening']['bn_analog_validation_penalized_rows'] == 1
    assert summary['screening']['chemical_plausibility_failed_formulas'] == ['AlBN']
