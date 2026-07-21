from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from runtime import io_utils
from runtime.io_utils import (
    assess_artifact_provenance,
    build_artifact_provenance,
    clear_project_cache,
    configure_matplotlib_cache,
    ensure_runtime_dirs,
    load_config,
    make_json_safe,
    read_json_file,
    validate_runtime_output_path,
    validate_json_payload,
    write_json_file,
)


def _make_myutils_file_layout(root: Path) -> None:
    file_utils_dir = root / 'file_utils'
    file_utils_dir.mkdir(parents=True)
    (file_utils_dir / 'filesystem.py').write_text('', encoding='utf-8')
    (file_utils_dir / 'json_io.py').write_text('', encoding='utf-8')


@pytest.mark.parametrize('checkout_prefix', [('aiforbn',), ('projects', 'aiforbn')])
def test_find_myutils_root_handles_nested_checkout_layouts(
    tmp_path: Path,
    monkeypatch,
    checkout_prefix,
):
    myutils_root = tmp_path / 'myutils'
    _make_myutils_file_layout(myutils_root)
    source_path = tmp_path.joinpath(*checkout_prefix, 'src', 'runtime', 'io_utils.py')
    monkeypatch.delenv('MYUTILS_ROOT', raising=False)

    assert io_utils._find_myutils_root(source_path) == myutils_root.resolve()


def test_find_myutils_root_honors_explicit_override(tmp_path: Path, monkeypatch):
    myutils_root = tmp_path / 'external-myutils'
    _make_myutils_file_layout(myutils_root)
    monkeypatch.setenv('MYUTILS_ROOT', str(myutils_root))

    assert io_utils._find_myutils_root('/unrelated/checkout/io_utils.py') == myutils_root.resolve()


def test_find_myutils_root_reports_actionable_failure(tmp_path: Path, monkeypatch):
    missing_root = tmp_path / 'missing-myutils'
    monkeypatch.setenv('MYUTILS_ROOT', str(missing_root))

    with pytest.raises(ModuleNotFoundError, match='Set MYUTILS_ROOT') as exc_info:
        io_utils._find_myutils_root('/unrelated/checkout/io_utils.py')

    assert str(missing_root) in str(exc_info.value)


def test_load_config_from_python_module(tmp_path: Path):
    cfg_path = tmp_path / 'temp_config.py'
    cfg_path.write_text("CONFIG = {'project': {'name': 'demo'}, 'value': 7}\n", encoding='utf-8')

    cfg = load_config(cfg_path)

    assert cfg['project']['name'] == 'demo'
    assert cfg['value'] == 7
    assert not (tmp_path / '__pycache__').exists()


def test_load_config_rejects_human_docs_before_execution(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    human_docs_dir = tmp_path / 'human_docs'
    human_docs_dir.mkdir()
    sentinel_path = tmp_path / 'executed.txt'
    cfg_path = human_docs_dir / 'user_config.py'
    cfg_path.write_text(
        'from pathlib import Path\n'
        f'Path({str(sentinel_path)!r}).write_text("executed", encoding="utf-8")\n'
        'CONFIG = {"value": 1}\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)

    with pytest.raises(ValueError, match='user-owned human_docs'):
        load_config(cfg_path)

    assert not sentinel_path.exists()
    assert not (human_docs_dir / '__pycache__').exists()


def test_ensure_runtime_dirs_only_creates_configured_runtime_dirs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
        },
        'project': {
            'artifact_dir': 'artifacts',
        },
    }

    ensure_runtime_dirs(cfg)

    assert (tmp_path / 'data' / 'raw').is_dir()
    assert (tmp_path / 'data' / 'processed').is_dir()
    assert (tmp_path / 'artifacts').is_dir()
    assert not (tmp_path / 'notebooks').exists()
    assert not (tmp_path / 'apps').exists()
    assert not (tmp_path / 'tests').exists()


def test_ensure_runtime_dirs_rejects_human_docs_output(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'human_docs/runtime-cache',
        },
        'project': {
            'artifact_dir': 'artifacts',
        },
    }

    with pytest.raises(ValueError, match='user-owned human_docs'):
        ensure_runtime_dirs(cfg)

    assert not (tmp_path / 'data').exists()
    assert not (tmp_path / 'artifacts').exists()
    assert not (tmp_path / 'human_docs').exists()


def test_ensure_runtime_dirs_rejects_non_directory_targets_before_any_creation(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    processed_file = tmp_path / 'processed-file'
    processed_file.write_text('keep', encoding='utf-8')
    cfg = {
        'data': {
            'raw_dir': 'new-raw',
            'processed_dir': str(processed_file),
        },
        'project': {
            'artifact_dir': 'new-artifacts',
        },
    }

    with pytest.raises(ValueError, match='directory'):
        ensure_runtime_dirs(cfg)

    assert not (tmp_path / 'new-raw').exists()
    assert not (tmp_path / 'new-artifacts').exists()
    assert processed_file.read_text(encoding='utf-8') == 'keep'


@pytest.mark.parametrize(
    'configured_value',
    [
        None,
        '',
        '   ',
        'relative-mpl',
        './nested/../normalized-mpl',
        '~/tilde-mpl',
        'absolute',
    ],
    ids=['unset', 'empty', 'whitespace', 'relative', 'normalized', 'tilde', 'absolute'],
)
def test_configure_matplotlib_cache_exports_one_canonical_idempotent_path(
    tmp_path: Path,
    monkeypatch,
    configured_value,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv('MPLBACKEND', raising=False)
    if configured_value is None or not configured_value.strip():
        if configured_value is None:
            monkeypatch.delenv('MPLCONFIGDIR', raising=False)
        else:
            monkeypatch.setenv('MPLCONFIGDIR', configured_value)
        expected_path = Path('/tmp/ai_for_bn_mplconfig').resolve(strict=False)
    else:
        declared_path = (
            tmp_path / 'absolute-mpl'
            if configured_value == 'absolute'
            else Path(configured_value)
        )
        monkeypatch.setenv('MPLCONFIGDIR', str(declared_path))
        expected_path = declared_path.expanduser().resolve(strict=False)
    existed_before = expected_path.exists()

    first_result = configure_matplotlib_cache()
    second_result = configure_matplotlib_cache()

    assert first_result == expected_path
    assert second_result == expected_path
    assert io_utils.os.environ['MPLCONFIGDIR'] == str(expected_path)
    assert io_utils.os.environ['MPLBACKEND'] == 'Agg'
    if not existed_before:
        assert not expected_path.exists()


def test_empty_matplotlib_cache_env_cannot_place_font_cache_in_cwd(tmp_path: Path):
    project_root = tmp_path / 'synthetic-project'
    project_root.mkdir()
    env = dict(os.environ)
    env.update({
        'AIFORBN_SYNTHETIC_PROJECT_ROOT': str(project_root),
        'MPLCONFIGDIR': '',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
    })
    script = (
        'import os\n'
        'from pathlib import Path\n'
        'from runtime import io_utils\n'
        "root = Path(os.environ['AIFORBN_SYNTHETIC_PROJECT_ROOT'])\n"
        'io_utils.PROJECT_ROOT = root\n'
        'import materials.plots\n'
        'import matplotlib\n'
        "expected = str(Path('/tmp/ai_for_bn_mplconfig').resolve())\n"
        "assert os.environ['MPLCONFIGDIR'] == expected\n"
        'assert matplotlib.get_configdir() == expected\n'
        "assert not list(root.glob('fontlist-*.json'))\n"
    )

    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(project_root.iterdir()) == []


def test_json_helpers_delegate_to_myutils_json_io(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    path = tmp_path / 'payload.json'
    payload = {
        'count': np.int64(2),
        'score': np.float64(1.5),
        'missing': pd.NA,
        'path': tmp_path / 'artifact.csv',
    }

    write_json_file(payload, path)

    assert read_json_file(path) == {
        'count': 2,
        'score': 1.5,
        'missing': None,
        'path': str(tmp_path / 'artifact.csv'),
    }
    assert make_json_safe({'nested': [np.int64(1), pd.NA]}) == {'nested': [1, None]}

    human_docs_path = tmp_path / 'human_docs' / 'runtime-state.json'
    with pytest.raises(ValueError, match='user-owned human_docs'):
        write_json_file({'forbidden': True}, human_docs_path)
    assert not human_docs_path.exists()
    assert validate_runtime_output_path(path) == path.resolve()


def test_artifact_provenance_tracks_source_config_and_dataset_identity(
    tmp_path: Path,
    monkeypatch,
):
    assert io_utils._read_local_source_state(tmp_path) == {
        'revision': None,
        'dirty': None,
    }
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg = {'project': {'artifact_dir': 'artifacts'}, 'value': 7}
    reordered_cfg = {'value': 7, 'project': {'artifact_dir': 'artifacts'}}
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }

    provenance = build_artifact_provenance(
        cfg,
        manifest,
        project_root_path=tmp_path,
    )

    assert provenance['schema'] == 'aiforbn.artifact_provenance.v1'
    assert provenance['source_revision'] == 'abc123'
    assert provenance['source_worktree_dirty'] is False
    assert len(provenance['config_sha256']) == 64
    assert len(provenance['dataset_manifest_sha256']) == 64
    assert build_artifact_provenance(
        reordered_cfg,
        manifest,
        project_root_path=tmp_path,
    )['config_sha256'] == provenance['config_sha256']
    assert assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['status'] == 'current'

    mismatch_cases = (
        ({**provenance, 'source_revision': 'older'}, cfg, manifest, 'source_revision_mismatch'),
        (provenance, {**cfg, 'value': 8}, manifest, 'effective_config_mismatch'),
        (provenance, cfg, {**manifest, 'name': 'other'}, 'dataset_manifest_mismatch'),
    )
    for stored, current_cfg, current_manifest, reason in mismatch_cases:
        assessment = assess_artifact_provenance(
            stored,
            current_cfg,
            current_manifest,
            project_root_path=tmp_path,
        )
        assert assessment == {'status': 'stale', 'reason': reason}

    provenance['source_worktree_dirty'] = True
    unverified = assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert unverified['status'] == 'unverified'
    assert unverified['reason'] == 'artifact_source_worktree_was_dirty'

    provenance['source_worktree_dirty'] = False
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': True},
    )
    assert assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    )['reason'] == 'current_source_worktree_is_dirty'

    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': None, 'dirty': None},
    )
    outside_git = build_artifact_provenance(
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    assert outside_git['source_revision'] is None
    assert outside_git['source_worktree_dirty'] is None
    assert assess_artifact_provenance(
        outside_git,
        cfg,
        manifest,
        project_root_path=tmp_path,
    ) == {
        'status': 'unverified',
        'reason': 'source_revision_unavailable',
    }


@pytest.mark.parametrize(
    ('manifest', 'reason'),
    [
        (None, 'dataset_manifest_missing'),
        ({}, 'dataset_manifest_invalid'),
        ([], 'dataset_manifest_invalid'),
        ('malformed', 'dataset_manifest_invalid'),
        ({'name': 'missing-required-fields'}, 'dataset_manifest_invalid'),
    ],
)
def test_artifact_provenance_never_accepts_missing_or_malformed_dataset_identity(
    tmp_path: Path,
    monkeypatch,
    manifest,
    reason,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg = {'project': {'artifact_dir': 'artifacts'}}
    valid_manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    provenance = build_artifact_provenance(
        cfg,
        manifest,
        project_root_path=tmp_path,
    )
    if manifest is None:
        provenance = build_artifact_provenance(
            cfg,
            valid_manifest,
            project_root_path=tmp_path,
        )

    assert assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    ) == {'status': 'unverified', 'reason': reason}


@pytest.mark.parametrize(
    ('field', 'value', 'remove_field'),
    [
        ('source_worktree_dirty', None, True),
        ('source_revision', '', False),
        ('source_worktree_dirty', 'clean', False),
        ('config_sha256', 'not-a-sha256', False),
        ('dataset_manifest_sha256', None, False),
    ],
)
def test_artifact_provenance_rejects_malformed_marker_fields(
    tmp_path: Path,
    monkeypatch,
    field,
    value,
    remove_field,
):
    monkeypatch.setattr(
        io_utils,
        '_read_local_source_state',
        lambda _root: {'revision': 'abc123', 'dirty': False},
    )
    cfg = {'project': {'artifact_dir': 'artifacts'}}
    manifest = {
        'name': 'twod_matpd',
        'source': 'jarvis-tools/figshare',
        'retrieved_at': '2026-07-21T00:00:00+00:00',
        'target_column': 'band_gap',
    }
    provenance = build_artifact_provenance(cfg, manifest, project_root_path=tmp_path)
    if remove_field:
        provenance.pop(field)
    else:
        provenance[field] = value

    assert assess_artifact_provenance(
        provenance,
        cfg,
        manifest,
        project_root_path=tmp_path,
    ) == {'status': 'unverified', 'reason': 'artifact_provenance_invalid'}


def test_artifact_provenance_source_identity_handles_detached_dirty_and_ignored_state(
    tmp_path: Path,
):
    project_root = tmp_path / 'project'
    (project_root / 'src').mkdir(parents=True)
    (project_root / '.gitignore').write_text(
        '__pycache__/\n.pytest_cache/\n',
        encoding='utf-8',
    )
    (project_root / 'main.py').write_text('VALUE = 1\n', encoding='utf-8')
    (project_root / 'src' / 'module.py').write_text('VALUE = 1\n', encoding='utf-8')
    (project_root / 'requirements.txt').write_text('pandas\n', encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=project_root, check=True)
    subprocess.run(['git', 'add', '.'], cwd=project_root, check=True)
    subprocess.run(
        [
            'git',
            '-c',
            'user.name=Round 10 Test',
            '-c',
            'user.email=round10@example.invalid',
            'commit',
            '-q',
            '-m',
            'fixture',
        ],
        cwd=project_root,
        check=True,
    )

    clean_state = io_utils._read_local_source_state(project_root)
    assert clean_state['revision']
    assert clean_state['dirty'] is False

    cache_path = project_root / 'src' / '__pycache__' / 'module.pyc'
    cache_path.parent.mkdir()
    cache_path.write_bytes(b'ignored')
    assert io_utils._read_local_source_state(project_root) == clean_state

    (project_root / 'untracked-scratch.tmp').write_text('irrelevant', encoding='utf-8')
    assert io_utils._read_local_source_state(project_root) == clean_state

    subprocess.run(['git', 'checkout', '--detach', '-q'], cwd=project_root, check=True)
    assert io_utils._read_local_source_state(project_root) == clean_state

    (project_root / 'src' / 'module.py').write_text('VALUE = 2\n', encoding='utf-8')
    dirty_state = io_utils._read_local_source_state(project_root)
    assert dirty_state['revision'] == clean_state['revision']
    assert dirty_state['dirty'] is True


def test_validate_json_payload_matches_write_serialization_contract():
    assert validate_json_payload({'value': np.int64(2)}) is None

    with pytest.raises(ValueError, match='not JSON-serializable'):
        validate_json_payload({'invalid': object()})


def test_write_json_file_uses_the_guarded_canonical_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    captured_paths = []

    def capture_write(payload, path, *args, **kwargs):
        captured_paths.append(Path(path))

    monkeypatch.setattr(io_utils, '_shared_write_json_file', capture_write)
    declared_path = Path('~/../guarded-output.json')

    write_json_file({'safe': True}, declared_path)

    assert captured_paths == [declared_path.expanduser().resolve(strict=False)]


def test_write_json_file_rejects_unserializable_payload_before_parent_creation(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    output_path = tmp_path / 'new-parent' / 'payload.json'

    with pytest.raises(ValueError, match='not JSON-serializable'):
        write_json_file({'invalid': object()}, output_path)

    assert not output_path.parent.exists()


def test_write_json_file_rejects_a_hardlinked_output_alias(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    human_docs_file = tmp_path / 'human_docs' / 'payload.json'
    human_docs_file.parent.mkdir()
    human_docs_file.write_text('{"user_owned": true}\n', encoding='utf-8')
    output_alias = tmp_path / 'artifacts' / 'payload.json'
    output_alias.parent.mkdir()
    output_alias.hardlink_to(human_docs_file)

    with pytest.raises(ValueError, match='multiple hard links'):
        write_json_file({'overwritten': True}, output_alias)

    assert human_docs_file.read_text(encoding='utf-8') == '{"user_owned": true}\n'


def test_write_json_file_rejects_symbolic_link_and_directory_leaves(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    target = tmp_path / 'allowed-target.json'
    target.write_text('{"keep": true}\n', encoding='utf-8')
    symlink_output = tmp_path / 'artifacts' / 'symlink.json'
    symlink_output.parent.mkdir()
    symlink_output.symlink_to(target)

    with pytest.raises(ValueError, match='symbolic-link'):
        write_json_file({'overwrite': True}, symlink_output)

    dangling_output = tmp_path / 'artifacts' / 'dangling.json'
    dangling_output.symlink_to(tmp_path / 'artifacts' / 'missing.json')
    with pytest.raises(ValueError, match='symbolic-link'):
        write_json_file({'overwrite': True}, dangling_output)

    directory_output = tmp_path / 'artifacts' / 'directory.json'
    directory_output.mkdir()
    with pytest.raises(ValueError, match='regular-file'):
        write_json_file({'overwrite': True}, directory_output)

    assert target.read_text(encoding='utf-8') == '{"keep": true}\n'


def test_runtime_output_guard_enforces_configured_root_and_concrete_file_leaf(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    output_root = tmp_path / 'artifacts'
    output_root.mkdir()
    normal_output = output_root / 'normal.json'

    assert validate_runtime_output_path(
        normal_output,
        required_parent_path=output_root,
        expected_output_kind='file',
    ) == normal_output.resolve()

    external_target = tmp_path / 'outside' / 'external.json'
    external_alias = output_root / 'external.json'
    external_alias.symlink_to(external_target)
    with pytest.raises(ValueError, match='configured output root'):
        validate_runtime_output_path(
            external_alias,
            required_parent_path=output_root,
            expected_output_kind='file',
        )

    internal_target = output_root / 'other.json'
    internal_alias = output_root / 'internal.json'
    internal_alias.symlink_to(internal_target)
    with pytest.raises(ValueError, match='symbolic-link'):
        validate_runtime_output_path(
            internal_alias,
            required_parent_path=output_root,
            expected_output_kind='file',
        )

    directory_leaf = output_root / 'directory.json'
    directory_leaf.mkdir()
    with pytest.raises(ValueError, match='regular-file'):
        validate_runtime_output_path(
            directory_leaf,
            required_parent_path=output_root,
            expected_output_kind='file',
        )

    file_parent = output_root / 'file-parent'
    file_parent.write_text('keep', encoding='utf-8')
    with pytest.raises(ValueError, match='parent paths'):
        validate_runtime_output_path(
            file_parent / 'child.json',
            required_parent_path=output_root,
            expected_output_kind='file',
        )


def test_runtime_output_guard_rejects_normalized_and_symlinked_human_docs_paths(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    human_docs_dir = tmp_path / 'human_docs'
    human_docs_dir.mkdir()
    symlinked_parent = tmp_path / 'symlinked-parent'
    symlinked_parent.symlink_to(human_docs_dir, target_is_directory=True)
    human_docs_leaf = human_docs_dir / 'existing.json'
    human_docs_leaf.write_text('user-owned', encoding='utf-8')
    symlinked_leaf = tmp_path / 'symlinked-leaf.json'
    symlinked_leaf.symlink_to(human_docs_leaf)

    forbidden_paths = [
        Path('human_docs'),
        Path('human_docs/nested/output.json'),
        human_docs_dir / 'absolute.json',
        Path('./human_docs/./normalized.json'),
        Path('data/../human_docs/traversal.json'),
        Path('~/human_docs/home-expanded.json'),
        Path('symlinked-parent/existing-child.json'),
        Path('symlinked-parent/not-yet/child.json'),
        Path('symlinked-leaf.json'),
    ]
    for forbidden_path in forbidden_paths:
        with pytest.raises(ValueError, match='user-owned human_docs'):
            validate_runtime_output_path(forbidden_path)

    assert validate_runtime_output_path('artifacts/output.json') == (
        tmp_path / 'artifacts/output.json'
    )
    assert validate_runtime_output_path('human_docs_backup/output.json') == (
        tmp_path / 'human_docs_backup/output.json'
    )


def test_runtime_output_guard_rejects_case_alias_of_human_docs(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    human_docs_dir = tmp_path / 'human_docs'
    human_docs_dir.mkdir()
    case_alias_dir = tmp_path / 'HUMAN_DOCS'
    if not case_alias_dir.exists() or not case_alias_dir.samefile(human_docs_dir):
        pytest.skip('filesystem is case-sensitive')
    output_path = case_alias_dir / 'case-alias.json'

    with pytest.raises(ValueError, match='user-owned human_docs'):
        write_json_file({'forbidden': True}, output_path)

    assert not (human_docs_dir / output_path.name).exists()


def test_ensure_runtime_dirs_does_not_trust_a_deceptive_project_root(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = {
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': str(tmp_path / 'human_docs' / 'processed'),
        },
        'project': {
            'artifact_dir': 'artifacts',
        },
    }

    with pytest.raises(ValueError, match='user-owned human_docs'):
        ensure_runtime_dirs(cfg, project_root_path=tmp_path / 'different-project')

    assert not (tmp_path / 'data').exists()
    assert not (tmp_path / 'artifacts').exists()
    assert not (tmp_path / 'human_docs').exists()


def test_clear_project_cache_uses_myutils_discovery_and_preserves_human_docs(tmp_path: Path):
    pycache_dir = tmp_path / 'pkg' / '__pycache__'
    pycache_dir.mkdir(parents=True)
    (pycache_dir / 'mod.pyc').write_text('x', encoding='utf-8')
    human_docs_cache_dir = tmp_path / 'human_docs' / 'notes' / '__pycache__'
    human_docs_cache_dir.mkdir(parents=True)
    (human_docs_cache_dir / 'user.pyc').write_text('user-owned', encoding='utf-8')
    cache_alias_parent = tmp_path / 'cache-alias'
    cache_alias_parent.mkdir()
    cache_alias = cache_alias_parent / '__pycache__'
    cache_alias.symlink_to(human_docs_cache_dir, target_is_directory=True)
    allowed_alias_target = tmp_path / 'allowed-cache-alias-target'
    allowed_alias_target.mkdir()
    allowed_alias_sentinel = allowed_alias_target / 'keep.pyc'
    allowed_alias_sentinel.write_text('keep', encoding='utf-8')
    allowed_cache_alias = tmp_path / 'allowed-alias' / '__pycache__'
    allowed_cache_alias.parent.mkdir()
    allowed_cache_alias.symlink_to(allowed_alias_target, target_is_directory=True)

    deleted = clear_project_cache(tmp_path)

    assert deleted == [pycache_dir]
    assert not pycache_dir.exists()
    assert human_docs_cache_dir.exists()
    assert cache_alias.exists()
    assert allowed_cache_alias.exists()
    assert allowed_alias_sentinel.exists()


@pytest.mark.parametrize('alias_position', ['leaf', 'parent'])
def test_clear_project_cache_rejects_symlinked_project_root_components(
    tmp_path: Path,
    monkeypatch,
    alias_position,
):
    canonical_project_root = tmp_path / 'canonical-project'
    canonical_project_root.mkdir()
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', canonical_project_root)
    outside_parent = tmp_path / 'outside'
    outside_project = (
        outside_parent if alias_position == 'leaf' else outside_parent / 'project'
    )
    outside_cache = outside_project / 'pkg' / '__pycache__'
    outside_cache.mkdir(parents=True)
    outside_sentinel = outside_cache / 'keep.pyc'
    outside_sentinel.write_text('keep', encoding='utf-8')
    root_alias = tmp_path / 'project-link'
    root_alias.symlink_to(
        outside_project if alias_position == 'leaf' else outside_parent,
        target_is_directory=True,
    )
    aliased_project_root = (
        root_alias if alias_position == 'leaf' else root_alias / outside_project.name
    )

    with pytest.raises(ValueError, match='project root.*symbolic link'):
        clear_project_cache(aliased_project_root)

    assert outside_sentinel.read_text(encoding='utf-8') == 'keep'


def test_clear_project_cache_skips_cache_reached_through_parent_symlink(
    tmp_path: Path,
    tmp_path_factory,
    monkeypatch,
):
    real_cache = tmp_path / 'pkg' / '__pycache__'
    real_cache.mkdir(parents=True)
    outside_root = tmp_path_factory.mktemp('outside-cache')
    outside_cache = outside_root / 'pkg' / '__pycache__'
    outside_cache.mkdir(parents=True)
    outside_sentinel = outside_cache / 'keep.pyc'
    outside_sentinel.write_text('keep', encoding='utf-8')
    escape_parent = tmp_path / 'escape'
    escape_parent.symlink_to(outside_root, target_is_directory=True)
    escaped_cache_path = escape_parent / 'pkg' / '__pycache__'
    monkeypatch.setattr(
        io_utils,
        'find_cache_dirs',
        lambda _root: [real_cache, escaped_cache_path],
    )

    deleted = clear_project_cache(tmp_path)

    assert deleted == [real_cache]
    assert not real_cache.exists()
    assert outside_sentinel.read_text(encoding='utf-8') == 'keep'


@pytest.mark.parametrize('use_symlink', [False, True])
def test_clear_project_cache_rejects_a_root_inside_human_docs_before_discovery(
    tmp_path: Path,
    monkeypatch,
    use_symlink,
):
    monkeypatch.setattr(io_utils, 'PROJECT_ROOT', tmp_path)
    human_docs_dir = tmp_path / 'human_docs'
    cache_dir = human_docs_dir / 'notes' / '__pycache__'
    cache_dir.mkdir(parents=True)
    sentinel = cache_dir / 'keep.pyc'
    sentinel.write_text('user-owned', encoding='utf-8')
    selected_root = human_docs_dir
    if use_symlink:
        selected_root = tmp_path / 'human-docs-link'
        selected_root.symlink_to(human_docs_dir, target_is_directory=True)
    discovery_calls = []

    def fail_if_called(path):
        discovery_calls.append(Path(path))
        raise AssertionError('cache discovery must not run inside human_docs')

    monkeypatch.setattr(io_utils, 'find_cache_dirs', fail_if_called)

    with pytest.raises(ValueError, match='user-owned human_docs'):
        clear_project_cache(selected_root)

    assert discovery_calls == []
    assert sentinel.read_text(encoding='utf-8') == 'user-owned'


def test_clear_project_cache_tolerates_concurrent_cache_deletion(tmp_path: Path, monkeypatch):
    calls = []

    def fake_find_cache_dirs(path):
        calls.append(Path(path))
        raise FileNotFoundError(tmp_path / '.pytest_cache')

    monkeypatch.setattr('runtime.io_utils.find_cache_dirs', fake_find_cache_dirs)

    assert clear_project_cache(tmp_path) is None
    assert calls == [tmp_path]


def test_clear_project_cache_raises_for_missing_project_root(tmp_path: Path):
    missing_root = tmp_path / 'missing-root'

    try:
        clear_project_cache(missing_root)
    except FileNotFoundError as exc:
        assert str(missing_root) in str(exc)
    else:
        raise AssertionError('missing project root should raise FileNotFoundError')


def test_clear_project_cache_rejects_a_non_directory_project_root(tmp_path: Path):
    root_file = tmp_path / 'not-a-project-root'
    root_file.write_text('keep', encoding='utf-8')

    with pytest.raises(NotADirectoryError, match='not a directory'):
        clear_project_cache(root_file)

    assert root_file.read_text(encoding='utf-8') == 'keep'


def test_clear_project_cache_reraises_non_cache_file_not_found(tmp_path: Path, monkeypatch):
    missing_payload = tmp_path / 'data' / 'missing.json'

    def fake_find_cache_dirs(_path):
        raise FileNotFoundError(missing_payload)

    monkeypatch.setattr('runtime.io_utils.find_cache_dirs', fake_find_cache_dirs)

    try:
        clear_project_cache(tmp_path)
    except FileNotFoundError as exc:
        assert exc.args == (missing_payload,)
    else:
        raise AssertionError('non-cache FileNotFoundError should not be swallowed')
