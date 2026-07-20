from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runtime import io_utils
from runtime.io_utils import (
    clear_project_cache,
    ensure_runtime_dirs,
    load_config,
    make_json_safe,
    read_json_file,
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


def test_json_helpers_delegate_to_myutils_json_io(tmp_path: Path):
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


def test_clear_project_cache_uses_myutils_discovery_and_preserves_human_docs(tmp_path: Path):
    pycache_dir = tmp_path / 'pkg' / '__pycache__'
    pycache_dir.mkdir(parents=True)
    (pycache_dir / 'mod.pyc').write_text('x', encoding='utf-8')
    human_docs_cache_dir = tmp_path / 'human_docs' / 'notes' / '__pycache__'
    human_docs_cache_dir.mkdir(parents=True)
    (human_docs_cache_dir / 'user.pyc').write_text('user-owned', encoding='utf-8')

    deleted = clear_project_cache(tmp_path)

    assert deleted == [pycache_dir]
    assert not pycache_dir.exists()
    assert human_docs_cache_dir.exists()


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
