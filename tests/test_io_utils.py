from pathlib import Path

import numpy as np
import pandas as pd

from core.io_utils import (
    clear_project_cache,
    ensure_runtime_dirs,
    load_config,
    make_json_safe,
    read_json_file,
    write_json_file,
)


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


def test_clear_project_cache_uses_myutils_filesystem_cleanup(tmp_path: Path):
    pycache_dir = tmp_path / 'pkg' / '__pycache__'
    pycache_dir.mkdir(parents=True)
    (pycache_dir / 'mod.pyc').write_text('x', encoding='utf-8')

    deleted = clear_project_cache(tmp_path)

    assert deleted == [pycache_dir]
    assert not pycache_dir.exists()
