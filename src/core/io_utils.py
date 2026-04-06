from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_MYUTILS_DIR = Path(__file__).resolve().parents[3] / 'myutils'
if str(_MYUTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_MYUTILS_DIR))

from file_utils import delete_cache as delete_cache_dirs, ensure_dirs


RUNTIME_DIR_KEYS = (
    ('data', 'raw_dir'),
    ('data', 'processed_dir'),
    ('project', 'artifact_dir'),
)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load config module from: {path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = getattr(module, 'CONFIG', None)
    if not isinstance(cfg, dict):
        raise TypeError(f'{path} must define CONFIG as a dict')
    return cfg


def ensure_runtime_dirs(cfg: dict) -> None:
    runtime_dirs = [cfg[section][key] for section, key in RUNTIME_DIR_KEYS]
    runtime_dirs.extend(['tasks', 'apps', 'tests', 'notebooks'])
    ensure_dirs(runtime_dirs)


def clear_project_cache(project_root_path: str | Path = '.'):
    return delete_cache_dirs(project_root_path)
