from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_MYUTILS_ROOT = Path(__file__).resolve().parents[3] / 'myutils'
_MYUTILS_MODULE_DIRS = (
    _MYUTILS_ROOT / 'file_utils',
    _MYUTILS_ROOT / 'ai_utils',
    _MYUTILS_ROOT / 'net_utils',
    _MYUTILS_ROOT / 'other_utils',
    _MYUTILS_ROOT / 'viz_utils',
)
for module_dir in _MYUTILS_MODULE_DIRS:
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

from filesystem import delete_cache as delete_cache_dirs, ensure_dirs
from json_io import make_json_safe, read_json_file, write_json_file


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
    ensure_dirs(runtime_dirs)


def clear_project_cache(project_root_path: str | Path = '.'):
    return delete_cache_dirs(project_root_path)
