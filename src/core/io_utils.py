from __future__ import annotations

from pathlib import Path
import sys

import yaml

_MYUTILS_DIR = Path(__file__).resolve().parents[3] / 'myutils'
if str(_MYUTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_MYUTILS_DIR))

from file_utils import delete_cache as delete_cache_dirs, ensure_dirs


RUNTIME_DIR_KEYS = (
    ('data', 'raw_dir'),
    ('data', 'processed_dir'),
    ('project', 'artifact_dir'),
)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def ensure_runtime_dirs(cfg: dict) -> None:
    runtime_dirs = [cfg[section][key] for section, key in RUNTIME_DIR_KEYS]
    runtime_dirs.extend(['tasks', 'apps', 'tests', 'notebooks'])
    ensure_dirs(runtime_dirs)


def clear_project_cache(project_root_path: str | Path = '.'):
    return delete_cache_dirs(project_root_path)
