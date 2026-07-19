from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

_REQUIRED_MYUTILS_PATHS = (
    Path('file_utils/filesystem.py'),
    Path('file_utils/json_io.py'),
)


def _find_myutils_root(source_path: str | Path | None = None) -> Path:
    override = os.environ.get('MYUTILS_ROOT', '').strip()
    if override:
        candidates = [Path(override).expanduser()]
    else:
        source = Path(source_path or __file__).expanduser().resolve()
        candidates = [parent / 'myutils' for parent in source.parents]

    checked: list[Path] = []
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in checked:
            continue
        checked.append(resolved_candidate)
        if all((resolved_candidate / relative_path).is_file() for relative_path in _REQUIRED_MYUTILS_PATHS):
            return resolved_candidate

    checked_text = ', '.join(str(path) for path in checked) or '<none>'
    raise ModuleNotFoundError(
        'Unable to locate the local myutils checkout containing '
        '`file_utils/filesystem.py` and `file_utils/json_io.py`. '
        'Set MYUTILS_ROOT to the myutils repository root. '
        f'Checked: {checked_text}'
    )


_MYUTILS_ROOT = _find_myutils_root()
_MYUTILS_FILE_UTILS_DIR = _MYUTILS_ROOT / 'file_utils'
if str(_MYUTILS_FILE_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_MYUTILS_FILE_UTILS_DIR))

from filesystem import delete_cache as delete_cache_dirs, ensure_dirs
from json_io import make_json_safe, read_json_file, write_json_file


RUNTIME_DIR_KEYS = (
    ('data', 'raw_dir'),
    ('data', 'processed_dir'),
    ('project', 'artifact_dir'),
)

CACHE_DIR_NAMES = frozenset({'__pycache__', '.pytest_cache'})


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


def _missing_path_from_file_not_found(exc: FileNotFoundError) -> Path | None:
    raw_path = exc.filename
    if raw_path is None and exc.args:
        raw_path = exc.args[0]
    if raw_path is None:
        return None
    try:
        return Path(raw_path)
    except TypeError:
        return None


def _is_cache_path(path: Path) -> bool:
    return any(part in CACHE_DIR_NAMES for part in path.parts)


def clear_project_cache(project_root_path: str | Path = '.'):
    root_path = Path(project_root_path)
    if not root_path.exists():
        raise FileNotFoundError(f'Project root does not exist: {root_path}')

    try:
        return delete_cache_dirs(root_path)
    except FileNotFoundError as exc:
        # Parallel agent validations can race while deleting the same cache tree.
        missing_path = _missing_path_from_file_not_found(exc)
        if missing_path is not None and _is_cache_path(missing_path):
            return None
        raise
