from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType

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

from filesystem import ensure_dirs, find_cache_dirs
from json_io import make_json_safe, read_json_file, write_json_file as _shared_write_json_file
from runtime.utils import _path_has_symlink_component, _path_is_same_or_descendant


RUNTIME_DIR_KEYS = (
    ('data', 'raw_dir'),
    ('data', 'processed_dir'),
    ('project', 'artifact_dir'),
)

CACHE_DIR_NAMES = frozenset({'__pycache__', '.pytest_cache'})
HUMAN_DOCS_DIR = Path('human_docs')
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve(strict=False)
    human_docs_root = (PROJECT_ROOT.resolve() / HUMAN_DOCS_DIR).resolve(strict=False)
    if _path_is_same_or_descendant(path, human_docs_root):
        raise ValueError('Config files must not be loaded from user-owned human_docs/')

    module = ModuleType(path.stem)
    module.__file__ = str(path)
    source = path.read_bytes()
    exec(compile(source, str(path), 'exec'), module.__dict__)

    cfg = getattr(module, 'CONFIG', None)
    if not isinstance(cfg, dict):
        raise TypeError(f'{path} must define CONFIG as a dict')
    return cfg


def validate_runtime_output_path(
    path: str | Path,
    project_root_path: str | Path | None = None,
    *,
    required_parent_path: str | Path | None = None,
    reject_leaf_symlink: bool = False,
    expected_output_kind: str | None = None,
) -> Path:
    expanded_path = Path(path).expanduser()
    resolved_path = expanded_path.resolve(strict=False)
    project_roots = [PROJECT_ROOT.resolve()]
    if project_root_path is not None:
        declared_project_root = Path(project_root_path).expanduser().resolve()
        if declared_project_root not in project_roots:
            project_roots.append(declared_project_root)

    for project_root in project_roots:
        human_docs_root = (project_root / HUMAN_DOCS_DIR).resolve(strict=False)
        if _path_is_same_or_descendant(resolved_path, human_docs_root):
            raise ValueError(
                'Runtime output paths must not be placed under user-owned human_docs/'
            )
    if required_parent_path is not None:
        resolved_parent = Path(required_parent_path).expanduser().resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_parent)
        except ValueError as exc:
            raise ValueError(
                'Runtime output paths must remain under their configured output root'
            ) from exc
        reject_leaf_symlink = True
    if reject_leaf_symlink and expanded_path.is_symlink():
        raise ValueError('Runtime output paths must not target symbolic-link leaves')
    if expected_output_kind not in (None, 'file', 'directory'):
        raise ValueError(f'Unsupported runtime output kind: {expected_output_kind}')
    if (
        expected_output_kind == 'file'
        and expanded_path.exists()
        and not expanded_path.is_file()
    ):
        raise ValueError('Runtime file outputs must target regular-file leaves')
    if (
        expected_output_kind == 'directory'
        and expanded_path.exists()
        and not expanded_path.is_dir()
    ):
        raise ValueError('Runtime directory outputs must target directory leaves')
    if expected_output_kind is not None:
        invalid_parent = next(
            (
                parent
                for parent in expanded_path.parents
                if (parent.exists() or parent.is_symlink()) and not parent.is_dir()
            ),
            None,
        )
        if invalid_parent is not None:
            raise ValueError(
                f'Runtime output parent paths must be directories: {invalid_parent}'
            )
    try:
        has_multiple_links = resolved_path.is_file() and resolved_path.stat().st_nlink > 1
    except OSError:
        has_multiple_links = False
    if has_multiple_links:
        raise ValueError('Runtime output paths must not target files with multiple hard links')
    return resolved_path


def configure_matplotlib_cache() -> Path:
    os.environ.setdefault('MPLBACKEND', 'Agg')
    configured_cache = os.environ.get('MPLCONFIGDIR')
    if configured_cache is None or not configured_cache.strip():
        configured_cache = '/tmp/ai_for_bn_mplconfig'
    config_dir = validate_runtime_output_path(
        configured_cache,
        expected_output_kind='directory',
    )
    os.environ['MPLCONFIGDIR'] = str(config_dir)
    return config_dir


def ensure_runtime_dirs(cfg: dict, project_root_path: str | Path = '.') -> None:
    runtime_dirs = [
        validate_runtime_output_path(
            runtime_dir,
            project_root_path=project_root_path,
            expected_output_kind='directory',
        )
        for runtime_dir in (cfg[section][key] for section, key in RUNTIME_DIR_KEYS)
    ]
    ensure_dirs(runtime_dirs)


def write_json_file(
    payload,
    path: str | Path,
    *,
    encoding: str = 'utf-8',
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    indent: int | None = 2,
):
    output_path = validate_runtime_output_path(
        path,
        reject_leaf_symlink=True,
        expected_output_kind='file',
    )
    safe_payload = make_json_safe(payload)
    try:
        serialized = json.dumps(
            safe_payload,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Payload is not JSON-serializable for {output_path}: {exc}'
        ) from exc
    if encoding is not None:
        serialized.encode(encoding)
    return _shared_write_json_file(
        safe_payload,
        output_path,
        encoding=encoding,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
    )


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
    root_path = Path(project_root_path).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f'Project root does not exist: {root_path}')
    if not root_path.is_dir():
        raise NotADirectoryError(f'Project root is not a directory: {root_path}')

    resolved_root = validate_runtime_output_path(root_path)
    if _path_has_symlink_component(root_path):
        raise ValueError(
            'Cache project root must not contain a symbolic link path component'
        )
    protected_human_docs_roots = {
        (PROJECT_ROOT.resolve() / HUMAN_DOCS_DIR).resolve(strict=False),
        (resolved_root / HUMAN_DOCS_DIR).resolve(strict=False),
    }
    try:
        cache_dirs = find_cache_dirs(resolved_root)
    except FileNotFoundError as exc:
        # Parallel agent validations can race while deleting the same cache tree.
        missing_path = _missing_path_from_file_not_found(exc)
        if missing_path is not None and _is_cache_path(missing_path):
            return None
        raise

    deleted: list[Path] = []
    for cache_dir in cache_dirs:
        if cache_dir.is_symlink():
            continue
        resolved_cache_dir = cache_dir.resolve(strict=False)
        if not _path_is_same_or_descendant(resolved_cache_dir, resolved_root):
            continue
        if any(
            _path_is_same_or_descendant(resolved_cache_dir, protected_root)
            for protected_root in protected_human_docs_roots
        ):
            continue
        try:
            shutil.rmtree(cache_dir)
        except FileNotFoundError as exc:
            missing_path = _missing_path_from_file_not_found(exc)
            if missing_path is not None and _is_cache_path(missing_path):
                continue
            raise
        deleted.append(cache_dir)
    return deleted
