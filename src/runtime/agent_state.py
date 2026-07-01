from __future__ import annotations

from datetime import datetime, timezone
import json
import importlib.util
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_AGENT_MANIFEST_PATH = Path('docs/AGENT_MANIFEST.json')


def _project_root(path: str | Path = '.') -> Path:
    return Path(path).expanduser().resolve()


def _read_text_if_present(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    return path.read_text(encoding='utf-8')


def _path_check(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    return {
        'path': relative_path,
        'exists': path.exists(),
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
    }


def _git_stdout(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def load_agent_manifest(
    project_root_path: str | Path = '.',
    manifest_path: str | Path = DEFAULT_AGENT_MANIFEST_PATH,
) -> dict[str, Any]:
    root = _project_root(project_root_path)
    manifest = json.loads((root / manifest_path).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise TypeError(f'Agent manifest must be a JSON object: {manifest_path}')
    return manifest


def validate_agent_layout(
    project_root_path: str | Path = '.',
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root_path)
    manifest_payload = manifest if manifest is not None else load_agent_manifest(root)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    checks: list[dict[str, object]] = []

    required_paths = list(manifest_payload.get('source_of_truth_files', []))
    required_paths.extend([
        'main.py',
        'src/config.py',
        'conftest.py',
        'requirements.txt',
    ])
    for relative_path in dict.fromkeys(str(path) for path in required_paths):
        check = _path_check(root, relative_path)
        checks.append({**check, 'kind': 'required_path'})
        if not check['exists']:
            errors.append({
                'code': 'missing_required_path',
                'path': relative_path,
                'message': f'Missing required AI-native source-of-truth path: {relative_path}',
            })

    modules = manifest_payload.get('modules', [])
    if not isinstance(modules, list):
        errors.append({
            'code': 'invalid_manifest_modules',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `modules` must be a list.',
        })
        modules = []
    for module in modules:
        if not isinstance(module, dict):
            errors.append({
                'code': 'invalid_manifest_module',
                'path': 'docs/AGENT_MANIFEST.json',
                'message': 'Every manifest module entry must be a JSON object.',
            })
            continue
        for field in ('path', 'public_surface', 'agent_rules', 'local_utils'):
            relative_path = str(module.get(field, ''))
            if not relative_path:
                errors.append({
                    'code': 'missing_module_field',
                    'path': str(module.get('name', '<unnamed>')),
                    'message': f'Module entry is missing `{field}`.',
                })
                continue
            check = _path_check(root, relative_path)
            checks.append({
                **check,
                'kind': f'module_{field}',
                'module': module.get('name', '<unnamed>'),
            })
            if not check['exists']:
                errors.append({
                    'code': f'missing_module_{field}',
                    'path': relative_path,
                    'message': f'Module `{module.get("name", "<unnamed>")}` is missing `{field}`.',
                })

    skill_text = _read_text_if_present(root / 'skill.txt')
    if 'skills' in skill_text and not any((root / 'skills').glob('*')):
        warnings.append({
            'code': 'empty_skills_directory',
            'path': 'skills/',
            'message': '`skill.txt` references skills/, but that directory has no files.',
        })

    if (root / 'README.md').exists():
        warnings.append({
            'code': 'root_readme_present',
            'path': 'README.md',
            'message': 'Root README.md exists; this repo prefers AGENTS.md as the agent entrypoint.',
        })

    stale_handoff = _read_text_if_present(root / 'docs/HANDOFF.md')
    if '$HOME/projects/ai_for_bn' in stale_handoff:
        warnings.append({
            'code': 'stale_handoff_project_path',
            'path': 'docs/HANDOFF.md',
            'message': 'Handoff still mentions the old `$HOME/projects/ai_for_bn` path.',
        })

    init_files = sorted(
        str(path.relative_to(root))
        for path in (root / 'src').glob('**/__init__.py')
    ) if (root / 'src').exists() else []
    if init_files:
        warnings.append({
            'code': 'package_init_files_present',
            'path': 'src/',
            'message': f'Flat-module contract prefers no __init__.py files; found: {init_files}',
        })

    dependency_checks = manifest_payload.get('dependency_imports', [])
    if isinstance(dependency_checks, list):
        for dependency in dependency_checks:
            if not isinstance(dependency, dict):
                continue
            module_name = str(dependency.get('module', '')).strip()
            if not module_name:
                continue
            available = importlib.util.find_spec(module_name) is not None
            checks.append({
                'kind': 'dependency_import',
                'package': dependency.get('package', module_name),
                'module': module_name,
                'required_for': dependency.get('required_for', ''),
                'available': available,
            })
            if not available:
                warnings.append({
                    'code': 'missing_declared_dependency',
                    'path': 'requirements.txt',
                    'message': (
                        f'Declared dependency `{dependency.get("package", module_name)}` '
                        f'is not importable as `{module_name}`; required for '
                        f'{dependency.get("required_for", "unspecified scope")}.'
                    ),
                })

    return {
        'status': 'ok' if not errors else 'error',
        'errors': errors,
        'warnings': warnings,
        'checks': checks,
    }


def build_agent_state(
    project_root_path: str | Path = '.',
    manifest_path: str | Path = DEFAULT_AGENT_MANIFEST_PATH,
) -> dict[str, Any]:
    root = _project_root(project_root_path)
    manifest = load_agent_manifest(root, manifest_path=manifest_path)
    validation = validate_agent_layout(root, manifest)
    git_status_short = _git_stdout(root, ['status', '--short', '--branch'])
    tracked_research_plan_count = _git_stdout(
        root,
        ['ls-files', 'docs/research_plan'],
    )
    tracked_research_paths = [
        line for line in (tracked_research_plan_count or '').splitlines() if line
    ]
    git_state = {
        'branch': _git_stdout(root, ['branch', '--show-current']),
        'head': _git_stdout(root, ['rev-parse', '--short', 'HEAD']),
        'status_short': git_status_short,
        'origin_main': _git_stdout(root, ['rev-parse', '--short', 'origin/main']),
        'tracked_research_plan_file_count': len(tracked_research_paths),
    }
    return {
        'schema_version': 'aiforbn.agent_state.v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'project_root': str(root),
        'manifest_path': str(manifest_path),
        'status': validation['status'],
        'manifest': manifest,
        'validation': validation,
        'git': git_state,
        'next_agent_recommended_order': [
            'Read AGENTS.md and nearest module AGENTS.md',
            'Run python3 main.py --agent-doctor',
            'Run python3 main.py --dry-run before expensive work',
            'Run focused pytest for touched modules',
            'Update PY_FILES_SUMMARY.md when public surfaces change',
        ],
    }


def agent_state_to_json(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)


def write_agent_state(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(agent_state_to_json(state) + '\n', encoding='utf-8')
