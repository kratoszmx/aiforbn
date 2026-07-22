from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / 'src'
MODULE_DIRS = {'runtime', 'materials', 'torch_models', 'ui', 'tests', 'template'}
PRODUCTION_MODULE_DIRS = {'runtime', 'materials', 'torch_models', 'ui'}
ROOT_FUNCTION_MODULES = {'main', 'runtime', 'materials', 'ui'}
PUBLIC_NAME_PATTERN = re.compile(r'`([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|`)')
FILE_HEADING_PATTERN = re.compile(
    r'^##\s+(?:`(?P<quoted>[^`]+\.py)`|(?P<plain>\S+\.py))\s*$'
)
PUBLIC_ENTRY_PATTERN = re.compile(r'^- `([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^`]*\))?`')
PUBLIC_CALLABLE_ENTRY_PATTERN = re.compile(
    r'^- `(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<parameters>[^`]*)\)`'
)
ROOT_MODULE_HEADING_PATTERN = re.compile(
    r'^## (?:(?P<main>main)\.py|src/(?P<module>[A-Za-z_][A-Za-z0-9_]*)/'
    r'(?:[^/\s]+\.py)?)$'
)
ROOT_CALLABLE_HEADING_PATTERN = re.compile(
    r'^### `(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<parameters>[^`]*)\)`$'
)
ROOT_FILE_HEADING_PATTERN = re.compile(
    r'^## (?P<path>main\.py|src/[A-Za-z_][A-Za-z0-9_]*/[^/\s]+\.py)$'
)
ROOT_PUBLIC_ENTRY_PATTERN = re.compile(
    r'^### `(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\([^`]*\))?`$'
)


def _documented_public_names(module_name: str) -> set[str]:
    summary_path = SRC_ROOT / module_name / 'PY_FILES_SUMMARY.md'
    return set(PUBLIC_NAME_PATTERN.findall(summary_path.read_text(encoding='utf-8')))


def _is_submodule_import(module_name: str, imported_name: str) -> bool:
    return (SRC_ROOT / module_name / f'{imported_name}.py').exists()


def _manifest_modules() -> dict[str, dict]:
    payload = json.loads((ROOT / 'docs' / 'AGENT_MANIFEST.json').read_text(encoding='utf-8'))
    return {entry['name']: entry for entry in payload['modules']}


def _source_module(source_path: Path) -> str | None:
    try:
        relative_path = source_path.relative_to(SRC_ROOT)
    except ValueError:
        return None
    return relative_path.parts[0] if relative_path.parts else None


def _is_test_source(source_path: Path) -> bool:
    try:
        relative_parts = source_path.relative_to(SRC_ROOT).parts
    except ValueError:
        return False
    return relative_parts[0] == 'tests' or 'tests' in relative_parts[1:-1]


def _documented_names_by_file(summary_path: Path) -> dict[str, set[str]]:
    documented: dict[str, set[str]] = {}
    current_file: str | None = None
    for line in summary_path.read_text(encoding='utf-8').splitlines():
        heading_match = FILE_HEADING_PATTERN.match(line)
        if heading_match:
            current_file = heading_match.group('quoted') or heading_match.group('plain')
            documented.setdefault(current_file, set())
            continue
        if line.startswith('## '):
            current_file = None
            continue
        entry_match = PUBLIC_ENTRY_PATTERN.match(line)
        if current_file is not None and entry_match:
            documented[current_file].add(entry_match.group(1))
    return documented


def _defined_top_level_names(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name != '*'
            )
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split('.')[0]
                for alias in node.names
            )
    return names


def _assigned_name_targets(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {
            name
            for element in target.elts
            for name in _assigned_name_targets(element)
        }
    return set()


def _locally_owned_top_level_names(source_text: str) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(source_text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                name
                for target in node.targets
                for name in _assigned_name_targets(target)
            )
        elif isinstance(node, ast.AnnAssign):
            names.update(_assigned_name_targets(node.target))
    return names


def _production_module_sources() -> dict[str, str]:
    return {
        '.'.join(source_path.relative_to(SRC_ROOT).with_suffix('').parts):
            source_path.read_text(encoding='utf-8')
        for module_name in PRODUCTION_MODULE_DIRS
        for source_path in (SRC_ROOT / module_name).rglob('*.py')
        if not _is_test_source(source_path)
    }


def _documented_production_names_by_module() -> dict[str, set[str]]:
    documented: dict[str, set[str]] = {}
    for module_name in PRODUCTION_MODULE_DIRS:
        summary_path = SRC_ROOT / module_name / 'PY_FILES_SUMMARY.md'
        for file_name, names in _documented_names_by_file(summary_path).items():
            source_module = '.'.join(
                [module_name, *Path(file_name).with_suffix('').parts]
            )
            documented[source_module] = names
    return documented


def _resolved_import_module(consumer_module: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ''
    parent_parts = consumer_module.split('.')[:-node.level]
    imported_parts = node.module.split('.') if node.module else []
    return '.'.join([*parent_parts, *imported_parts])


def _production_import_ownership_violations(
    source_by_module: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    locally_owned_names = {
        module_name: _locally_owned_top_level_names(source_text)
        for module_name, source_text in source_by_module.items()
    }
    documented_names = _documented_production_names_by_module()
    violations: list[tuple[str, str, str, str]] = []
    for consumer_module, source_text in source_by_module.items():
        consumer_root = consumer_module.split('.')[0]
        for node in ast.walk(ast.parse(source_text)):
            if not isinstance(node, ast.ImportFrom):
                continue
            imported_module = _resolved_import_module(consumer_module, node)
            for alias in node.names:
                if alias.name == '*':
                    violations.append((
                        consumer_module,
                        imported_module,
                        alias.name,
                        'production_wildcard_import',
                    ))
                    continue
                if (
                    imported_module.split('.')[0] == consumer_root
                    and imported_module in locally_owned_names
                    and alias.name not in locally_owned_names[imported_module]
                    and alias.name not in documented_names.get(imported_module, set())
                ):
                    violations.append((
                        consumer_module,
                        imported_module,
                        alias.name,
                        'implicit_same_module_facade_import',
                    ))
    return sorted(violations)


def _add_import_after_future(source_text: str, import_line: str) -> str:
    future_line = 'from __future__ import annotations\n'
    if source_text.startswith(future_line):
        return source_text.replace(future_line, f'{future_line}\n{import_line}\n', 1)
    return f'{import_line}\n{source_text}'


def _documented_callable_parameters_by_file(
    summary_path: Path,
) -> dict[str, dict[str, tuple[list[str], bool]]]:
    documented: dict[str, dict[str, tuple[list[str], bool]]] = {}
    current_file: str | None = None
    for line in summary_path.read_text(encoding='utf-8').splitlines():
        heading_match = FILE_HEADING_PATTERN.match(line)
        if heading_match:
            current_file = heading_match.group('quoted') or heading_match.group('plain')
            documented.setdefault(current_file, {})
            continue
        if line.startswith('## '):
            current_file = None
            continue
        entry_match = PUBLIC_CALLABLE_ENTRY_PATTERN.match(line)
        if current_file is None or entry_match is None:
            continue
        documented[current_file][entry_match.group('name')] = (
            _parse_documented_parameters(entry_match.group('parameters'))
        )
    return documented


def _parse_documented_parameters(parameters_text: str) -> tuple[list[str], bool]:
    raw_parameters = [
        parameter.strip() for parameter in parameters_text.split(',')
    ]
    has_ellipsis = '...' in raw_parameters
    parameter_names = [
        (
            parameter
            if parameter in {'*', '/'}
            else parameter.split('=', 1)[0].strip()
        )
        for parameter in raw_parameters
        if parameter not in {'', '...'}
    ]
    return parameter_names, has_ellipsis


def _root_documented_callable_parameters() -> dict[str, dict[str, tuple[list[str], bool]]]:
    documented: dict[str, dict[str, tuple[list[str], bool]]] = {}
    current_module: str | None = None
    for line in (ROOT / 'docs' / 'PY_FILES_SUMMARY.md').read_text(
        encoding='utf-8'
    ).splitlines():
        module_match = ROOT_MODULE_HEADING_PATTERN.match(line)
        if module_match:
            current_module = module_match.group('main') or module_match.group('module')
            documented.setdefault(current_module, {})
            continue
        if line.startswith('## '):
            current_module = None
            continue
        callable_match = ROOT_CALLABLE_HEADING_PATTERN.match(line)
        if current_module is None or callable_match is None:
            continue
        documented[current_module][callable_match.group('name')] = (
            _parse_documented_parameters(callable_match.group('parameters'))
        )
    return documented


def _root_documented_names_by_file() -> dict[str, set[str]]:
    documented: dict[str, set[str]] = {}
    current_file: str | None = None
    for line in (ROOT / 'docs' / 'PY_FILES_SUMMARY.md').read_text(
        encoding='utf-8'
    ).splitlines():
        file_match = ROOT_FILE_HEADING_PATTERN.match(line)
        if file_match:
            current_file = file_match.group('path')
            documented.setdefault(current_file, set())
            continue
        if line.startswith('## '):
            current_file = None
            continue
        entry_match = ROOT_PUBLIC_ENTRY_PATTERN.match(line)
        if current_file is not None and entry_match:
            documented[current_file].add(entry_match.group('name'))
    return documented


def _main_control_flags() -> set[str]:
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
    return {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'add_argument'
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith('--')
    }


def _manifest_entrypoint_flags() -> set[str]:
    manifest = json.loads(
        (ROOT / 'docs' / 'AGENT_MANIFEST.json').read_text(encoding='utf-8')
    )
    return {
        token
        for entry in manifest['entrypoints']
        for token in entry['command'].split()
        if token.startswith('--')
    }


def _defined_top_level_function_parameters(source_path: Path) -> dict[str, list[str]]:
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    signatures: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = [
            argument.arg
            for argument in node.args.posonlyargs
        ]
        if node.args.posonlyargs:
            parameters.append('/')
        parameters.extend(argument.arg for argument in node.args.args)
        if node.args.vararg is not None:
            parameters.append(f'*{node.args.vararg.arg}')
        elif node.args.kwonlyargs:
            parameters.append('*')
        parameters.extend(argument.arg for argument in node.args.kwonlyargs)
        if node.args.kwarg is not None:
            parameters.append(f'**{node.args.kwarg.arg}')
        signatures[node.name] = parameters
    return signatures


def _documented_parameters_match(
    documented_parameters: list[str],
    has_ellipsis: bool,
    live_parameters: list[str],
) -> bool:
    return (
        live_parameters[:len(documented_parameters)] == documented_parameters
        if has_ellipsis
        else live_parameters == documented_parameters
    )


def test_explicit_cross_module_imports_are_documented_public_surfaces():
    documented_names = {
        module_name: _documented_public_names(module_name)
        for module_name in MODULE_DIRS
    }
    missing: list[tuple[str, str, str]] = []

    for source_path in [ROOT / 'main.py', *SRC_ROOT.rglob('*.py')]:
        consumer_module = _source_module(source_path)
        tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue

            source_module = node.module.split('.')[0]
            if source_module not in MODULE_DIRS:
                continue
            if source_module == consumer_module:
                continue

            for alias in node.names:
                imported_name = alias.name
                if (
                    imported_name == '*'
                    or imported_name.startswith('_')
                    or _is_submodule_import(source_module, imported_name)
                ):
                    continue
                if imported_name not in documented_names[source_module]:
                    missing.append((
                        str(source_path.relative_to(ROOT)),
                        node.module,
                        imported_name,
                    ))

    assert missing == []


def test_production_cross_module_imports_follow_manifest_boundaries():
    modules = _manifest_modules()
    violations: list[tuple[str, str, str]] = []

    for source_path in SRC_ROOT.rglob('*.py'):
        source_module = _source_module(source_path)
        if source_module not in modules or _is_test_source(source_path):
            continue
        allowed_dependencies = set(modules[source_module].get('allowed_dependencies', []))
        tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_module = node.module.split('.')[0]
                if imported_module not in modules or imported_module == source_module:
                    continue
                if imported_module not in allowed_dependencies:
                    violations.append((
                        str(source_path.relative_to(ROOT)),
                        imported_module,
                        'dependency_not_allowed',
                    ))
                for alias in node.names:
                    if alias.name == '*' or alias.name.startswith('_'):
                        violations.append((
                            str(source_path.relative_to(ROOT)),
                            f'{node.module}.{alias.name}',
                            'private_or_wildcard_cross_module_import',
                        ))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_module = alias.name.split('.')[0]
                    if (
                        imported_module in modules
                        and imported_module != source_module
                        and imported_module not in allowed_dependencies
                    ):
                        violations.append((
                            str(source_path.relative_to(ROOT)),
                            imported_module,
                            'dependency_not_allowed',
                        ))

    assert violations == []


def test_main_entrypoint_avoids_private_or_wildcard_module_imports():
    violations: list[str] = []
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        imported_module = node.module.split('.')[0]
        if imported_module not in MODULE_DIRS:
            continue
        violations.extend(
            f'{node.module}.{alias.name}'
            for alias in node.names
            if alias.name == '*' or alias.name.startswith('_')
        )

    assert violations == []


def test_production_imports_are_explicit_and_owned_by_the_imported_module():
    source_by_module = _production_module_sources()

    assert source_by_module
    assert any(name.startswith('materials.') for name in source_by_module)
    assert _production_import_ownership_violations(source_by_module) == []


def test_production_import_guard_rejects_wildcard_and_facade_mutations():
    source_by_module = _production_module_sources()
    target_module = 'materials.plots'
    mutations = (
        (
            'from materials.constants import *',
            '*',
            'production_wildcard_import',
        ),
        (
            'from materials.constants import REFERENCE_PROPERTY_COLUMNS',
            'REFERENCE_PROPERTY_COLUMNS',
            'implicit_same_module_facade_import',
        ),
        (
            'from .constants import REFERENCE_PROPERTY_COLUMNS',
            'REFERENCE_PROPERTY_COLUMNS',
            'implicit_same_module_facade_import',
        ),
    )

    for import_line, imported_name, reason in mutations:
        mutated_sources = dict(source_by_module)
        mutated_sources[target_module] = _add_import_after_future(
            mutated_sources[target_module], import_line
        )
        compile(mutated_sources[target_module], target_module, 'exec')
        assert (
            target_module,
            'materials.constants',
            imported_name,
            reason,
        ) in _production_import_ownership_violations(mutated_sources)


def test_every_documented_public_symbol_exists_in_its_declared_file():
    missing: list[tuple[str, str]] = []
    checked_symbol_count = 0
    checked_symbol_counts_by_module: dict[str, int] = {}
    for module_name in MODULE_DIRS:
        summary_path = SRC_ROOT / module_name / 'PY_FILES_SUMMARY.md'
        module_symbol_count = 0
        for file_name, documented_names in _documented_names_by_file(summary_path).items():
            checked_symbol_count += len(documented_names)
            module_symbol_count += len(documented_names)
            source_path = SRC_ROOT / module_name / file_name
            if not source_path.is_file():
                missing.extend((str(source_path.relative_to(ROOT)), name) for name in documented_names)
                continue
            defined_names = _defined_top_level_names(source_path)
            missing.extend(
                (str(source_path.relative_to(ROOT)), name)
                for name in sorted(documented_names - defined_names)
            )
        checked_symbol_counts_by_module[module_name] = module_symbol_count

    assert checked_symbol_count > 0, 'No documented public symbols were parsed from module summaries'
    empty_production_modules = sorted(
        module_name
        for module_name in PRODUCTION_MODULE_DIRS
        if checked_symbol_counts_by_module.get(module_name, 0) == 0
    )
    assert empty_production_modules == [], (
        'No documented public symbols were parsed for production modules: '
        f'{empty_production_modules}'
    )
    assert missing == []


def test_every_root_documented_symbol_exists_in_its_declared_file():
    missing: list[tuple[str, str]] = []
    documented = _root_documented_names_by_file()
    checked_symbol_count = sum(len(names) for names in documented.values())
    for relative_path, documented_names in documented.items():
        source_path = ROOT / relative_path
        if not source_path.is_file():
            missing.extend((relative_path, name) for name in sorted(documented_names))
            continue
        defined_names = _defined_top_level_names(source_path)
        missing.extend(
            (relative_path, name)
            for name in sorted(documented_names - defined_names)
        )

    assert checked_symbol_count > 0, 'No root-summary public symbols were parsed'
    assert missing == []


def test_every_main_control_flag_is_declared_as_an_agent_entrypoint():
    implemented_flags = _main_control_flags()
    manifested_flags = _manifest_entrypoint_flags()

    assert implemented_flags
    assert manifested_flags == implemented_flags


def test_documented_public_callable_parameter_order_matches_live_source():
    mismatches: list[tuple[str, str, list[str], list[str]]] = []
    checked_callable_count = 0
    for module_name in MODULE_DIRS:
        summary_path = SRC_ROOT / module_name / 'PY_FILES_SUMMARY.md'
        for file_name, documented_callables in (
            _documented_callable_parameters_by_file(summary_path).items()
        ):
            source_path = SRC_ROOT / module_name / file_name
            if not source_path.is_file():
                continue
            live_signatures = _defined_top_level_function_parameters(source_path)
            for callable_name, (documented_parameters, has_ellipsis) in (
                documented_callables.items()
            ):
                live_parameters = live_signatures.get(callable_name)
                if live_parameters is None:
                    continue
                checked_callable_count += 1
                matches = _documented_parameters_match(
                    documented_parameters,
                    has_ellipsis,
                    live_parameters,
                )
                if not matches:
                    mismatches.append((
                        str(source_path.relative_to(ROOT)),
                        callable_name,
                        documented_parameters,
                        live_parameters,
                    ))

    assert checked_callable_count > 0, 'No documented callable signatures were checked'
    assert mismatches == []


def test_root_public_summary_callable_parameter_order_matches_live_source():
    mismatches: list[tuple[str, str, list[str], list[str]]] = []
    unresolved: list[tuple[str, str, int]] = []
    checked_callable_count = 0
    root_documented_callables = _root_documented_callable_parameters()
    empty_root_modules = sorted(
        module_name
        for module_name in ROOT_FUNCTION_MODULES
        if not root_documented_callables.get(module_name)
    )
    assert empty_root_modules == [], (
        'No root-summary callable signatures were parsed for modules: '
        f'{empty_root_modules}'
    )
    for module_name, documented_callables in root_documented_callables.items():
        source_paths = (
            [ROOT / 'main.py']
            if module_name == 'main'
            else list((SRC_ROOT / module_name).glob('*.py'))
        )
        live_candidates: dict[str, list[tuple[Path, list[str]]]] = {}
        for source_path in source_paths:
            for callable_name, live_parameters in (
                _defined_top_level_function_parameters(source_path).items()
            ):
                live_candidates.setdefault(callable_name, []).append(
                    (source_path, live_parameters)
                )
        for callable_name, (documented_parameters, has_ellipsis) in (
            documented_callables.items()
        ):
            candidates = live_candidates.get(callable_name, [])
            if len(candidates) != 1:
                unresolved.append((module_name, callable_name, len(candidates)))
                continue
            source_path, live_parameters = candidates[0]
            checked_callable_count += 1
            if not _documented_parameters_match(
                documented_parameters,
                has_ellipsis,
                live_parameters,
            ):
                mismatches.append((
                    str(source_path.relative_to(ROOT)),
                    callable_name,
                    documented_parameters,
                    live_parameters,
                ))

    assert checked_callable_count > 0, 'No root-summary callable signatures were checked'
    assert unresolved == []
    assert mismatches == []


def test_imported_runtime_json_helpers_count_as_live_public_reexports():
    defined_names = _defined_top_level_names(SRC_ROOT / 'runtime' / 'io_utils.py')

    assert {'make_json_safe', 'read_json_file'}.issubset(defined_names)


def test_ui_renderer_profile_requires_a_passed_call_phase(tmp_path):
    project_root = tmp_path / 'project'
    runtime_dir = project_root / 'src' / 'runtime'
    ui_test_dir = project_root / 'src' / 'ui' / 'tests'
    runtime_dir.mkdir(parents=True)
    ui_test_dir.mkdir(parents=True)
    shutil.copyfile(ROOT / 'conftest.py', project_root / 'conftest.py')
    (runtime_dir / 'io_utils.py').write_text(
        'def clear_project_cache(project_root_path):\n    return None\n',
        encoding='utf-8',
    )
    ui_test_path = ui_test_dir / 'test_streamlit_app.py'

    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTEST_ADDOPTS', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'

    def run_pytest(*arguments):
        return subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    relative_target = 'src/ui/tests/test_streamlit_app.py'
    ui_test_path.write_text(
        "import pytest\npytestmark = pytest.mark.skip(reason='disabled')\n"
        'def test_renderer_contract():\n    assert False\n',
        encoding='utf-8',
    )
    skipped = run_pytest(relative_target)
    assert skipped.returncode == 1
    assert '1 skipped' in skipped.stdout
    assert 'passed no non-xfail renderer test calls' in skipped.stdout
    assert run_pytest('--collect-only', relative_target).returncode == 0

    ui_test_path.write_text(
        'def test_renderer_contract():\n    assert True\n', encoding='utf-8'
    )
    absolute_nodeid = f'{ui_test_path.resolve()}::test_renderer_contract'
    passed = run_pytest(absolute_nodeid)
    assert passed.returncode == 0
    assert '1 passed' in passed.stdout
