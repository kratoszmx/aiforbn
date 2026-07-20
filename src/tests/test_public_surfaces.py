from __future__ import annotations

import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / 'src'
MODULE_DIRS = {'runtime', 'materials', 'torch_models', 'ui', 'tests', 'template'}
PRODUCTION_MODULE_DIRS = {'runtime', 'materials', 'torch_models', 'ui'}
PUBLIC_NAME_PATTERN = re.compile(r'`([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|`)')
FILE_HEADING_PATTERN = re.compile(
    r'^##\s+(?:`(?P<quoted>[^`]+\.py)`|(?P<plain>\S+\.py))\s*$'
)
PUBLIC_ENTRY_PATTERN = re.compile(r'^- `([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^`]*\))?`')


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


def test_explicit_cross_module_imports_are_documented_public_surfaces():
    documented_names = {
        module_name: _documented_public_names(module_name)
        for module_name in MODULE_DIRS
    }
    missing: list[tuple[str, str, str]] = []

    for source_path in [ROOT / 'main.py', *SRC_ROOT.rglob('*.py')]:
        tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue

            source_module = node.module.split('.')[0]
            if source_module not in MODULE_DIRS:
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


def test_imported_runtime_json_helpers_count_as_live_public_reexports():
    defined_names = _defined_top_level_names(SRC_ROOT / 'runtime' / 'io_utils.py')

    assert {'make_json_safe', 'read_json_file'}.issubset(defined_names)
