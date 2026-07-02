from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / 'src'
MODULE_DIRS = {'runtime', 'materials', 'torch_models', 'ui', 'tests', 'template'}
PUBLIC_NAME_PATTERN = re.compile(r'`([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|`)')


def _documented_public_names(module_name: str) -> set[str]:
    summary_path = SRC_ROOT / module_name / 'PY_FILES_SUMMARY.md'
    return set(PUBLIC_NAME_PATTERN.findall(summary_path.read_text(encoding='utf-8')))


def _is_submodule_import(module_name: str, imported_name: str) -> bool:
    return (SRC_ROOT / module_name / f'{imported_name}.py').exists()


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
