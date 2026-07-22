from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from runtime.utils import _path_is_same_or_descendant


DEFAULT_AGENT_MANIFEST_PATH = Path('docs/AGENT_MANIFEST.json')
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_RESEARCH_PLAN_SOURCE_FILES = (
    'human_docs/research_plan/ai_for_bn_research_plan_v18.tex',
    'human_docs/research_plan/ai_for_bn_research_plan_v18.bib',
)

REQUIRED_RESEARCH_PLAN_ALIGNMENT_STATUS = 'v18_alignment_contract'

REQUIRED_RESEARCH_PLAN_ALIGNMENT_ANCHORS = {
    'bounded_bn_centered_design_space',
    'provenance_aware_bn_data_layer',
    'formula_only_candidate_compatible_screening',
    'structure_resolved_followup_after_handoff',
    'grouped_formula_and_bn_family_holdout_diagnostics',
    'uncertainty_calibration_rank_stability_domain_support_novelty_action_labels',
    'conservative_formula_stage_directness_and_structure_properties',
    'validation_ready_structure_handoff_not_synthesis_proof',
    'machine_verifiable_deliverable_chain',
}

REQUIRED_RESEARCH_PLAN_DELIVERABLE_CHAIN = (
    'bn_dataset',
    'benchmarked_models',
    'ranked_candidates',
    'structure_handoff',
    'technical_report',
)

REQUIRED_RESEARCH_PLAN_DELIVERABLES = set(REQUIRED_RESEARCH_PLAN_DELIVERABLE_CHAIN)

REQUIRED_RESEARCH_PLAN_NON_CLAIMS = {
    'open_ended_material_discovery',
    'experimental_synthesis_proof',
    'formula_stage_structure_dependent_property_claims',
    'direct_gap_claim_before_structure_review',
}

REQUIRED_ENTRYPOINTS = {
    'full_pipeline': {
        'command': 'python3 main.py',
        'writes_artifacts': True,
        'expected_output': 'stdout status summary plus files under artifacts/',
    },
    'fast_smoke': {
        'command': 'python3 main.py --dry-run',
        'writes_artifacts': False,
        'expected_output': 'stdout compatibility report',
    },
    'emit_agent_state': {
        'command': 'python3 main.py --emit-agent-state',
        'writes_artifacts': False,
        'expected_output': 'JSON project state on stdout',
    },
    'write_agent_state': {
        'command': 'python3 main.py --write-agent-state /tmp/aiforbn-agent-state.json',
        'writes_artifacts': True,
        'expected_output': 'JSON project state on stdout and at the requested output path',
    },
    'emit_agent_commands': {
        'command': 'python3 main.py --emit-agent-commands',
        'writes_artifacts': False,
        'expected_output': 'JSON entrypoint and validation-command index on stdout',
    },
    'verify_agent_contract': {
        'command': 'python3 main.py --verify-agent-contract',
        'writes_artifacts': False,
        'expected_output': (
            'JSON project state on stdout; nonzero only for blocking layout errors'
        ),
    },
}

REQUIRED_ENTRYPOINT_NAMES = set(REQUIRED_ENTRYPOINTS)

REQUIRED_VALIDATION_COMMANDS = {
    'verify_agent_contract': {
        'command': 'python3 main.py --verify-agent-contract',
        'scope': 'layout_manifest_skill_dependency_checks_and_git_state_reporting',
        'provides': [
            'agent_contract',
            'dependency_declarations',
            'dependency_import_availability',
            'project_skill_metadata',
        ],
    },
    'fast_smoke': {
        'command': 'python3 main.py --dry-run',
        'scope': 'config_candidate_features_model_imports',
        'provides': ['pipeline_wiring_smoke'],
    },
    'focused_regression': {
        'command': (
            'python3 -m pytest -q src/tests/test_main.py '
            'src/tests/test_public_surfaces.py '
            'src/runtime/tests/test_agent_state.py '
            'src/runtime/tests/test_io_utils.py'
        ),
        'scope': 'entrypoints_runtime_agent_state_public_surfaces',
        'provides': ['entrypoint_runtime_public_surface_regressions'],
    },
    'full_src_tests': {
        'command': 'python3 -m pytest -q src',
        'scope': 'complete_src_test_suite_with_declared_dependencies',
        'provides': ['complete_src_test_suite'],
    },
    'ui_render_smoke': {
        'command': 'python3 -m pytest -q src/ui/tests/test_streamlit_app.py',
        'scope': 'real_streamlit_render_and_artifact_viewer_contract',
        'provides': ['streamlit_renderer_contract'],
    },
}

REQUIRED_VALIDATION_COMMAND_NAMES = set(REQUIRED_VALIDATION_COMMANDS)

REQUIRED_VALIDATION_PROFILES = {
    'architecture_doc_skill_edit': {
        'use_when': (
            'agent contract, project skills, handoff, docs, or entrypoint metadata changed'
        ),
        'requires': [
            'agent_contract',
            'dependency_declarations',
            'dependency_import_availability',
            'entrypoint_runtime_public_surface_regressions',
            'pipeline_wiring_smoke',
            'project_skill_metadata',
        ],
    },
    'module_logic_edit': {
        'use_when': (
            'public Python functions, module boundaries, feature/model logic, '
            'or artifact writers changed'
        ),
        'requires': [
            'agent_contract',
            'complete_src_test_suite',
            'dependency_declarations',
            'dependency_import_availability',
            'entrypoint_runtime_public_surface_regressions',
            'pipeline_wiring_smoke',
        ],
    },
    'scientific_pipeline_edit': {
        'use_when': (
            'default pipeline behavior or generated scientific artifacts changed; '
            'run full_pipeline only when the task needs regenerated artifacts'
        ),
        'requires': [
            'agent_contract',
            'complete_src_test_suite',
            'dependency_declarations',
            'dependency_import_availability',
            'pipeline_wiring_smoke',
        ],
    },
    'ui_edit': {
        'use_when': (
            'Streamlit imports, rendering, artifact display, or UI dependency wiring changed'
        ),
        'requires': [
            'agent_contract',
            'dependency_declarations',
            'dependency_import_availability',
            'entrypoint_runtime_public_surface_regressions',
            'streamlit_renderer_contract',
        ],
    },
}

REQUIRED_SOURCE_OF_TRUTH_FILES = {
    'AGENTS.md',
    '.agents/skills/aiforbn-workflow/SKILL.md',
    '.agents/skills/aiforbn-overleaf-proposal/SKILL.md',
    'docs/AGENT_MANIFEST.json',
    'docs/HANDOFF.md',
    'docs/PY_FILES_SUMMARY.md',
    'skills/ai_native_workflow.txt',
}

REQUIRED_PROJECT_SKILLS = [
    {
        'name': 'aiforbn-workflow',
        'path': '.agents/skills/aiforbn-workflow/SKILL.md',
        'scope': 'repo_scoped_codex_skill',
        'status': 'active',
    },
    {
        'name': 'aiforbn-overleaf-proposal',
        'path': '.agents/skills/aiforbn-overleaf-proposal/SKILL.md',
        'scope': 'repo_scoped_codex_skill',
        'status': 'active',
    },
    {
        'name': 'ai_native_workflow',
        'path': 'skills/ai_native_workflow.txt',
        'scope': 'plain_text_agent_runtime_guidance',
        'status': 'active',
    },
]

REQUIRED_RETIRED_GUIDANCE_FILES = [
    'skills/codex_skill.txt',
    'skills/coding_skill.txt',
    'skills/docs_skill.txt',
    'skills/model_skill.txt',
    'skills/python_skill.txt',
    'skills/template.txt',
    'skills/workflow.txt',
]


REQUIRED_MODULE_CONTRACTS = {
    'runtime': {
        'name': 'runtime',
        'path': 'src/runtime',
        'role': 'config_loading_runtime_dirs_cache_schema_agent_state_and_command_index',
        'public_surface': 'src/runtime/PY_FILES_SUMMARY.md',
        'agent_rules': 'src/runtime/AGENTS.md',
        'local_utils': 'src/runtime/utils.py',
        'allowed_dependencies': [],
    },
    'materials': {
        'name': 'materials',
        'path': 'src/materials',
        'role': 'materials_data_features_models_benchmarks_screening_reporting_structure_handoff',
        'public_surface': 'src/materials/PY_FILES_SUMMARY.md',
        'agent_rules': 'src/materials/AGENTS.md',
        'local_utils': 'src/materials/utils.py',
        'allowed_dependencies': ['runtime', 'torch_models'],
    },
    'torch_models': {
        'name': 'torch_models',
        'path': 'src/torch_models',
        'role': 'repo_local_sklearn_style_torch_regressors',
        'public_surface': 'src/torch_models/PY_FILES_SUMMARY.md',
        'agent_rules': 'src/torch_models/AGENTS.md',
        'local_utils': 'src/torch_models/utils.py',
        'allowed_dependencies': [],
    },
    'ui': {
        'name': 'ui',
        'path': 'src/ui',
        'role': 'text_verifiable_streamlit_artifact_viewer',
        'public_surface': 'src/ui/PY_FILES_SUMMARY.md',
        'agent_rules': 'src/ui/AGENTS.md',
        'local_utils': 'src/ui/utils.py',
        'allowed_dependencies': ['runtime'],
    },
    'tests': {
        'name': 'tests',
        'path': 'src/tests',
        'role': 'top_level_pytest_entrypoint_and_config_coverage',
        'public_surface': 'src/tests/PY_FILES_SUMMARY.md',
        'agent_rules': 'src/tests/AGENTS.md',
        'local_utils': 'src/tests/utils.py',
        'allowed_dependencies': [],
    },
    'template': {
        'name': 'template',
        'path': 'src/template',
        'role': 'copyable_ai_native_module_template',
        'public_surface': 'src/template/PY_FILES_SUMMARY.md',
        'agent_rules': 'src/template/AGENTS.md',
        'local_utils': 'src/template/utils.py',
        'allowed_dependencies': [],
    },
}
REQUIRED_MODULE_NAMES = set(REQUIRED_MODULE_CONTRACTS)

HUMAN_DOCS_POLICY_MARKER = (
    'HUMAN_DOCS_POLICY=user_owned_read_only_unless_explicit_human_document_task'
)

REQUIRED_HUMAN_DOCS_POLICY = {
    'policy_id': 'user_owned_read_only_unless_explicit_human_document_task',
    'path': 'human_docs/',
    'owner': 'user',
    'default_access': 'read_only',
    'write_condition': 'explicit_human_document_task',
    'agent_contract_authority': False,
}

BACKTICKED_LOCAL_PATH_PATTERN = re.compile(r'`((?:/[^`\n]+)|(?:~/[^`\n]+))`')
SKILL_FRONTMATTER_PATTERN = re.compile(
    r'\A---\r?\n(?P<body>.*?)\r?\n---(?:\r?\n|\Z)',
    re.DOTALL,
)
REQUIREMENT_DECLARATION_PATTERN = re.compile(
    r'^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)'
    r'(?P<specifier>(?:(?:===|~=|==|!=|<=|>=|<|>)[^,;]+'
    r'(?:,(?:===|~=|==|!=|<=|>=|<|>)[^,;]+)*)?)$'
)
DEPENDENCY_ROLES = {
    'core_runtime',
    'scientific_pipeline',
    'optional_lazy',
    'ui',
    'test_tool',
}
DEPENDENCY_IMPORT_KINDS = {'direct', 'backend'}
IMPORT_MODULE_PATTERN = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$'
)


def _project_root(path: str | Path = '.') -> Path:
    return Path(path).expanduser().resolve()


def _read_text_if_present(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    return path.read_text(encoding='utf-8')


def _skill_frontmatter_fields(text: str) -> dict[str, str]:
    match = SKILL_FRONTMATTER_PATTERN.match(text)
    if match is None:
        return {}
    fields: dict[str, str] = {}
    for line in match.group('body').splitlines():
        field_match = re.match(r'^([A-Za-z][A-Za-z0-9_-]*):\s*(.*?)\s*$', line)
        if field_match is None:
            continue
        key, value = field_match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        fields[key] = value.strip()
    return fields


def _normalized_requirement_name(value: str) -> str:
    return re.sub(r'[-_.]+', '-', value).casefold()


def _normalized_requirement_specifier(value: str) -> str:
    return re.sub(r'\s+', '', value)


def _declared_requirements(
    text: str,
) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    declarations: dict[str, dict[str, str]] = {}
    duplicate_names: list[str] = []
    invalid_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        compact_line = re.sub(r'\s+', '', line)
        match = REQUIREMENT_DECLARATION_PATTERN.fullmatch(compact_line)
        if match is None:
            invalid_lines.append(line)
            continue
        package_name = match.group('name')
        normalized_name = _normalized_requirement_name(package_name)
        if normalized_name in declarations:
            duplicate_names.append(normalized_name)
            continue
        declarations[normalized_name] = {
            'package': package_name,
            'specifier': _normalized_requirement_specifier(
                match.group('specifier') or ''
            ),
        }
    return declarations, sorted(set(duplicate_names)), invalid_lines


def _project_python_source_paths(root: Path) -> list[Path]:
    root_sources = [
        path
        for path in (root / 'main.py', root / 'conftest.py')
        if path.is_file()
    ]
    src_root = root / 'src'
    src_sources = sorted(src_root.rglob('*.py')) if src_root.is_dir() else []
    return [*root_sources, *src_sources]


def _source_import_roots(source_text: str) -> set[str]:
    tree = ast.parse(source_text)
    import_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_roots.update(alias.name.split('.', 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            import_roots.add(node.module.split('.', 1)[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'import_module'
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            import_roots.add(node.args[0].value.split('.', 1)[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == '_bind_missing'
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            import_roots.add(node.args[1].value.split('.', 1)[0])
    return import_roots


def _source_import_consumers(
    root: Path,
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    consumers: dict[str, list[str]] = {}
    parse_errors: list[tuple[str, str]] = []
    for source_path in _project_python_source_paths(root):
        relative_path = str(source_path.relative_to(root))
        try:
            import_roots = _source_import_roots(_read_text_if_present(source_path))
        except SyntaxError as exc:
            parse_errors.append((relative_path, str(exc)))
            continue
        for module_name in import_roots:
            consumers.setdefault(module_name, []).append(relative_path)
    return {
        module_name: sorted(set(paths))
        for module_name, paths in consumers.items()
    }, parse_errors


def _project_import_roots(root: Path) -> set[str]:
    src_root = root / 'src'
    roots = {'main', 'conftest', 'src'}
    if src_root.is_dir():
        roots.update(path.stem for path in src_root.glob('*.py'))
        roots.update(
            path.name
            for path in src_root.iterdir()
            if path.is_dir() and not path.name.startswith('.')
        )
    return roots


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


def _validate_command_entries(
    manifest_payload: dict[str, Any],
    field_name: str,
    errors: list[dict[str, str]],
) -> set[str]:
    value = manifest_payload.get(field_name, [])
    command_names: set[str] = set()
    if not isinstance(value, list) or not value:
        errors.append({
            'code': f'invalid_{field_name}',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': f'Manifest field `{field_name}` must be a non-empty list.',
        })
        return command_names

    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            errors.append({
                'code': f'invalid_{field_name}_entry',
                'path': f'docs/AGENT_MANIFEST.json:{field_name}[{index}]',
                'message': f'Every `{field_name}` entry must be a JSON object.',
            })
            continue
        name = str(entry.get('name', '')).strip()
        command = str(entry.get('command', '')).strip()
        if not name:
            errors.append({
                'code': f'missing_{field_name}_name',
                'path': f'docs/AGENT_MANIFEST.json:{field_name}[{index}]',
                'message': f'Every `{field_name}` entry needs a stable `name`.',
            })
        elif name in command_names:
            errors.append({
                'code': f'duplicate_{field_name}_name',
                'path': f'docs/AGENT_MANIFEST.json:{field_name}[{index}]',
                'message': f'Duplicate `{field_name}` command name: {name}',
            })
        else:
            command_names.add(name)
        if not command:
            errors.append({
                'code': f'missing_{field_name}_command',
                'path': f'docs/AGENT_MANIFEST.json:{field_name}[{index}]',
                'message': f'Every `{field_name}` entry needs a non-empty `command`.',
            })
    return command_names


def _validate_required_entrypoint_contracts(
    manifest_payload: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    entries = manifest_payload.get('entrypoints', [])
    if not isinstance(entries, list):
        return
    entries_by_name = {
        str(entry.get('name', '')).strip(): entry
        for entry in entries
        if isinstance(entry, dict) and str(entry.get('name', '')).strip()
    }
    for name, required_contract in REQUIRED_ENTRYPOINTS.items():
        entry = entries_by_name.get(name)
        if entry is None:
            continue
        actual_contract = {
            field: entry.get(field)
            for field in ('command', 'writes_artifacts', 'expected_output')
        }
        if actual_contract != required_contract:
            errors.append({
                'code': 'unexpected_entrypoint_contract',
                'path': f'docs/AGENT_MANIFEST.json:entrypoints.{name}',
                'message': (
                    f'Entrypoint `{name}` must retain its exact command, write '
                    'behavior, and expected-output contract.'
                ),
            })


def _validate_required_validation_command_contracts(
    manifest_payload: dict[str, Any],
    errors: list[dict[str, str]],
) -> dict[str, set[str]]:
    entries = manifest_payload.get('validation_commands', [])
    if not isinstance(entries, list):
        return {}
    entries_by_name = {
        str(entry.get('name', '')).strip(): entry
        for entry in entries
        if isinstance(entry, dict) and str(entry.get('name', '')).strip()
    }
    capabilities_by_name: dict[str, set[str]] = {}
    for name, required_contract in REQUIRED_VALIDATION_COMMANDS.items():
        entry = entries_by_name.get(name)
        if entry is None:
            continue
        actual_contract = {
            field: entry.get(field)
            for field in ('command', 'scope', 'provides')
        }
        if actual_contract != required_contract:
            errors.append({
                'code': 'unexpected_validation_command_contract',
                'path': f'docs/AGENT_MANIFEST.json:validation_commands.{name}',
                'message': (
                    f'Validation command `{name}` must retain its exact command, '
                    'scope, and provided-capability contract.'
                ),
            })
        provides = entry.get('provides', [])
        if isinstance(provides, list) and all(
            isinstance(capability, str) and capability.strip()
            for capability in provides
        ):
            capabilities_by_name[name] = {
                capability.strip() for capability in provides
            }
    return capabilities_by_name


def _validate_local_instruction_paths(
    root: Path,
    manifest_payload: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    relative_paths: list[str] = []
    for field_name, path_key in (('project_skills', 'path'), ('modules', 'agent_rules')):
        entries = manifest_payload.get(field_name, [])
        if not isinstance(entries, list):
            continue
        relative_paths.extend(
            str(entry.get(path_key, '')).strip()
            for entry in entries
            if isinstance(entry, dict) and str(entry.get(path_key, '')).strip()
        )

    for relative_path in dict.fromkeys(relative_paths):
        instruction_path = root / relative_path
        text = _read_text_if_present(instruction_path)
        for raw_path in BACKTICKED_LOCAL_PATH_PATTERN.findall(text):
            local_path = Path(raw_path).expanduser()
            if local_path.exists():
                continue
            errors.append({
                'code': 'stale_local_instruction_path',
                'path': relative_path,
                'message': (
                    f'Agent instruction references a missing local path: {raw_path}'
                ),
            })


def _validate_dependency_contract(
    root: Path,
    manifest_payload: dict[str, Any],
    errors: list[dict[str, Any]],
    checks: list[dict[str, object]],
) -> None:
    def add_error(code: str, path: str, message: str, **extra: str) -> None:
        errors.append({'code': code, 'path': path, 'message': message, **extra})

    requirements, duplicate_requirements, invalid_requirements = _declared_requirements(
        _read_text_if_present(root / 'requirements.txt')
    )
    for name in duplicate_requirements:
        add_error('duplicate_requirement', 'requirements.txt', f'Duplicate normalized requirement: {name}.')
    for line in invalid_requirements:
        add_error('invalid_requirement_declaration', 'requirements.txt', f'Unsupported requirement declaration: {line!r}.')

    entries = manifest_payload.get('dependency_imports')
    if not isinstance(entries, list) or not entries:
        add_error('invalid_dependency_imports', 'docs/AGENT_MANIFEST.json:dependency_imports', '`dependency_imports` must be a non-empty object list.')
        entries = []
    dependencies: dict[str, dict[str, str]] = {}
    module_owners: dict[str, str] = {}
    for index, entry in enumerate(entries):
        path = f'docs/AGENT_MANIFEST.json:dependency_imports[{index}]'
        if not isinstance(entry, dict):
            add_error('invalid_dependency_import_entry', path, 'Dependency entry must be an object.')
            continue
        record = {
            field: str(entry.get(field, '')).strip()
            for field in ('package', 'specifier', 'module', 'role', 'import_kind', 'required_for')
        }
        record['specifier'] = _normalized_requirement_specifier(record['specifier'])
        if not all(record[field] for field in ('package', 'module', 'required_for')) or not IMPORT_MODULE_PATTERN.fullmatch(record['module']):
            add_error('invalid_dependency_import_entry', path, 'Dependency entry requires package, importable module, and purpose.')
            continue
        if record['role'] not in DEPENDENCY_ROLES:
            add_error('invalid_dependency_role', path, f'Unsupported dependency role: {record["role"]}.')
            continue
        if record['import_kind'] not in DEPENDENCY_IMPORT_KINDS:
            add_error('invalid_dependency_import_kind', path, f'Unsupported dependency import kind: {record["import_kind"]}.')
            continue
        package_key = _normalized_requirement_name(record['package'])
        if package_key in dependencies:
            add_error('duplicate_dependency_package', path, f'Duplicate normalized dependency: {package_key}.')
            continue
        if record['module'] in module_owners:
            add_error('duplicate_dependency_module', path, f'Duplicate dependency module: {record["module"]}.')
            continue
        dependencies[package_key] = record
        module_owners[record['module']] = record['package']

    for key in sorted(set(dependencies) - set(requirements)):
        add_error('missing_dependency_requirement', 'requirements.txt', f'Manifest dependency `{dependencies[key]["package"]}` is not required.')
    for key in sorted(set(requirements) - set(dependencies)):
        add_error('unmanifested_requirement', 'docs/AGENT_MANIFEST.json:dependency_imports', f'Requirement `{requirements[key]["package"]}` has no dependency record.')

    local_entries = manifest_payload.get('local_shared_imports')
    if not isinstance(local_entries, list) or not local_entries:
        add_error('invalid_local_shared_imports', 'docs/AGENT_MANIFEST.json:local_shared_imports', '`local_shared_imports` must be a non-empty object list.')
        local_entries = []
    local_modules: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(local_entries):
        path = f'docs/AGENT_MANIFEST.json:local_shared_imports[{index}]'
        if not isinstance(entry, dict):
            add_error('invalid_local_shared_import_entry', path, 'Local shared import entry must be an object.')
            continue
        module = str(entry.get('module', '')).strip()
        owner = str(entry.get('owner', '')).strip()
        purpose = str(entry.get('required_for', '')).strip()
        if not IMPORT_MODULE_PATTERN.fullmatch(module) or not owner or not purpose:
            add_error('invalid_local_shared_import_entry', path, 'Local shared import requires module, owner, and purpose.')
            continue
        if module in local_modules or module in module_owners:
            add_error('duplicate_local_shared_import_module', path, f'Duplicate import owner for `{module}`.')
            continue
        local_modules[module] = {'owner': owner, 'required_for': purpose}

    consumers_by_module, parse_errors = _source_import_consumers(root)
    for path, detail in parse_errors:
        add_error('dependency_source_parse_error', path, f'Cannot classify imports: {detail}')
    external_modules = {module.split('.', 1)[0] for module in module_owners}
    shared_modules = {module.split('.', 1)[0] for module in local_modules}
    classified = set(sys.stdlib_module_names) | {'__future__'} | _project_import_roots(root) | external_modules | shared_modules
    unclassified = sorted(set(consumers_by_module) - classified)
    for module in unclassified:
        for path in consumers_by_module[module]:
            add_error('undeclared_external_import', path, f'Unclassified external import: {module}.', module=module)

    for package_key, record in dependencies.items():
        declaration = requirements.get(package_key)
        matches = declaration is not None and declaration['specifier'] == record['specifier']
        checks.append({
            'kind': 'dependency_requirement', 'package': record['package'],
            'specifier': record['specifier'], 'declared_specifier': declaration['specifier'] if declaration else None,
            'declared': declaration is not None, 'matches': matches, 'role': record['role'],
        })
        if declaration is not None and not matches:
            add_error('dependency_specifier_mismatch', 'requirements.txt', f'Specifier mismatch for `{record["package"]}`.')
        module = record['module']
        consumers = consumers_by_module.get(module.split('.', 1)[0], [])
        checks.append({
            'kind': 'dependency_source_imports', 'package': record['package'],
            'module': module, 'role': record['role'], 'import_kind': record['import_kind'],
            'consumers': consumers,
        })
        if record['import_kind'] == 'direct' and not consumers:
            add_error('dependency_module_not_imported', 'docs/AGENT_MANIFEST.json:dependency_imports', f'Direct dependency `{module}` has no source consumer.', module=module)
        if record['role'] == 'test_tool':
            production_consumers = [path for path in consumers if '/tests/' not in f'/{path}/' and not Path(path).name.startswith('test_') and Path(path).name != 'conftest.py']
            if production_consumers:
                add_error('test_tool_imported_by_production', production_consumers[0], f'Test tool `{module}` has production consumers.', module=module)
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, AttributeError, ValueError):
            available = False
        checks.append({
            'kind': 'dependency_import', 'package': record['package'], 'module': module,
            'required_for': record['required_for'], 'role': record['role'], 'available': available,
        })
        if not available:
            add_error('missing_declared_dependency', 'requirements.txt', f'`{record["package"]}` is not importable as `{module}`.', module=module)

    for module, record in sorted(local_modules.items()):
        consumers = consumers_by_module.get(module.split('.', 1)[0], [])
        checks.append({
            'kind': 'local_shared_source_imports', 'module': module, 'owner': record['owner'],
            'required_for': record['required_for'], 'consumers': consumers,
        })
        if not consumers:
            add_error('local_shared_module_not_imported', 'docs/AGENT_MANIFEST.json:local_shared_imports', f'Local shared module `{module}` has no consumer.', module=module)
    checks.append({
        'kind': 'source_import_classification', 'source_file_count': len(_project_python_source_paths(root)),
        'external_modules': sorted(external_modules), 'local_shared_modules': sorted(shared_modules),
        'unclassified_modules': unclassified,
    })


def _validate_human_docs_policy(
    root: Path,
    manifest_payload: dict[str, Any],
    errors: list[dict[str, str]],
    checks: list[dict[str, object]],
) -> None:
    policy = manifest_payload.get('human_docs_policy')
    if not isinstance(policy, dict):
        errors.append({
            'code': 'invalid_human_docs_policy',
            'path': 'docs/AGENT_MANIFEST.json:human_docs_policy',
            'message': 'Manifest field `human_docs_policy` must be a JSON object.',
        })
    else:
        mismatches = {
            key: {'expected': expected, 'actual': policy.get(key)}
            for key, expected in REQUIRED_HUMAN_DOCS_POLICY.items()
            if policy.get(key) != expected
        }
        if mismatches:
            errors.append({
                'code': 'unexpected_human_docs_policy',
                'path': 'docs/AGENT_MANIFEST.json:human_docs_policy',
                'message': (
                    'Human-document ownership policy must match the required '
                    f'user-owned read-only contract; mismatches: {mismatches}'
                ),
            })

    human_docs_check = _path_check(root, REQUIRED_HUMAN_DOCS_POLICY['path'])
    checks.append({**human_docs_check, 'kind': 'human_docs_root'})
    if not human_docs_check['is_dir']:
        errors.append({
            'code': 'missing_human_docs_root',
            'path': REQUIRED_HUMAN_DOCS_POLICY['path'],
            'message': 'The declared user-owned human_docs/ root is missing or not a directory.',
        })

    policy_surfaces: list[str] = []
    source_of_truth_files = manifest_payload.get('source_of_truth_files', [])
    if isinstance(source_of_truth_files, list):
        policy_surfaces.extend(
            path.strip()
            for path in source_of_truth_files
            if isinstance(path, str)
            and path.strip()
            and path.strip() != str(DEFAULT_AGENT_MANIFEST_PATH)
        )
    modules = manifest_payload.get('modules', [])
    if isinstance(modules, list):
        for module in modules:
            if not isinstance(module, dict):
                continue
            policy_surfaces.extend(
                path
                for field_name in ('agent_rules', 'public_surface')
                if (path := str(module.get(field_name, '')).strip())
            )

    for relative_path in dict.fromkeys(policy_surfaces):
        check = _path_check(root, relative_path)
        checks.append({**check, 'kind': 'human_docs_policy_surface'})
        if not check['is_file']:
            continue
        if HUMAN_DOCS_POLICY_MARKER in _read_text_if_present(root / relative_path):
            continue
        errors.append({
            'code': 'missing_human_docs_policy_marker',
            'path': relative_path,
            'message': (
                'Declared agent instruction surface is missing the stable '
                f'human-document policy marker: {HUMAN_DOCS_POLICY_MARKER}'
            ),
        })


def _validate_research_plan_alignment(
    root: Path,
    manifest_payload: dict[str, Any],
    errors: list[dict[str, str]],
    checks: list[dict[str, object]],
) -> None:
    alignment = manifest_payload.get('research_plan_alignment')
    if not isinstance(alignment, dict):
        errors.append({
            'code': 'invalid_research_plan_alignment',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `research_plan_alignment` must be a JSON object.',
        })
        return

    if alignment.get('status') != REQUIRED_RESEARCH_PLAN_ALIGNMENT_STATUS:
        errors.append({
            'code': 'unexpected_research_plan_alignment_status',
            'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.status',
            'message': (
                '`research_plan_alignment.status` must be '
                f'`{REQUIRED_RESEARCH_PLAN_ALIGNMENT_STATUS}`.'
            ),
        })

    source_files = alignment.get('source_files', [])
    if not isinstance(source_files, list) or not source_files:
        errors.append({
            'code': 'invalid_research_plan_alignment_sources',
            'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.source_files',
            'message': '`research_plan_alignment.source_files` must be a non-empty string list.',
        })
    else:
        for index, source_file in enumerate(source_files):
            if not isinstance(source_file, str):
                errors.append({
                    'code': 'invalid_research_plan_alignment_source',
                    'path': f'docs/AGENT_MANIFEST.json:research_plan_alignment.source_files[{index}]',
                    'message': 'Every research-plan alignment source path must be a string.',
                })
                continue
            relative_path = source_file.strip()
            if not relative_path:
                errors.append({
                    'code': 'missing_research_plan_alignment_source',
                    'path': f'docs/AGENT_MANIFEST.json:research_plan_alignment.source_files[{index}]',
                    'message': 'Every research-plan alignment source needs a non-empty path.',
                })
                continue
            check = _path_check(root, relative_path)
            checks.append({**check, 'kind': 'research_plan_alignment_source'})
            if not check['is_file']:
                errors.append({
                    'code': 'missing_research_plan_alignment_source_file',
                    'path': relative_path,
                    'message': f'Research-plan alignment source is missing or not a file: {relative_path}',
                })
        if all(isinstance(source_file, str) for source_file in source_files) and (
            source_files != list(REQUIRED_RESEARCH_PLAN_SOURCE_FILES)
        ):
            errors.append({
                'code': 'unexpected_research_plan_alignment_sources',
                'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.source_files',
                'message': (
                    '`research_plan_alignment.source_files` must exactly match '
                    f'{list(REQUIRED_RESEARCH_PLAN_SOURCE_FILES)}.'
                ),
            })

    anchors = alignment.get('implementation_anchors', [])
    if not isinstance(anchors, list) or not all(
        isinstance(anchor, str) and anchor.strip() for anchor in anchors
    ):
        errors.append({
            'code': 'invalid_research_plan_alignment_anchors',
            'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.implementation_anchors',
            'message': '`research_plan_alignment.implementation_anchors` must be a string list.',
        })
    else:
        missing_anchors = sorted(REQUIRED_RESEARCH_PLAN_ALIGNMENT_ANCHORS - set(anchors))
        if missing_anchors:
            errors.append({
                'code': 'missing_research_plan_alignment_anchors',
                'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.implementation_anchors',
                'message': f'Missing v18 alignment anchors: {missing_anchors}',
            })

    deliverable_chain = alignment.get('deliverable_chain', [])
    if not isinstance(deliverable_chain, list) or not all(
        isinstance(deliverable, str) and deliverable.strip()
        for deliverable in deliverable_chain
    ):
        errors.append({
            'code': 'invalid_research_plan_deliverable_chain',
            'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.deliverable_chain',
            'message': '`research_plan_alignment.deliverable_chain` must be a string list.',
        })
    else:
        missing_deliverables = sorted(
            REQUIRED_RESEARCH_PLAN_DELIVERABLES - set(deliverable_chain)
        )
        if missing_deliverables:
            errors.append({
                'code': 'missing_research_plan_deliverables',
                'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.deliverable_chain',
                'message': f'Missing v18 deliverable-chain entries: {missing_deliverables}',
            })
        elif deliverable_chain != list(REQUIRED_RESEARCH_PLAN_DELIVERABLE_CHAIN):
            errors.append({
                'code': 'unexpected_research_plan_deliverable_chain',
                'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.deliverable_chain',
                'message': (
                    '`research_plan_alignment.deliverable_chain` must exactly match '
                    f'{list(REQUIRED_RESEARCH_PLAN_DELIVERABLE_CHAIN)}.'
                ),
            })

    non_claims = alignment.get('non_claims', [])
    if not isinstance(non_claims, list) or not all(
        isinstance(non_claim, str) and non_claim.strip() for non_claim in non_claims
    ):
        errors.append({
            'code': 'invalid_research_plan_non_claims',
            'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.non_claims',
            'message': '`research_plan_alignment.non_claims` must be a string list.',
        })
    else:
        missing_non_claims = sorted(REQUIRED_RESEARCH_PLAN_NON_CLAIMS - set(non_claims))
        if missing_non_claims:
            errors.append({
                'code': 'missing_research_plan_non_claims',
                'path': 'docs/AGENT_MANIFEST.json:research_plan_alignment.non_claims',
                'message': f'Missing v18 non-claim safety boundaries: {missing_non_claims}',
            })


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

    project_payload = manifest_payload.get('project', {})
    if not isinstance(project_payload, dict):
        errors.append({
            'code': 'invalid_manifest_project',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `project` must be a JSON object.',
        })
        project_payload = {}
    if project_payload.get('manual_operation_supported') is not False:
        errors.append({
            'code': 'manual_operation_not_disabled',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'AI-native contract requires `project.manual_operation_supported` to be false.',
        })
    if project_payload.get('primary_entrypoint') != 'AGENTS.md':
        errors.append({
            'code': 'primary_entrypoint_not_agents_md',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'AI-native contract requires `project.primary_entrypoint` to be AGENTS.md.',
        })

    _validate_research_plan_alignment(root, manifest_payload, errors, checks)

    entrypoint_names = _validate_command_entries(manifest_payload, 'entrypoints', errors)
    _validate_required_entrypoint_contracts(manifest_payload, errors)
    missing_entrypoints = sorted(REQUIRED_ENTRYPOINT_NAMES - entrypoint_names)
    if missing_entrypoints:
        errors.append({
            'code': 'missing_required_entrypoints',
            'path': 'docs/AGENT_MANIFEST.json:entrypoints',
            'message': f'Missing required agent entrypoint names: {missing_entrypoints}',
        })
    validation_command_names = _validate_command_entries(
        manifest_payload,
        'validation_commands',
        errors,
    )
    validation_command_capabilities = (
        _validate_required_validation_command_contracts(
            manifest_payload,
            errors,
        )
    )
    missing_validation_commands = sorted(
        REQUIRED_VALIDATION_COMMAND_NAMES - validation_command_names
    )
    if missing_validation_commands:
        errors.append({
            'code': 'missing_required_validation_commands',
            'path': 'docs/AGENT_MANIFEST.json:validation_commands',
            'message': (
                f'Missing required validation command names: {missing_validation_commands}'
            ),
        })
    validation_profiles = manifest_payload.get('validation_profiles', [])
    if not isinstance(validation_profiles, list) or not validation_profiles:
        errors.append({
            'code': 'invalid_validation_profiles',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `validation_profiles` must be a non-empty list.',
        })
    else:
        seen_profiles: set[str] = set()
        for index, profile in enumerate(validation_profiles):
            if not isinstance(profile, dict):
                errors.append({
                    'code': 'invalid_validation_profile',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': 'Every validation profile must be a JSON object.',
                })
                continue
            profile_name = str(profile.get('name', '')).strip()
            if not profile_name:
                errors.append({
                    'code': 'missing_validation_profile_name',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': 'Every validation profile needs a stable `name`.',
                })
            elif profile_name in seen_profiles:
                errors.append({
                    'code': 'duplicate_validation_profile_name',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': f'Duplicate validation profile name: {profile_name}',
                })
            else:
                seen_profiles.add(profile_name)
            use_when = str(profile.get('use_when', '')).strip()
            if not use_when:
                errors.append({
                    'code': 'missing_validation_profile_use_when',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': 'Every validation profile needs a non-empty `use_when` selector.',
                })
            commands = profile.get('commands', [])
            if not isinstance(commands, list) or not commands or not all(
                isinstance(command, str) and command.strip() for command in commands
            ):
                errors.append({
                    'code': 'invalid_validation_profile_commands',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': 'Validation profile `commands` must be a non-empty string list.',
                })
                continue
            if len(commands) != len(set(commands)):
                errors.append({
                    'code': 'duplicate_validation_profile_commands',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': (
                        f'Validation profile `{profile_name}` repeats a command name.'
                    ),
                })
            missing_commands = sorted(set(commands) - validation_command_names)
            if missing_commands:
                errors.append({
                    'code': 'validation_profile_unknown_command',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': (
                        f'Validation profile `{profile_name}` references unknown '
                        f'validation command names: {missing_commands}'
                    ),
                })
            requires = profile.get('requires', [])
            if not isinstance(requires, list) or not requires or not all(
                isinstance(capability, str) and capability.strip()
                for capability in requires
            ):
                errors.append({
                    'code': 'invalid_validation_profile_requirements',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': (
                        'Validation profile `requires` must be a non-empty string list.'
                    ),
                })
                continue
            if len(requires) != len(set(requires)):
                errors.append({
                    'code': 'duplicate_validation_profile_requirements',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': (
                        f'Validation profile `{profile_name}` repeats a required capability.'
                    ),
                })
            provided_capabilities: set[str] = set()
            for command in commands:
                provided_capabilities.update(
                    validation_command_capabilities.get(command, set())
                )
            missing_capabilities = sorted(
                set(requires) - provided_capabilities
            )
            if missing_capabilities:
                errors.append({
                    'code': 'validation_profile_missing_capabilities',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': (
                        f'Validation profile `{profile_name}` does not reach required '
                        f'capabilities: {missing_capabilities}'
                    ),
                })

        profiles_by_name = {
            str(profile.get('name', '')).strip(): profile
            for profile in validation_profiles
            if isinstance(profile, dict) and str(profile.get('name', '')).strip()
        }
        missing_required_profiles = sorted(
            set(REQUIRED_VALIDATION_PROFILES) - set(profiles_by_name)
        )
        if missing_required_profiles:
            errors.append({
                'code': 'missing_required_validation_profiles',
                'path': 'docs/AGENT_MANIFEST.json:validation_profiles',
                'message': (
                    'Missing required validation profile names: '
                    f'{missing_required_profiles}'
                ),
            })
        for profile_name, required_contract in REQUIRED_VALIDATION_PROFILES.items():
            profile = profiles_by_name.get(profile_name)
            if profile is None:
                continue
            actual_contract = {
                field: profile.get(field)
                for field in ('use_when', 'requires')
            }
            if actual_contract != required_contract:
                errors.append({
                    'code': 'unexpected_validation_profile_contract',
                    'path': (
                        'docs/AGENT_MANIFEST.json:'
                        f'validation_profiles.{profile_name}'
                    ),
                    'message': (
                        f'Validation profile `{profile_name}` must retain its exact '
                        '`use_when` selector and required-capability contract.'
                    ),
                })

        workflow_guidance_path = root / 'skills/ai_native_workflow.txt'
        workflow_guidance = _read_text_if_present(workflow_guidance_path)
        missing_guidance_profiles = sorted(
            profile_name
            for profile_name in REQUIRED_VALIDATION_PROFILES
            if f'`{profile_name}`' not in workflow_guidance
        )
        checks.append({
            'kind': 'validation_profile_guidance',
            'path': 'skills/ai_native_workflow.txt',
            'missing_profiles': missing_guidance_profiles,
        })
        if missing_guidance_profiles:
            errors.append({
                'code': 'missing_validation_profile_guidance',
                'path': 'skills/ai_native_workflow.txt',
                'message': (
                    'Active compact guidance must route through every emitted '
                    f'validation profile name; missing: {missing_guidance_profiles}'
                ),
            })

    source_of_truth_files = manifest_payload.get('source_of_truth_files', [])
    if not isinstance(source_of_truth_files, list) or not all(
        isinstance(path, str) and path.strip() for path in source_of_truth_files
    ):
        errors.append({
            'code': 'invalid_source_of_truth_files',
            'path': 'docs/AGENT_MANIFEST.json:source_of_truth_files',
            'message': '`source_of_truth_files` must be a non-empty string list.',
        })
        source_of_truth_files = []
    else:
        missing_source_files = sorted(
            REQUIRED_SOURCE_OF_TRUTH_FILES - set(source_of_truth_files)
        )
        if missing_source_files:
            errors.append({
                'code': 'missing_source_of_truth_files',
                'path': 'docs/AGENT_MANIFEST.json:source_of_truth_files',
                'message': f'Missing required source-of-truth files: {missing_source_files}',
            })

    required_paths = list(source_of_truth_files)
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
    else:
        module_name_list = [
            module.get('name')
            for module in modules
            if isinstance(module, dict) and isinstance(module.get('name'), str)
        ]
        module_names = set(module_name_list)
        missing_modules = sorted(REQUIRED_MODULE_NAMES - module_names)
        if missing_modules:
            errors.append({
                'code': 'missing_required_modules',
                'path': 'docs/AGENT_MANIFEST.json:modules',
                'message': f'Missing required module contract names: {missing_modules}',
            })
        duplicate_modules = sorted({
            module_name
            for module_name in module_name_list
            if module_name_list.count(module_name) > 1
        })
        if duplicate_modules:
            errors.append({
                'code': 'duplicate_module_contracts',
                'path': 'docs/AGENT_MANIFEST.json:modules',
                'message': f'Duplicate module contract names: {duplicate_modules}',
            })
    for module in modules:
        if not isinstance(module, dict):
            errors.append({
                'code': 'invalid_manifest_module',
                'path': 'docs/AGENT_MANIFEST.json',
                'message': 'Every manifest module entry must be a JSON object.',
            })
            continue
        module_name = module.get('name')
        expected_module = REQUIRED_MODULE_CONTRACTS.get(module_name)
        if expected_module is None or module != expected_module:
            errors.append({
                'code': 'unexpected_module_contract',
                'path': f'docs/AGENT_MANIFEST.json:modules:{module_name or "<unnamed>"}',
                'message': (
                    f'Module `{module_name or "<unnamed>"}` does not match the required '
                    'path, role, public-surface, agent-rules, local-utils, and dependency contract.'
                ),
            })
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

    project_skills = manifest_payload.get('project_skills', [])
    if not isinstance(project_skills, list):
        errors.append({
            'code': 'invalid_project_skills',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `project_skills` must be a list when present.',
        })
        project_skills = []
    if project_skills != REQUIRED_PROJECT_SKILLS:
        errors.append({
            'code': 'unexpected_project_skills',
            'path': 'docs/AGENT_MANIFEST.json:project_skills',
            'message': (
                'Project skill names, paths, scopes, and active statuses must match '
                'the required agent-routing contract.'
            ),
        })
    for index, skill in enumerate(project_skills):
        if not isinstance(skill, dict):
            errors.append({
                'code': 'invalid_project_skill',
                'path': f'docs/AGENT_MANIFEST.json:project_skills[{index}]',
                'message': 'Every project skill entry must be a JSON object.',
            })
            continue
        relative_path = str(skill.get('path', '')).strip()
        if not relative_path:
            errors.append({
                'code': 'missing_project_skill_path',
                'path': str(skill.get('name', '<unnamed>')),
                'message': 'Every project skill entry needs a `path`.',
            })
            continue
        check = _path_check(root, relative_path)
        checks.append({
            **check,
            'kind': 'project_skill',
            'skill': skill.get('name', '<unnamed>'),
        })
        if skill.get('status', 'active') == 'active' and not check['exists']:
            errors.append({
                'code': 'missing_project_skill',
                'path': relative_path,
                'message': f'Active project skill `{skill.get("name", "<unnamed>")}` is missing.',
            })
        if skill.get('scope') == 'repo_scoped_codex_skill' and check['exists']:
            frontmatter = _skill_frontmatter_fields(
                _read_text_if_present(root / relative_path)
            )
            expected_name = str(skill.get('name', '')).strip()
            frontmatter_valid = (
                frontmatter.get('name') == expected_name
                and bool(frontmatter.get('description'))
            )
            checks.append({
                'kind': 'project_skill_frontmatter',
                'path': relative_path,
                'name': expected_name,
                'valid': frontmatter_valid,
            })
            if not frontmatter_valid:
                errors.append({
                    'code': 'unexpected_project_skill_frontmatter',
                    'path': relative_path,
                    'message': (
                        f'Repo-scoped skill frontmatter must declare exact name '
                        f'`{expected_name}` and a non-empty description.'
                    ),
                })

    _validate_human_docs_policy(root, manifest_payload, errors, checks)

    retired_paths = ['skill.txt', *REQUIRED_RETIRED_GUIDANCE_FILES]
    retired_guidance_files = manifest_payload.get('retired_guidance_files', [])
    if not isinstance(retired_guidance_files, list):
        errors.append({
            'code': 'invalid_retired_guidance_files',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `retired_guidance_files` must be a list when present.',
        })
        retired_guidance_files = []
    if retired_guidance_files != REQUIRED_RETIRED_GUIDANCE_FILES:
        errors.append({
            'code': 'unexpected_retired_guidance_files',
            'path': 'docs/AGENT_MANIFEST.json:retired_guidance_files',
            'message': (
                'Retired guidance paths must match the required stale-shard '
                'detection contract.'
            ),
        })
    retired_paths.extend(str(path) for path in retired_guidance_files)
    for relative_path in dict.fromkeys(retired_paths):
        check = _path_check(root, relative_path)
        checks.append({**check, 'kind': 'retired_guidance_path'})
        if check['exists']:
            errors.append({
                'code': 'retired_guidance_file_present',
                'path': relative_path,
                'message': (
                    f'Retired guidance file `{relative_path}` is present; consolidate into '
                    'AGENTS.md, project skills, or docs/AGENT_MANIFEST.json.'
                ),
            })

    _validate_local_instruction_paths(root, manifest_payload, errors)

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

    _validate_dependency_contract(
        root,
        manifest_payload,
        errors,
        checks,
    )

    return {
        'status': 'ok' if not errors else 'error',
        'errors': errors,
        'warnings': warnings,
        'checks': checks,
    }


def build_agent_command_index(
    project_root_path: str | Path = '.',
    manifest_path: str | Path = DEFAULT_AGENT_MANIFEST_PATH,
) -> dict[str, Any]:
    root = _project_root(project_root_path)
    manifest = load_agent_manifest(root, manifest_path=manifest_path)
    return {
        'schema_version': 'aiforbn.agent_command_index.v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'project_root': str(root),
        'manifest_path': str(manifest_path),
        'project': manifest.get('project', {}),
        'first_inspection_command': 'python3 main.py --verify-agent-contract',
        'entrypoints': manifest.get('entrypoints', []),
        'validation_commands': manifest.get('validation_commands', []),
        'validation_profiles': manifest.get('validation_profiles', []),
        'source_of_truth_files': manifest.get('source_of_truth_files', []),
        'project_skills': manifest.get('project_skills', []),
        'modules': manifest.get('modules', []),
        'human_docs_policy': manifest.get('human_docs_policy', {}),
        'retired_guidance_files': manifest.get('retired_guidance_files', []),
        'research_plan_alignment': manifest.get('research_plan_alignment', {}),
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
        ['ls-files', 'human_docs/research_plan'],
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
            'Run python3 main.py --emit-agent-commands to choose the smallest validation profile',
            'Run python3 main.py --verify-agent-contract',
            'Run python3 main.py --dry-run before expensive work',
            'Run focused pytest for touched modules',
            'Update PY_FILES_SUMMARY.md when public surfaces change',
        ],
    }


def agent_state_to_json(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)


def write_agent_state(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser()
    resolved_path = path.resolve(strict=False)
    project_roots = [PROJECT_ROOT.resolve()]
    declared_project_root = _project_root(state.get('project_root', '.'))
    if declared_project_root not in project_roots:
        project_roots.append(declared_project_root)
    for project_root in project_roots:
        human_docs_root = (
            project_root / REQUIRED_HUMAN_DOCS_POLICY['path']
        ).resolve(strict=False)
        if _path_is_same_or_descendant(resolved_path, human_docs_root):
            raise ValueError(
                'Agent runtime state must not be written under user-owned human_docs/'
            )
    if path.is_symlink():
        raise ValueError('Agent runtime state must not target a symbolic-link leaf')
    if path.exists() and not path.is_file():
        raise ValueError('Agent runtime state must target a regular-file leaf')
    invalid_parent = next(
        (
            parent
            for parent in path.parents
            if (parent.exists() or parent.is_symlink()) and not parent.is_dir()
        ),
        None,
    )
    if invalid_parent is not None:
        raise ValueError(
            f'Agent runtime state parent paths must be directories: {invalid_parent}'
        )
    try:
        has_multiple_links = resolved_path.is_file() and resolved_path.stat().st_nlink > 1
    except OSError:
        has_multiple_links = False
    if has_multiple_links:
        raise ValueError(
            'Agent runtime state must not target a file with multiple hard links'
        )
    serialized_state = agent_state_to_json(state) + '\n'
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(serialized_state, encoding='utf-8')
