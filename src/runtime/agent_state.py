from __future__ import annotations

import ast
from datetime import datetime, timezone
from importlib import metadata
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
SKILL_REFERENCE_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_])\$([a-z][a-z0-9]*(?:[-_][a-z0-9]+)*)'
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


def _source_import_analysis(
    source_text: str,
    *,
    relative_path: str,
) -> tuple[set[str], list[int]]:
    tree = ast.parse(source_text)
    parent_by_node = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    scope_types = (
        ast.Module,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
    )

    def lexical_scope(node: ast.AST) -> ast.AST:
        current = node
        while not isinstance(current, scope_types):
            current = parent_by_node[id(current)]
        return current

    def enclosing_scope(scope: ast.AST) -> ast.AST | None:
        current = parent_by_node.get(id(scope))
        while current is not None:
            if isinstance(
                current,
                (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
            ):
                return current
            current = parent_by_node.get(id(current))
        return None

    def target_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, ast.Starred):
            return target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return {
                name
                for child in target.elts
                for name in target_names(child)
            }
        return set()

    comprehension_types = (
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def is_descendant(node: ast.AST, ancestor: ast.AST) -> bool:
        current: ast.AST | None = node
        while current is not None:
            if current is ancestor:
                return True
            current = parent_by_node.get(id(current))
        return False

    def comprehension_bound_names(
        node: ast.AST,
        comprehension: ast.AST,
    ) -> set[str]:
        bound_names: set[str] = set()
        for generator in comprehension.generators:
            if is_descendant(node, generator.iter):
                return bound_names
            if is_descendant(node, generator.target):
                return bound_names
            bound_names.update(target_names(generator.target))
            if any(is_descendant(node, condition) for condition in generator.ifs):
                return bound_names
        return bound_names

    def is_comprehension_shadowed(node: ast.AST, name: str) -> bool:
        current = parent_by_node.get(id(node))
        while current is not None:
            if (
                isinstance(current, comprehension_types)
                and name in comprehension_bound_names(node, current)
            ):
                return True
            current = parent_by_node.get(id(current))
        return False

    def evaluation_scope(node: ast.AST) -> ast.AST:
        current: ast.AST = node
        while True:
            parent = parent_by_node[id(current)]
            if isinstance(parent, ast.Module):
                return parent
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(is_descendant(node, statement) for statement in parent.body):
                    return parent
                current = parent
                continue
            if isinstance(parent, ast.Lambda):
                if is_descendant(node, parent.body):
                    return parent
                current = parent
                continue
            if isinstance(parent, ast.ClassDef):
                if any(is_descendant(node, statement) for statement in parent.body):
                    return parent
                current = parent
                continue
            current = parent

    import_roots: set[str] = set()
    local_names: dict[int, set[str]] = {}
    owner_event_kinds: dict[tuple[int, str, int], set[str]] = {}
    assignment_nodes: list[
        tuple[ast.AST, ast.expr | None, list[ast.AST], ast.AST]
    ] = []
    binding_events: list[tuple[ast.AST, str, ast.AST]] = []
    deleted_binding_events: set[tuple[int, str, int]] = set()
    implicit_handler_cleanup_events: set[tuple[int, str, int]] = set()
    global_names: dict[int, set[str]] = {}
    nonlocal_names: dict[int, set[str]] = {}

    def add_local(scope: ast.AST, name: str) -> None:
        local_names.setdefault(id(scope), set()).add(name)

    def add_nonowner(scope: ast.AST, name: str) -> None:
        add_local(scope, name)

    def add_owner(
        scope: ast.AST,
        name: str,
        kind: str,
        binding_node: ast.AST,
    ) -> None:
        add_local(scope, name)
        owner_event_kinds.setdefault(
            (id(scope), name, id(binding_node)),
            set(),
        ).add(kind)

    for node in ast.walk(tree):
        scope = lexical_scope(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent = parent_by_node.get(id(node))
            if parent is not None:
                parent_scope = lexical_scope(parent)
                add_nonowner(parent_scope, node.name)
                binding_events.append((parent_scope, node.name, node))
        elif isinstance(node, ast.arg):
            add_nonowner(scope, node.arg)
            binding_events.append((scope, node.arg, node))
        elif isinstance(node, ast.Global):
            global_names.setdefault(id(scope), set()).update(node.names)
        elif isinstance(node, ast.Nonlocal):
            nonlocal_names.setdefault(id(scope), set()).update(node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_roots.add(alias.name.split('.', 1)[0])
                bound_name = alias.asname or alias.name.split('.', 1)[0]
                if alias.name == 'importlib' or (
                    alias.name.startswith('importlib.')
                    and alias.asname is None
                ):
                    add_owner(
                        scope,
                        bound_name,
                        'importlib_module',
                        node,
                    )
                elif alias.name == 'builtins':
                    add_owner(
                        scope,
                        bound_name,
                        'builtins_module',
                        node,
                    )
                else:
                    add_nonowner(scope, bound_name)
                binding_events.append((scope, bound_name, node))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                import_roots.add(node.module.split('.', 1)[0])
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if (
                    node.level == 0
                    and node.module == 'importlib'
                    and alias.name == 'import_module'
                ):
                    add_owner(
                        scope,
                        bound_name,
                        'importlib_callable',
                        node,
                    )
                elif (
                    node.level == 0
                    and node.module == 'builtins'
                    and alias.name == '__import__'
                ):
                    add_owner(
                        scope,
                        bound_name,
                        'builtins_callable',
                        node,
                    )
                else:
                    add_nonowner(scope, bound_name)
                binding_events.append((scope, bound_name, node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assignment_nodes.append((scope, node.value, targets, node))
            for target in targets:
                for name in target_names(target):
                    add_local(scope, name)
                    binding_events.append((scope, name, node))
        elif isinstance(node, (ast.AugAssign, ast.Delete)):
            targets = [node.target] if isinstance(node, ast.AugAssign) else node.targets
            for target in targets:
                for name in target_names(target):
                    add_nonowner(scope, name)
                    binding_events.append((scope, name, node))
                    if isinstance(node, ast.Delete):
                        deleted_binding_events.add(
                            (id(scope), name, id(node))
                        )
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for name in target_names(node.target):
                add_nonowner(scope, name)
                binding_events.append((scope, name, node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for name in target_names(item.optional_vars):
                        add_nonowner(scope, name)
                        binding_events.append((scope, name, item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add_nonowner(scope, node.name)
            binding_events.append((scope, node.name, node.type or node))
            binding_events.append((scope, node.name, node))
            cleanup_event = (id(scope), node.name, id(node))
            deleted_binding_events.add(cleanup_event)
            implicit_handler_cleanup_events.add(cleanup_event)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            add_nonowner(scope, node.name)
            binding_events.append((scope, node.name, node))
        elif isinstance(node, ast.MatchMapping) and node.rest:
            add_nonowner(scope, node.rest)
            binding_events.append((scope, node.rest, node))

    def source_end(node: ast.AST) -> tuple[int, int]:
        return (
            getattr(node, 'end_lineno', None) or node.lineno,
            getattr(node, 'end_col_offset', None) or node.col_offset,
        )

    def source_start(node: ast.AST) -> tuple[int, int]:
        return node.lineno, node.col_offset

    def binding_target_scope(scope: ast.AST, name: str) -> ast.AST:
        if name in global_names.get(id(scope), set()):
            return tree
        if name not in nonlocal_names.get(id(scope), set()):
            return scope
        parent_scope = enclosing_scope(scope)
        while parent_scope is not None:
            if name in global_names.get(id(parent_scope), set()):
                return tree
            if (
                name in local_names.get(id(parent_scope), set())
                and name not in nonlocal_names.get(id(parent_scope), set())
            ):
                return parent_scope
            parent_scope = enclosing_scope(parent_scope)
        return scope

    conditional_binding_ancestors = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        getattr(ast, 'TryStar', ast.Try),
        ast.Match,
        ast.BoolOp,
        ast.IfExp,
    )

    def control_region(
        node: ast.AST,
        control: ast.AST,
    ) -> tuple[str, int | None] | None:
        regions: list[tuple[str, int | None, list[ast.AST]]] = []
        if isinstance(control, ast.If):
            regions = [
                ('test', None, [control.test]),
                ('body', None, control.body),
                ('orelse', None, control.orelse),
            ]
        elif isinstance(control, (ast.For, ast.AsyncFor)):
            regions = [
                ('iter', None, [control.iter]),
                ('target', None, [control.target]),
                ('body', None, control.body),
                ('orelse', None, control.orelse),
            ]
        elif isinstance(control, ast.While):
            regions = [
                ('test', None, [control.test]),
                ('body', None, control.body),
                ('orelse', None, control.orelse),
            ]
        elif isinstance(control, (ast.With, ast.AsyncWith)):
            regions = [
                *[
                    ('context', index, [item.context_expr])
                    for index, item in enumerate(control.items)
                ],
                *[
                    ('target', index, [item.optional_vars])
                    for index, item in enumerate(control.items)
                    if item.optional_vars is not None
                ],
                ('body', None, control.body),
            ]
        elif isinstance(control, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
            regions = [
                ('body', None, control.body),
                *[
                    ('handler', index, [handler])
                    for index, handler in enumerate(control.handlers)
                ],
                ('orelse', None, control.orelse),
                ('finalbody', None, control.finalbody),
            ]
        elif isinstance(control, ast.Match):
            regions = [
                ('subject', None, [control.subject]),
                *[
                    ('case', index, [case])
                    for index, case in enumerate(control.cases)
                ],
            ]
        elif isinstance(control, ast.BoolOp):
            regions = [
                ('value', index, [value])
                for index, value in enumerate(control.values)
            ]
        elif isinstance(control, ast.IfExp):
            regions = [
                ('test', None, [control.test]),
                ('body', None, [control.body]),
                ('orelse', None, [control.orelse]),
            ]
        for part, index, roots in regions:
            if any(is_descendant(node, root) for root in roots):
                return part, index
        return None

    def control_ancestors(
        node: ast.AST,
        scope: ast.AST,
    ) -> list[ast.AST]:
        controls: list[ast.AST] = []
        current = parent_by_node.get(id(node))
        while current is not None and current is not scope:
            if isinstance(current, conditional_binding_ancestors):
                controls.append(current)
            current = parent_by_node.get(id(current))
        return controls

    def control_regions_are_compatible(
        control: ast.AST,
        event_region: tuple[str, int | None] | None,
        use_region: tuple[str, int | None] | None,
    ) -> bool:
        if event_region is None or use_region is None:
            return True
        event_part, event_index = event_region
        use_part, use_index = use_region
        if isinstance(control, ast.If):
            return {event_part, use_part} != {'body', 'orelse'}
        if isinstance(control, ast.Match):
            return not (
                event_part == use_part == 'case'
                and event_index != use_index
            )
        if isinstance(control, ast.Try) and not isinstance(
            control,
            getattr(ast, 'TryStar', ()),
        ):
            if event_part == use_part == 'handler':
                return event_index == use_index
            return {event_part, use_part} != {'handler', 'orelse'}
        if isinstance(control, getattr(ast, 'TryStar', ())):
            return {event_part, use_part} != {'handler', 'orelse'}
        if isinstance(control, (ast.With, ast.AsyncWith)):
            if event_part == 'body':
                return use_part not in {'context', 'target'}
            if event_part == 'target' and use_part in {'context', 'target'}:
                return (
                    event_index is not None
                    and use_index is not None
                    and event_index < use_index
                )
            if event_part == 'context' and use_part in {'context', 'target'}:
                return (
                    event_index is not None
                    and use_index is not None
                    and event_index <= use_index
                )
        if isinstance(control, ast.BoolOp):
            return not (
                event_part == use_part == 'value'
                and event_index is not None
                and use_index is not None
                and event_index > use_index
            )
        if isinstance(control, ast.IfExp):
            return {event_part, use_part} != {'body', 'orelse'}
        return True

    def binding_may_reach_use(
        binding_node: ast.AST,
        use_node: ast.AST,
        scope: ast.AST,
    ) -> bool:
        use_controls = {
            id(control): control
            for control in control_ancestors(use_node, scope)
        }
        for control in control_ancestors(binding_node, scope):
            shared_control = use_controls.get(id(control))
            if shared_control is None:
                continue
            if not control_regions_are_compatible(
                control,
                control_region(binding_node, control),
                control_region(use_node, shared_control),
            ):
                return False
        return True

    def nested_finally_binding_precedes_outer_finally(
        binding_node: ast.AST,
        control: ast.AST,
    ) -> bool:
        current = binding_node
        saw_nested_finally = False
        while current is not control:
            parent = parent_by_node.get(id(current))
            if parent is None:
                return False
            if isinstance(parent, conditional_binding_ancestors):
                if isinstance(
                    parent,
                    (ast.Try, getattr(ast, 'TryStar', ast.Try)),
                ) and control_region(binding_node, parent) == (
                    'finalbody',
                    None,
                ):
                    if not parent.finalbody or current is not parent.finalbody[0]:
                        return False
                    saw_nested_finally = True
                elif parent is not control:
                    return False
            if parent is control:
                return bool(
                    saw_nested_finally
                    and current in control.body
                    and control.body.index(current) == 0
                )
            current = parent
        return False

    def binding_is_path_definite(
        binding_node: ast.AST,
        use_node: ast.AST,
        scope: ast.AST,
    ) -> bool:
        use_controls = {
            id(control): control
            for control in control_ancestors(use_node, scope)
        }
        for control in control_ancestors(binding_node, scope):
            event_part, event_index = control_region(binding_node, control) or (
                '',
                None,
            )
            use_region = (
                control_region(use_node, use_controls[id(control)])
                if id(control) in use_controls
                else None
            )
            use_part, use_index = use_region or ('', None)
            if isinstance(control, ast.If):
                if event_part == 'test' or (
                    event_part == use_part
                    and event_part in {'body', 'orelse'}
                ):
                    continue
                return False
            if isinstance(control, (ast.For, ast.AsyncFor)):
                if event_part == 'iter' or (
                    event_part in {'target', 'body'}
                    and use_part == 'body'
                ) or (event_part == use_part == 'orelse') or (
                    event_part == 'orelse'
                    and use_part == ''
                    and not loop_has_nearest_break(control)
                ):
                    continue
                return False
            if isinstance(control, ast.While):
                if event_part == 'test' or (
                    event_part == use_part
                    and event_part in {'body', 'orelse'}
                ) or (
                    event_part == 'orelse'
                    and use_part == ''
                    and not loop_has_nearest_break(control)
                ):
                    continue
                return False
            if isinstance(control, (ast.With, ast.AsyncWith)):
                if event_part == 'context' and (
                    (
                        use_part in {'context', 'target'}
                        and event_index is not None
                        and use_index is not None
                        and event_index <= use_index
                    )
                    or use_part == 'body'
                    or (use_part == '' and event_index == 0)
                ):
                    continue
                if event_part == 'target' and (
                    (
                        use_part in {'context', 'target'}
                        and event_index is not None
                        and use_index is not None
                        and event_index < use_index
                    )
                    or use_part == 'body'
                    or (
                        use_part == ''
                        and event_index == 0
                        and isinstance(binding_node, ast.Name)
                    )
                ):
                    continue
                if event_part == use_part == 'body':
                    continue
                return False
            if isinstance(control, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
                if event_part == 'finalbody' and use_part in {
                    '',
                    'finalbody',
                }:
                    continue
                if (
                    event_part == 'body'
                    and use_part == 'finalbody'
                    and nested_finally_binding_precedes_outer_finally(
                        binding_node,
                        control,
                    )
                ):
                    continue
                if event_part == 'body' and use_part in {'body', 'orelse'}:
                    continue
                if (
                    event_part == use_part == 'handler'
                    and event_index == use_index
                ) or event_part == use_part == 'orelse':
                    continue
                return False
            if isinstance(control, ast.Match):
                if event_part == 'subject' or (
                    event_part == use_part == 'case'
                    and event_index == use_index
                ):
                    continue
                return False
            if isinstance(control, ast.BoolOp):
                if event_part == use_part == 'value' and (
                    event_index is not None
                    and use_index is not None
                    and event_index <= use_index
                ):
                    continue
                if event_part == 'value' and event_index == 0:
                    continue
                return False
            if isinstance(control, ast.IfExp):
                if event_part == 'test' or (
                    event_part == use_part
                    and event_part in {'body', 'orelse'}
                ):
                    continue
                return False
        return True

    def is_conditional_binding(
        event_scope: ast.AST,
        target_scope: ast.AST,
        binding_node: ast.AST,
    ) -> bool:
        if event_scope is not target_scope:
            return True
        current = parent_by_node.get(id(binding_node))
        while current is not None and current is not target_scope:
            if isinstance(current, ast.BoolOp):
                region = control_region(binding_node, current)
                if region != ('value', 0):
                    return True
            elif isinstance(current, ast.IfExp):
                region = control_region(binding_node, current)
                if region != ('test', None):
                    return True
            elif isinstance(current, conditional_binding_ancestors):
                return True
            current = parent_by_node.get(id(current))
        return False

    loop_types = (ast.For, ast.AsyncFor, ast.While)

    def nearest_enclosing_loop(node: ast.AST) -> ast.AST | None:
        current = parent_by_node.get(id(node))
        while current is not None:
            if isinstance(current, loop_types):
                return current
            current = parent_by_node.get(id(current))
        return None

    def loop_has_nearest_break(loop: ast.AST) -> bool:
        return any(
            isinstance(node, ast.Break)
            and nearest_enclosing_loop(node) is loop
            for node in ast.walk(loop)
        )

    def abrupt_exit_runs_finally_binding(
        abrupt_node: ast.AST,
        binding_node: ast.AST,
        loop: ast.AST,
    ) -> bool:
        current = parent_by_node.get(id(abrupt_node))
        while current is not None and current is not loop:
            if isinstance(current, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
                if (
                    control_region(abrupt_node, current)
                    != ('finalbody', None)
                    and control_region(binding_node, current)
                    == ('finalbody', None)
                ):
                    return True
            current = parent_by_node.get(id(current))
        return False

    def binding_is_definite_on_abrupt_loop_exit(
        binding_node: ast.AST,
        abrupt_node: ast.AST,
        loop: ast.AST,
        scope: ast.AST,
    ) -> bool:
        if (
            source_end(binding_node) <= source_start(abrupt_node)
            and binding_may_reach_use(binding_node, abrupt_node, scope)
            and binding_is_path_definite(binding_node, abrupt_node, scope)
        ):
            return True
        current = parent_by_node.get(id(abrupt_node))
        while current is not None and current is not loop:
            if isinstance(current, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
                if (
                    control_region(abrupt_node, current)
                    != ('finalbody', None)
                    and control_region(binding_node, current)
                    == ('finalbody', None)
                    and binding_is_structurally_definite_until(
                        binding_node,
                        current,
                        scope,
                    )
                ):
                    return True
            current = parent_by_node.get(id(current))
        return False

    def prior_loop_exit_may_skip_event(
        binding_node: ast.AST,
        loop: ast.AST,
        *,
        include_break_paths: bool,
    ) -> bool:
        abrupt_types = (
            (ast.Break, ast.Continue)
            if include_break_paths
            else (ast.Continue,)
        )
        return any(
            isinstance(node, abrupt_types)
            and nearest_enclosing_loop(node) is loop
            and source_start(node) < source_start(binding_node)
            and not abrupt_exit_runs_finally_binding(
                node,
                binding_node,
                loop,
            )
            for node in ast.walk(loop)
        )

    def binding_is_structurally_definite_until(
        binding_node: ast.AST,
        stop_control: ast.AST,
        scope: ast.AST,
    ) -> bool:
        for control in control_ancestors(binding_node, scope):
            if control is stop_control:
                return True
            part, index = control_region(binding_node, control) or ('', None)
            if isinstance(control, ast.If) and part == 'test':
                continue
            if isinstance(control, ast.BoolOp) and (part, index) == (
                'value',
                0,
            ):
                continue
            if isinstance(control, ast.IfExp) and part == 'test':
                continue
            if isinstance(control, ast.Match) and part == 'subject':
                continue
            if isinstance(control, (ast.For, ast.AsyncFor)) and (
                part == 'orelse'
                and not loop_has_nearest_break(control)
            ):
                continue
            if isinstance(control, ast.While) and (
                part == 'test'
                or (
                    part == 'orelse'
                    and not loop_has_nearest_break(control)
                )
            ):
                continue
            if isinstance(
                control,
                (ast.Try, getattr(ast, 'TryStar', ast.Try)),
            ) and part == 'finalbody':
                continue
            return False
        return False

    def binding_is_structurally_cycle_definite(
        binding_node: ast.AST,
        loop: ast.AST,
        scope: ast.AST,
    ) -> bool:
        return binding_is_structurally_definite_until(
            binding_node,
            loop,
            scope,
        )

    def binding_is_cycle_definite(
        binding_node: ast.AST,
        loop: ast.AST,
        scope: ast.AST,
        *,
        include_break_paths: bool = False,
    ) -> bool:
        return binding_is_structurally_cycle_definite(
            binding_node,
            loop,
            scope,
        ) and not prior_loop_exit_may_skip_event(
            binding_node,
            loop,
            include_break_paths=include_break_paths,
        )

    def statement_guarantees_loop_break(
        statement: ast.AST,
        loop: ast.AST,
    ) -> bool:
        if isinstance(statement, ast.Break):
            return nearest_enclosing_loop(statement) is loop
        if not isinstance(
            statement,
            (ast.Try, getattr(ast, 'TryStar', ast.Try)),
        ) or statement.handlers:
            return False
        if not statement.body or not statement_guarantees_loop_break(
            statement.body[-1],
            loop,
        ):
            return False
        possible_continues = (
            *statement.body[:-1],
            *statement.finalbody,
        )
        return not any(
            isinstance(node, ast.Continue)
            and nearest_enclosing_loop(node) is loop
            for root in possible_continues
            for node in ast.walk(root)
        )

    def statement_prevents_loop_fallthrough(
        statement: ast.AST,
        loop: ast.AST,
    ) -> bool:
        if isinstance(statement, (ast.Break, ast.Continue)):
            return nearest_enclosing_loop(statement) is loop
        return bool(
            isinstance(
                statement,
                (ast.Try, getattr(ast, 'TryStar', ast.Try)),
            )
            and not statement.handlers
            and statement.body
            and statement_prevents_loop_fallthrough(
                statement.body[-1],
                loop,
            )
        )

    def event_follows_guaranteed_loop_exit(
        binding_node: ast.AST,
        loop: ast.AST,
    ) -> bool:
        current = binding_node
        while current is not loop:
            parent = parent_by_node.get(id(current))
            if parent is None:
                return False
            for field_name in ('body', 'orelse', 'finalbody'):
                statements = getattr(parent, field_name, None)
                if not isinstance(statements, list) or current not in statements:
                    continue
                index = statements.index(current)
                if any(
                    statement_prevents_loop_fallthrough(statement, loop)
                    for statement in statements[:index]
                ):
                    return True
            current = parent
        return False

    def loop_cycle_events(
        loop: ast.AST,
        scope: ast.AST,
        events: list[tuple[ast.AST, ast.AST]],
    ) -> list[tuple[ast.AST, ast.AST]]:
        if isinstance(loop, (ast.For, ast.AsyncFor)):
            cycle_parts = {'target', 'body'}
        else:
            cycle_parts = {'test', 'body'}
        cycle_events = []
        for event in events:
            region = control_region(event[1], loop)
            if (
                event[0] is scope
                and region
                and region[0] in cycle_parts
                and not event_follows_guaranteed_loop_exit(event[1], loop)
            ):
                cycle_events.append(event)
        return cycle_events

    def final_reaching_events(
        loop: ast.AST,
        scope: ast.AST,
        events: list[tuple[ast.AST, ast.AST]],
        *,
        include_break_paths: bool,
        structural_only: bool = False,
    ) -> list[tuple[ast.AST, ast.AST]]:
        definite_events = [
            event
            for event in events
            if (
                binding_is_structurally_cycle_definite(
                    event[1],
                    loop,
                    scope,
                )
                if structural_only
                else binding_is_cycle_definite(
                    event[1],
                    loop,
                    scope,
                    include_break_paths=include_break_paths,
                )
            )
        ]
        if not definite_events:
            return events
        boundary = max(source_end(event[1]) for event in definite_events)
        return [
            event
            for event in events
            if source_end(event[1]) >= boundary
        ]

    def loop_final_reaching_events(
        loop: ast.AST,
        scope: ast.AST,
        events: list[tuple[ast.AST, ast.AST]],
        *,
        include_break_paths: bool = False,
    ) -> list[tuple[ast.AST, ast.AST]]:
        cycle_events = loop_cycle_events(loop, scope, events)
        if not include_break_paths:
            return final_reaching_events(
                loop,
                scope,
                cycle_events,
                include_break_paths=False,
            )

        normal_events = cycle_events
        if isinstance(loop, ast.While):
            test_events = [
                event
                for event in cycle_events
                if control_region(event[1], loop) == ('test', None)
            ]
            if test_events:
                test_is_definite = any(
                    binding_is_cycle_definite(
                        event[1],
                        loop,
                        scope,
                    )
                    for event in test_events
                )
                test_final_events = final_reaching_events(
                    loop,
                    scope,
                    test_events,
                    include_break_paths=False,
                )
                if test_is_definite:
                    normal_events = test_final_events
                else:
                    normal_events = [
                        *final_reaching_events(
                            loop,
                            scope,
                            [
                                event
                                for event in cycle_events
                                if control_region(event[1], loop)
                                == ('body', None)
                            ],
                            include_break_paths=False,
                        ),
                        *test_final_events,
                    ]
            else:
                normal_events = final_reaching_events(
                    loop,
                    scope,
                    cycle_events,
                    include_break_paths=False,
                )
        else:
            normal_events = final_reaching_events(
                loop,
                scope,
                cycle_events,
                include_break_paths=False,
            )

        orelse_events = [
            event
            for event in events
            if (
                event[0] is scope
                and control_region(event[1], loop) == ('orelse', None)
            )
        ]
        if orelse_events:
            orelse_final_events = final_reaching_events(
                loop,
                scope,
                orelse_events,
                include_break_paths=False,
                structural_only=True,
            )
            if any(
                binding_is_structurally_cycle_definite(
                    event[1],
                    loop,
                    scope,
                )
                for event in orelse_events
            ):
                normal_events = orelse_final_events
            else:
                normal_events = [*normal_events, *orelse_final_events]

        break_events = []
        if loop_has_nearest_break(loop):
            break_events = final_reaching_events(
                loop,
                scope,
                cycle_events,
                include_break_paths=True,
            )
        return list({
            (id(event[0]), id(event[1])): event
            for event in (*normal_events, *break_events)
        }.values())

    def summarize_completed_loop_events(
        scope: ast.AST,
        use_node: ast.AST,
        events: list[tuple[ast.AST, ast.AST]],
        *,
        within_loop: ast.AST | None = None,
    ) -> tuple[list[tuple[ast.AST, ast.AST]], ast.AST | None]:
        completed_loops: dict[int, ast.AST] = {}
        for _, binding_node in events:
            for control in control_ancestors(binding_node, scope):
                if not isinstance(control, loop_types):
                    continue
                if within_loop is not None:
                    if (
                        control is not within_loop
                        and is_descendant(control, within_loop)
                    ):
                        completed_loops[id(control)] = control
                    continue
                use_region = control_region(use_node, control)
                if (
                    source_end(control) <= source_start(use_node)
                    or use_region == ('orelse', None)
                ):
                    completed_loops[id(control)] = control
        retained_events = events
        bound_loops = []
        for loop in sorted(
            completed_loops.values(),
            key=lambda control: len(control_ancestors(control, scope)),
            reverse=True,
        ):
            raw_cycle_events = [
                event
                for event in retained_events
                if (
                    event[0] is scope
                    and control_region(event[1], loop)
                    and control_region(event[1], loop)[0]
                    in {'target', 'test', 'body'}
                )
            ]
            if not raw_cycle_events:
                continue
            final_events = loop_final_reaching_events(
                loop,
                scope,
                retained_events,
                include_break_paths=True,
            )
            if (
                source_end(loop) <= source_start(use_node)
                and binding_is_path_definite(loop, use_node, scope)
                and (
                    any(
                        control_region(event, loop) == ('orelse', None)
                        and binding_is_structurally_definite_until(
                            event,
                            loop,
                            scope,
                        )
                        for _, event in retained_events
                    )
                    or (
                        isinstance(loop, ast.While)
                        and any(
                            control_region(event, loop) == ('test', None)
                            and binding_is_cycle_definite(event, loop, scope)
                            for _, event in retained_events
                        )
                    )
                )
                and all(
                    any(
                        binding_is_definite_on_abrupt_loop_exit(
                            event,
                            break_node,
                            loop,
                            scope,
                        )
                        for _, event in raw_cycle_events
                    )
                    for break_node in ast.walk(loop)
                    if isinstance(break_node, ast.Break)
                    and nearest_enclosing_loop(break_node) is loop
                )
            ):
                bound_loops.append(loop)
            cycle_event_ids = {
                id(event[1])
                for event in raw_cycle_events
            }
            cycle_event_ids.update(
                id(event[1])
                for event in retained_events
                if control_region(event[1], loop) == ('orelse', None)
            )
            retained_events = [
                event
                for event in retained_events
                if id(event[1]) not in cycle_event_ids
            ]
            retained_events.extend(final_events)
        boundary = max(bound_loops, key=source_end, default=None)
        return retained_events, boundary

    def event_has_following_direct_break(
        binding_node: ast.AST,
        loop: ast.AST,
    ) -> bool:
        current = binding_node
        while current is not loop:
            if statement_guarantees_loop_break(current, loop):
                return True
            parent = parent_by_node.get(id(current))
            if parent is None:
                return False
            for field_name in ('body', 'orelse', 'finalbody'):
                statements = getattr(parent, field_name, None)
                if not isinstance(statements, list) or current not in statements:
                    continue
                index = statements.index(current)
                if any(
                    statement_guarantees_loop_break(statement, loop)
                    for statement in statements[index + 1:]
                ):
                    return True
            current = parent
        return False

    def handler_cleanup_crosses_finally(
        handler: ast.ExceptHandler,
        use_node: ast.AST,
        scope: ast.AST,
    ) -> bool:
        current = parent_by_node.get(id(handler))
        while current is not None and current is not scope:
            if (
                isinstance(
                    current,
                    (ast.Try, getattr(ast, 'TryStar', ast.Try)),
                )
                and control_region(handler, current) != ('finalbody', None)
                and control_region(use_node, current) == ('finalbody', None)
            ):
                return True
            current = parent_by_node.get(id(current))
        return False

    def handler_cleanup_reaches_use(
        handler: ast.ExceptHandler,
        use_node: ast.AST,
        scope: ast.AST,
    ) -> bool:
        if not binding_may_reach_use(handler, use_node, scope):
            return False
        if not handler.body:
            return True
        terminal = handler.body[-1]
        if not isinstance(terminal, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return True
        if handler_cleanup_crosses_finally(handler, use_node, scope):
            return True
        if isinstance(terminal, ast.Continue):
            return True
        if isinstance(terminal, ast.Break):
            loop = nearest_enclosing_loop(terminal)
            return bool(loop is not None and not is_descendant(use_node, loop))
        if isinstance(terminal, ast.Raise):
            current = parent_by_node.get(id(handler))
            while current is not None and current is not scope:
                if isinstance(
                    current,
                    (ast.Try, getattr(ast, 'TryStar', ast.Try)),
                ):
                    use_region = control_region(use_node, current)
                    event_region = control_region(handler, current)
                    if (
                        event_region == ('body', None)
                        and current.handlers
                        and (
                            (use_region and use_region[0] == 'handler')
                            or (
                                use_region is None
                                and source_end(current) <= source_start(use_node)
                            )
                        )
                    ):
                        return True
                current = parent_by_node.get(id(current))
        return False

    def summarize_completed_handler_events(
        name: str,
        use_node: ast.AST,
        events: list[tuple[ast.AST, ast.AST]],
        *,
        include_later_loop_events: bool = False,
    ) -> list[tuple[ast.AST, ast.AST]]:
        retained_events = events
        cleanup_events = [
            event
            for event in events
            if (
                id(event[0]),
                name,
                id(event[1]),
            ) in implicit_handler_cleanup_events
            and not is_descendant(use_node, event[1])
            and (
                include_later_loop_events
                or source_end(event[1]) <= source_start(use_node)
            )
        ]
        for cleanup_scope, handler in cleanup_events:
            cleanup_reaches_use = handler_cleanup_reaches_use(
                handler,
                use_node,
                cleanup_scope,
            )
            retained_events = [
                event
                for event in retained_events
                if (
                    event == (cleanup_scope, handler)
                    and cleanup_reaches_use
                )
                or event[0] is not cleanup_scope
                or not is_descendant(event[1], handler)
            ]
        return retained_events

    def loop_carried_event_resolution(
        scope: ast.AST,
        name: str,
        use_node: ast.AST,
    ) -> tuple[bool, set[str], bool]:
        carried_events: list[tuple[ast.AST, ast.AST]] = []
        all_scope_events = [
            (event_scope, binding_node)
            for event_scope, event_name, binding_node in binding_events
            if event_scope is scope and event_name == name
        ]
        all_scope_events = summarize_completed_handler_events(
            name,
            use_node,
            all_scope_events,
            include_later_loop_events=True,
        )
        for loop in control_ancestors(use_node, scope):
            if not isinstance(loop, loop_types):
                continue
            use_region = control_region(use_node, loop)
            if isinstance(loop, (ast.For, ast.AsyncFor)):
                if use_region != ('body', None):
                    continue
                if name in target_names(loop.target):
                    continue
            elif use_region not in {('test', None), ('body', None)}:
                continue

            (
                resolved_scope_events,
                completed_prefix_boundary,
            ) = summarize_completed_loop_events(
                scope,
                use_node,
                all_scope_events,
                within_loop=loop,
            )
            cycle_events = loop_cycle_events(
                loop,
                scope,
                resolved_scope_events,
            )
            definite_prefix_events = [
                event
                for event in cycle_events
                if (
                    source_end(event[1]) <= source_start(use_node)
                    and binding_may_reach_use(event[1], use_node, scope)
                    and (
                        binding_is_cycle_definite(event[1], loop, scope)
                        or binding_is_path_definite(
                            event[1],
                            use_node,
                            scope,
                        )
                    )
                )
            ]
            if definite_prefix_events or completed_prefix_boundary is not None:
                continue

            carried_events.extend(
                event
                for event in loop_final_reaching_events(
                    loop,
                    scope,
                    resolved_scope_events,
                )
                if not event_has_following_direct_break(event[1], loop)
            )

        unique_events = {
            (id(event_scope), id(binding_node)): (event_scope, binding_node)
            for event_scope, binding_node in carried_events
        }
        retained_events = list(unique_events.values())
        has_event, kinds = resolved_event_kinds(name, retained_events)
        fallback_possible = any(
            (id(event_scope), name, id(binding_node))
            in deleted_binding_events
            for event_scope, binding_node in retained_events
        )
        return has_event, kinds, fallback_possible

    def resolved_event_kinds(
        name: str,
        events: list[tuple[ast.AST, ast.AST]],
    ) -> tuple[bool, set[str]]:
        if not events:
            return False, set()
        kinds: set[str] = set()
        for event_scope, binding_node in events:
            kinds.update(
                owner_event_kinds.get(
                    (id(event_scope), name, id(binding_node)),
                    set(),
                )
            )
        return True, kinds

    def event_kinds_before(
        scope: ast.AST,
        name: str,
        use_node: ast.AST,
    ) -> tuple[bool, set[str], bool, bool]:
        preceding_events = [
            (event_scope, binding_node)
            for event_scope, event_name, binding_node in binding_events
            if (
                event_scope is scope
                and event_name == name
                and source_end(binding_node) <= source_start(use_node)
                and binding_may_reach_use(binding_node, use_node, scope)
            )
        ]
        preceding_events = summarize_completed_handler_events(
            name,
            use_node,
            preceding_events,
        )
        preceding_events, completed_loop_boundary = summarize_completed_loop_events(
            scope,
            use_node,
            preceding_events,
        )
        if not preceding_events:
            has_event, kinds, deleted_fallback = loop_carried_event_resolution(
                scope,
                name,
                use_node,
            )
            return has_event, kinds, deleted_fallback, False

        definite_events = [
            event
            for event in preceding_events
            if binding_is_path_definite(event[1], use_node, scope)
        ]
        boundaries: list[tuple[tuple[int, int], str, ast.AST]] = [
            (source_end(binding_node), 'event', binding_node)
            for _, binding_node in definite_events
        ]
        if completed_loop_boundary is not None:
            boundaries.append(
                (
                    source_end(completed_loop_boundary),
                    'control',
                    completed_loop_boundary,
                )
            )

        def direct_region_latest_event(
            control: ast.AST,
            region: tuple[str, int | None],
        ) -> tuple[ast.AST, ast.AST] | None:
            candidates = []
            for event in preceding_events:
                binding_node = event[1]
                if control_region(binding_node, control) != region:
                    continue
                nested_controls = []
                for ancestor in control_ancestors(binding_node, scope):
                    if ancestor is control:
                        break
                    nested_controls.append(ancestor)
                if not nested_controls:
                    candidates.append(event)
            if not candidates:
                return None
            return max(candidates, key=lambda event: source_end(event[1]))

        seen_controls: dict[int, ast.AST] = {}
        for _, binding_node in preceding_events:
            for control in control_ancestors(binding_node, scope):
                seen_controls[id(control)] = control
        for control in seen_controls.values():
            if (
                source_end(control) > source_start(use_node)
                or is_descendant(use_node, control)
                or not binding_is_path_definite(control, use_node, scope)
            ):
                continue
            regions: list[tuple[str, int | None]] = []
            if isinstance(control, ast.If) and control.orelse:
                regions = [('body', None), ('orelse', None)]
            elif isinstance(control, ast.Match) and control.cases:
                final_case = control.cases[-1]
                if (
                    final_case.guard is None
                    and isinstance(final_case.pattern, ast.MatchAs)
                    and final_case.pattern.pattern is None
                ):
                    regions = [
                        ('case', index)
                        for index in range(len(control.cases))
                    ]
            if not regions:
                continue
            latest_by_region = [
                direct_region_latest_event(control, region)
                for region in regions
            ]
            if any(event is None for event in latest_by_region):
                continue
            if any(
                (id(event[0]), name, id(event[1]))
                in deleted_binding_events
                for event in latest_by_region
                if event is not None
            ):
                continue
            boundaries.append((source_end(control), 'control', control))

        retained_events = preceding_events
        boundary_kind = ''
        boundary_node: ast.AST | None = None
        if boundaries:
            _, boundary_kind, boundary_node = max(
                boundaries,
                key=lambda boundary: boundary[0],
            )
            if boundary_kind == 'event':
                boundary_position = source_end(boundary_node)
                retained_events = [
                    event
                    for event in preceding_events
                    if source_end(event[1]) >= boundary_position
                ]
            else:
                control_end = source_end(boundary_node)
                retained_events = [
                    event
                    for event in preceding_events
                    if (
                        is_descendant(event[1], boundary_node)
                        or source_end(event[1]) >= control_end
                    )
                ]

        fallback_possible = not boundaries
        definite_delete = False
        if boundary_kind == 'event' and boundary_node is not None:
            fallback_possible = (
                id(scope),
                name,
                id(boundary_node),
            ) in deleted_binding_events
            definite_delete = fallback_possible
        if any(
            (id(event_scope), name, id(binding_node))
            in deleted_binding_events
            for event_scope, binding_node in retained_events
        ):
            fallback_possible = True
            if boundary_kind == 'control':
                definite_delete = True
        has_event, kinds = resolved_event_kinds(name, retained_events)
        (
            carried_has_event,
            carried_kinds,
            carried_fallback_possible,
        ) = loop_carried_event_resolution(scope, name, use_node)
        kinds.update(carried_kinds)
        return (
            has_event or carried_has_event,
            kinds,
            fallback_possible or carried_fallback_possible,
            definite_delete,
        )

    def event_kinds_anywhere(
        scope: ast.AST,
        name: str,
        use_node: ast.AST,
    ) -> tuple[bool, set[str]]:
        events = [
            (event_scope, binding_node)
            for event_scope, event_name, binding_node in binding_events
            if (
                binding_target_scope(event_scope, event_name) is scope
                and event_name == name
            )
        ]
        events = summarize_completed_handler_events(
            name,
            use_node,
            events,
        )
        return resolved_event_kinds(name, events)

    def event_kinds_possible_before(
        scope: ast.AST,
        name: str,
        use_node: ast.AST,
    ) -> tuple[bool, set[str]]:
        events = [
            (event_scope, binding_node)
            for event_scope, event_name, binding_node in binding_events
            if (
                event_scope is scope
                and event_name == name
                and source_end(binding_node) <= source_start(use_node)
            )
        ]
        events = summarize_completed_handler_events(
            name,
            use_node,
            events,
        )
        return runtime_reaching_event_kinds(scope, name, events)

    def runtime_reaching_event_kinds(
        scope: ast.AST,
        name: str,
        events: list[tuple[ast.AST, ast.AST]],
    ) -> tuple[bool, set[str]]:
        unconditional_events = [
            event
            for event in events
            if not is_conditional_binding(event[0], scope, event[1])
        ]
        if unconditional_events:
            latest_unconditional = max(
                source_end(binding_node)
                for _, binding_node in unconditional_events
            )
            events = [
                event
                for event in events
                if (
                    event[0] is not scope
                    or source_end(event[1]) >= latest_unconditional
                )
            ]
        return resolved_event_kinds(name, events)

    def module_builtin_fallback_possible(
        name: str,
        *,
        before_node: ast.AST | None = None,
        observation_node: ast.AST | None = None,
    ) -> bool:
        events = [
            (event_scope, binding_node)
            for event_scope, event_name, binding_node in binding_events
            if (
                binding_target_scope(event_scope, event_name) is tree
                and event_name == name
                and (
                    before_node is None
                    or source_end(binding_node) <= source_start(before_node)
                )
            )
        ]
        if observation_node is not None:
            events = summarize_completed_handler_events(
                name,
                observation_node,
                events,
            )
        unconditional_events = [
            event
            for event in events
            if event[0] is tree
            and not is_conditional_binding(event[0], tree, event[1])
        ]
        if not unconditional_events:
            return True
        latest_unconditional = max(
            source_end(binding_node)
            for _, binding_node in unconditional_events
        )
        return any(
            (id(event_scope), name, id(binding_node))
            in deleted_binding_events
            and (
                event_scope is not tree
                or source_end(binding_node) >= latest_unconditional
            )
            for event_scope, binding_node in events
        )

    def resolve_name_at(
        scope: ast.AST,
        name: str,
        use_node: ast.AST,
        *,
        runtime_lookup: bool = False,
        possible_before_lookup: bool = False,
    ) -> set[str]:
        def fallback_name_kinds() -> set[str]:
            class_definition_lookup = isinstance(scope, ast.ClassDef)
            if name in global_names.get(id(scope), set()):
                return resolve_name_at(
                    tree,
                    name,
                    scope if class_definition_lookup else use_node,
                    runtime_lookup=not class_definition_lookup,
                    possible_before_lookup=class_definition_lookup,
                )
            if name in nonlocal_names.get(id(scope), set()):
                parent_scope = enclosing_scope(scope)
                return (
                    resolve_name_at(
                        parent_scope,
                        name,
                        scope if class_definition_lookup else use_node,
                        runtime_lookup=not class_definition_lookup,
                        possible_before_lookup=class_definition_lookup,
                    )
                    if parent_scope is not None
                    else set()
                )
            if (
                isinstance(
                    scope,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
                )
                and name in local_names.get(id(scope), set())
            ):
                return set()
            parent_scope = enclosing_scope(scope)
            if parent_scope is not None:
                parent_runtime_lookup = (
                    runtime_lookup
                    or isinstance(
                        scope,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
                    )
                )
                return resolve_name_at(
                    parent_scope,
                    name,
                    scope if class_definition_lookup else use_node,
                    runtime_lookup=parent_runtime_lookup,
                    possible_before_lookup=(
                        not parent_runtime_lookup
                        and (possible_before_lookup or class_definition_lookup)
                    ),
                )
            return {'builtins_callable'} if name == '__import__' else set()

        def deleted_name_fallback_kinds() -> set[str]:
            if name in nonlocal_names.get(id(scope), set()):
                return set()
            if name in global_names.get(id(scope), set()):
                return (
                    {'builtins_callable'}
                    if name == '__import__'
                    else set()
                )
            return fallback_name_kinds()

        direct_fallback_possible = False
        direct_delete_is_definite = False
        if runtime_lookup:
            has_event, event_kinds = event_kinds_anywhere(
                scope,
                name,
                use_node,
            )
        elif possible_before_lookup:
            has_event, event_kinds = event_kinds_possible_before(
                scope,
                name,
                use_node,
            )
        else:
            (
                has_event,
                event_kinds,
                direct_fallback_possible,
                direct_delete_is_definite,
            ) = event_kinds_before(
                scope,
                name,
                use_node,
            )
        if has_event:
            if (
                scope is tree
                and name == '__import__'
                and (runtime_lookup or possible_before_lookup)
                and not event_kinds
                and module_builtin_fallback_possible(
                    name,
                    before_node=(
                        use_node if possible_before_lookup else None
                    ),
                    observation_node=use_node,
                )
            ):
                return {'builtins_callable'}
            resolved_kinds = set(event_kinds)
            if direct_delete_is_definite:
                resolved_kinds.update(deleted_name_fallback_kinds())
            elif direct_fallback_possible:
                resolved_kinds.update(fallback_name_kinds())
            return resolved_kinds
        return fallback_name_kinds()

    def resolve_dynamic_callable(node: ast.expr) -> set[str]:
        if isinstance(node, ast.Name):
            if is_comprehension_shadowed(node, node.id):
                return set()
            return resolve_name_at(
                evaluation_scope(node),
                node.id,
                node,
            ) & {'importlib_callable', 'builtins_callable'}
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            return set()
        if is_comprehension_shadowed(node.value, node.value.id):
            return set()
        owner_kinds = resolve_name_at(
            evaluation_scope(node.value),
            node.value.id,
            node.value,
        )
        callable_kinds: set[str] = set()
        if (
            'importlib_module' in owner_kinds
            and node.attr == 'import_module'
        ):
            callable_kinds.add('importlib_callable')
        if 'builtins_module' in owner_kinds and node.attr == '__import__':
            callable_kinds.add('builtins_callable')
        return callable_kinds

    unresolved_assignments = assignment_nodes
    while unresolved_assignments:
        pending_assignments = []
        made_progress = False
        for scope, value, targets, binding_node in unresolved_assignments:
            kinds = resolve_dynamic_callable(value) if value is not None else set()
            if not kinds:
                pending_assignments.append(
                    (scope, value, targets, binding_node)
                )
                continue
            for target in targets:
                for name in target_names(target):
                    for kind in kinds:
                        add_owner(scope, name, kind, binding_node)
            made_progress = True
        if not made_progress:
            break
        unresolved_assignments = pending_assignments

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    module_bind_missing_bindings = [
        node
        for scope, name, node in binding_events
        if scope is tree and name == '_bind_missing'
    ]
    global_bind_missing_rebindings = [
        node
        for scope, name, node in binding_events
        if (
            name == '_bind_missing'
            and name in global_names.get(id(scope), set())
        )
    ]
    bind_missing_candidate = (
        module_bind_missing_bindings[0]
        if (
            relative_path == 'main.py'
            and len(module_bind_missing_bindings) == 1
            and isinstance(
                module_bind_missing_bindings[0],
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and not global_bind_missing_rebindings
        )
        else None
    )

    def resolves_to_bind_missing_candidate(node: ast.Name) -> bool:
        if (
            bind_missing_candidate is None
            or is_comprehension_shadowed(node, '_bind_missing')
        ):
            return False
        scope = evaluation_scope(node)
        while scope is not None:
            if scope is tree:
                return True
            if '_bind_missing' in global_names.get(id(scope), set()):
                scope = tree
                continue
            if '_bind_missing' in nonlocal_names.get(id(scope), set()):
                scope = enclosing_scope(scope)
                continue
            if '_bind_missing' in local_names.get(id(scope), set()):
                return False
            scope = enclosing_scope(scope)
        return False

    direct_bind_missing_calls = [
        node
        for node in calls
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == '_bind_missing'
            and resolves_to_bind_missing_candidate(node.func)
        )
    ]
    direct_bind_missing_name_nodes = {
        id(node.func) for node in direct_bind_missing_calls
    }
    bind_missing_name_nodes = {
        id(node)
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == '_bind_missing'
            and resolves_to_bind_missing_candidate(node)
        )
    }
    bind_missing_definition = (
        bind_missing_candidate
        if (
            bind_missing_candidate is not None
            and direct_bind_missing_calls
            and bind_missing_name_nodes == direct_bind_missing_name_nodes
            and all(
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                for node in direct_bind_missing_calls
            )
        )
        else None
    )
    direct_bind_missing_call_ids = {
        id(node) for node in direct_bind_missing_calls
    }
    allowed_nonliteral_import_calls = {
        id(node)
        for node in calls
        if (
            bind_missing_definition is not None
            and evaluation_scope(node) is bind_missing_definition
            and resolve_dynamic_callable(node.func) == {'importlib_callable'}
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == 'module_name'
        )
    }
    unsupported_dynamic_import_lines: set[int] = set()

    def record_literal_module(module_name: str, line_number: int) -> None:
        if module_name.startswith('.'):
            return
        if not IMPORT_MODULE_PATTERN.fullmatch(module_name):
            unsupported_dynamic_import_lines.add(line_number)
            return
        import_roots.add(module_name.split('.', 1)[0])

    for node in calls:
        if id(node) in direct_bind_missing_call_ids:
            if (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                record_literal_module(node.args[1].value, node.lineno)
            else:
                unsupported_dynamic_import_lines.add(node.lineno)
            continue
        if not resolve_dynamic_callable(node.func):
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            record_literal_module(node.args[0].value, node.lineno)
        elif id(node) not in allowed_nonliteral_import_calls:
            unsupported_dynamic_import_lines.add(node.lineno)

    return import_roots, sorted(unsupported_dynamic_import_lines)


def _source_import_consumers(
    root: Path,
) -> tuple[
    dict[str, list[str]],
    list[tuple[str, str]],
    list[tuple[str, int]],
]:
    consumers: dict[str, list[str]] = {}
    parse_errors: list[tuple[str, str]] = []
    unsupported_dynamic_imports: list[tuple[str, int]] = []
    for source_path in _project_python_source_paths(root):
        relative_path = str(source_path.relative_to(root))
        try:
            import_roots, unsupported_lines = _source_import_analysis(
                _read_text_if_present(source_path),
                relative_path=relative_path,
            )
        except SyntaxError as exc:
            parse_errors.append((relative_path, str(exc)))
            continue
        unsupported_dynamic_imports.extend(
            (relative_path, line_number)
            for line_number in unsupported_lines
        )
        for module_name in import_roots:
            consumers.setdefault(module_name, []).append(relative_path)
    return {
        module_name: sorted(set(paths))
        for module_name, paths in consumers.items()
    }, parse_errors, unsupported_dynamic_imports


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


def _is_test_source_path(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        path.name == 'conftest.py'
        or path.name.startswith('test_')
        or 'tests' in path.parts
    )


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
        owner_path = Path(owner)
        owner_is_valid = (
            not owner_path.is_absolute()
            and owner_path.parts[:1] == ('myutils',)
            and all(part not in {'.', '..'} for part in owner_path.parts)
            and owner_path.suffix == '.py'
            and owner_path.stem == module.rsplit('.', 1)[-1]
        )
        if (
            not IMPORT_MODULE_PATTERN.fullmatch(module)
            or not owner_is_valid
            or not purpose
        ):
            add_error('invalid_local_shared_import_entry', path, 'Local shared import requires module, owner, and purpose.')
            continue
        if module in local_modules or module in module_owners:
            add_error('duplicate_local_shared_import_module', path, f'Duplicate import owner for `{module}`.')
            continue
        local_modules[module] = {'owner': owner, 'required_for': purpose}

    (
        consumers_by_module,
        parse_errors,
        unsupported_dynamic_imports,
    ) = _source_import_consumers(root)
    for path, detail in parse_errors:
        add_error('dependency_source_parse_error', path, f'Cannot classify imports: {detail}')
    for path, line_number in unsupported_dynamic_imports:
        add_error(
            'unsupported_dynamic_import',
            path,
            (
                'Dynamic dependency imports must use a literal module name; '
                f'unsupported call at line {line_number}.'
            ),
        )
    external_modules = {module.split('.', 1)[0] for module in module_owners}
    shared_modules = {module.split('.', 1)[0] for module in local_modules}
    classified = set(sys.stdlib_module_names) | {'__future__'} | _project_import_roots(root) | external_modules | shared_modules
    unclassified = sorted(set(consumers_by_module) - classified)
    for module in unclassified:
        for path in consumers_by_module[module]:
            add_error('undeclared_external_import', path, f'Unclassified external import: {module}.', module=module)

    distributions_by_module = metadata.packages_distributions()
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
        module_root = module.split('.', 1)[0]
        owning_distributions = sorted(
            distributions_by_module.get(module_root, [])
        )
        normalized_owners = {
            _normalized_requirement_name(distribution)
            for distribution in owning_distributions
        }
        ownership_matches = package_key in normalized_owners
        checks.append({
            'kind': 'dependency_distribution_ownership',
            'package': record['package'],
            'module': module,
            'owning_distributions': owning_distributions,
            'matches': ownership_matches,
        })
        if not ownership_matches:
            add_error(
                'dependency_distribution_mismatch',
                'docs/AGENT_MANIFEST.json:dependency_imports',
                f'Distribution `{record["package"]}` does not own import module `{module}`.',
                module=module,
            )
        consumers = consumers_by_module.get(module_root, [])
        checks.append({
            'kind': 'dependency_source_imports', 'package': record['package'],
            'module': module, 'role': record['role'], 'import_kind': record['import_kind'],
            'consumers': consumers,
        })
        if record['import_kind'] == 'direct' and not consumers:
            add_error('dependency_module_not_imported', 'docs/AGENT_MANIFEST.json:dependency_imports', f'Direct dependency `{module}` has no source consumer.', module=module)
        if record['import_kind'] == 'backend' and consumers:
            add_error(
                'backend_dependency_imported_directly',
                'docs/AGENT_MANIFEST.json:dependency_imports',
                f'Backend dependency `{module}` has direct source consumers.',
                module=module,
            )
        production_consumers = [
            path for path in consumers if not _is_test_source_path(path)
        ]
        if record['role'] == 'test_tool':
            if production_consumers:
                add_error('test_tool_imported_by_production', production_consumers[0], f'Test tool `{module}` has production consumers.', module=module)
        if record['role'] == 'ui' and (
            not production_consumers
            or any(
                Path(path).parts[:2] != ('src', 'ui')
                for path in production_consumers
            )
        ):
            add_error(
                'ui_dependency_outside_ui',
                'docs/AGENT_MANIFEST.json:dependency_imports',
                f'UI dependency `{module}` must have only UI production consumers.',
                module=module,
            )
        role_consumer_roots = {
            'core_runtime': {'main.py', 'src/runtime'},
            'scientific_pipeline': {'src/materials', 'src/torch_models'},
        }
        required_consumer_roots = role_consumer_roots.get(record['role'])
        if required_consumer_roots is not None and not any(
            (path == 'main.py' and 'main.py' in required_consumer_roots)
            or any(
                Path(path).parts[:2] == tuple(root_path.split('/'))
                for root_path in required_consumer_roots - {'main.py'}
            )
            for path in production_consumers
        ):
            add_error(
                'dependency_role_consumer_mismatch',
                'docs/AGENT_MANIFEST.json:dependency_imports',
                f'Dependency `{module}` has no production consumer for role `{record["role"]}`.',
                module=module,
            )
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
        'source_files': [
            str(path.relative_to(root))
            for path in _project_python_source_paths(root)
        ],
        'external_modules': sorted(external_modules), 'local_shared_modules': sorted(shared_modules),
        'unclassified_modules': unclassified,
        'unsupported_dynamic_imports': [
            {'path': path, 'line': line_number}
            for path, line_number in unsupported_dynamic_imports
        ],
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
            required_dependency_capabilities = {
                'dependency_declarations',
                'dependency_import_availability',
            }
            missing_dependency_capabilities = sorted(
                required_dependency_capabilities - set(requires)
            )
            if missing_dependency_capabilities:
                errors.append({
                    'code': 'validation_profile_missing_dependency_capabilities',
                    'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                    'message': (
                        f'Validation profile `{profile_name}` must reach dependency '
                        'declaration and import-availability checks.'
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
    active_repo_skill_names = {
        str(skill.get('name', '')).strip()
        for skill in project_skills
        if isinstance(skill, dict)
        and skill.get('scope') == 'repo_scoped_codex_skill'
        and skill.get('status', 'active') == 'active'
    }
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
            skill_text = _read_text_if_present(root / relative_path)
            frontmatter = _skill_frontmatter_fields(skill_text)
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
            references = sorted(set(SKILL_REFERENCE_PATTERN.findall(skill_text)))
            unresolved_references = sorted(
                set(references) - active_repo_skill_names
            )
            checks.append({
                'kind': 'project_skill_references',
                'path': relative_path,
                'references': references,
                'unresolved_references': unresolved_references,
            })
            if unresolved_references:
                errors.append({
                    'code': 'unresolved_project_skill_reference',
                    'path': relative_path,
                    'message': (
                        'Repo-scoped `$skill` references must resolve to an active '
                        'repo-scoped skill; describe environment-specific global '
                        'roles or capabilities without trigger syntax. Unresolved: '
                        f'{unresolved_references}'
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
