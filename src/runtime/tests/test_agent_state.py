import ast
from pathlib import Path
import json

import pytest

from runtime import agent_state
from runtime.agent_state import (
    agent_state_to_json,
    build_agent_command_index,
    build_agent_state,
    load_agent_manifest,
    validate_agent_layout,
)


ROOT = Path(__file__).resolve().parents[3]


def test_agent_manifest_loads_machine_readable_contract():
    manifest = load_agent_manifest(ROOT)

    assert manifest['schema_version'] == 'aiforbn.agent_manifest.v1'
    assert manifest['project']['primary_entrypoint'] == 'AGENTS.md'
    assert manifest['project']['manual_operation_supported'] is False
    assert {module['name'] for module in manifest['modules']} >= {
        'runtime',
        'materials',
        'torch_models',
        'ui',
    }
    assert {skill['name'] for skill in manifest['project_skills']} >= {
        'aiforbn-workflow',
        'aiforbn-overleaf-proposal',
        'ai_native_workflow',
    }
    assert 'skills/codex_skill.txt' in manifest['retired_guidance_files']
    assert any(entry['name'] == 'verify_agent_contract' for entry in manifest['entrypoints'])
    assert any(entry['name'] == 'emit_agent_commands' for entry in manifest['entrypoints'])
    assert any(entry['name'] == 'write_agent_state' for entry in manifest['entrypoints'])
    assert manifest['human_docs_policy'] == {
        'policy_id': 'user_owned_read_only_unless_explicit_human_document_task',
        'path': 'human_docs/',
        'owner': 'user',
        'default_access': 'read_only',
        'write_condition': 'explicit_human_document_task',
        'agent_contract_authority': False,
    }
    research_alignment = manifest['research_plan_alignment']
    assert research_alignment['source_files'] == [
        'human_docs/research_plan/ai_for_bn_research_plan_v18.tex',
        'human_docs/research_plan/ai_for_bn_research_plan_v18.bib',
    ]
    assert {
        'bounded_bn_centered_design_space',
        'formula_only_candidate_compatible_screening',
        'validation_ready_structure_handoff_not_synthesis_proof',
    }.issubset(research_alignment['implementation_anchors'])
    assert research_alignment['deliverable_chain'][-2:] == [
        'structure_handoff',
        'technical_report',
    ]


def test_validate_agent_layout_accepts_current_repo_contract():
    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'ok'
    assert validation['errors'] == []
    checked_paths = {check['path'] for check in validation['checks'] if 'path' in check}
    assert 'docs/AGENT_MANIFEST.json' in checked_paths
    assert '.agents/skills/aiforbn-workflow/SKILL.md' in checked_paths
    assert '.agents/skills/aiforbn-overleaf-proposal/SKILL.md' in checked_paths
    assert 'src/runtime/PY_FILES_SUMMARY.md' in checked_paths
    assert 'skills/ai_native_workflow.txt' in checked_paths
    assert 'human_docs/research_plan/ai_for_bn_research_plan_v18.tex' in checked_paths
    assert 'human_docs/research_plan/ai_for_bn_research_plan_v18.bib' in checked_paths
    research_source_checks = {
        check['path']
        for check in validation['checks']
        if check['kind'] == 'research_plan_alignment_source'
    }
    assert research_source_checks == {
        'human_docs/research_plan/ai_for_bn_research_plan_v18.tex',
        'human_docs/research_plan/ai_for_bn_research_plan_v18.bib',
    }
    required_checked_paths = {
        check['path']
        for check in validation['checks']
        if check['kind'] == 'required_path'
    }
    assert 'skill.txt' not in required_checked_paths
    retired_checks = {
        check['path']
        for check in validation['checks']
        if check['kind'] == 'retired_guidance_path' and check['exists']
    }
    assert retired_checks == set()
    human_docs_root_checks = [
        check for check in validation['checks'] if check['kind'] == 'human_docs_root'
    ]
    assert human_docs_root_checks == [{
        'path': 'human_docs/',
        'exists': True,
        'is_file': False,
        'is_dir': True,
        'kind': 'human_docs_root',
    }]
    policy_surfaces = {
        check['path']
        for check in validation['checks']
        if check['kind'] == 'human_docs_policy_surface'
    }
    assert {
        'AGENTS.md',
        '.agents/skills/aiforbn-workflow/SKILL.md',
        '.agents/skills/aiforbn-overleaf-proposal/SKILL.md',
        'docs/HANDOFF.md',
        'docs/PY_FILES_SUMMARY.md',
        'skills/ai_native_workflow.txt',
        'src/runtime/AGENTS.md',
        'src/materials/AGENTS.md',
        'src/torch_models/AGENTS.md',
        'src/ui/AGENTS.md',
        'src/tests/AGENTS.md',
        'src/template/AGENTS.md',
    }.issubset(policy_surfaces)
    dependency_modules = {
        check['module']
        for check in validation['checks']
        if check['kind'] == 'dependency_import'
    }
    assert {
        'pandas',
        'numpy',
        'sklearn',
        'pyarrow',
        'torch',
        'streamlit',
        'jarvis',
    }.issubset(dependency_modules)
    requirement_packages = {
        check['package']
        for check in validation['checks']
        if check['kind'] == 'dependency_requirement' and check['declared']
    }
    assert {'pandas', 'streamlit', 'jarvis-tools'}.issubset(requirement_packages)
    skill_frontmatter_checks = [
        check
        for check in validation['checks']
        if check['kind'] == 'project_skill_frontmatter'
    ]
    assert {check['name'] for check in skill_frontmatter_checks} == {
        'aiforbn-workflow',
        'aiforbn-overleaf-proposal',
    }
    assert all(check['valid'] for check in skill_frontmatter_checks)
    skill_reference_checks = {
        check['path']: check
        for check in validation['checks']
        if check['kind'] == 'project_skill_references'
    }
    assert skill_reference_checks[
        '.agents/skills/aiforbn-workflow/SKILL.md'
    ]['references'] == ['aiforbn-overleaf-proposal']
    assert skill_reference_checks[
        '.agents/skills/aiforbn-overleaf-proposal/SKILL.md'
    ]['references'] == []
    assert all(
        not check['unresolved_references']
        for check in skill_reference_checks.values()
    )


def test_build_agent_command_index_returns_validation_profiles():
    command_index = build_agent_command_index(ROOT)
    manifest = load_agent_manifest(ROOT)

    assert command_index['schema_version'] == 'aiforbn.agent_command_index.v1'
    assert command_index['first_inspection_command'] == 'python3 main.py --verify-agent-contract'
    assert {entry['name'] for entry in command_index['entrypoints']} >= {
        'fast_smoke',
        'emit_agent_commands',
        'verify_agent_contract',
        'write_agent_state',
    }
    validation_names = {entry['name'] for entry in command_index['validation_commands']}
    assert {
        'verify_agent_contract',
        'fast_smoke',
        'full_src_tests',
        'ui_render_smoke',
    }.issubset(validation_names)
    assert any(profile['name'] == 'architecture_doc_skill_edit' for profile in command_index['validation_profiles'])
    ui_profile = next(
        profile
        for profile in command_index['validation_profiles']
        if profile['name'] == 'ui_edit'
    )
    assert ui_profile['commands'] == [
        'verify_agent_contract',
        'focused_regression',
        'ui_render_smoke',
    ]
    assert ui_profile['requires'] == [
        'agent_contract',
        'dependency_declarations',
        'dependency_import_availability',
        'entrypoint_runtime_public_surface_regressions',
        'streamlit_renderer_contract',
    ]
    focused_command = next(
        command
        for command in command_index['validation_commands']
        if command['name'] == 'focused_regression'
    )
    assert focused_command['provides'] == [
        'entrypoint_runtime_public_surface_regressions'
    ]
    assert command_index['modules'] == manifest['modules']
    assert command_index['human_docs_policy']['policy_id'] == (
        'user_owned_read_only_unless_explicit_human_document_task'
    )
    research_alignment = command_index['research_plan_alignment']
    assert set(research_alignment) >= {
        'status',
        'source_files',
        'implementation_anchors',
        'non_claims',
        'deliverable_chain',
    }
    assert research_alignment['status'] == 'v18_alignment_contract'
    assert research_alignment['source_files'] == [
        'human_docs/research_plan/ai_for_bn_research_plan_v18.tex',
        'human_docs/research_plan/ai_for_bn_research_plan_v18.bib',
    ]
    assert {
        'bounded_bn_centered_design_space',
        'formula_only_candidate_compatible_screening',
        'machine_verifiable_deliverable_chain',
    }.issubset(research_alignment['implementation_anchors'])
    assert {
        'open_ended_material_discovery',
        'experimental_synthesis_proof',
        'formula_stage_structure_dependent_property_claims',
        'direct_gap_claim_before_structure_review',
    }.issubset(research_alignment['non_claims'])
    assert research_alignment['deliverable_chain'] == [
        'bn_dataset',
        'benchmarked_models',
        'ranked_candidates',
        'structure_handoff',
        'technical_report',
    ]


def test_agent_command_index_round_trips_every_manifest_contract_section():
    command_index = build_agent_command_index(ROOT)
    manifest = load_agent_manifest(ROOT)

    for field in (
        'entrypoints',
        'validation_commands',
        'validation_profiles',
        'project_skills',
        'source_of_truth_files',
        'retired_guidance_files',
        'research_plan_alignment',
        'modules',
        'human_docs_policy',
    ):
        assert command_index[field] == manifest[field]


def _dependency_by_package(manifest, package):
    return next(
        dependency
        for dependency in manifest['dependency_imports']
        if dependency['package'] == package
    )


def _swap_dependency_modules(manifest, left_package, right_package):
    left = _dependency_by_package(manifest, left_package)
    right = _dependency_by_package(manifest, right_package)
    left['module'], right['module'] = right['module'], left['module']


def test_dependency_contract_covers_requirements_source_imports_and_profiles():
    manifest = load_agent_manifest(ROOT)
    validation = validate_agent_layout(ROOT, manifest)

    dependencies = {
        dependency['package']: dependency
        for dependency in manifest['dependency_imports']
    }
    assert {
        'pydantic',
        'matplotlib',
        'matminer',
        'pymatgen',
        'pytest',
    }.issubset(dependencies)
    assert dependencies['scikit-learn']['module'] == 'sklearn'
    assert dependencies['jarvis-tools']['module'] == 'jarvis'
    assert dependencies['pyarrow']['import_kind'] == 'backend'
    assert dependencies['pytest']['role'] == 'test_tool'
    assert all(
        {'dependency_declarations', 'dependency_import_availability'}.issubset(
            profile['requires']
        )
        for profile in manifest['validation_profiles']
    )

    source_checks = {
        check['package']: check
        for check in validation['checks']
        if check['kind'] == 'dependency_source_imports'
    }
    assert source_checks['pandas']['consumers']
    assert 'main.py' in source_checks['pandas']['consumers']
    assert source_checks['pytest']['consumers']
    assert source_checks['pyarrow']['consumers'] == []
    ownership_checks = {
        check['package']: check
        for check in validation['checks']
        if check['kind'] == 'dependency_distribution_ownership'
    }
    assert ownership_checks.keys() == dependencies.keys()
    assert all(check['matches'] for check in ownership_checks.values())
    source_classification = next(
        check
        for check in validation['checks']
        if check['kind'] == 'source_import_classification'
    )
    discovered_python_files = set(source_classification['source_files'])
    expected_python_files = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob('*.py')
        if path.relative_to(ROOT).parts[0]
        not in {'human_docs', 'data', 'artifacts'}
        and '__pycache__' not in path.parts
    }
    assert source_classification['source_file_count'] == len(
        discovered_python_files
    )
    assert discovered_python_files == expected_python_files
    assert source_classification['unsupported_dynamic_imports'] == []
    assert validation['status'] == 'ok'


@pytest.mark.parametrize(
    ('mutate_manifest', 'mutate_requirements', 'expected_error_code'),
    [
        (
            lambda manifest: None,
            lambda text: text + 'requests>=2.0\n',
            'unmanifested_requirement',
        ),
        (
            lambda manifest: None,
            lambda text: text + 'Pandas>=2.2\n',
            'duplicate_requirement',
        ),
        (
            lambda manifest: manifest.pop('dependency_imports'),
            lambda text: text,
            'invalid_dependency_imports',
        ),
        (
            lambda manifest: manifest['dependency_imports'].append({}),
            lambda text: text,
            'invalid_dependency_import_entry',
        ),
        (
            lambda manifest: manifest['dependency_imports'].append(
                json.loads(json.dumps(manifest['dependency_imports'][0]))
            ),
            lambda text: text,
            'duplicate_dependency_package',
        ),
        (
            lambda manifest: manifest.update({
                'dependency_imports': [
                    dependency
                    for dependency in manifest['dependency_imports']
                    if dependency['package'] != 'pydantic'
                ],
            }),
            lambda text: text,
            'unmanifested_requirement',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pandas'
            ).update({'module': 'json'}),
            lambda text: text,
            'undeclared_external_import',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pandas'
            ).pop('module'),
            lambda text: text,
            'invalid_dependency_import_entry',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pandas'
            ).update({'specifier': '>=999'}),
            lambda text: text,
            'dependency_specifier_mismatch',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pandas'
            ).update({'role': 'unclassified'}),
            lambda text: text,
            'invalid_dependency_role',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pandas'
            ).update({'import_kind': 'unknown'}),
            lambda text: text,
            'invalid_dependency_import_kind',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pandas'
            ).update({'import_kind': 'backend'}),
            lambda text: text,
            'backend_dependency_imported_directly',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pandas'
            ).update({'role': 'ui'}),
            lambda text: text,
            'ui_dependency_outside_ui',
        ),
        (
            lambda manifest: _swap_dependency_modules(
                manifest, 'pandas', 'numpy'
            ),
            lambda text: text,
            'dependency_distribution_mismatch',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'pytest'
            ).update({'role': 'core_runtime'}),
            lambda text: text,
            'dependency_role_consumer_mismatch',
        ),
        (
            lambda manifest: _dependency_by_package(
                manifest, 'numpy'
            ).update({'role': 'core_runtime'}),
            lambda text: text,
            'dependency_role_consumer_mismatch',
        ),
        (
            lambda manifest: next(
                dependency
                for dependency in manifest['local_shared_imports']
                if dependency['module'] == 'filesystem'
            ).update({'owner': 'myutils/file_utils/not_the_owner.py'}),
            lambda text: text,
            'invalid_local_shared_import_entry',
        ),
        (
            lambda manifest: manifest.update({
                'local_shared_imports': [
                    dependency
                    for dependency in manifest['local_shared_imports']
                    if dependency['module'] != 'filesystem'
                ],
            }),
            lambda text: text,
            'undeclared_external_import',
        ),
    ],
)
def test_validate_agent_layout_rejects_dependency_contract_drift(
    monkeypatch,
    mutate_manifest,
    mutate_requirements,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)
    original_read = agent_state._read_text_if_present
    requirements_path = (ROOT / 'requirements.txt').resolve()

    def read_with_requirements_mutation(path):
        text = original_read(path)
        if Path(path).resolve() == requirements_path:
            return mutate_requirements(text)
        return text

    monkeypatch.setattr(
        agent_state,
        '_read_text_if_present',
        read_with_requirements_mutation,
    )

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == expected_error_code
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'source_prefix',
    [
        'import requests as client\n',
        'from requests import get as fetch\n',
    ],
    ids=('import-alias', 'import-from-alias'),
)
def test_validate_agent_layout_rejects_compile_valid_undeclared_source_import(
    monkeypatch,
    source_prefix,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'undeclared_external_import'
        and error['path'] == 'src/config.py'
        for error in validation['errors']
    )


def _validate_agent_layout_with_config_prefix(monkeypatch, source_prefix):
    original_read = agent_state._read_text_if_present
    target_path = (ROOT / 'src' / 'config.py').resolve()

    def read_with_external_import(path):
        text = original_read(path)
        if Path(path).resolve() != target_path:
            return text
        mutated = f'{source_prefix}{text}'
        compile(mutated, str(path), 'exec')
        return mutated

    monkeypatch.setattr(agent_state, '_read_text_if_present', read_with_external_import)

    return validate_agent_layout(ROOT)


def _validate_agent_layout_with_main_suffix(monkeypatch, source_suffix):
    original_read = agent_state._read_text_if_present
    target_path = (ROOT / 'main.py').resolve()

    def read_with_indirect_loader(path):
        text = original_read(path)
        if Path(path).resolve() != target_path:
            return text
        mutated = f'{text}\n{source_suffix}'
        compile(mutated, str(path), 'exec')
        return mutated

    monkeypatch.setattr(agent_state, '_read_text_if_present', read_with_indirect_loader)

    return validate_agent_layout(ROOT)


@pytest.mark.parametrize(
    ('source_prefix', 'expected_error_code'),
    [
        (
            'from importlib import import_module as load\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'import importlib\n'
            'load = importlib.import_module\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        ('__import__("requests")\n', 'undeclared_external_import'),
        (
            'import importlib as loader\n'
            'loader.import_module("requests")\n',
            'undeclared_external_import',
        ),
        (
            'import importlib\n'
            'module_name = "requests"\n'
            'importlib.import_module(module_name)\n',
            'unsupported_dynamic_import',
        ),
        (
            'from importlib import import_module as load\n'
            'module_name = "requests"\n'
            'load(module_name)\n',
            'unsupported_dynamic_import',
        ),
        (
            'import importlib\n'
            'load = importlib.import_module\n'
            'module_name = "requests"\n'
            'load(module_name)\n',
            'unsupported_dynamic_import',
        ),
        (
            'module_name = "requests"\n'
            '__import__(module_name)\n',
            'unsupported_dynamic_import',
        ),
        (
            'import builtins as runtime_builtins\n'
            'runtime_builtins.__import__("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from builtins import __import__ as load\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from typing import TYPE_CHECKING\n'
            'from importlib import import_module as load\n'
            'if TYPE_CHECKING:\n'
            '    load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'try:\n'
            '    from importlib import import_module as load\n'
            '    def load_optional_dependency():\n'
            '        return load("requests")\n'
            'except ImportError:\n'
            '    load_optional_dependency = None\n',
            'undeclared_external_import',
        ),
        (
            'import builtins\n'
            'real_import = builtins.__import__\n'
            'def delegated_import(*args, **kwargs):\n'
            '    return real_import("requests", *args, **kwargs)\n',
            'undeclared_external_import',
        ),
    ],
    ids=(
        'aliased-import-module-literal',
        'assigned-import-module-literal',
        'builtin-import-literal',
        'aliased-importlib-literal',
        'importlib-computed-name',
        'aliased-import-module-computed-name',
        'assigned-import-module-computed-name',
        'builtin-import-computed-name',
        'aliased-builtins-literal',
        'aliased-builtin-import-literal',
        'type-checking-literal',
        'nested-optional-literal',
        'delegated-wrapper-hardcoded-literal',
    ),
)
def test_validate_agent_layout_classifies_dynamic_import_forms(
    monkeypatch,
    source_prefix,
    expected_error_code,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == expected_error_code
        and error['path'] == 'src/config.py'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    ('source_prefix', 'expected_error_code'),
    [
        (
            'from importlib import import_module as load\n'
            'def use_default(load, value=load("requests")):\n'
            '    return value\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'module_name = "requests"\n'
            'async def use_kw_default(\n'
            '    *, load=None, value=load(module_name)\n'
            '):\n'
            '    return value\n',
            'unsupported_dynamic_import',
        ),
        (
            'from importlib import import_module as load\n'
            '@load("requests")\n'
            'def decorated(load):\n'
            '    return load\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'def annotated(load: load("requests")):\n'
            '    return load\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'def annotated_return() -> load("requests"):\n'
            '    load = None\n'
            '    return load\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            '@load("requests")\n'
            'class Decorated:\n'
            '    load = None\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'class Derived(load("requests")):\n'
            '    load = None\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'class Meta(metaclass=load("requests")):\n'
            '    load = None\n',
            'undeclared_external_import',
        ),
        (
            'import importlib\n'
            'def use_attribute(\n'
            '    importlib, value=importlib.import_module("requests")\n'
            '):\n'
            '    return value\n',
            'undeclared_external_import',
        ),
        (
            'import builtins\n'
            'value = (\n'
            '    lambda builtins, loaded=builtins.__import__("requests"): loaded\n'
            ')(None)\n',
            'undeclared_external_import',
        ),
        (
            'def use_builtin(\n'
            '    __import__, value=__import__("requests")\n'
            '):\n'
            '    return value\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'value = (lambda load=load("requests"): load)()\n',
            'undeclared_external_import',
        ),
    ],
    ids=(
        'function-default',
        'async-kw-default-computed',
        'function-decorator',
        'parameter-annotation',
        'return-annotation',
        'class-decorator',
        'class-base',
        'class-keyword',
        'attribute-owner-default',
        'builtins-attribute-lambda-default',
        'bare-builtin-default',
        'lambda-default',
    ),
)
def test_validate_agent_layout_classifies_definition_time_dynamic_imports(
    monkeypatch,
    source_prefix,
    expected_error_code,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == expected_error_code
        and error['path'] == 'src/config.py'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'source_prefix',
    [
        (
            'from importlib import import_module as load\n'
            'load("requests")\n'
            'load = lambda value: value\n'
        ),
        (
            'load = lambda value: value\n'
            'from importlib import import_module as load\n'
            'load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    from importlib import import_module as load\n'
            '    load("requests")\n'
            '    load = lambda value: value\n'
        ),
        (
            'from importlib import import_module as load\n'
            'class Loader:\n'
            '    value = load("requests")\n'
            '    load = lambda value: value\n'
        ),
        (
            'from importlib import import_module as load\n'
            'values = [\n'
            '    item for item in [1] for load in load("requests")\n'
            ']\n'
        ),
        (
            'from importlib import import_module as load\n'
            'def use_global():\n'
            '    global load\n'
            '    load("requests")\n'
            '    load = lambda value: value\n'
        ),
        (
            'def outer():\n'
            '    from importlib import import_module as load\n'
            '    def inner():\n'
            '        nonlocal load\n'
            '        load("requests")\n'
            '        load = lambda value: value\n'
            '    return inner\n'
        ),
        (
            'def use_late_module_owner():\n'
            '    return load("requests")\n'
            'from importlib import import_module as load\n'
            'use_late_module_owner()\n'
        ),
        (
            'def outer():\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    from importlib import import_module as load\n'
            '    return inner()\n'
        ),
    ],
    ids=(
        'module-call-before-rebind',
        'module-call-after-owner-rebind',
        'function-call-before-rebind',
        'class-call-before-shadow',
        'comprehension-own-target-iterable',
        'global-call-before-rebind',
        'nonlocal-call-before-rebind',
        'module-owner-after-function-definition',
        'closure-owner-after-inner-definition',
    ),
)
def test_validate_agent_layout_uses_dynamic_owner_at_call_position(
    monkeypatch,
    source_prefix,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'undeclared_external_import'
        and error['path'] == 'src/config.py'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'source_prefix',
    [
        (
            'from importlib import import_module as load\n'
            'values = [load("requests") for load in []]\n'
        ),
        (
            'from importlib import import_module as load\n'
            'values = [\n'
            '    item for load in [] if load("requests") for item in []\n'
            ']\n'
        ),
        (
            'from importlib import import_module as load\n'
            'values = [\n'
            '    item for load in [] for item in load("requests")\n'
            ']\n'
        ),
        (
            'import importlib\n'
            'values = [\n'
            '    importlib.import_module("requests") for importlib in []\n'
            ']\n'
        ),
        'values = [__import__("requests") for __import__ in []]\n',
        (
            'from importlib import import_module as load\n'
            'values = [[load("requests") for item in []] for load in []]\n'
        ),
        (
            'from importlib import import_module as load\n'
            'load = lambda value: value\n'
            'load("requests")\n'
        ),
        (
            'from importlib import import_module as load\n'
            'class Loader:\n'
            '    load = lambda value: value\n'
            '    value = load("requests")\n'
        ),
        (
            'from importlib import import_module as load\n'
            'def use_lambda():\n'
            '    return (lambda load: load("requests"))(lambda value: value)\n'
        ),
        (
            'from importlib import import_module as load\n'
            'def use_relative(load, value=load(".utils", __package__)):\n'
            '    return value\n'
        ),
    ],
    ids=(
        'comprehension-body-target',
        'comprehension-filter-target',
        'comprehension-later-iterator-target',
        'comprehension-attribute-target',
        'comprehension-builtin-target',
        'nested-comprehension-outer-target',
        'module-call-after-rebind',
        'class-call-after-shadow',
        'lambda-parameter',
        'relative-definition-time-import',
    ),
)
def test_validate_agent_layout_ignores_positionally_shadowed_dynamic_calls(
    monkeypatch,
    source_prefix,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'ok'
    assert validation['errors'] == []


@pytest.mark.parametrize(
    'source_prefix',
    [
        (
            'class LocalLoader:\n'
            '    def import_module(self, module_name):\n'
            '        return module_name\n'
            'local_loader = LocalLoader()\n'
            'local_loader.import_module("requests")\n'
        ),
        (
            'import importlib\n'
            'importlib.import_module(".utils", __package__)\n'
        ),
        (
            'import importlib\n'
            'class LocalLoader:\n'
            '    def import_module(self, module_name):\n'
            '        return module_name\n'
            'def use_local_loader():\n'
            '    importlib = LocalLoader()\n'
            '    return importlib.import_module("requests")\n'
        ),
        (
            'def bind_loader():\n'
            '    import importlib as loader\n'
            '    return loader\n'
            'def use_unbound_name():\n'
            '    return loader.import_module("requests")\n'
        ),
    ],
    ids=(
        'unrelated-method',
        'relative-local-dynamic-import',
        'shadowed-importlib-owner',
        'cross-scope-importlib-alias',
    ),
)
def test_validate_agent_layout_ignores_non_dependency_import_forms(
    monkeypatch,
    source_prefix,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'ok'
    assert validation['errors'] == []


@pytest.mark.parametrize(
    'caller_source',
    [
        'delegated_import("requests")\n',
        'module_name = "requests"\ndelegated_import(module_name)\n',
        'alias = delegated_import\nalias("requests")\n',
        (
            'def expose_loader():\n'
            '    return delegated_import\n'
            'expose_loader()("requests")\n'
        ),
        'loaders = [delegated_import]\nloaders[0]("requests")\n',
    ],
    ids=(
        'literal-caller',
        'computed-caller',
        'simple-alias',
        'returned-wrapper',
        'stored-wrapper',
    ),
)
def test_validate_agent_layout_rejects_delegated_import_passthrough(
    monkeypatch,
    caller_source,
):
    source_prefix = (
        'import builtins\n'
        'real_import = builtins.__import__\n'
        'def delegated_import(name, *args, **kwargs):\n'
        '    return real_import(name, *args, **kwargs)\n'
        f'{caller_source}'
    )

    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unsupported_dynamic_import'
        and error['path'] == 'src/config.py'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'caller_source',
    [
        'load = _bind_missing\nload("optional", "requests")\n',
        (
            'def expose_loader():\n'
            '    return _bind_missing\n'
            'expose_loader()("optional", "requests")\n'
        ),
        'loaders = [_bind_missing]\nloaders[0]("optional", "requests")\n',
    ],
    ids=('simple-alias', 'returned-wrapper', 'stored-wrapper'),
)
def test_validate_agent_layout_rejects_bind_missing_outside_main(
    monkeypatch,
    caller_source,
):
    source_prefix = (
        'import importlib\n'
        'def _bind_missing(name, module_name):\n'
        '    return importlib.import_module(module_name)\n'
        f'{caller_source}'
    )

    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unsupported_dynamic_import'
        and error['path'] == 'src/config.py'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'source_suffix',
    [
        'load = _bind_missing\nload("optional", "requests")\n',
        (
            'def expose_loader():\n'
            '    return _bind_missing\n'
            'expose_loader()("optional", "requests")\n'
        ),
        'loaders = [_bind_missing]\nloaders[0]("optional", "requests")\n',
    ],
    ids=('simple-alias', 'returned-wrapper', 'stored-wrapper'),
)
def test_validate_agent_layout_rejects_indirect_bind_missing_calls_in_main(
    monkeypatch,
    source_suffix,
):
    validation = _validate_agent_layout_with_main_suffix(
        monkeypatch,
        source_suffix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unsupported_dynamic_import'
        and error['path'] == 'main.py'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'source_suffix',
    [
        (
            'def use_local_loader():\n'
            '    def _bind_missing(name, module_name):\n'
            '        return module_name\n'
            '    return _bind_missing("local", "requests")\n'
        ),
        (
            'def use_parameter_loader(_bind_missing):\n'
            '    return _bind_missing("local", "requests")\n'
        ),
        (
            'def outer_loader():\n'
            '    def _bind_missing(name, module_name):\n'
            '        return module_name\n'
            '    def inner_loader():\n'
            '        nonlocal _bind_missing\n'
            '        return _bind_missing("local", "requests")\n'
        ),
        'local_loader = lambda _bind_missing: _bind_missing("local", "requests")\n',
        (
            'local_results = [\n'
            '    _bind_missing("local", "requests")\n'
            '    for _bind_missing in []\n'
            ']\n'
        ),
        (
            'def use_assigned_loader():\n'
            '    result = _bind_missing("local", "requests")\n'
            '    _bind_missing = lambda name, module_name: module_name\n'
            '    return result\n'
        ),
        (
            'def use_imported_loader():\n'
            '    from runtime.io_utils import load_config as _bind_missing\n'
            '    return _bind_missing("local", "requests")\n'
        ),
        (
            'class LocalLoader:\n'
            '    _bind_missing = staticmethod(lambda name, module_name: module_name)\n'
            '    result = _bind_missing("local", "requests")\n'
        ),
        (
            'class AttributeLoader:\n'
            '    def use_loader(self):\n'
            '        return self._bind_missing("local", "requests")\n'
        ),
        (
            'def use_exception_loader():\n'
            '    try:\n'
            '        pass\n'
            '    except Exception as _bind_missing:\n'
            '        return _bind_missing("local", "requests")\n'
        ),
        (
            'def use_context_loader(manager):\n'
            '    with manager as _bind_missing:\n'
            '        return _bind_missing("local", "requests")\n'
        ),
    ],
    ids=(
        'nested-local-function',
        'parameter',
        'closure-nonlocal',
        'lambda-parameter',
        'comprehension-target',
        'later-local-assignment',
        'local-import-alias',
        'class-local-binding',
        'class-attribute-call',
        'exception-target',
        'with-target',
    ),
)
def test_validate_agent_layout_ignores_shadowed_bind_missing_calls_in_main(
    monkeypatch,
    source_suffix,
):
    validation = _validate_agent_layout_with_main_suffix(
        monkeypatch,
        source_suffix,
    )

    assert validation['status'] == 'ok'
    assert validation['errors'] == []


@pytest.mark.parametrize(
    'source_suffix',
    [
        (
            'def use_global_loader():\n'
            '    global _bind_missing\n'
            '    return _bind_missing("local", "requests")\n'
        ),
        (
            'comprehension = [\n'
            '    None\n'
            '    for _bind_missing in _bind_missing("local", "requests")\n'
            ']\n'
        ),
        (
            'def use_default_loader(\n'
            '    _bind_missing,\n'
            '    value=_bind_missing("local", "requests"),\n'
            '):\n'
            '    return value\n'
        ),
        (
            'class LoaderBase(_bind_missing("local", "requests")):\n'
            '    _bind_missing = None\n'
        ),
    ],
    ids=(
        'global-name',
        'comprehension-first-iterable',
        'function-default-outer-scope',
        'class-base-outer-scope',
    ),
)
def test_validate_agent_layout_keeps_real_bind_missing_calls_identity_bound(
    monkeypatch,
    source_suffix,
):
    validation = _validate_agent_layout_with_main_suffix(
        monkeypatch,
        source_suffix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'undeclared_external_import'
        and error['path'] == 'main.py'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'source_suffix',
    [
        '_bind_missing = lambda name, module_name, attr_name=None: None\n',
        '_bind_missing += ()\n',
        'del _bind_missing\n',
        'from runtime.io_utils import load_config as _bind_missing\n',
        (
            'def replace_loader():\n'
            '    global _bind_missing\n'
            '    _bind_missing = lambda name, module_name, attr_name=None: None\n'
        ),
    ],
    ids=(
        'module-assignment',
        'module-augmented-assignment',
        'module-delete',
        'module-import-alias',
        'global-assignment',
    ),
)
def test_validate_agent_layout_rejects_rebound_bind_missing_in_main(
    monkeypatch,
    source_suffix,
):
    validation = _validate_agent_layout_with_main_suffix(
        monkeypatch,
        source_suffix,
    )

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unsupported_dynamic_import'
        and error['path'] == 'main.py'
        for error in validation['errors']
    )


def test_shadowed_call_cannot_authorize_bind_missing_definition(monkeypatch):
    original_read = agent_state._read_text_if_present
    target_path = (ROOT / 'main.py').resolve()

    def read_without_real_loader_calls(path):
        text = original_read(path)
        if Path(path).resolve() != target_path:
            return text
        tree = ast.parse(text)
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in {
                    '_ensure_dry_run_dependencies_loaded',
                    '_ensure_pipeline_dependencies_loaded',
                }
            ):
                node.body = [ast.Pass()]
        tree.body.extend(
            ast.parse(
                'def use_local_loader():\n'
                '    def _bind_missing(name, module_name):\n'
                '        return module_name\n'
                '    return _bind_missing("local", "pandas")\n'
            ).body
        )
        ast.fix_missing_locations(tree)
        mutated = ast.unparse(tree)
        compile(mutated, str(path), 'exec')
        return mutated

    monkeypatch.setattr(
        agent_state,
        '_read_text_if_present',
        read_without_real_loader_calls,
    )

    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unsupported_dynamic_import'
        and error['path'] == 'main.py'
        for error in validation['errors']
    )


def test_validate_agent_layout_rejects_missing_declared_dependency(monkeypatch):
    real_find_spec = agent_state.importlib.util.find_spec

    def find_spec_with_missing_pydantic(module_name):
        return None if module_name == 'pydantic' else real_find_spec(module_name)

    monkeypatch.setattr(
        agent_state.importlib.util,
        'find_spec',
        find_spec_with_missing_pydantic,
    )

    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'missing_declared_dependency'
        and error['module'] == 'pydantic'
        for error in validation['errors']
    )


def test_validate_agent_layout_rejects_unresolved_project_skill_reference(
    monkeypatch,
):
    original_read = agent_state._read_text_if_present
    skill_path = (
        ROOT / '.agents' / 'skills' / 'aiforbn-workflow' / 'SKILL.md'
    ).resolve()

    def read_with_missing_skill_reference(path):
        text = original_read(path)
        if Path(path).resolve() == skill_path:
            return f'{text}\nUse `$missing-project-skill`.\n'
        return text

    monkeypatch.setattr(
        agent_state,
        '_read_text_if_present',
        read_with_missing_skill_reference,
    )

    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unresolved_project_skill_reference'
        and error['path'] == '.agents/skills/aiforbn-workflow/SKILL.md'
        for error in validation['errors']
    )


def test_write_agent_state_rejects_human_docs_output(tmp_path):
    state = {
        'project_root': str(tmp_path),
        'schema_version': 'aiforbn.agent_state.v1',
    }
    output_path = tmp_path / 'human_docs' / 'agent_state.json'

    with pytest.raises(ValueError, match='user-owned human_docs'):
        agent_state.write_agent_state(state, output_path)

    assert not output_path.exists()


def test_write_agent_state_rejects_case_alias_of_human_docs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(agent_state, 'PROJECT_ROOT', tmp_path)
    human_docs_dir = tmp_path / 'human_docs'
    human_docs_dir.mkdir()
    case_alias_dir = tmp_path / 'HUMAN_DOCS'
    if not case_alias_dir.exists() or not case_alias_dir.samefile(human_docs_dir):
        pytest.skip('filesystem is case-sensitive')
    output_path = case_alias_dir / 'agent-state.json'
    state = {
        'project_root': str(tmp_path),
        'schema_version': 'aiforbn.agent_state.v1',
    }

    with pytest.raises(ValueError, match='user-owned human_docs'):
        agent_state.write_agent_state(state, output_path)

    assert not (human_docs_dir / output_path.name).exists()


def test_write_agent_state_serializes_before_parent_creation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(agent_state, 'PROJECT_ROOT', tmp_path)
    output_path = tmp_path / 'new-parent' / 'agent-state.json'
    state = {
        'project_root': str(tmp_path),
        'schema_version': 'aiforbn.agent_state.v1',
        'invalid': object(),
    }

    with pytest.raises(TypeError, match='JSON serializable'):
        agent_state.write_agent_state(state, output_path)

    assert not output_path.parent.exists()


def test_write_agent_state_does_not_trust_a_deceptive_state_project_root(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(agent_state, 'PROJECT_ROOT', tmp_path)
    state = {
        'project_root': str(tmp_path / 'different-project'),
        'schema_version': 'aiforbn.agent_state.v1',
    }
    output_path = tmp_path / 'human_docs' / 'agent_state.json'

    with pytest.raises(ValueError, match='user-owned human_docs'):
        agent_state.write_agent_state(state, output_path)

    assert not output_path.parent.exists()


def test_write_agent_state_rejects_a_hardlinked_output_alias(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(agent_state, 'PROJECT_ROOT', tmp_path)
    human_docs_file = tmp_path / 'human_docs' / 'agent_state.json'
    human_docs_file.parent.mkdir()
    human_docs_file.write_text('{"user_owned": true}\n', encoding='utf-8')
    output_alias = tmp_path / 'artifacts' / 'agent_state.json'
    output_alias.parent.mkdir()
    output_alias.hardlink_to(human_docs_file)
    state = {
        'project_root': str(tmp_path),
        'schema_version': 'aiforbn.agent_state.v1',
    }

    with pytest.raises(ValueError, match='multiple hard links'):
        agent_state.write_agent_state(state, output_alias)

    assert human_docs_file.read_text(encoding='utf-8') == '{"user_owned": true}\n'


def test_write_agent_state_rejects_symbolic_link_directory_and_file_parent_outputs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(agent_state, 'PROJECT_ROOT', tmp_path)
    state = {
        'project_root': str(tmp_path),
        'schema_version': 'aiforbn.agent_state.v1',
    }
    target = tmp_path / 'allowed-target.json'
    target.write_text('{"keep": true}\n', encoding='utf-8')
    symlink_output = tmp_path / 'artifacts' / 'symlink.json'
    symlink_output.parent.mkdir()
    symlink_output.symlink_to(target)
    with pytest.raises(ValueError, match='symbolic-link'):
        agent_state.write_agent_state(state, symlink_output)

    directory_output = tmp_path / 'artifacts' / 'directory.json'
    directory_output.mkdir()
    with pytest.raises(ValueError, match='regular-file'):
        agent_state.write_agent_state(state, directory_output)

    file_parent = tmp_path / 'artifacts' / 'file-parent'
    file_parent.write_text('keep', encoding='utf-8')
    with pytest.raises(ValueError, match='parent paths'):
        agent_state.write_agent_state(state, file_parent / 'state.json')

    assert target.read_text(encoding='utf-8') == '{"keep": true}\n'


def test_build_agent_state_returns_json_serializable_status():
    state = build_agent_state(ROOT)

    assert state['schema_version'] == 'aiforbn.agent_state.v1'
    assert state['status'] == 'ok'
    assert state['manifest']['project']['name'] == 'aiforbn'
    assert state['git']['branch'] is not None
    parsed = json.loads(agent_state_to_json(state))
    assert parsed['schema_version'] == 'aiforbn.agent_state.v1'


def test_validate_agent_layout_rejects_incomplete_v18_alignment_contract():
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    manifest['research_plan_alignment']['implementation_anchors'] = [
        'bounded_bn_centered_design_space'
    ]

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'missing_research_plan_alignment_anchors'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    ('mutate_manifest', 'expected_error_code'),
    [
        (
            lambda manifest: manifest.pop('human_docs_policy'),
            'invalid_human_docs_policy',
        ),
        (
            lambda manifest: manifest.update({'human_docs_policy': []}),
            'invalid_human_docs_policy',
        ),
        (
            lambda manifest: manifest['human_docs_policy'].update({
                'policy_id': 'agent_owned',
            }),
            'unexpected_human_docs_policy',
        ),
        (
            lambda manifest: manifest['human_docs_policy'].update({
                'path': 'agent_docs/',
            }),
            'unexpected_human_docs_policy',
        ),
        (
            lambda manifest: manifest['human_docs_policy'].update({
                'owner': 'agent',
            }),
            'unexpected_human_docs_policy',
        ),
        (
            lambda manifest: manifest['human_docs_policy'].update({
                'default_access': 'read_write',
            }),
            'unexpected_human_docs_policy',
        ),
        (
            lambda manifest: manifest['human_docs_policy'].update({
                'write_condition': 'implicit_agent_task',
            }),
            'unexpected_human_docs_policy',
        ),
        (
            lambda manifest: manifest['human_docs_policy'].update({
                'agent_contract_authority': True,
            }),
            'unexpected_human_docs_policy',
        ),
    ],
)
def test_validate_agent_layout_rejects_weakened_human_docs_policy(
    mutate_manifest,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == expected_error_code for error in validation['errors'])


@pytest.mark.parametrize(
    'relative_path',
    ['AGENTS.md', 'src/runtime/PY_FILES_SUMMARY.md'],
)
def test_validate_agent_layout_rejects_missing_human_docs_marker(monkeypatch, relative_path):
    original_read = agent_state._read_text_if_present

    def read_without_root_marker(path: Path) -> str:
        text = original_read(path)
        if path == ROOT / relative_path:
            return text.replace(agent_state.HUMAN_DOCS_POLICY_MARKER, '')
        return text

    monkeypatch.setattr(agent_state, '_read_text_if_present', read_without_root_marker)

    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'missing_human_docs_policy_marker'
        and error['path'] == relative_path
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    ('mutate_manifest', 'expected_error_code'),
    [
        (
            lambda manifest: manifest.pop('validation_profiles'),
            'invalid_validation_profiles',
        ),
        (
            lambda manifest: manifest.update({'validation_profiles': []}),
            'invalid_validation_profiles',
        ),
        (
            lambda manifest: manifest['validation_profiles'][0].pop('use_when'),
            'missing_validation_profile_use_when',
        ),
        (
            lambda manifest: manifest['validation_profiles'][0].update(
                {'commands': []}
            ),
            'invalid_validation_profile_commands',
        ),
        (
            lambda manifest: manifest['validation_profiles'][0].update({
                'commands': [
                    command
                    for command in manifest['validation_profiles'][0]['commands']
                    if command != 'focused_regression'
                ],
            }),
            'validation_profile_missing_capabilities',
        ),
        (
            lambda manifest: manifest.update({
                'validation_profiles': [
                    profile
                    for profile in manifest['validation_profiles']
                    if profile['name'] != 'ui_edit'
                ],
            }),
            'missing_required_validation_profiles',
        ),
        (
            lambda manifest: manifest['validation_profiles'].append({
                'name': 'dependency_blind_extra_profile',
                'commands': ['fast_smoke'],
                'use_when': 'a bounded new edit class',
                'requires': ['pipeline_wiring_smoke'],
            }),
            'validation_profile_missing_dependency_capabilities',
        ),
    ],
)
def test_validate_agent_layout_rejects_incomplete_validation_profiles(
    mutate_manifest,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == expected_error_code for error in validation['errors'])


@pytest.mark.parametrize(
    ('mutate_manifest', 'expected_error_code'),
    [
        (
            lambda manifest: next(
                entry
                for entry in manifest['validation_commands']
                if entry['name'] == 'focused_regression'
            ).update({'scope': 'does_not_cover_public_surfaces'}),
            'unexpected_validation_command_contract',
        ),
        (
            lambda manifest: next(
                profile
                for profile in manifest['validation_profiles']
                if profile['name'] == 'ui_edit'
            ).update({'use_when': 'docs only'}),
            'unexpected_validation_profile_contract',
        ),
        (
            lambda manifest: next(
                entry
                for entry in manifest['validation_commands']
                if entry['name'] == 'focused_regression'
            ).update({'provides': []}),
            'unexpected_validation_command_contract',
        ),
        (
            lambda manifest: next(
                profile
                for profile in manifest['validation_profiles']
                if profile['name'] == 'ui_edit'
            ).update({
                'requires': [
                    'agent_contract',
                    'dependency_declarations',
                    'streamlit_renderer_contract',
                ],
            }),
            'unexpected_validation_profile_contract',
        ),
    ],
)
def test_validate_agent_layout_rejects_validation_scope_claim_drift(
    mutate_manifest,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == expected_error_code for error in validation['errors'])


@pytest.mark.parametrize(
    ('mutate_manifest', 'expected_error_code'),
    [
        (
            lambda manifest: manifest.update({
                'entrypoints': [
                    entry
                    for entry in manifest['entrypoints']
                    if entry['name'] != 'emit_agent_commands'
                ],
            }),
            'missing_required_entrypoints',
        ),
        (
            lambda manifest: (
                manifest.update({
                    'validation_commands': [
                        command
                        for command in manifest['validation_commands']
                        if command['name'] != 'full_src_tests'
                    ],
                }),
                [
                    profile.update({
                        'commands': [
                            command
                            for command in profile['commands']
                            if command != 'full_src_tests'
                        ],
                    })
                    for profile in manifest['validation_profiles']
                ],
            ),
            'missing_required_validation_commands',
        ),
        (
            lambda manifest: next(
                entry
                for entry in manifest['entrypoints']
                if entry['name'] == 'fast_smoke'
            ).update({'command': 'python3 broken.py'}),
            'unexpected_entrypoint_contract',
        ),
        (
            lambda manifest: next(
                entry
                for entry in manifest['entrypoints']
                if entry['name'] == 'fast_smoke'
            ).update({'writes_artifacts': True}),
            'unexpected_entrypoint_contract',
        ),
        (
            lambda manifest: next(
                entry
                for entry in manifest['validation_commands']
                if entry['name'] == 'full_src_tests'
            ).update({'command': 'python3 -m pytest -q tests'}),
            'unexpected_validation_command_contract',
        ),
    ],
)
def test_validate_agent_layout_rejects_incomplete_command_surface(
    mutate_manifest,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == expected_error_code for error in validation['errors'])


@pytest.mark.parametrize(
    ('mutate_manifest', 'expected_error_code'),
    [
        (
            lambda manifest: manifest.update({'project_skills': []}),
            'unexpected_project_skills',
        ),
        (
            lambda manifest: [
                skill.update({'name': f'renamed-{index}'})
                for index, skill in enumerate(manifest['project_skills'])
            ],
            'unexpected_project_skills',
        ),
        (
            lambda manifest: manifest.update({'retired_guidance_files': []}),
            'unexpected_retired_guidance_files',
        ),
        (
            lambda manifest: manifest.pop('retired_guidance_files'),
            'unexpected_retired_guidance_files',
        ),
    ],
)
def test_validate_agent_layout_pins_skills_and_retired_guidance(
    mutate_manifest,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == expected_error_code for error in validation['errors'])


@pytest.mark.parametrize(
    ('relative_path', 'mutate_text', 'expected_error_code'),
    [
        (
            '.agents/skills/aiforbn-workflow/SKILL.md',
            lambda text: text.replace(
                'name: aiforbn-workflow',
                'name: renamed-workflow',
                1,
            ),
            'unexpected_project_skill_frontmatter',
        ),
        (
            'requirements.txt',
            lambda text: '\n'.join(
                line
                for line in text.splitlines()
                if not line.startswith('streamlit')
            ) + '\n',
            'missing_dependency_requirement',
        ),
        (
            'skills/ai_native_workflow.txt',
            lambda text: text.replace('`ui_edit`', '`renamed_ui_profile`'),
            'missing_validation_profile_guidance',
        ),
    ],
)
def test_validate_agent_layout_rejects_source_of_truth_reachability_drift(
    monkeypatch,
    relative_path,
    mutate_text,
    expected_error_code,
):
    original_read = agent_state._read_text_if_present
    target_path = (ROOT / relative_path).resolve()

    def read_with_mutation(path):
        text = original_read(path)
        return mutate_text(text) if Path(path).resolve() == target_path else text

    monkeypatch.setattr(agent_state, '_read_text_if_present', read_with_mutation)

    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == expected_error_code
        for error in validation['errors']
    )


def test_local_instruction_path_validation_rejects_missing_absolute_path(tmp_path: Path):
    skill_path = tmp_path / '.agents' / 'skills' / 'demo' / 'SKILL.md'
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text('Read `/definitely/missing/aiforbn/AGENTS.md`.\n', encoding='utf-8')
    errors = []

    agent_state._validate_local_instruction_paths(
        tmp_path,
        {
            'project_skills': [{'path': '.agents/skills/demo/SKILL.md'}],
            'modules': [],
        },
        errors,
    )

    assert [error['code'] for error in errors] == ['stale_local_instruction_path']


@pytest.mark.parametrize(
    ('mutate_manifest', 'expected_error_code'),
    [
        (
            lambda manifest: manifest.update({'source_of_truth_files': []}),
            'missing_source_of_truth_files',
        ),
        (
            lambda manifest: manifest.update({
                'source_of_truth_files': [
                    path
                    for path in manifest['source_of_truth_files']
                    if path != 'AGENTS.md'
                ],
            }),
            'missing_source_of_truth_files',
        ),
        (
            lambda manifest: manifest.update({'source_of_truth_files': 'AGENTS.md'}),
            'invalid_source_of_truth_files',
        ),
    ],
)
def test_validate_agent_layout_rejects_incomplete_source_of_truth_surface(
    mutate_manifest,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == expected_error_code for error in validation['errors'])


@pytest.mark.parametrize(
    'mutate_manifest',
    [
        lambda manifest: manifest.update({'modules': []}),
        lambda manifest: manifest.update({
            'modules': [
                module
                for module in manifest['modules']
                if module['name'] != 'runtime'
            ],
        }),
    ],
)
def test_validate_agent_layout_rejects_incomplete_module_contract_surface(
    mutate_manifest,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_manifest(manifest)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == 'missing_required_modules' for error in validation['errors'])


@pytest.mark.parametrize(
    'mutate_module',
    [
        lambda module: module.update({'allowed_dependencies': ['ui']}),
        lambda module: module.update({'path': 'src/materials'}),
        lambda module: module.update({'public_surface': 'src/materials/PY_FILES_SUMMARY.md'}),
        lambda module: module.update({'agent_rules': 'src/materials/AGENTS.md'}),
        lambda module: module.update({'local_utils': 'src/materials/utils.py'}),
        lambda module: module.update({'role': 'weakened_runtime_role'}),
    ],
)
def test_validate_agent_layout_rejects_mutated_module_contract(
    mutate_module,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    runtime_module = next(
        module for module in manifest['modules'] if module['name'] == 'runtime'
    )
    mutate_module(runtime_module)

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unexpected_module_contract'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'module_name',
    ['runtime', 'materials', 'torch_models', 'ui', 'tests', 'template'],
)
def test_validate_agent_layout_pins_every_declared_module_contract(module_name):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    module = next(
        entry for entry in manifest['modules'] if entry['name'] == module_name
    )
    module['role'] = f'weakened_{module_name}_role'

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'unexpected_module_contract'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    ('mutate_modules', 'expected_error_code'),
    [
        (
            lambda modules: modules.append(json.loads(json.dumps(modules[0]))),
            'duplicate_module_contracts',
        ),
        (
            lambda modules: modules.append({
                'name': 'unknown',
                'path': 'src/runtime',
                'role': 'unexpected',
                'public_surface': 'src/runtime/PY_FILES_SUMMARY.md',
                'agent_rules': 'src/runtime/AGENTS.md',
                'local_utils': 'src/runtime/utils.py',
                'allowed_dependencies': [],
            }),
            'unexpected_module_contract',
        ),
    ],
)
def test_validate_agent_layout_rejects_duplicate_or_unknown_module_contracts(
    mutate_modules,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_modules(manifest['modules'])

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == expected_error_code
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    ('mutate_alignment', 'expected_error_code'),
    [
        (
            lambda alignment: alignment.update({'status': 'weakened'}),
            'unexpected_research_plan_alignment_status',
        ),
        (
            lambda alignment: alignment.update({'source_files': [123]}),
            'invalid_research_plan_alignment_source',
        ),
        (
            lambda alignment: alignment.update({
                'source_files': [
                    'human_docs/research_plan/ai_for_bn_research_plan_v18.tex',
                    'human_docs/research_plan/ai_for_bn_research_plan_v18.bib',
                    'AGENTS.md',
                ],
            }),
            'unexpected_research_plan_alignment_sources',
        ),
        (
            lambda alignment: alignment.update({
                'non_claims': ['open_ended_material_discovery'],
            }),
            'missing_research_plan_non_claims',
        ),
        (
            lambda alignment: alignment.update({
                'deliverable_chain': ['bn_dataset', 'benchmarked_models'],
            }),
            'missing_research_plan_deliverables',
        ),
        (
            lambda alignment: alignment.update({
                'deliverable_chain': [
                    'bn_dataset',
                    'benchmarked_models',
                    'ranked_candidates',
                    'structure_handoff',
                    'ranked_candidates',
                    'structure_handoff',
                    'technical_report',
                ],
            }),
            'unexpected_research_plan_deliverable_chain',
        ),
    ],
)
def test_validate_agent_layout_rejects_incomplete_v18_alignment_fields(
    mutate_alignment,
    expected_error_code,
):
    manifest = json.loads(json.dumps(load_agent_manifest(ROOT)))
    mutate_alignment(manifest['research_plan_alignment'])

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(error['code'] == expected_error_code for error in validation['errors'])
