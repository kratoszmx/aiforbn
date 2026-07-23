import ast
import json
import os
from pathlib import Path
import subprocess
import sys

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
JARVIS_TARGET_SYMBOLS = {
    'jarvis.db.figshare': (
        'get_db_info',
        'get_request_data',
    ),
}
MATMINER_TARGET_SYMBOLS = {
    'matminer.featurizers.base': (
        'MultipleFeaturizer',
    ),
    'matminer.featurizers.composition': (
        'ElementProperty',
        'Stoichiometry',
    ),
}
PYMATGEN_CORE_SYMBOLS = (
    'Composition',
    'Element',
    'Structure',
    'Lattice',
)


@pytest.fixture
def cleared_dependency_import_probe_cache():
    agent_state._clear_dependency_import_probe_cache()
    yield
    agent_state._clear_dependency_import_probe_cache()


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
    pytest_non_vacuity_targets = {
        entry['name']: entry['pytest_non_vacuity_targets']
        for entry in command_index['validation_commands']
        if 'pytest_non_vacuity_targets' in entry
    }
    assert pytest_non_vacuity_targets == {
        'focused_regression': [
            'src/tests/test_main.py',
            'src/tests/test_public_surfaces.py',
            'src/runtime/tests/test_agent_state.py',
            'src/runtime/tests/test_io_utils.py',
        ],
        'full_src_tests': ['src'],
        'ui_render_smoke': ['src/ui/tests/test_streamlit_app.py'],
    }
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
    assert dependencies['torch']['import_probe_preloads'] == ['numpy', 'sklearn']
    assert dependencies['jarvis-tools']['import_probe_targets'] == list(
        JARVIS_TARGET_SYMBOLS
    )
    assert dependencies['jarvis-tools']['import_probe_symbols'] == {
        target: list(symbols)
        for target, symbols in JARVIS_TARGET_SYMBOLS.items()
    }
    assert dependencies['matminer']['import_probe_targets'] == list(
        MATMINER_TARGET_SYMBOLS
    )
    assert dependencies['matminer']['import_probe_symbols'] == {
        target: list(symbols)
        for target, symbols in MATMINER_TARGET_SYMBOLS.items()
    }
    assert dependencies['pymatgen']['import_probe_targets'] == [
        'pymatgen.core'
    ]
    assert dependencies['pymatgen']['import_probe_symbols'] == {
        'pymatgen.core': list(PYMATGEN_CORE_SYMBOLS),
    }
    assert all(
        'import_probe_preloads' not in dependency
        for package, dependency in dependencies.items()
        if package != 'torch'
    )
    assert all(
        'import_probe_targets' not in dependency
        for package, dependency in dependencies.items()
        if package not in {'jarvis-tools', 'matminer', 'pymatgen'}
    )
    assert all(
        'import_probe_symbols' not in dependency
        for package, dependency in dependencies.items()
        if package not in {'jarvis-tools', 'matminer', 'pymatgen'}
    )
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
    import_checks = {
        check['module']: check
        for check in validation['checks']
        if check['kind'] == 'dependency_import'
    }
    assert import_checks['torch']['import_probe_preloads'] == ['numpy', 'sklearn']
    assert import_checks['pyarrow']['import_probe_preloads'] == []
    assert import_checks['jarvis']['import_probe_targets'] == list(
        JARVIS_TARGET_SYMBOLS
    )
    assert import_checks['jarvis']['import_probe_symbols'] == {
        target: list(symbols)
        for target, symbols in JARVIS_TARGET_SYMBOLS.items()
    }
    assert import_checks['matminer']['import_probe_targets'] == list(
        MATMINER_TARGET_SYMBOLS
    )
    assert import_checks['matminer']['import_probe_symbols'] == {
        target: list(symbols)
        for target, symbols in MATMINER_TARGET_SYMBOLS.items()
    }
    assert import_checks['pymatgen']['import_probe_targets'] == [
        'pymatgen.core'
    ]
    assert import_checks['pymatgen']['import_probe_symbols'] == {
        'pymatgen.core': list(PYMATGEN_CORE_SYMBOLS),
    }
    source_import_symbols = {
        target: set()
        for target in (
            *JARVIS_TARGET_SYMBOLS,
            *MATMINER_TARGET_SYMBOLS,
            'pymatgen.core',
        )
    }
    for python_path in ROOT.rglob('*.py'):
        relative_path = python_path.relative_to(ROOT)
        if (
            relative_path.parts[0] in {'human_docs', 'data', 'artifacts'}
            or '__pycache__' in relative_path.parts
        ):
            continue
        for node in ast.walk(
            ast.parse(python_path.read_text(encoding='utf-8'))
        ):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module in source_import_symbols
            ):
                source_import_symbols[node.module].update(
                    alias.name for alias in node.names
                )
    assert {
        target: source_import_symbols[target]
        for target in JARVIS_TARGET_SYMBOLS
    } == {
        target: set(symbols)
        for target, symbols in JARVIS_TARGET_SYMBOLS.items()
    }
    assert {
        target: source_import_symbols[target]
        for target in MATMINER_TARGET_SYMBOLS
    } == {
        target: set(symbols)
        for target, symbols in MATMINER_TARGET_SYMBOLS.items()
    }
    assert source_import_symbols['pymatgen.core'] == set(
        PYMATGEN_CORE_SYMBOLS
    )
    assert all(check['available'] for check in import_checks.values())
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
    'invalid_preloads',
    [
        [],
        'numpy',
        ['torch'],
        ['numpy', 'numpy'],
        ['not-valid!'],
        ['numpy', 'missing_dependency_module'],
    ],
)
def test_dependency_contract_rejects_invalid_import_probe_preloads(
    invalid_preloads,
):
    manifest = load_agent_manifest(ROOT)
    _dependency_by_package(manifest, 'torch')['import_probe_preloads'] = (
        invalid_preloads
    )

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'invalid_dependency_import_probe_preloads'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'invalid_targets',
    [
        [],
        'pymatgen.core',
        ['pymatgen'],
        ['pymatgen.core', 'pymatgen.core'],
        ['not-valid!'],
        ['jarvis.db'],
    ],
)
def test_dependency_contract_rejects_invalid_import_probe_targets(
    invalid_targets,
):
    manifest = load_agent_manifest(ROOT)
    _dependency_by_package(manifest, 'pymatgen')['import_probe_targets'] = (
        invalid_targets
    )

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'invalid_dependency_import_probe_targets'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    'invalid_symbols',
    [
        [],
        {},
        {'pymatgen.io.cif': ['CifParser']},
        {'pymatgen.core': []},
        {'pymatgen.core': 'Composition'},
        {'pymatgen.core': ['Composition', 'Composition']},
        {'pymatgen.core': ['not-valid!']},
        {'pymatgen.core': ['class']},
        {'pymatgen.core': [1]},
    ],
)
def test_dependency_contract_rejects_invalid_import_probe_symbols(
    invalid_symbols,
):
    manifest = load_agent_manifest(ROOT)
    _dependency_by_package(manifest, 'pymatgen')['import_probe_symbols'] = (
        invalid_symbols
    )

    validation = validate_agent_layout(ROOT, manifest)

    assert validation['status'] == 'error'
    assert any(
        error['code'] == 'invalid_dependency_import_probe_symbols'
        for error in validation['errors']
    )


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
    ('source_prefix', 'expected_error_code'),
    [
        (
            'def outer(flag):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        load = lambda name: name\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(flag, module_name):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        load = lambda name: name\n'
            '    def inner():\n'
            '        return load(module_name)\n'
            '    return inner()\n',
            'unsupported_dynamic_import',
        ),
        (
            'def outer():\n'
            '    try:\n'
            '        from importlib import import_module as load\n'
            '    except ImportError:\n'
            '        load = lambda name: name\n'
            '    finally:\n'
            '        completed = True\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer():\n'
            '    try:\n'
            '        from importlib import import_module as load\n'
            '    except* ImportError:\n'
            '        load = lambda name: name\n'
            '    class Inner:\n'
            '        value = load("requests")\n'
            '    return Inner\n',
            'undeclared_external_import',
        ),
        (
            'def outer(value):\n'
            '    match value:\n'
            '        case 0:\n'
            '            from importlib import import_module as load\n'
            '        case _:\n'
            '            load = lambda name: name\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(values):\n'
            '    from importlib import import_module as load\n'
            '    for load in values:\n'
            '        pass\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(manager, flag):\n'
            '    if flag:\n'
            '        import importlib as loader\n'
            '    else:\n'
            '        with manager as loader:\n'
            '            pass\n'
            '    def inner():\n'
            '        return loader.import_module("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(flag):\n'
            '    from importlib import import_module as load\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    if flag:\n'
            '        load = lambda name: name\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'if FLAG:\n'
            '    from importlib import import_module as load\n'
            'else:\n'
            '    load = lambda name: name\n'
            'def inner():\n'
            '    global load\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def outer(flag):\n'
            '    if flag:\n'
            '        from builtins import __import__ as load\n'
            '    else:\n'
            '        load = lambda name: name\n'
            '    def inner():\n'
            '        nonlocal load\n'
            '        return load("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(flag):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        from builtins import __import__ as load\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(flag):\n'
            '    if flag:\n'
            '        import builtins as loader\n'
            '    else:\n'
            '        loader = object()\n'
            '    def inner():\n'
            '        return loader.__import__("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(flag):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        load = lambda name: name\n'
            '    inner = lambda: load("requests")\n'
            '    return inner()\n',
            'undeclared_external_import',
        ),
        (
            'def outer(flag):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        load = lambda name: name\n'
            '    class Inner:\n'
            '        value = load("requests")\n'
            '    return Inner\n',
            'undeclared_external_import',
        ),
        (
            'def set_owner():\n'
            '    global load\n'
            '    from importlib import import_module as load\n'
            'load = lambda name: name\n'
            'def use_loader():\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def outer():\n'
            '    def set_owner():\n'
            '        nonlocal load\n'
            '        from importlib import import_module as load\n'
            '    load = lambda name: name\n'
            '    def use_loader():\n'
            '        return load("requests")\n'
            '    return use_loader\n',
            'undeclared_external_import',
        ),
        (
            'if FLAG:\n'
            '    __import__ = lambda name: name\n'
            'def use_loader():\n'
            '    return __import__("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'class Loader:\n'
            '    global load\n'
            '    value = load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def outer():\n'
            '    from importlib import import_module as load\n'
            '    class Loader:\n'
            '        nonlocal load\n'
            '        value = load("requests")\n'
            '    return Loader\n',
            'undeclared_external_import',
        ),
        (
            '__import__ = lambda name: name\n'
            'del __import__\n'
            'def use_loader():\n'
            '    return __import__("requests")\n',
            'undeclared_external_import',
        ),
        (
            '__import__ = lambda name: name\n'
            'if FLAG:\n'
            '    del __import__\n'
            'def use_loader():\n'
            '    return __import__("requests")\n',
            'undeclared_external_import',
        ),
    ],
    ids=(
        'if-else-literal',
        'if-else-computed',
        'try-except-finally',
        'try-star-class-definition',
        'match-case',
        'loop-target',
        'with-target-attribute',
        'owner-before-definition-shadow-after',
        'global-mixed-binding',
        'nonlocal-mixed-binding',
        'mixed-callable-owner-kinds',
        'mixed-builtins-module-owner',
        'lambda-mixed-binding',
        'class-mixed-binding',
        'child-global-owner',
        'child-nonlocal-owner',
        'conditional-builtin-shadow',
        'class-global-current-owner',
        'class-nonlocal-current-owner',
        'builtin-shadow-then-delete',
        'builtin-shadow-then-conditional-delete',
    ),
)
def test_validate_agent_layout_rejects_ambiguous_runtime_import_owners(
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
            'def outer(flag):\n'
            '    if flag:\n'
            '        load = lambda name: name\n'
            '    else:\n'
            '        def load(name):\n'
            '            return name\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    return inner()\n'
        ),
        (
            'from importlib import import_module as load\n'
            'def sibling(load):\n'
            '    return load("requests")\n'
            'def owner_call():\n'
            '    return load(".helpers", __package__)\n'
        ),
        (
            'def outer(flag):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        load = lambda name, package=None: name\n'
            '    def inner():\n'
            '        return load(".helpers", __package__)\n'
            '    return inner()\n'
        ),
        (
            'import importlib\n'
            'def use_capture(subject):\n'
            '    match subject:\n'
            '        case {"loader": importlib}:\n'
            '            return importlib.import_module("requests")\n'
        ),
        (
            'import importlib\n'
            'import builtins\n'
            'def use_captures(subject):\n'
            '    match subject:\n'
            '        case [importlib, *builtins]:\n'
            '            return (\n'
            '                importlib.import_module("requests"),\n'
            '                builtins.__import__("requests"),\n'
            '            )\n'
        ),
        (
            'import importlib\n'
            'def use_rest(subject):\n'
            '    match subject:\n'
            '        case {"loader": loader, **importlib}:\n'
            '            return importlib.import_module("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'class Loader:\n'
            '    global load\n'
            '    value = load("requests")\n'
            'from importlib import import_module as load\n'
        ),
        (
            'def outer():\n'
            '    load = lambda name: name\n'
            '    class Loader:\n'
            '        nonlocal load\n'
            '        value = load("requests")\n'
            '    from importlib import import_module as load\n'
            '    return Loader\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'del __import__\n'
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    return __import__("requests")\n'
        ),
    ],
    ids=(
        'pure-nonowner-branches',
        'unrelated-sibling-and-relative-owner',
        'ambiguous-relative-local-import',
        'match-as-capture',
        'match-star-captures',
        'match-mapping-rest-capture',
        'class-global-future-owner',
        'class-nonlocal-future-owner',
        'builtin-delete-then-shadow',
    ),
)
def test_validate_agent_layout_ignores_non_dependency_runtime_owner_controls(
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
            'from importlib import import_module as load\n'
            'def use_loader():\n'
            '    global load\n'
            '    del load\n'
            '    return load("requests")\n'
        ),
        (
            'from builtins import __import__ as load\n'
            'def use_loader():\n'
            '    global load\n'
            '    del load\n'
            '    return load("requests")\n'
        ),
        (
            'def outer():\n'
            '    from importlib import import_module as load\n'
            '    def use_loader():\n'
            '        nonlocal load\n'
            '        del load\n'
            '        return load("requests")\n'
            '    return use_loader\n'
        ),
        (
            'def outer():\n'
            '    from builtins import __import__\n'
            '    def use_loader():\n'
            '        nonlocal __import__\n'
            '        del __import__\n'
            '        return __import__("requests")\n'
            '    return use_loader\n'
        ),
        (
            'import importlib\n'
            'def use_loader():\n'
            '    global importlib\n'
            '    del importlib\n'
            '    return importlib.import_module("requests")\n'
        ),
        (
            'def outer():\n'
            '    from importlib import import_module as load\n'
            '    def use_loader(outer_values, inner_values):\n'
            '        nonlocal load\n'
            '        for outer in outer_values:\n'
            '            for inner in inner_values:\n'
            '                try:\n'
            '                    observe(inner)\n'
            '                finally:\n'
            '                    del load\n'
            '                return load("requests")\n'
            '    return use_loader\n'
        ),
    ],
    ids=(
        'global-importlib-alias',
        'global-builtin-alias',
        'nonlocal-importlib-alias',
        'nonlocal-builtin-name',
        'global-importlib-module',
        'nested-nonlocal-finally-delete',
    ),
)
def test_validate_agent_layout_ignores_definitely_deleted_declared_targets(
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
    ('source_prefix', 'expected_error_code'),
    [
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    del __import__\n'
            '    return __import__("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'def use_loader(flag):\n'
            '    global load\n'
            '    if flag:\n'
            '        del load\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def outer():\n'
            '    from importlib import import_module as load\n'
            '    def use_loader(flag):\n'
            '        nonlocal load\n'
            '        if flag:\n'
            '            del load\n'
            '        return load("requests")\n'
            '    return use_loader\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values):\n'
            '    from importlib import import_module as load\n'
            '    for value in values:\n'
            '        del load\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values):\n'
            '    from importlib import import_module as load\n'
            '    for value in values:\n'
            '        break\n'
            '    else:\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values, flag):\n'
            '    from importlib import import_module as load\n'
            '    for value in values:\n'
            '        try:\n'
            '            break\n'
            '        finally:\n'
            '            if flag:\n'
            '                del load\n'
            '    else:\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'def use_loader(values):\n'
            '    global load\n'
            '    for value in values:\n'
            '        try:\n'
            '            maybe_fail()\n'
            '            try:\n'
            '                observe(value)\n'
            '            finally:\n'
            '                del load\n'
            '        finally:\n'
            '            load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'def use_loader(values):\n'
            '    global load\n'
            '    for value in values:\n'
            '        try:\n'
            '            try:\n'
            '                observe(value)\n'
            '            finally:\n'
            '                maybe_fail()\n'
            '                del load\n'
            '        finally:\n'
            '            load("requests")\n',
            'undeclared_external_import',
        ),
    ],
    ids=(
        'global-builtin-fallback',
        'conditional-global-delete',
        'conditional-nonlocal-delete',
        'zero-iteration-owner-fallback',
        'uncovered-break-owner-fallback',
        'conditional-finally-delete',
        'outer-try-prefix-may-skip-nested-delete',
        'inner-finally-prefix-may-skip-delete',
    ),
)
def test_validate_agent_layout_preserves_feasible_deleted_binding_fallbacks(
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
            'if FLAG:\n'
            '    from importlib import import_module as load\n'
            'else:\n'
            '    load = lambda name: name\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(flag, module_name):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        load = lambda name: name\n'
            '    return load(module_name)\n',
            'unsupported_dynamic_import',
        ),
        (
            'class Loader:\n'
            '    if FLAG:\n'
            '        import importlib as loader\n'
            '    else:\n'
            '        loader = object()\n'
            '    value = loader.import_module("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'if FLAG:\n'
            '    if OTHER_FLAG:\n'
            '        load = lambda name: name\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    try:\n'
            '        from importlib import import_module as load\n'
            '    except ImportError:\n'
            '        load = lambda name: name\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    try:\n'
            '        from importlib import import_module as load\n'
            '    except ImportError:\n'
            '        load = lambda name: name\n'
            '    finally:\n'
            '        result = load("requests")\n'
            '    return result\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except* ValueError:\n'
            '        from importlib import import_module as load\n'
            '    except* TypeError:\n'
            '        load = lambda name: name\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(value):\n'
            '    match value:\n'
            '        case 0:\n'
            '            from importlib import import_module as load\n'
            '        case _:\n'
            '            load = lambda name: name\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'if FLAG:\n'
            '    __import__ = lambda name: name\n'
            '__import__("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from builtins import __import__ as load\n'
            'if FLAG:\n'
            '    del load\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(flag):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        load = lambda name: name\n'
            '    alias = load\n'
            '    return alias("requests")\n',
            'undeclared_external_import',
        ),
    ],
    ids=(
        'module-if-else-join',
        'function-if-else-computed-join',
        'class-if-else-attribute-join',
        'nested-optional-shadow-join',
        'try-except-join',
        'try-finally-call',
        'try-star-handler-join',
        'match-case-join',
        'module-conditional-builtin-shadow',
        'conditional-delete-owner-path',
        'post-join-alias',
    ),
)
def test_validate_agent_layout_rejects_direct_conditional_owner_paths(
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
            'load = lambda name: name\n'
            'load("requests")\n'
            'if FLAG:\n'
            '    from importlib import import_module as load\n'
        ),
        (
            'def use_loader(flag):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        return load("requests")\n'
        ),
        (
            'def use_loader(flag, other):\n'
            '    if flag:\n'
            '        from importlib import import_module as load\n'
            '    elif other:\n'
            '        return load("requests")\n'
            '    return None\n'
        ),
        (
            'def use_loader():\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError:\n'
            '        from importlib import import_module as load\n'
            '    except TypeError:\n'
            '        return load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError:\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        return load("requests")\n'
        ),
        (
            'def use_loader(value):\n'
            '    match value:\n'
            '        case 0:\n'
            '            from importlib import import_module as load\n'
            '        case 1:\n'
            '            return load("requests")\n'
            '    return None\n'
        ),
        (
            'from importlib import import_module as load\n'
            'class Loader:\n'
            '    if FLAG:\n'
            '        load = lambda name: name\n'
            '    else:\n'
            '        def load(name):\n'
            '            return name\n'
            '    value = load("requests")\n'
        ),
        (
            'def use_loader(flag):\n'
            '    if flag:\n'
            '        __import__ = lambda name: name\n'
            '    return __import__("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    from importlib import import_module as load\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except Exception:\n'
            '        pass\n'
            '    finally:\n'
            '        load = lambda name: name\n'
            '    return load("requests")\n'
        ),
    ],
    ids=(
        'call-before-conditional',
        'exclusive-if-else-arm',
        'exclusive-if-elif-arm',
        'exclusive-except-handler',
        'exclusive-try-else',
        'exclusive-match-case',
        'exhaustive-class-nonowner-branches',
        'function-local-conditional-builtin-shadow',
        'finally-nonowner-dominates',
    ),
)
def test_validate_agent_layout_ignores_infeasible_direct_conditional_owners(
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
            'load = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as load:\n'
            '    from importlib import import_module as load\n'
            'load("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'class Loader:\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '    value = load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '    return load("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'def use_loader():\n'
            '    global load\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '    return load("requests")\n'
        ),
        (
            'def outer():\n'
            '    load = lambda name: name\n'
            '    def use_loader():\n'
            '        nonlocal load\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as load:\n'
            '            from importlib import import_module as load\n'
            '        return load("requests")\n'
            '    return use_loader\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            'finally:\n'
            '    load("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as load:\n'
            '    from importlib import import_module as load\n'
            '    del load\n'
            'except TypeError as load:\n'
            '    del load\n'
            '    from importlib import import_module as load\n'
            'load("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as load:\n'
            '    try:\n'
            '        nested_operation()\n'
            '    except TypeError as load:\n'
            '        from importlib import import_module as load\n'
            'load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as load:\n'
            '            from importlib import import_module as load\n'
            '            return None\n'
            '    finally:\n'
            '        load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as load:\n'
            '            from importlib import import_module as load\n'
            '            raise\n'
            '    finally:\n'
            '        load("requests")\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name: name\n'
            '    for value in values:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as load:\n'
            '            from importlib import import_module as load\n'
            '            break\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name: name\n'
            '    for value in values:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as load:\n'
            '            from importlib import import_module as load\n'
            '            continue\n'
            '        load("requests")\n'
            '    return load("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '        return None\n'
            '    return __import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '    raise\n'
            '__import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '        break\n'
            '    __import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '        break\n'
            'else:\n'
            '    __import__("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    risky_group_operation()\n'
            'except* ValueError as load:\n'
            '    from importlib import import_module as load\n'
            'load("requests")\n'
        ),
    ],
    ids=(
        'module-normal-cleanup',
        'class-normal-cleanup',
        'function-local-cleanup',
        'explicit-global-cleanup',
        'explicit-nonlocal-cleanup',
        'cleanup-before-finally',
        'multiple-handlers-explicit-del-rebind',
        'nested-same-name-handler',
        'return-cleanup-before-finally',
        'reraise-cleanup-before-finally',
        'break-cleanup-post-loop',
        'continue-cleanup-carried-state',
        'return-cleanup-skips-post-handler-call',
        'reraise-cleanup-skips-post-handler-call',
        'break-cleanup-skips-later-loop-body',
        'break-cleanup-skips-loop-else',
        'try-star-cleanup',
    ),
)
def test_validate_agent_layout_drops_implicitly_cleaned_handler_owners(
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
            '__import__ = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '__import__("requests")\n'
        ),
        (
            'class Loader:\n'
            '    __import__ = lambda name: name\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '    value = __import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '    return __import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            'finally:\n'
            '    __import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    risky_group_operation()\n'
            'except* ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '__import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '        break\n'
            '__import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as __import__:\n'
            '            __import__ = lambda name: name\n'
            '            return None\n'
            '    finally:\n'
            '        __import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '        raise\n'
            'finally:\n'
            '    __import__("requests")\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    risky_group_operation()\n'
            'except* ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            'except* __import__("requests").exceptions.RequestException:\n'
            '    pass\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '        raise\n'
            'except ValueError:\n'
            '    pass\n'
            '__import__("requests")\n'
        ),
    ],
    ids=(
        'module-builtin-fallback',
        'class-builtin-fallback',
        'explicit-global-builtin-fallback',
        'builtin-fallback-before-finally',
        'try-star-builtin-fallback',
        'break-builtin-fallback-post-loop',
        'return-cleanup-before-finally-builtin-fallback',
        'reraise-cleanup-before-finally-builtin-fallback',
        'try-star-cleanup-before-later-handler-type',
        'reraise-cleanup-through-outer-handler',
    ),
)
def test_validate_agent_layout_detects_builtin_after_handler_cleanup(
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
            'def use_loader():\n'
            '    __import__ = lambda name: name\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '    return __import__("requests")\n'
        ),
        (
            'def outer():\n'
            '    __import__ = lambda name: name\n'
            '    def use_loader():\n'
            '        nonlocal __import__\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as __import__:\n'
            '            __import__ = lambda name: name\n'
            '        return __import__("requests")\n'
            '    return use_loader\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as load:\n'
            '    load("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    risky_group_operation()\n'
            'except* ValueError as load:\n'
            '    from importlib import import_module as load\n'
            'except* load("requests").exceptions.RequestException:\n'
            '    pass\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '    except TypeError:\n'
            '        return __import__("requests")\n'
            '    return None\n'
        ),
        (
            '__import__ = lambda name: name\n'
            'class Loader:\n'
            '    __import__ = lambda name: name\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        __import__ = lambda name: name\n'
            '    value = __import__("requests")\n'
        ),
    ],
    ids=(
        'function-local-no-builtin-fallback',
        'nonlocal-no-builtin-fallback',
        'handler-body-alias-is-nonowner',
        'try-star-owner-cleanup-before-later-handler-type',
        'ordinary-sibling-handler-bypasses-cleanup',
        'class-cleanup-falls-through-module-nonowner',
    ),
)
def test_validate_agent_layout_preserves_handler_cleanup_scope_controls(
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
    ('owner_binding', 'post_call'),
    [
        ('import importlib as load', 'load.import_module("requests")'),
        ('from builtins import __import__ as load', 'load("requests")'),
        ('import builtins as load', 'load.__import__("requests")'),
        (
            'from importlib import import_module as load',
            'load(module_name)',
        ),
    ],
    ids=(
        'importlib-module',
        'aliased-builtin-callable',
        'builtins-module',
        'computed-import-module-alias',
    ),
)
def test_validate_agent_layout_cleans_every_handler_owner_kind(
    monkeypatch,
    owner_binding,
    post_call,
):
    source_prefix = (
        'load = lambda name: name\n'
        'try:\n'
        '    risky_operation()\n'
        'except ValueError as load:\n'
        f'    {owner_binding}\n'
        f'{post_call}\n'
    )

    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    assert validation['status'] == 'ok'
    assert validation['errors'] == []


def test_validate_agent_layout_fails_closed_for_computed_builtin_cleanup(
    monkeypatch,
):
    source_prefix = (
        '__import__ = lambda name: name\n'
        'try:\n'
        '    risky_operation()\n'
        'except ValueError as __import__:\n'
        '    __import__ = lambda name: name\n'
        '__import__(module_name)\n'
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
    'source_prefix',
    [
        (
            'from importlib import import_module as load\n'
            'try:\n'
            '    risky_operation()\n'
            'except load("requests").exceptions.RequestException as load:\n'
            '    pass\n'
        ),
        (
            'from importlib import import_module as load\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as load:\n'
            '    load = lambda name: name\n'
            'load("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as error:\n'
            '    from importlib import import_module as load\n'
            'load("requests")\n'
        ),
        (
            'from importlib import import_module as load\n'
            'class Loader:\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as load:\n'
            '        load = lambda name: name\n'
            '    value = load("requests")\n'
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as load:\n'
            '    from importlib import import_module as load\n'
            '    def inner():\n'
            '        return load("requests")\n'
            '    inner()\n'
        ),
    ],
    ids=(
        'handler-type-before-target-binding',
        'prior-owner-bypass',
        'different-name-owner-survives',
        'class-cleanup-falls-through-module-owner',
        'closure-called-before-handler-cleanup',
    ),
)
def test_validate_agent_layout_preserves_handler_cleanup_owner_controls(
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
    ('source_prefix', 'expects_dependency'),
    [
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if flag:\n'
            '            return None\n'
            '        else:\n'
            '            return None\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(first, second):\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if first:\n'
            '            if second:\n'
            '                return None\n'
            '            else:\n'
            '                return None\n'
            '        else:\n'
            '            return None\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if flag:\n'
            '            raise RuntimeError\n'
            '        else:\n'
            '            raise TypeError\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if flag:\n'
            '            break\n'
            '        else:\n'
            '            break\n'
            '    __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if flag:\n'
            '            return None\n'
            '    return __import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if flag:\n'
            '            raise RuntimeError\n'
            '        else:\n'
            '            raise TypeError\n'
            'except (RuntimeError, TypeError):\n'
            '    pass\n'
            '__import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if flag:\n'
            '            break\n'
            '        else:\n'
            '            break\n'
            '__import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        if flag:\n'
            '            continue\n'
            '        else:\n'
            '            continue\n'
            '    __import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as __import__:\n'
            '            from builtins import __import__ as __import__\n'
            '            if flag:\n'
            '                return None\n'
            '            else:\n'
            '                return None\n'
            '    finally:\n'
            '        __import__("requests")\n',
            True,
        ),
    ],
    ids=(
        'exhaustive-if-return-skips-post-handler',
        'nested-exhaustive-if-return-skips-post-handler',
        'exhaustive-if-raise-skips-post-handler',
        'exhaustive-if-break-skips-later-loop-body',
        'nonexhaustive-if-fallthrough',
        'exhaustive-if-raise-reaches-outer-handler',
        'exhaustive-if-break-reaches-post-loop',
        'exhaustive-if-continue-reaches-later-iteration',
        'exhaustive-if-return-cleanup-before-finally',
    ),
)
def test_validate_agent_layout_resolves_exhaustive_handler_if_termination(
    monkeypatch,
    source_prefix,
    expects_dependency,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    dependency_errors = [
        error
        for error in validation['errors']
        if error['code'] == 'undeclared_external_import'
        and error['path'] == 'src/config.py'
    ]
    assert bool(dependency_errors) is expects_dependency


@pytest.mark.filterwarnings(
    "ignore:'(return|break|continue)' in a 'finally' block:SyntaxWarning"
)
@pytest.mark.parametrize(
    ('source_prefix', 'expects_dependency'),
    [
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            return None\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            raise RuntimeError\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            break\n'
            '    __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            break\n'
            '__import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'for value in values:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            continue\n'
            '    __import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            raise RuntimeError\n'
            'except RuntimeError:\n'
            '    pass\n'
            '__import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader():\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            __import__("requests")\n'
            '            return None\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as __import__:\n'
            '    from builtins import __import__ as __import__\n'
            '    try:\n'
            '        pass\n'
            '    except KeyError:\n'
            '        pass\n'
            '__import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    risky_operation()\n'
            'except ValueError as __import__:\n'
            '    from builtins import __import__ as __import__\n'
            '    try:\n'
            '        pass\n'
            '    finally:\n'
            '        marker = 1\n'
            '__import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        risky_operation()\n'
            '    except ValueError as __import__:\n'
            '        from builtins import __import__ as __import__\n'
            '        try:\n'
            '            pass\n'
            '        finally:\n'
            '            if flag:\n'
            '                return None\n'
            '            else:\n'
            '                return None\n'
            '    return __import__("requests")\n',
            False,
        ),
    ],
    ids=(
        'finally-return-skips-post-handler',
        'finally-raise-skips-post-handler',
        'finally-break-skips-later-loop-body',
        'finally-break-reaches-post-loop',
        'finally-continue-reaches-later-iteration',
        'finally-raise-reaches-outer-handler',
        'finally-use-precedes-handler-cleanup',
        'try-without-finally-falls-through',
        'nonterminal-finally-falls-through',
        'recursive-if-finally-return-skips-post-handler',
    ),
)
def test_validate_agent_layout_resolves_terminal_handler_try_finally(
    monkeypatch,
    source_prefix,
    expects_dependency,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    dependency_errors = [
        error
        for error in validation['errors']
        if error['code'] == 'undeclared_external_import'
        and error['path'] == 'src/config.py'
    ]
    assert bool(dependency_errors) is expects_dependency


@pytest.mark.filterwarnings(
    "ignore:'(return|break|continue)' in a 'finally' block:SyntaxWarning"
)
@pytest.mark.parametrize(
    ('source_prefix', 'expects_dependency'),
    [
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            try:\n'
            '                risky_operation()\n'
            '            except ValueError as __import__:\n'
            '                from builtins import __import__ as __import__\n'
            '                if flag:\n'
            '                    return None\n'
            '                else:\n'
            '                    return None\n'
            '        finally:\n'
            '            raise RuntimeError\n'
            '    except RuntimeError:\n'
            '        pass\n'
            '    return __import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as __import__:\n'
            '            from builtins import __import__ as __import__\n'
            '            if flag:\n'
            '                return None\n'
            '            else:\n'
            '                return None\n'
            '    finally:\n'
            '        raise RuntimeError\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as __import__:\n'
            '            from builtins import __import__ as __import__\n'
            '            if flag:\n'
            '                return None\n'
            '            else:\n'
            '                return None\n'
            '    finally:\n'
            '        marker = 1\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as __import__:\n'
            '            from builtins import __import__ as __import__\n'
            '            if flag:\n'
            '                return None\n'
            '            else:\n'
            '                return None\n'
            '    finally:\n'
            '        __import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    for value in values:\n'
            '        try:\n'
            '            try:\n'
            '                risky_operation()\n'
            '            except ValueError as __import__:\n'
            '                from builtins import __import__ as __import__\n'
            '                if flag:\n'
            '                    return None\n'
            '                else:\n'
            '                    return None\n'
            '        finally:\n'
            '            break\n'
            '        __import__("flask")\n'
            '    return __import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    for value in values:\n'
            '        try:\n'
            '            try:\n'
            '                risky_operation()\n'
            '            except ValueError as __import__:\n'
            '                from builtins import __import__ as __import__\n'
            '                if flag:\n'
            '                    return None\n'
            '                else:\n'
            '                    return None\n'
            '        finally:\n'
            '            continue\n'
            '        __import__("flask")\n'
            '    return __import__("requests")\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            risky_operation()\n'
            '        except ValueError as __import__:\n'
            '            from builtins import __import__ as __import__\n'
            '            if flag:\n'
            '                raise KeyError\n'
            '            else:\n'
            '                raise TypeError\n'
            '    finally:\n'
            '        return None\n'
            '    return __import__("requests")\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'def use_loader(flag):\n'
            '    global __import__\n'
            '    try:\n'
            '        try:\n'
            '            try:\n'
            '                risky_operation()\n'
            '            except ValueError as __import__:\n'
            '                from builtins import __import__ as __import__\n'
            '                if flag:\n'
            '                    raise KeyError\n'
            '                else:\n'
            '                    raise TypeError\n'
            '        finally:\n'
            '            raise RuntimeError\n'
            '    except RuntimeError:\n'
            '        pass\n'
            '    return __import__("requests")\n',
            True,
        ),
    ],
    ids=(
        'caught-finalizer-raise-overrides-exhaustive-return',
        'uncaught-finalizer-raise-keeps-post-handler-unreachable',
        'fallthrough-finalizer-preserves-exhaustive-return',
        'finalizer-use-runs-after-handler-cleanup',
        'finalizer-break-overrides-return-and-reaches-post-loop',
        'finalizer-continue-overrides-return-and-reaches-post-loop',
        'finalizer-return-keeps-post-handler-unreachable',
        'caught-finalizer-raise-after-exhaustive-handler-raise',
    ),
)
def test_validate_agent_layout_resolves_enclosing_finalizer_overrides(
    monkeypatch,
    source_prefix,
    expects_dependency,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    dependency_errors = [
        error
        for error in validation['errors']
        if error['code'] == 'undeclared_external_import'
        and error['path'] == 'src/config.py'
    ]
    assert bool(dependency_errors) is expects_dependency


@pytest.mark.parametrize(
    ('source_prefix', 'expects_dependency'),
    [
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    raise ValueError\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '    def inner():\n'
            '        return __import__("requests")\n'
            '    inner()\n',
            False,
        ),
        (
            'load = lambda name: name\n'
            'try:\n'
            '    raise ValueError\n'
            'except ValueError as load:\n'
            '    from importlib import import_module as load\n'
            '    def inner():\n'
            '        return load("requests")\n'
            'inner()\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    raise ValueError\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '    def inner():\n'
            '        return __import__("requests")\n'
            'inner()\n',
            True,
        ),
        (
            'async def outer():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        raise ValueError\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '        async def inner():\n'
            '            return load("requests")\n'
            '        return await inner()\n',
            True,
        ),
        (
            'async def outer():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        raise ValueError\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '        async def inner():\n'
            '            return load("requests")\n'
            '    return await inner()\n',
            False,
        ),
        (
            'def outer():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        raise ValueError\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '        def inner():\n'
            '            yield load("requests")\n'
            '        yield from inner()\n',
            True,
        ),
        (
            'def outer():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        raise ValueError\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '        def inner():\n'
            '            yield load("requests")\n'
            '    yield from inner()\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    raise ValueError\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '    inner = lambda: __import__("requests")\n'
            '    inner()\n',
            False,
        ),
    ],
    ids=(
        'immediate-function-global-nonowner',
        'post-handler-function-local-empty-cell',
        'post-handler-function-global-builtin',
        'immediate-coroutine-local-owner',
        'post-handler-coroutine-local-empty-cell',
        'immediate-generator-local-owner',
        'post-handler-generator-local-empty-cell',
        'immediate-lambda-global-nonowner',
    ),
)
def test_validate_agent_layout_resolves_direct_handler_nested_execution_timing(
    monkeypatch,
    source_prefix,
    expects_dependency,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    dependency_errors = [
        error
        for error in validation['errors']
        if error['code'] == 'undeclared_external_import'
        and error['path'] == 'src/config.py'
    ]
    assert bool(dependency_errors) is expects_dependency


@pytest.mark.parametrize(
    'source_prefix',
    [
        (
            'def outer():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        raise ValueError\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '        return lambda load=load: load("requests")\n'
            'outer()()\n'
        ),
        (
            'async def outer():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        raise ValueError\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '        async def inner(load=load):\n'
            '            return load("requests")\n'
            '    return await inner()\n'
        ),
        (
            'def outer():\n'
            '    load = lambda name: name\n'
            '    try:\n'
            '        raise ValueError\n'
            '    except ValueError as load:\n'
            '        from importlib import import_module as load\n'
            '        def inner(load=load):\n'
            '            yield load("requests")\n'
            '    yield from inner()\n'
        ),
    ],
    ids=('lambda-default', 'coroutine-default', 'generator-default'),
)
def test_validate_agent_layout_preserves_handler_callable_default_snapshots(
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
    ('source_prefix', 'expects_dependency'),
    [
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    raise ValueError\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '    pending = (__import__("requests") for _ in (0,))\n'
            'next(pending)\n',
            True,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    raise ValueError\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '    next((__import__("requests") for _ in (0,)))\n',
            False,
        ),
        (
            '__import__ = lambda name: name\n'
            'try:\n'
            '    raise ValueError\n'
            'except ValueError as __import__:\n'
            '    __import__ = lambda name: name\n'
            '    pending = (value for value in __import__("requests"))\n',
            False,
        ),
    ],
    ids=(
        'generator-expression-driven-after-cleanup',
        'generator-expression-driven-before-cleanup',
        'generator-expression-eager-first-iterable',
    ),
)
def test_validate_agent_layout_resolves_handler_generator_expression_timing(
    monkeypatch,
    source_prefix,
    expects_dependency,
):
    validation = _validate_agent_layout_with_config_prefix(
        monkeypatch,
        source_prefix,
    )

    dependency_errors = [
        error
        for error in validation['errors']
        if error['code'] == 'undeclared_external_import'
        and error['path'] == 'src/config.py'
    ]
    assert bool(dependency_errors) is expects_dependency


@pytest.mark.parametrize(
    ('source_prefix', 'expected_error_code'),
    [
        (
            'load = lambda name, package=None: name\n'
            'for item in values:\n'
            '    load("requests")\n'
            '    from importlib import import_module as load\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values, module_name):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load(module_name)\n'
            '        from importlib import import_module as load\n',
            'unsupported_dynamic_import',
        ),
        (
            'async def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    async for item in values:\n'
            '        load("requests")\n'
            '        from importlib import import_module as load\n',
            'undeclared_external_import',
        ),
        (
            'class Loader:\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load("requests")\n'
            '        from importlib import import_module as load\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    while condition(load("requests")):\n'
            '        from importlib import import_module as load\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    loader = object()\n'
            '    while condition():\n'
            '        loader.import_module("requests")\n'
            '        import importlib as loader\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values, select_owner):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        if select_owner:\n'
            '            load("requests")\n'
            '        else:\n'
            '            from importlib import import_module as load\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values, select_owner):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load("requests")\n'
            '        if select_owner:\n'
            '            from builtins import __import__ as load\n'
            '            continue\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(select_owner):\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        load("requests")\n'
            '        if select_owner:\n'
            '            from importlib import import_module as load\n'
            '            continue\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        from importlib import import_module as load\n'
            '        break\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        from importlib import import_module as load\n'
            '        break\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(values, stop):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        from importlib import import_module as load\n'
            '        if stop:\n'
            '            break\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(stop):\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        from importlib import import_module as load\n'
            '        if stop:\n'
            '            break\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'for load in values:\n'
            '    pass\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        from importlib import import_module as load\n'
            '    else:\n'
            '        return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'for (load, *rest) in load("requests"):\n'
            '    pass\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'with manager(load("requests")) as load:\n'
            '    pass\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'with manager() as item, manager(load("requests")) as load:\n'
            '    pass\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'with manager() as item, manager(load("requests")) as (load, *rest):\n'
            '    pass\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'async def use_loader():\n'
            '    global load\n'
            '    async with manager() as item, manager(load("requests")) as load:\n'
            '        pass\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'while flag and (load := lambda name, package=None: name):\n'
            '    pass\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'for item in ((load := []) if flag else []):\n'
            '    pass\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'while ((load := lambda name: name) if flag else load("requests")):\n'
            '    pass\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'for item in ((load := []) if flag else [load("requests")]):\n'
            '    pass\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'with suppressing_manager() as first, failing_manager() as load:\n'
            '    pass\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'module_name = "requests"\n'
            'with suppressing_manager() as first, failing_manager() as load:\n'
            '    pass\n'
            'load(module_name)\n',
            'unsupported_dynamic_import',
        ),
        (
            'async def use_loader():\n'
            '    from importlib import import_module as load\n'
            '    async with suppressing_manager() as first, failing_manager() as load:\n'
            '        pass\n'
            '    return load("requests")\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as load\n'
            'with suppressing_manager() as (head, *load):\n'
            '    pass\n'
            'load("requests")\n',
            'undeclared_external_import',
        ),
    ],
    ids=(
        'module-for-body-next-iteration',
        'function-for-body-computed-next-iteration',
        'async-for-body-next-iteration',
        'class-for-body-next-iteration',
        'while-test-next-iteration',
        'while-body-attribute-next-iteration',
        'for-opposite-branch-next-iteration',
        'for-continue-next-iteration',
        'while-continue-next-iteration',
        'for-break-post-loop',
        'while-break-post-loop',
        'for-conditional-break-preserves-owner-post-loop',
        'while-conditional-break-preserves-owner-post-loop',
        'for-target-zero-iteration-post-loop',
        'while-else-cycle-owner',
        'for-iter-before-tuple-star-target',
        'with-context-before-own-target',
        'with-later-context-before-own-target',
        'with-later-context-before-tuple-star-target',
        'async-with-later-context-before-own-target',
        'while-short-circuit-shadow-post-loop',
        'for-conditional-iter-shadow-post-loop',
        'while-if-expression-exclusive-call',
        'for-if-expression-exclusive-call',
        'with-suppressed-later-target-post-loop',
        'with-suppressed-later-target-computed-post-loop',
        'async-with-suppressed-later-target-post-loop',
        'with-suppressed-first-destructuring-target-post-loop',
    ),
)
def test_validate_agent_layout_respects_loop_and_with_evaluation_order(
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
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
            '            break\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader():\n'
            '    loader = object()\n'
            '    while outer_condition():\n'
            '        loader.import_module("requests")\n'
            '        while inner_condition():\n'
            '            import importlib as loader\n'
            '            break\n',
            'undeclared_external_import',
        ),
        (
            'async def use_loader(outer_values, inner_values, module_name):\n'
            '    load = lambda name, package=None: name\n'
            '    async for outer in outer_values:\n'
            '        load(module_name)\n'
            '        async for inner in inner_values:\n'
            '            from builtins import __import__ as load\n'
            '            break\n',
            'unsupported_dynamic_import',
        ),
        (
            'def use_loader(outer_values):\n'
            '    loader = object()\n'
            '    for outer in outer_values:\n'
            '        loader.__import__("requests")\n'
            '        while inner_condition():\n'
            '            import builtins as loader\n'
            '            break\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(outer_values, inner_values, select_owner):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            if select_owner:\n'
            '                from importlib import import_module as load\n'
            '            break\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    while outer_condition():\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            try:\n'
            '                from importlib import import_module as load\n'
            '                break\n'
            '            finally:\n'
            '                observe(inner)\n',
            'undeclared_external_import',
        ),
        (
            'def use_loader(outer_values, inner_values, skip):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        if skip:\n'
            '            continue\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as owner\n'
            'def use_loader(outer_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        while (load := owner) and inner_condition():\n'
            '            load = lambda name, package=None: name\n',
            'undeclared_external_import',
        ),
        (
            'from importlib import import_module as owner\n'
            'def use_loader(outer_values, stop):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        while (load := (lambda name, package=None: name)) '
            'and inner_condition():\n'
            '            load = owner\n'
            '            if stop:\n'
            '                break\n'
            '        load("requests")\n',
            'undeclared_external_import',
        ),
    ],
    ids=(
        'for-for-inner-break-owner',
        'while-while-inner-break-module-owner',
        'async-for-inner-break-computed-owner',
        'for-while-inner-break-builtins-owner',
        'conditional-inner-owner-before-break',
        'inner-try-break-owner',
        'conditional-outer-continue-preserves-inner-owner-path',
        'inner-while-final-test-owner-reaches-next-outer-cycle',
        'inner-while-break-retains-body-owner',
    ),
)
def test_validate_agent_layout_isolates_nested_loop_breaks(
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
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            try:\n'
            '                from importlib import import_module as load\n'
            '                break\n'
            '            finally:\n'
            '                load = lambda name, package=None: name\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    while outer_condition():\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
            '            load = lambda name, package=None: name\n'
            '            break\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
            '            break\n'
            '        load = lambda name, package=None: name\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
            '            break\n'
            '        break\n'
        ),
        (
            'from importlib import import_module as owner\n'
            'def use_loader(outer_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        while (load := (lambda name, package=None: name)) '
            'and inner_condition():\n'
            '            load = owner\n'
            '        load("requests")\n'
        ),
        (
            'from importlib import import_module as owner\n'
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            load = owner\n'
            '        else:\n'
            '            load = lambda name, package=None: name\n'
        ),
        (
            'from importlib import import_module as owner\n'
            'async def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    async for outer in outer_values:\n'
            '        load("requests")\n'
            '        async for inner in inner_values:\n'
            '            load = owner\n'
            '        else:\n'
            '            load = lambda name, package=None: name\n'
        ),
        (
            'from importlib import import_module as owner\n'
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            load = owner\n'
            '            continue\n'
            '        else:\n'
            '            load = lambda name, package=None: name\n'
        ),
        (
            'from importlib import import_module as owner\n'
            'def use_loader(outer_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        while inner_condition():\n'
            '            load = owner\n'
            '        else:\n'
            '            load = lambda name, package=None: name\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
            '            break\n'
            '        try:\n'
            '            break\n'
            '        finally:\n'
            '            observe(outer)\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        try:\n'
            '            for inner in inner_values:\n'
            '                observe(inner)\n'
            '            break\n'
            '        finally:\n'
            '            from importlib import import_module as load\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        continue\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    while outer_condition():\n'
            '        load("requests")\n'
            '        break\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            continue\n'
            '            from importlib import import_module as load\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load(".helpers", __package__)\n'
            '        for inner in inner_values:\n'
            '            from importlib import import_module as load\n'
            '            break\n'
        ),
        (
            'def use_loader(outer_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        def inner():\n'
            '            from importlib import import_module as load\n'
            '            return load("json")\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        load("requests")\n'
            '        for inner in inner_values:\n'
            '            try:\n'
            '                break\n'
            '            finally:\n'
            '                from importlib import import_module as load\n'
            '        break\n'
        ),
    ],
    ids=(
        'inner-break-finally-nonowner-dominates',
        'inner-final-nonowner-before-break',
        'outer-final-nonowner-dominates',
        'genuine-outer-break-stops-backedge',
        'inner-while-final-test-nonowner-dominates-body-owner',
        'inner-for-else-nonowner-dominates-normal-completion',
        'inner-async-for-else-nonowner-dominates-normal-completion',
        'inner-continue-still-reaches-for-else-nonowner',
        'inner-while-else-nonowner-dominates-normal-completion',
        'outer-try-finally-break-stops-backedge',
        'outer-break-finally-owner-stops-backedge',
        'outer-continue-before-inner-owner',
        'outer-break-before-inner-owner',
        'inner-continue-before-inner-owner',
        'relative-inner-owner-call',
        'nested-function-owner-isolated',
        'outer-break-after-inner-finally-owner',
    ),
)
def test_validate_agent_layout_preserves_nested_loop_exit_controls(
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
            'def use_loader(inner_values):\n'
            '    from importlib import import_module as load\n'
            '    for inner in inner_values:\n'
            '        del load\n'
            '        break\n'
            '    else:\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    load = lambda name, package=None: name\n'
            '    for outer in outer_values:\n'
            '        from importlib import import_module as load\n'
            '        for inner in inner_values:\n'
            '            del load\n'
            '            break\n'
            '        else:\n'
            '            load = lambda name, package=None: name\n'
            '        load("requests")\n'
        ),
        (
            'def use_loader(outer_values, inner_values):\n'
            '    from importlib import import_module as load\n'
            '    for outer in outer_values:\n'
            '        for inner in inner_values:\n'
            '            try:\n'
            '                try:\n'
            '                    observe(inner)\n'
            '                finally:\n'
            '                    del load\n'
            '            finally:\n'
            '                load("requests")\n'
        ),
        (
            'def use_loader(inner_values, stop):\n'
            '    from importlib import import_module as load\n'
            '    for inner in inner_values:\n'
            '        if stop:\n'
            '            del load\n'
            '            break\n'
            '    else:\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader(inner_values):\n'
            '    from importlib import import_module as load\n'
            '    for inner in inner_values:\n'
            '        try:\n'
            '            break\n'
            '        finally:\n'
            '            del load\n'
            '    else:\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n'
        ),
        (
            'from importlib import import_module as owner\n'
            'def use_loader(stop):\n'
            '    load = owner\n'
            '    while (load := (lambda name: name)) and condition():\n'
            '        if stop:\n'
            '            del load\n'
            '            break\n'
            '    return load("requests")\n'
        ),
    ],
    ids=(
        'break-delete-or-else-nonowner',
        'nested-break-delete-or-else-nonowner',
        'nested-finally-delete-before-call',
        'conditional-break-delete-or-else-nonowner',
        'break-finally-delete-or-else-nonowner',
        'while-final-test-nonowner-or-break-delete',
    ),
)
def test_validate_agent_layout_preserves_deleted_completed_loop_outcomes(
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
            'from importlib import import_module as load\n'
            'for load in values:\n'
            '    load("requests")\n'
        ),
        (
            'from importlib import import_module as load\n'
            'async def use_loader(values):\n'
            '    async for (load, *rest) in values:\n'
            '        load("requests")\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    load("requests")\n'
            '    for item in values:\n'
            '        from importlib import import_module as load\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load(".helpers", __package__)\n'
            '        from importlib import import_module as load\n'
            '        load(".helpers", __package__)\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load("requests")\n'
            '        load = lambda name, package=None: name\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    load("requests")\n'
            '    while condition():\n'
            '        from importlib import import_module as load\n'
        ),
        (
            'from importlib import import_module as load\n'
            'with manager() as load, manager(load("requests")):\n'
            '    pass\n'
        ),
        (
            'from importlib import import_module as load\n'
            'with manager() as (load, *rest):\n'
            '    load("requests")\n'
        ),
        (
            'from importlib import import_module as load\n'
            'with manager() as load:\n'
            '    pass\n'
            'load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    with manager(load("requests")):\n'
            '        from importlib import import_module as load\n'
        ),
        (
            'from importlib import import_module as load\n'
            'async def use_loader():\n'
            '    async with manager() as load, manager(load("requests")):\n'
            '        pass\n'
        ),
        (
            'from importlib import import_module as load\n'
            'for (head, *load) in values:\n'
            '    load("requests")\n'
        ),
        (
            'from importlib import import_module as load\n'
            'with manager() as (head, *load):\n'
            '    load("requests")\n'
        ),
        (
            'from importlib import import_module as load\n'
            'async def use_loader():\n'
            '    async with manager() as item, manager(load("requests")) as load:\n'
            '        pass\n'
        ),
        (
            'from importlib import import_module as load\n'
            'while flag and (load := lambda name, package=None: name) and load("requests"):\n'
            '    pass\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        from importlib import import_module as load\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        from importlib import import_module as load\n'
            '        load = lambda name, package=None: name\n'
            '    else:\n'
            '        return load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        from importlib import import_module as load\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        if select_owner:\n'
            '            from importlib import import_module as load\n'
            '        load = lambda name, package=None: name\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load("requests")\n'
            '        from importlib import import_module as load\n'
            '        load = lambda name, package=None: name\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load("requests")\n'
            '        from importlib import import_module as load\n'
            '        break\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        load("requests")\n'
            '        from importlib import import_module as load\n'
            '        break\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        load("requests")\n'
            '        try:\n'
            '            from importlib import import_module as load\n'
            '            continue\n'
            '        finally:\n'
            '            load = lambda name, package=None: name\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        try:\n'
            '            from importlib import import_module as load\n'
            '            break\n'
            '        finally:\n'
            '            load = lambda name, package=None: name\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader(values):\n'
            '    load = lambda name, package=None: name\n'
            '    for item in values:\n'
            '        from importlib import import_module as load\n'
            '        del load\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        from importlib import import_module as load\n'
            '        del load\n'
            '    return load("requests")\n'
        ),
        (
            'def use_loader():\n'
            '    load = lambda name, package=None: name\n'
            '    while condition():\n'
            '        from importlib import import_module as load\n'
            '        load = lambda name, package=None: name\n'
            '    else:\n'
            '        return load("requests")\n'
        ),
    ],
    ids=(
        'for-target-shadows-body',
        'async-for-tuple-star-target-shadows-body',
        'call-before-loop-body-owner',
        'relative-loop-carried-owner',
        'pure-nonowner-loop',
        'call-before-while-body-owner',
        'with-earlier-target-shadows-later-context',
        'with-tuple-star-target-shadows-body',
        'with-target-shadows-post-body',
        'with-body-owner-not-retroactive',
        'async-with-earlier-target-shadows-later-context',
        'for-starred-target-shadows-body',
        'with-starred-target-shadows-body',
        'async-with-local-later-target-is-unbound',
        'while-same-path-short-circuit-shadow',
        'for-body-final-nonowner-post-loop',
        'for-body-final-nonowner-else',
        'while-body-final-nonowner-post-loop',
        'conditional-owner-final-nonowner-post-loop',
        'for-backedge-final-nonowner',
        'for-owner-then-break-no-backedge',
        'while-owner-then-break-no-backedge',
        'for-continue-finally-nonowner-no-backedge',
        'for-break-finally-nonowner-post-loop',
        'for-body-final-delete-post-loop',
        'while-body-final-delete-post-loop',
        'while-body-final-nonowner-else',
    ),
)
def test_validate_agent_layout_ignores_non_dependency_loop_and_with_controls(
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
        and error['failure_category'] == 'spec_missing'
        for error in validation['errors']
    )


def test_validate_agent_layout_classifies_dependency_spec_errors(monkeypatch):
    real_find_spec = agent_state.importlib.util.find_spec

    def find_spec_with_broken_pydantic(module_name):
        if module_name == 'pydantic':
            raise ValueError('synthetic spec failure that must stay private')
        return real_find_spec(module_name)

    monkeypatch.setattr(
        agent_state.importlib.util,
        'find_spec',
        find_spec_with_broken_pydantic,
    )

    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    error = next(
        error
        for error in validation['errors']
        if error['code'] == 'missing_declared_dependency'
        and error['module'] == 'pydantic'
    )
    assert error['failure_category'] == 'spec_error'
    assert 'synthetic spec failure' not in json.dumps(validation)


def test_dependency_import_probe_propagates_preloads_and_caches_by_full_key(
    monkeypatch,
    tmp_path,
    cleared_dependency_import_probe_cache,
):
    calls = []

    def successful_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b'', stderr=b'')

    monkeypatch.setattr(agent_state.subprocess, 'run', successful_run)
    preloads = ('numpy', 'sklearn')

    assert agent_state._probe_dependency_import(
        ROOT,
        'torch',
        preloads,
        timeout_seconds=10.0,
    ) == (True, None)
    assert agent_state._probe_dependency_import(
        ROOT,
        'torch',
        preloads,
        timeout_seconds=1.0,
    ) == (True, None)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert command[-2] == 'torch'
    assert json.loads(command[-1]) == {
        'preloads': ['numpy', 'sklearn'],
        'targets': [],
        'symbols': {},
    }
    assert command[-3] == str(ROOT.resolve())
    assert kwargs['cwd'] == ROOT.resolve()
    assert kwargs['capture_output'] is True
    assert kwargs['check'] is False
    assert kwargs['timeout'] == 10.0
    assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'

    assert agent_state._probe_dependency_import(
        ROOT,
        'torch',
        ('numpy',),
        timeout_seconds=10.0,
    ) == (True, None)
    other_root = tmp_path / 'other-root'
    other_root.mkdir()
    assert agent_state._probe_dependency_import(
        other_root,
        'torch',
        preloads,
        timeout_seconds=10.0,
    ) == (True, None)
    assert len(calls) == 3

    agent_state._clear_dependency_import_probe_cache()
    assert agent_state._probe_dependency_import(
        ROOT,
        'torch',
        preloads,
        timeout_seconds=10.0,
    ) == (True, None)
    assert len(calls) == 4

    target = ('pymatgen.core',)
    symbols = (('pymatgen.core', ('Composition',)),)
    assert agent_state._probe_dependency_import(
        ROOT,
        'pymatgen',
        (),
        target,
        symbols,
        timeout_seconds=10.0,
    ) == (True, None)
    assert agent_state._probe_dependency_import(
        ROOT,
        'pymatgen',
        (),
        target,
        symbols,
        timeout_seconds=1.0,
    ) == (True, None)
    assert len(calls) == 5
    assert json.loads(calls[-1][0][-1]) == {
        'preloads': [],
        'targets': ['pymatgen.core'],
        'symbols': {'pymatgen.core': ['Composition']},
    }

    assert agent_state._probe_dependency_import(
        ROOT,
        'pymatgen',
        (),
        target,
        (('pymatgen.core', ('Element',)),),
        timeout_seconds=10.0,
    ) == (True, None)
    assert len(calls) == 6


def test_public_dependency_probe_cache_invalidates_after_pythonpath_change(
    monkeypatch,
    cleared_dependency_import_probe_cache,
):
    before_context = '/private/sensitive-probe-context-before'
    after_context = '/private/sensitive-probe-context-after'
    pydantic_calls = 0

    def environment_sensitive_run(command, **kwargs):
        nonlocal pydantic_calls
        module = command[-2]
        if module == 'pydantic':
            pydantic_calls += 1
        return subprocess.CompletedProcess(
            command,
            1
            if module == 'pydantic'
            and os.environ['PYTHONPATH'] == after_context
            else 0,
            stdout=b'',
            stderr=b'',
        )

    monkeypatch.setattr(
        agent_state.subprocess,
        'run',
        environment_sensitive_run,
    )
    monkeypatch.setenv('PYTHONPATH', before_context)

    assert validate_agent_layout(ROOT)['status'] == 'ok'

    monkeypatch.setenv('PYTHONPATH', after_context)
    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    assert pydantic_calls == 2
    assert any(
        error['code'] == 'unimportable_declared_dependency'
        and error['module'] == 'pydantic'
        and error['failure_category'] == 'nonzero_exit'
        for error in validation['errors']
    )
    cache_keys = repr(tuple(agent_state._DEPENDENCY_IMPORT_PROBE_CACHE))
    assert before_context not in cache_keys
    assert after_context not in cache_keys


def test_public_dependency_probe_cache_invalidates_after_module_bytes_change(
    monkeypatch,
    tmp_path,
    cleared_dependency_import_probe_cache,
):
    shim_root = tmp_path / 'shim'
    shim_root.mkdir()
    shim_path = shim_root / 'jarvis.py'
    shim_path.write_text('SENTINEL = 1\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(shim_root))
    monkeypatch.setenv('PYTHONPATH', str(shim_root))
    monkeypatch.delitem(sys.modules, 'jarvis', raising=False)
    agent_state.importlib.invalidate_caches()
    jarvis_calls = 0

    def module_sensitive_run(command, **kwargs):
        nonlocal jarvis_calls
        module = command[-2]
        if module == 'jarvis':
            jarvis_calls += 1
        return subprocess.CompletedProcess(
            command,
            1
            if module == 'jarvis'
            and shim_path.read_text(encoding='utf-8').startswith('raise ')
            else 0,
            stdout=b'',
            stderr=b'',
        )

    monkeypatch.setattr(
        agent_state.subprocess,
        'run',
        module_sensitive_run,
    )

    assert validate_agent_layout(ROOT)['status'] == 'ok'

    shim_path.write_text(
        'raise RuntimeError("synthetic import failure")\n',
        encoding='utf-8',
    )
    agent_state.importlib.invalidate_caches()
    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'error'
    assert jarvis_calls == 2
    assert any(
        error['code'] == 'unimportable_declared_dependency'
        and error['module'] == 'jarvis'
        and error['failure_category'] == 'nonzero_exit'
        for error in validation['errors']
    )


@pytest.mark.parametrize(
    ('mutation', 'missing_symbol'),
    [
        ('missing-target', None),
        ('missing-symbol', 'get_db_info'),
        ('missing-symbol', 'get_request_data'),
        ('raising-target', None),
    ],
)
def test_public_and_cli_jarvis_probe_rejects_invalid_consumer_import(
    monkeypatch,
    tmp_path,
    cleared_dependency_import_probe_cache,
    mutation,
    missing_symbol,
):
    package_dir = tmp_path / 'jarvis'
    db_dir = package_dir / 'db'
    db_dir.mkdir(parents=True)
    (package_dir / '__init__.py').write_text(
        'OWNER = True\n',
        encoding='utf-8',
    )
    (db_dir / '__init__.py').write_text('', encoding='utf-8')
    target = next(iter(JARVIS_TARGET_SYMBOLS))
    target_path = db_dir / 'figshare.py'
    target_path.write_text(
        ''.join(
            f'def {symbol}(*args, **kwargs):\n    return None\n'
            for symbol in JARVIS_TARGET_SYMBOLS[target]
        ),
        encoding='utf-8',
    )

    def reset_jarvis_modules():
        for module_name in tuple(sys.modules):
            if (
                module_name == 'jarvis'
                or module_name.startswith('jarvis.')
            ):
                monkeypatch.delitem(
                    sys.modules,
                    module_name,
                    raising=False,
                )
        agent_state.importlib.invalidate_caches()

    reset_jarvis_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    real_run = subprocess.run
    jarvis_calls = 0

    def selective_run(command, **kwargs):
        nonlocal jarvis_calls
        if command[-2] != 'jarvis':
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b'',
                stderr=b'',
            )
        jarvis_calls += 1
        return real_run(command, **kwargs)

    monkeypatch.setattr(agent_state.subprocess, 'run', selective_run)
    probe_environment = os.environ.copy()
    probe_environment['PYTHONDONTWRITEBYTECODE'] = '1'
    exact_consumer = (
        'from jarvis.db.figshare import get_db_info, get_request_data'
    )
    baseline_consumer = real_run(
        [sys.executable, '-c', exact_consumer],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    baseline_validation = validate_agent_layout(ROOT)
    cached_baseline_validation = validate_agent_layout(ROOT)

    assert baseline_consumer.returncode == 0
    assert baseline_validation['status'] == 'ok'
    assert cached_baseline_validation['status'] == 'ok'
    assert jarvis_calls == 1

    private_diagnostic = 'synthetic private jarvis consumer failure'
    if mutation == 'missing-target':
        target_path.unlink()
    elif mutation == 'raising-target':
        target_path.write_text(
            f'raise RuntimeError({private_diagnostic!r})\n',
            encoding='utf-8',
        )
    else:
        target_path.write_text(
            ''.join(
                f'def {symbol}(*args, **kwargs):\n    return None\n'
                for symbol in JARVIS_TARGET_SYMBOLS[target]
                if symbol != missing_symbol
            )
            + (
                'def __getattr__(name):\n'
                f'    raise AttributeError({private_diagnostic!r})\n'
            ),
            encoding='utf-8',
        )
    reset_jarvis_modules()
    direct_consumer = real_run(
        [sys.executable, '-c', exact_consumer],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    repeated_validation = validate_agent_layout(ROOT)
    agent_state._clear_dependency_import_probe_cache()
    uncached_validation = validate_agent_layout(ROOT)
    completed = real_run(
        [
            sys.executable,
            str(ROOT / 'main.py'),
            '--verify-agent-contract',
        ],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert direct_consumer.returncode == 1
    assert repeated_validation['status'] == 'error'
    assert uncached_validation['status'] == 'error'
    assert jarvis_calls == 3
    for validation in (repeated_validation, uncached_validation):
        jarvis_check = next(
            check
            for check in validation['checks']
            if check.get('kind') == 'dependency_import'
            and check.get('module') == 'jarvis'
        )
        assert jarvis_check['import_probe_targets'] == list(
            JARVIS_TARGET_SYMBOLS
        )
        assert jarvis_check['import_probe_symbols'] == {
            target_name: list(symbols)
            for target_name, symbols in JARVIS_TARGET_SYMBOLS.items()
        }
        assert jarvis_check['available'] is False
        assert jarvis_check['failure_category'] == 'nonzero_exit'

    assert completed.returncode == 1
    assert completed.stderr == ''
    payload = json.loads(completed.stdout)
    assert payload['status'] == 'error'
    assert private_diagnostic not in completed.stdout


@pytest.mark.parametrize(
    ('mutation', 'target', 'missing_symbol'),
    [
        (
            'missing-submodule',
            'matminer.featurizers.base',
            None,
        ),
        (
            'missing-submodule',
            'matminer.featurizers.composition',
            None,
        ),
        (
            'missing-symbol',
            'matminer.featurizers.base',
            'MultipleFeaturizer',
        ),
        (
            'missing-symbol',
            'matminer.featurizers.composition',
            'ElementProperty',
        ),
        (
            'missing-symbol',
            'matminer.featurizers.composition',
            'Stoichiometry',
        ),
        (
            'raising-target',
            'matminer.featurizers.base',
            None,
        ),
    ],
)
def test_public_and_cli_matminer_probe_rejects_invalid_consumer_import(
    monkeypatch,
    tmp_path,
    cleared_dependency_import_probe_cache,
    mutation,
    target,
    missing_symbol,
):
    package_dir = tmp_path / 'matminer'
    featurizers_dir = package_dir / 'featurizers'
    featurizers_dir.mkdir(parents=True)
    (package_dir / '__init__.py').write_text(
        'OWNER = True\n',
        encoding='utf-8',
    )
    (featurizers_dir / '__init__.py').write_text('', encoding='utf-8')
    target_paths = {
        'matminer.featurizers.base': featurizers_dir / 'base.py',
        'matminer.featurizers.composition': (
            featurizers_dir / 'composition.py'
        ),
    }
    for target_name, symbols in MATMINER_TARGET_SYMBOLS.items():
        target_paths[target_name].write_text(
            ''.join(
                f'class {symbol}:\n    pass\n'
                for symbol in symbols
            ),
            encoding='utf-8',
        )

    def reset_matminer_modules():
        for module_name in tuple(sys.modules):
            if (
                module_name == 'matminer'
                or module_name.startswith('matminer.')
            ):
                monkeypatch.delitem(
                    sys.modules,
                    module_name,
                    raising=False,
                )
        agent_state.importlib.invalidate_caches()

    reset_matminer_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    real_run = subprocess.run
    matminer_calls = 0

    def selective_run(command, **kwargs):
        nonlocal matminer_calls
        if command[-2] != 'matminer':
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b'',
                stderr=b'',
            )
        matminer_calls += 1
        return real_run(command, **kwargs)

    monkeypatch.setattr(agent_state.subprocess, 'run', selective_run)

    baseline_validation = validate_agent_layout(ROOT)
    cached_baseline_validation = validate_agent_layout(ROOT)

    assert baseline_validation['status'] == 'ok'
    assert cached_baseline_validation['status'] == 'ok'
    assert matminer_calls == 1

    target_path = target_paths[target]
    private_diagnostic = 'synthetic private matminer consumer failure'
    if mutation == 'missing-submodule':
        target_path.unlink()
    elif mutation == 'raising-target':
        target_path.write_text(
            f'raise RuntimeError({private_diagnostic!r})\n',
            encoding='utf-8',
        )
    else:
        target_path.write_text(
            ''.join(
                f'class {symbol}:\n    pass\n'
                for symbol in MATMINER_TARGET_SYMBOLS[target]
                if symbol != missing_symbol
            )
            + (
                'def __getattr__(name):\n'
                f'    raise AttributeError({private_diagnostic!r})\n'
            ),
            encoding='utf-8',
        )
    reset_matminer_modules()
    probe_environment = os.environ.copy()
    probe_environment['PYTHONDONTWRITEBYTECODE'] = '1'
    direct_consumer = real_run(
        [
            sys.executable,
            '-c',
            (
                'from matminer.featurizers.base import MultipleFeaturizer; '
                'from matminer.featurizers.composition import '
                'ElementProperty, Stoichiometry'
            ),
        ],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    repeated_validation = validate_agent_layout(ROOT)
    agent_state._clear_dependency_import_probe_cache()
    uncached_validation = validate_agent_layout(ROOT)
    completed = real_run(
        [
            sys.executable,
            str(ROOT / 'main.py'),
            '--verify-agent-contract',
        ],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert direct_consumer.returncode == 1
    assert repeated_validation['status'] == 'error'
    assert uncached_validation['status'] == 'error'
    assert matminer_calls == 3
    for validation in (repeated_validation, uncached_validation):
        matminer_check = next(
            check
            for check in validation['checks']
            if check.get('kind') == 'dependency_import'
            and check.get('module') == 'matminer'
        )
        assert matminer_check['import_probe_targets'] == list(
            MATMINER_TARGET_SYMBOLS
        )
        assert matminer_check['import_probe_symbols'] == {
            target_name: list(symbols)
            for target_name, symbols in MATMINER_TARGET_SYMBOLS.items()
        }
        assert matminer_check['available'] is False
        assert matminer_check['failure_category'] == 'nonzero_exit'

    assert completed.returncode == 1
    assert completed.stderr == ''
    payload = json.loads(completed.stdout)
    assert payload['status'] == 'error'
    assert private_diagnostic not in completed.stdout


def test_public_pymatgen_probe_rejects_changed_required_namespace_submodule(
    monkeypatch,
    tmp_path,
    cleared_dependency_import_probe_cache,
):
    shim_root = tmp_path / 'shim'
    core_dir = shim_root / 'pymatgen' / 'core'
    core_dir.mkdir(parents=True)
    core_init = core_dir / '__init__.py'
    core_init.write_text(
        'Composition = Element = Structure = Lattice = object\n',
        encoding='utf-8',
    )
    monkeypatch.syspath_prepend(str(shim_root))
    monkeypatch.setenv('PYTHONPATH', str(shim_root))
    monkeypatch.delitem(sys.modules, 'pymatgen', raising=False)
    monkeypatch.delitem(sys.modules, 'pymatgen.core', raising=False)
    agent_state.importlib.invalidate_caches()
    real_run = subprocess.run
    pymatgen_calls = 0

    def selective_run(command, **kwargs):
        nonlocal pymatgen_calls
        if command[-2] != 'pymatgen':
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b'',
                stderr=b'',
            )
        pymatgen_calls += 1
        return real_run(command, **kwargs)

    monkeypatch.setattr(agent_state.subprocess, 'run', selective_run)

    assert validate_agent_layout(ROOT)['status'] == 'ok'

    core_init.write_text(
        'raise RuntimeError("synthetic required consumer failure")\n',
        encoding='utf-8',
    )
    agent_state.importlib.invalidate_caches()
    probe_environment = os.environ.copy()
    probe_environment['PYTHONDONTWRITEBYTECODE'] = '1'
    direct_consumer = real_run(
        [
            sys.executable,
            '-c',
            'from pymatgen.core import Composition, Element, Structure',
        ],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        check=False,
    )
    validation = validate_agent_layout(ROOT)

    assert direct_consumer.returncode == 1
    assert validation['status'] == 'error'
    assert pymatgen_calls == 2
    pymatgen_check = next(
        check
        for check in validation['checks']
        if check.get('kind') == 'dependency_import'
        and check.get('module') == 'pymatgen'
    )
    assert pymatgen_check['import_probe_targets'] == ['pymatgen.core']
    assert pymatgen_check['available'] is False
    assert pymatgen_check['failure_category'] == 'nonzero_exit'


@pytest.mark.parametrize('missing_symbol', PYMATGEN_CORE_SYMBOLS)
def test_public_pymatgen_probe_rejects_missing_required_consumer_symbol(
    monkeypatch,
    tmp_path,
    cleared_dependency_import_probe_cache,
    missing_symbol,
):
    shim_root = tmp_path / 'shim'
    core_dir = shim_root / 'pymatgen' / 'core'
    core_dir.mkdir(parents=True)
    core_init = core_dir / '__init__.py'

    def write_symbols(symbols):
        core_init.write_text(
            ''.join(f'{symbol} = object\n' for symbol in symbols)
            + (
                'def __getattr__(name):\n'
                '    raise AttributeError("synthetic private symbol failure")\n'
            ),
            encoding='utf-8',
        )
        monkeypatch.delitem(sys.modules, 'pymatgen', raising=False)
        monkeypatch.delitem(sys.modules, 'pymatgen.core', raising=False)
        agent_state.importlib.invalidate_caches()

    write_symbols(PYMATGEN_CORE_SYMBOLS)
    monkeypatch.syspath_prepend(str(shim_root))
    monkeypatch.setenv('PYTHONPATH', str(shim_root))
    real_run = subprocess.run
    pymatgen_calls = 0

    def selective_run(command, **kwargs):
        nonlocal pymatgen_calls
        if command[-2] != 'pymatgen':
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b'',
                stderr=b'',
            )
        pymatgen_calls += 1
        return real_run(command, **kwargs)

    monkeypatch.setattr(agent_state.subprocess, 'run', selective_run)

    assert validate_agent_layout(ROOT)['status'] == 'ok'

    write_symbols(
        symbol
        for symbol in PYMATGEN_CORE_SYMBOLS
        if symbol != missing_symbol
    )
    probe_environment = os.environ.copy()
    probe_environment['PYTHONDONTWRITEBYTECODE'] = '1'
    direct_module = real_run(
        [sys.executable, '-c', 'import pymatgen.core'],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        check=False,
    )
    direct_symbol = real_run(
        [
            sys.executable,
            '-c',
            f'from pymatgen.core import {missing_symbol}',
        ],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        check=False,
    )
    repeated_validation = validate_agent_layout(ROOT)
    agent_state._clear_dependency_import_probe_cache()
    uncached_validation = validate_agent_layout(ROOT)

    assert direct_module.returncode == 0
    assert direct_symbol.returncode == 1
    assert repeated_validation['status'] == 'error'
    assert uncached_validation['status'] == 'error'
    assert pymatgen_calls == 3
    for validation in (repeated_validation, uncached_validation):
        pymatgen_check = next(
            check
            for check in validation['checks']
            if check.get('kind') == 'dependency_import'
            and check.get('module') == 'pymatgen'
        )
        assert pymatgen_check['import_probe_symbols'] == {
            'pymatgen.core': list(PYMATGEN_CORE_SYMBOLS),
        }
        assert pymatgen_check['available'] is False
        assert pymatgen_check['failure_category'] == 'nonzero_exit'


@pytest.mark.parametrize(
    ('failure_mode', 'expected_category'),
    [
        ('nonzero', 'nonzero_exit'),
        ('signal', 'signal'),
        ('timeout', 'timeout'),
        ('launch', 'launch_error'),
    ],
)
def test_dependency_import_probe_redacts_bounded_failure_categories(
    monkeypatch,
    cleared_dependency_import_probe_cache,
    failure_mode,
    expected_category,
):
    private_diagnostic = b'synthetic-private-dependency-diagnostic'

    def failed_run(command, **kwargs):
        if failure_mode == 'timeout':
            raise subprocess.TimeoutExpired(
                command,
                timeout=kwargs['timeout'],
                output=private_diagnostic,
                stderr=private_diagnostic,
            )
        if failure_mode == 'launch':
            raise OSError(private_diagnostic.decode())
        return subprocess.CompletedProcess(
            command,
            -6 if failure_mode == 'signal' else 1,
            stdout=private_diagnostic,
            stderr=private_diagnostic,
        )

    monkeypatch.setattr(agent_state.subprocess, 'run', failed_run)

    result = agent_state._probe_dependency_import(
        ROOT,
        'pydantic',
        (),
        timeout_seconds=10.0,
    )

    assert result == (False, expected_category)
    assert private_diagnostic.decode() not in repr(result)


def test_dependency_import_probe_does_not_cache_transient_failures(
    monkeypatch,
    cleared_dependency_import_probe_cache,
):
    timeouts = []

    def timeout_then_succeed(command, **kwargs):
        timeouts.append(kwargs['timeout'])
        if len(timeouts) == 1:
            raise subprocess.TimeoutExpired(
                command,
                timeout=kwargs['timeout'],
            )
        return subprocess.CompletedProcess(command, 0, stdout=b'', stderr=b'')

    monkeypatch.setattr(agent_state.subprocess, 'run', timeout_then_succeed)

    assert agent_state._probe_dependency_import(
        ROOT,
        'pydantic',
        (),
        timeout_seconds=0.01,
    ) == (False, 'timeout')
    assert agent_state._probe_dependency_import(
        ROOT,
        'pydantic',
        (),
        timeout_seconds=10.0,
    ) == (True, None)
    assert timeouts == [0.01, 10.0]


def test_dependency_import_probe_cache_is_consulted_before_shared_budget(
    monkeypatch,
    cleared_dependency_import_probe_cache,
):
    cached_key = agent_state._dependency_import_probe_cache_key(
        ROOT,
        'pydantic',
        (),
    )
    agent_state._DEPENDENCY_IMPORT_PROBE_CACHE[cached_key] = (True, None)
    monkeypatch.setattr(
        agent_state,
        'DEPENDENCY_IMPORT_PROBE_TOTAL_BUDGET_SECONDS',
        0.0,
    )
    monkeypatch.setattr(
        agent_state,
        '_probe_dependency_import',
        lambda *args, **kwargs: pytest.fail('budget exhaustion must skip new probes'),
    )

    validation = validate_agent_layout(ROOT)
    checks = {
        check['module']: check
        for check in validation['checks']
        if check['kind'] == 'dependency_import'
    }

    assert checks['pydantic']['available'] is True
    assert 'failure_category' not in checks['pydantic']
    assert checks['pyarrow']['available'] is False
    assert checks['pyarrow']['failure_category'] == 'budget_exhausted'
    assert any(
        error['code'] == 'unimportable_declared_dependency'
        and error['module'] == 'pyarrow'
        and error['failure_category'] == 'budget_exhausted'
        for error in validation['errors']
    )


def test_verify_agent_contract_rejects_spec_present_broken_pyarrow_import(
    tmp_path,
):
    shim_path = tmp_path / 'pyarrow.py'
    shim_path.write_text(
        'from synthetic_missing_native_backend import ABI_SYMBOL\n',
        encoding='utf-8',
    )
    probe_environment = os.environ.copy()
    existing_pythonpath = probe_environment.get('PYTHONPATH', '')
    probe_environment['PYTHONPATH'] = os.pathsep.join(
        part
        for part in (str(tmp_path), existing_pythonpath)
        if part
    )
    probe_environment['PYTHONDONTWRITEBYTECODE'] = '1'

    completed = subprocess.run(
        [sys.executable, str(ROOT / 'main.py'), '--verify-agent-contract'],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 1
    assert completed.stderr == ''
    payload = json.loads(completed.stdout)
    assert payload['status'] == 'error'
    pyarrow_check = next(
        check
        for check in payload['validation']['checks']
        if check.get('kind') == 'dependency_import'
        and check.get('module') == 'pyarrow'
    )
    assert pyarrow_check == {
        'kind': 'dependency_import',
        'package': 'pyarrow',
        'module': 'pyarrow',
        'required_for': 'pandas_parquet_dataset_cache',
        'role': 'optional_lazy',
        'available': False,
        'import_probe_preloads': [],
        'import_probe_targets': [],
        'import_probe_symbols': {},
        'failure_category': 'nonzero_exit',
    }
    assert any(
        error['code'] == 'unimportable_declared_dependency'
        and error['module'] == 'pyarrow'
        and error['failure_category'] == 'nonzero_exit'
        for error in payload['validation']['errors']
    )
    assert 'synthetic_missing_native_backend' not in completed.stdout


def test_verify_agent_contract_rejects_broken_pymatgen_consumer_probe(
    tmp_path,
):
    core_dir = tmp_path / 'pymatgen' / 'core'
    core_dir.mkdir(parents=True)
    (core_dir / '__init__.py').write_text(
        'raise RuntimeError("synthetic required consumer failure")\n',
        encoding='utf-8',
    )
    probe_environment = os.environ.copy()
    existing_pythonpath = probe_environment.get('PYTHONPATH', '')
    probe_environment['PYTHONPATH'] = os.pathsep.join(
        part
        for part in (str(tmp_path), existing_pythonpath)
        if part
    )
    probe_environment['PYTHONDONTWRITEBYTECODE'] = '1'

    completed = subprocess.run(
        [sys.executable, str(ROOT / 'main.py'), '--verify-agent-contract'],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 1
    assert completed.stderr == ''
    payload = json.loads(completed.stdout)
    assert payload['status'] == 'error'
    pymatgen_check = next(
        check
        for check in payload['validation']['checks']
        if check.get('kind') == 'dependency_import'
        and check.get('module') == 'pymatgen'
    )
    assert pymatgen_check == {
        'kind': 'dependency_import',
        'package': 'pymatgen',
        'module': 'pymatgen',
        'required_for': 'formula_and_structure_processing',
        'role': 'scientific_pipeline',
        'available': False,
        'import_probe_preloads': [],
        'import_probe_targets': ['pymatgen.core'],
        'import_probe_symbols': {
            'pymatgen.core': list(PYMATGEN_CORE_SYMBOLS),
        },
        'failure_category': 'nonzero_exit',
    }
    assert any(
        error['code'] == 'unimportable_declared_dependency'
        and error['module'] == 'pymatgen'
        and error['failure_category'] == 'nonzero_exit'
        for error in payload['validation']['errors']
    )
    assert 'synthetic required consumer failure' not in completed.stdout


@pytest.mark.parametrize('missing_symbol', PYMATGEN_CORE_SYMBOLS)
def test_verify_agent_contract_rejects_missing_pymatgen_consumer_symbol(
    tmp_path,
    missing_symbol,
):
    core_dir = tmp_path / 'pymatgen' / 'core'
    core_dir.mkdir(parents=True)
    (core_dir / '__init__.py').write_text(
        ''.join(
            f'{symbol} = object\n'
            for symbol in PYMATGEN_CORE_SYMBOLS
            if symbol != missing_symbol
        )
        + (
            'def __getattr__(name):\n'
            '    raise AttributeError("synthetic private symbol failure")\n'
        ),
        encoding='utf-8',
    )
    probe_environment = os.environ.copy()
    existing_pythonpath = probe_environment.get('PYTHONPATH', '')
    probe_environment['PYTHONPATH'] = os.pathsep.join(
        part
        for part in (str(tmp_path), existing_pythonpath)
        if part
    )
    probe_environment['PYTHONDONTWRITEBYTECODE'] = '1'

    direct_module = subprocess.run(
        [sys.executable, '-c', 'import pymatgen.core'],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    direct_symbol = subprocess.run(
        [
            sys.executable,
            '-c',
            f'from pymatgen.core import {missing_symbol}',
        ],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / 'main.py'), '--verify-agent-contract'],
        cwd=ROOT,
        env=probe_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert direct_module.returncode == 0
    assert direct_symbol.returncode == 1
    assert completed.returncode == 1
    assert completed.stderr == ''
    payload = json.loads(completed.stdout)
    assert payload['status'] == 'error'
    pymatgen_check = next(
        check
        for check in payload['validation']['checks']
        if check.get('kind') == 'dependency_import'
        and check.get('module') == 'pymatgen'
    )
    assert pymatgen_check['import_probe_symbols'] == {
        'pymatgen.core': list(PYMATGEN_CORE_SYMBOLS),
    }
    assert pymatgen_check['available'] is False
    assert pymatgen_check['failure_category'] == 'nonzero_exit'
    assert 'synthetic private symbol failure' not in completed.stdout


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
            lambda manifest: next(
                entry
                for entry in manifest['validation_commands']
                if entry['name'] == 'focused_regression'
            ).pop('pytest_non_vacuity_targets'),
            'missing_pytest_non_vacuity_targets',
        ),
        (
            lambda manifest: next(
                entry
                for entry in manifest['validation_commands']
                if entry['name'] == 'full_src_tests'
            ).update({'pytest_non_vacuity_targets': ['src/tests']}),
            'pytest_non_vacuity_command_target_mismatch',
        ),
        (
            lambda manifest: next(
                entry
                for entry in manifest['validation_commands']
                if entry['name'] == 'fast_smoke'
            ).update({'pytest_non_vacuity_targets': ['src']}),
            'forbidden_pytest_non_vacuity_targets',
        ),
        (
            lambda manifest: next(
                entry
                for entry in manifest['validation_commands']
                if entry['name'] == 'ui_render_smoke'
            ).update({
                'pytest_non_vacuity_targets': ['.'],
            }),
            'invalid_pytest_non_vacuity_targets',
        ),
    ],
)
def test_validate_agent_layout_rejects_invalid_pytest_non_vacuity_contracts(
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
