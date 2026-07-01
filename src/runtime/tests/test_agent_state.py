from pathlib import Path
import json

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
    dependency_modules = {
        check['module']
        for check in validation['checks']
        if check['kind'] == 'dependency_import'
    }
    assert {'pandas', 'numpy', 'sklearn', 'pyarrow', 'torch'}.issubset(dependency_modules)


def test_build_agent_command_index_returns_validation_profiles():
    command_index = build_agent_command_index(ROOT)

    assert command_index['schema_version'] == 'aiforbn.agent_command_index.v1'
    assert command_index['first_inspection_command'] == 'python3 main.py --verify-agent-contract'
    assert {entry['name'] for entry in command_index['entrypoints']} >= {
        'fast_smoke',
        'emit_agent_commands',
        'verify_agent_contract',
    }
    validation_names = {entry['name'] for entry in command_index['validation_commands']}
    assert {'verify_agent_contract', 'fast_smoke', 'full_src_tests'}.issubset(validation_names)
    assert any(profile['name'] == 'architecture_doc_skill_edit' for profile in command_index['validation_profiles'])


def test_build_agent_state_returns_json_serializable_status():
    state = build_agent_state(ROOT)

    assert state['schema_version'] == 'aiforbn.agent_state.v1'
    assert state['status'] == 'ok'
    assert state['manifest']['project']['name'] == 'aiforbn'
    assert state['git']['branch'] is not None
    parsed = json.loads(agent_state_to_json(state))
    assert parsed['schema_version'] == 'aiforbn.agent_state.v1'
