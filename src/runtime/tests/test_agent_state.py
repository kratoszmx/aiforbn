from pathlib import Path
import json

from runtime.agent_state import (
    agent_state_to_json,
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
    assert any(entry['name'] == 'verify_agent_contract' for entry in manifest['entrypoints'])


def test_validate_agent_layout_accepts_current_repo_contract():
    validation = validate_agent_layout(ROOT)

    assert validation['status'] == 'ok'
    assert validation['errors'] == []
    checked_paths = {check['path'] for check in validation['checks'] if 'path' in check}
    assert 'docs/AGENT_MANIFEST.json' in checked_paths
    assert 'src/runtime/PY_FILES_SUMMARY.md' in checked_paths
    assert 'skills/ai_native_workflow.txt' in checked_paths
    assert 'skill.txt' not in checked_paths
    dependency_modules = {
        check['module']
        for check in validation['checks']
        if check['kind'] == 'dependency_import'
    }
    assert {'pandas', 'numpy', 'sklearn', 'pyarrow', 'torch'}.issubset(dependency_modules)


def test_build_agent_state_returns_json_serializable_status():
    state = build_agent_state(ROOT)

    assert state['schema_version'] == 'aiforbn.agent_state.v1'
    assert state['status'] == 'ok'
    assert state['manifest']['project']['name'] == 'aiforbn'
    assert state['git']['branch'] is not None
    parsed = json.loads(agent_state_to_json(state))
    assert parsed['schema_version'] == 'aiforbn.agent_state.v1'
