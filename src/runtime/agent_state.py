from __future__ import annotations

from datetime import datetime, timezone
import json
import importlib.util
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_AGENT_MANIFEST_PATH = Path('docs/AGENT_MANIFEST.json')

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

REQUIRED_RESEARCH_PLAN_DELIVERABLES = {
    'bn_dataset',
    'benchmarked_models',
    'ranked_candidates',
    'structure_handoff',
    'technical_report',
}

REQUIRED_RESEARCH_PLAN_NON_CLAIMS = {
    'open_ended_material_discovery',
    'experimental_synthesis_proof',
    'formula_stage_structure_dependent_property_claims',
    'direct_gap_claim_before_structure_review',
}


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

    _validate_command_entries(manifest_payload, 'entrypoints', errors)
    validation_command_names = _validate_command_entries(
        manifest_payload,
        'validation_commands',
        errors,
    )
    validation_profiles = manifest_payload.get('validation_profiles', [])
    if validation_profiles:
        if not isinstance(validation_profiles, list):
            errors.append({
                'code': 'invalid_validation_profiles',
                'path': 'docs/AGENT_MANIFEST.json',
                'message': 'Manifest field `validation_profiles` must be a list when present.',
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
                commands = profile.get('commands', [])
                if not isinstance(commands, list) or not all(
                    isinstance(command, str) and command.strip() for command in commands
                ):
                    errors.append({
                        'code': 'invalid_validation_profile_commands',
                        'path': f'docs/AGENT_MANIFEST.json:validation_profiles[{index}]',
                        'message': 'Validation profile `commands` must be a non-empty string list.',
                    })
                    continue
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

    project_skills = manifest_payload.get('project_skills', [])
    if not isinstance(project_skills, list):
        errors.append({
            'code': 'invalid_project_skills',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `project_skills` must be a list when present.',
        })
        project_skills = []
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

    retired_paths = ['skill.txt']
    retired_guidance_files = manifest_payload.get('retired_guidance_files', [])
    if not isinstance(retired_guidance_files, list):
        errors.append({
            'code': 'invalid_retired_guidance_files',
            'path': 'docs/AGENT_MANIFEST.json',
            'message': 'Manifest field `retired_guidance_files` must be a list when present.',
        })
        retired_guidance_files = []
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(agent_state_to_json(state) + '\n', encoding='utf-8')
