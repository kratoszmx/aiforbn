from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
AGENT_MANIFEST_PATH = ROOT / 'docs' / 'AGENT_MANIFEST.json'
PYTEST_NON_VACUITY_TARGETS_FIELD = 'pytest_non_vacuity_targets'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime.io_utils import clear_project_cache, read_json_file

clear_project_cache(str(ROOT))


def _manifest_pytest_non_vacuity_contracts() -> dict[str, frozenset[Path]]:
    manifest = read_json_file(AGENT_MANIFEST_PATH)
    contracts = {}
    for entry in manifest['validation_commands']:
        targets = entry.get(PYTEST_NON_VACUITY_TARGETS_FIELD)
        if targets is not None:
            contracts[entry['name']] = frozenset(
                (ROOT / target).resolve() for target in targets
            )
    return contracts


def _resolved_explicit_targets(config) -> frozenset[Path]:
    invocation_dir = Path(config.invocation_params.dir)
    resolved_targets = set()
    for argument in config.args:
        path_text = str(argument).split('::', 1)[0]
        candidate_path = Path(path_text)
        if not candidate_path.is_absolute():
            candidate_path = invocation_dir / candidate_path
        try:
            resolved_targets.add(candidate_path.resolve())
        except OSError:
            return frozenset()
    return frozenset(resolved_targets)


def _active_pytest_non_vacuity_contract(config):
    requested_targets = _resolved_explicit_targets(config)
    for command_name, declared_targets in (
        _manifest_pytest_non_vacuity_contracts().items()
    ):
        if requested_targets == declared_targets:
            return command_name, declared_targets
    return None


def _item_is_under_declared_target(item, declared_targets: frozenset[Path]) -> bool:
    try:
        item_path = Path(item.path).resolve()
    except OSError:
        return False
    return any(
        item_path == target or target in item_path.parents
        for target in declared_targets
    )


def pytest_configure(config):
    config._aiforbn_pytest_non_vacuity_contract = (
        None
        if bool(getattr(config.option, 'collectonly', False))
        else _active_pytest_non_vacuity_contract(config)
    )
    config._aiforbn_pytest_non_vacuity_passed_calls = 0


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    active_contract = getattr(
        item.config,
        '_aiforbn_pytest_non_vacuity_contract',
        None,
    )
    if (
        active_contract is not None
        and _item_is_under_declared_target(item, active_contract[1])
        and report.when == 'call'
        and report.passed
        and getattr(report, 'wasxfail', None) is None
    ):
        item.config._aiforbn_pytest_non_vacuity_passed_calls += 1


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    config = session.config
    active_contract = getattr(
        config,
        '_aiforbn_pytest_non_vacuity_contract',
        None,
    )
    if (
        active_contract is None
        or getattr(config, '_aiforbn_pytest_non_vacuity_passed_calls', 0) > 0
        or exitstatus != pytest.ExitCode.OK
    ):
        return
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
    terminal_reporter = config.pluginmanager.get_plugin('terminalreporter')
    if terminal_reporter is not None:
        terminal_reporter.write_line(
            f'ERROR: {active_contract[0]} passed no non-xfail test calls.'
        )
