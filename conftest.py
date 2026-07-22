from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
_UI_RENDER_TEST_PATH = (ROOT / 'src' / 'ui' / 'tests' / 'test_streamlit_app.py').resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime.io_utils import clear_project_cache

clear_project_cache(str(ROOT))


def _explicit_ui_renderer_target_requested(config) -> bool:
    invocation_dir = Path(config.invocation_params.dir)
    for argument in config.args:
        path_text = str(argument).split('::', 1)[0]
        candidate_path = Path(path_text)
        if not candidate_path.is_absolute():
            candidate_path = invocation_dir / candidate_path
        try:
            if candidate_path.resolve() == _UI_RENDER_TEST_PATH:
                return True
        except OSError:
            continue
    return False


def pytest_configure(config):
    config._aiforbn_ui_renderer_required = (
        not bool(getattr(config.option, 'collectonly', False))
        and _explicit_ui_renderer_target_requested(config)
    )
    config._aiforbn_ui_renderer_passed_calls = 0


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        getattr(item.config, '_aiforbn_ui_renderer_required', False)
        and Path(item.path).resolve() == _UI_RENDER_TEST_PATH
        and report.when == 'call'
        and report.passed
        and getattr(report, 'wasxfail', None) is None
    ):
        item.config._aiforbn_ui_renderer_passed_calls += 1


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if (
        not getattr(config, '_aiforbn_ui_renderer_required', False)
        or getattr(config, '_aiforbn_ui_renderer_passed_calls', 0) > 0
        or exitstatus != pytest.ExitCode.OK
    ):
        return
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
    terminal_reporter = config.pluginmanager.get_plugin('terminalreporter')
    if terminal_reporter is not None:
        terminal_reporter.write_line(
            'ERROR: ui_render_smoke passed no non-xfail renderer test calls.'
        )
