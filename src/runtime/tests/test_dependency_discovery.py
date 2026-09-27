"""Dependency cache discovery must not run the code it is inspecting."""

import os
import sys

from runtime import agent_state


def test_descendant_fingerprint_does_not_execute_packages(tmp_path, monkeypatch):
    owner = 'aiforbn_discovery_fixture'
    package = tmp_path / owner
    branch = package / 'branch'
    branch.mkdir(parents=True)
    marker = tmp_path / 'unexpected-import.txt'
    initializer = (
        'from pathlib import Path\n'
        f'Path({str(marker)!r}).write_text("executed")\n'
        'raise RuntimeError("private package initialization failure")\n'
    )
    (package / '__init__.py').write_text(initializer)
    (branch / '__init__.py').write_text(initializer)
    target = branch / 'target.py'
    target.write_text('value = 1\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    descendants = (f'{owner}.branch.target',)

    before = agent_state._dependency_import_probe_context_digest(owner, (), descendants)
    assert before == agent_state._dependency_import_probe_context_digest(owner, (), descendants)
    assert not marker.exists()
    assert not any(name == owner or name.startswith(owner + '.') for name in sys.modules)

    original_stat = target.stat()
    target.write_text('value = 2\n')
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    changed = agent_state._dependency_import_probe_context_digest(owner, (), descendants)
    assert changed != before
    target.unlink()
    assert agent_state._dependency_import_probe_context_digest(owner, (), descendants) != changed
    assert not marker.exists()
