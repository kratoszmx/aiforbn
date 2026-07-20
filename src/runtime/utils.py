from __future__ import annotations

from pathlib import Path


def _path_is_same_or_descendant(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        pass

    if not parent.exists():
        return False
    for candidate in (path, *path.parents):
        try:
            if candidate.exists() and candidate.samefile(parent):
                return True
        except OSError:
            continue
    return False


def _path_has_symlink_component(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents))
