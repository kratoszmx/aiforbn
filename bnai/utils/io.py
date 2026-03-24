from __future__ import annotations

from pathlib import Path
import shutil
import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def ensure_runtime_dirs(cfg: dict) -> None:
    for path in [cfg['data']['raw_dir'], cfg['data']['processed_dir'], cfg['project']['artifact_dir'], 'tasks', 'apps', 'tests', 'notebooks']:
        Path(path).mkdir(parents=True, exist_ok=True)


def delete_cache(project_root_path: str = '.') -> None:
    """
    删除指定目录下的所有 __pycache__ / pycache 文件夹。

    参数:
        project_root_path: 项目根目录路径，默认当前目录。
    """
    root_path = Path(project_root_path)
    cache_dirs = list(root_path.rglob('__pycache__')) + list(root_path.rglob('pycache'))
    seen = set()
    for cache_dir in cache_dirs:
        resolved = str(cache_dir.resolve())
        if resolved in seen or not cache_dir.exists():
            continue
        seen.add(resolved)
        print(f'Deleting: {cache_dir}')
        shutil.rmtree(cache_dir)
    print('All pycache directories have been deleted.')
