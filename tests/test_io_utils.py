from pathlib import Path

from core.io_utils import load_config


def test_load_config_from_python_module(tmp_path: Path):
    cfg_path = tmp_path / 'temp_config.py'
    cfg_path.write_text("CONFIG = {'project': {'name': 'demo'}, 'value': 7}\n", encoding='utf-8')

    cfg = load_config(cfg_path)

    assert cfg['project']['name'] == 'demo'
    assert cfg['value'] == 7
