from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_has_expected_poc_defaults():
    cfg_path = ROOT / 'configs' / 'default.py'
    spec = spec_from_file_location('default_config', cfg_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cfg = module.CONFIG
    assert cfg['data']['dataset'] == 'twod_matpd'
    assert cfg['data']['target_column'] == 'band_gap'
    assert cfg['screening']['candidate_strategy'] == 'simple_bn_substitutions'
    assert cfg['ui']['streamlit_enabled'] is True
