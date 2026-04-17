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
    assert cfg['split']['method'] == 'group_by_formula'
    assert cfg['features']['feature_set'] == 'basic_formula_composition'
    assert cfg['features']['candidate_sets'] == [
        'basic_formula_composition',
        'matminer_composition',
        'matminer_composition_plus_structure_summary',
    ]
    assert cfg['features']['feature_family'] == 'mixed_formula_and_structure'
    assert cfg['model']['candidate_types'] == ['linear_regression', 'hist_gradient_boosting']
    assert cfg['screening']['candidate_space_name'] == 'toy_iii_v_demo_grid'
    assert cfg['screening']['use_model_disagreement'] is True
    assert cfg['screening']['uncertainty_method'] == 'small_feature_model_disagreement'
    assert cfg['screening']['uncertainty_penalty'] == 0.5
    assert cfg['screening']['domain_support']['enabled'] is True
    assert (
        cfg['screening']['domain_support']['method']
        == 'train_plus_val_knn_feature_space_support'
    )
    assert cfg['screening']['domain_support']['distance_metric'] == 'z_scored_euclidean_rms'
    assert cfg['screening']['domain_support']['k_neighbors'] == 5
    assert cfg['screening']['domain_support']['ranking_penalty_enabled'] is True
    assert cfg['screening']['domain_support']['ranking_penalty_weight'] == 0.15
    assert cfg['screening']['domain_support']['penalize_below_percentile'] == 25.0
    assert cfg['screening']['chemical_plausibility']['enabled'] is True
    assert (
        cfg['screening']['chemical_plausibility']['method']
        == 'pymatgen_common_oxidation_state_balance'
    )
    assert cfg['ui']['streamlit_enabled'] is True
