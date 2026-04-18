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
    assert cfg['robustness']['enabled'] is True
    assert cfg['robustness']['method'] == 'group_kfold_by_formula'
    assert cfg['robustness']['group_column'] == 'formula'
    assert cfg['robustness']['n_splits'] == 5
    assert cfg['screening']['candidate_generation_strategy'] == 'bn_anchored_formula_family_grid'
    assert cfg['screening']['candidate_space_name'] == 'bn_anchored_formula_family_grid'
    assert cfg['screening']['candidate_space_kind'] == 'bn_family_demo'
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
    assert cfg['screening']['bn_support']['enabled'] is True
    assert (
        cfg['screening']['bn_support']['method']
        == 'train_plus_val_bn_knn_feature_space_support'
    )
    assert cfg['screening']['bn_support']['distance_metric'] == 'z_scored_euclidean_rms'
    assert cfg['screening']['bn_support']['k_neighbors'] == 3
    assert cfg['screening']['bn_support']['ranking_penalty_enabled'] is True
    assert cfg['screening']['bn_support']['ranking_penalty_weight'] == 0.1
    assert cfg['screening']['bn_support']['penalize_below_percentile'] == 25.0
    assert cfg['screening']['bn_analog_evidence']['enabled'] is True
    assert (
        cfg['screening']['bn_analog_evidence']['aggregation']
        == 'mean_over_k_nearest_bn_formulas'
    )
    assert (
        cfg['screening']['bn_analog_evidence']['reference_split']
        == 'train_plus_val_bn_unique_formulas'
    )
    assert (
        cfg['screening']['bn_analog_evidence']['exfoliation_reference']
        == 'train_plus_val_bn_formula_median'
    )
    assert cfg['screening']['bn_analog_validation']['enabled'] is True
    assert (
        cfg['screening']['bn_analog_validation']['method']
        == 'bn_analog_alignment_vote_fraction'
    )
    assert cfg['screening']['bn_analog_validation']['ranking_penalty_enabled'] is True
    assert cfg['screening']['bn_analog_validation']['ranking_penalty_weight'] == 0.12
    assert cfg['screening']['chemical_plausibility']['enabled'] is True
    assert (
        cfg['screening']['chemical_plausibility']['method']
        == 'pymatgen_common_oxidation_state_balance'
    )
    assert cfg['ui']['streamlit_enabled'] is True
