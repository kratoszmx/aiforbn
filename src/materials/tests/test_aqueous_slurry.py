"""Scientific data boundaries: original cells, cohort isolation and label-blind checks."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from materials.aqueous_slurry import (
    AQUEOUS_DATA_PATH, AqueousSlurryRecipe, evaluate_aqueous_viscosity,
    load_aqueous_slurries, predict_aqueous_viscosity,
)


def test_source_cells_units_and_missing_failure_are_preserved():
    data = load_aqueous_slurries()
    rows = {r['record_id']: r for r in data['records']}
    assert len(rows) == 12
    assert [rows[f'LG-{i:02}']['viscosity_mPa_s'] for i in range(1,9)] == [7.1,7.8,11.6,6.2,5.4,31.5,None,8.3]
    assert [rows[f'PI-{i:02}']['viscosity_mPa_s'] for i in range(1,5)] == [27.,18.,24.,20.]
    assert rows['LG-07']['coating_result'] == '膏狀，無法塗布'
    assert rows['LG-08']['cohort'] == 'Al2O3_SBA_water'
    assert rows['PI-01']['solids_weight_pct'] is None
    assert '水 433 g' in rows['PI-02']['formulation']
    assert all(r['measurement_temperature_c'] is None and r['shear_rate_s_inv'] is None for r in rows.values())


@pytest.mark.parametrize('field,value', [
    ('source_file','../../etc/passwd'), ('sha256','0'*64),
])
def test_sources_fail_closed_on_manifest_tampering(tmp_path,field,value):
    data = json.loads(AQUEOUS_DATA_PATH.read_text())
    data['sources'][0][field] = value
    path = tmp_path/'sources.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_aqueous_slurries(path)


@pytest.mark.parametrize('bn,expected', [(0,5.4),(.5,5.8),(1,6.2),(2,6.65),
    (3,7.1),(4,7.45),(5,7.8),(6.5,9.7),(8,11.6),(10,21.55),(12,31.5)])
def test_matched_series_and_intermediate_inputs(bn,expected):
    result = predict_aqueous_viscosity(load_aqueous_slurries(), {'bn_volume_pct':bn})
    assert result['value'] == pytest.approx(expected)
    assert result['unit'] == 'mPa·s'
    assert len(result['supporting_records']) == 6
    assert not set(result['supporting_records']) & {'LG-07','LG-08','PI-01','PI-02','PI-03','PI-04'}
    assert result['matched_observation_mPa_s'] == (expected if bn in (0,1,3,5,8,12) else None)


def test_incompatible_formulations_cannot_change_prediction():
    data = load_aqueous_slurries()
    original = predict_aqueous_viscosity(data, {'bn_volume_pct':4})
    for row in data['records']:
        if row['cohort'] != 'BaTiO3_BN_SBA_water':
            row['viscosity_mPa_s'] = 9999
    assert predict_aqueous_viscosity(data, {'bn_volume_pct':4}) == original


@pytest.mark.parametrize('changes', [
    {'bn_volume_pct':-.1}, {'bn_volume_pct':12.1}, {'bn_volume_pct':float('nan')},
    {'solids_weight_pct':40}, {'binder_volume_pct':20}, {'binder':'PVDF'},
    {'solvent':'NMP'}, {'filler':'Al2O3'}, {'bn_diameter_nm':30},
    {'filler_diameter_nm':100}, {'preparation_temperature_c':40}, {'mixing_minutes':30},
    {'shear_rate_s_inv':100}, {'model_name':'anything'},
])
def test_unmatched_measurement_and_recipe_settings_are_rejected(changes):
    with pytest.raises(ValidationError):
        AqueousSlurryRecipe.model_validate(changes)


def test_development_evaluation_hides_each_target_and_keeps_all_methods():
    data = load_aqueous_slurries(); report = evaluate_aqueous_viscosity(data)
    assert len(report['cases']) == 4
    assert not report['independent_validation'] and not report['experimental_savings_measured']
    assert set(report['mae_mPa_s']) == {'interpolation','nearest','linear'}
    for case in report['cases']:
        assert case['record_id'] not in case['train_ids']
        changed = deepcopy(data)
        next(r for r in changed['records'] if r['record_id']==case['record_id'])['viscosity_mPa_s'] += 100
        rerun = next(c for c in evaluate_aqueous_viscosity(changed)['cases'] if c['record_id']==case['record_id'])
        assert rerun['predictions_mPa_s'] == case['predictions_mPa_s']
