"""Measured-data grouping, answer isolation and sequential evaluation checks."""
from copy import deepcopy
import csv
import hashlib
import json

import numpy as np
import pytest

from materials.experiment_planning import (
    SOURCE_DIR, PROTOCOL_PATH, ElectrolyteRecipe, ExperimentPlan,
    load_electrolyte_records, predict_electrolyte, suggest_experiments,
    evaluate_experiment_planner,
)
from materials.separator_data import CoatingRecipe, load_separator_dataset, predict_coating_thickness


@pytest.fixture(scope='module')
def records():
    return load_electrolyte_records()


def test_measurements_group_whole_formulations_and_reconstruct_source(records,tmp_path):
    with (SOURCE_DIR/'clio_2022_measurements.csv').open() as stream:
        raw=list(csv.DictReader(stream))
    assert len(records)==38 and len(raw)==125
    lines=[]
    compositions=[]
    for r in records:
        original=[raw[line-2] for line in r['source_csv_lines']]
        values=[float(x['conductivity']) for x in original]
        assert r['conductivity_mS_cm']==pytest.approx(sum(values)/len(values))
        assert r['repeat_measurements']==len(values)
        assert all(float(x['LiPF6_molalitt'])==r['salt_molality'] for x in original)
        lines.extend(r['source_csv_lines'])
        compositions.append((r['salt_molality'],r['ec_fraction'],r['dmc_ratio']))
    assert sorted(lines)==list(range(2,127)) and len(set(compositions))==38
    (tmp_path/'source_manifest.json').write_bytes((SOURCE_DIR/'source_manifest.json').read_bytes())
    (tmp_path/'clio_2022_measurements.csv').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):
        load_electrolyte_records(tmp_path)


@pytest.mark.parametrize('salt',[.14,1.,2.])
@pytest.mark.parametrize('ec',[.3,.4,.5])
@pytest.mark.parametrize('dmc',[0.,.5,1.])
def test_numerical_grid_spans_composition_domain(records,salt,ec,dmc):
    result=predict_electrolyte(records,ElectrolyteRecipe(salt_molality=salt,ec_fraction=ec,dmc_ratio=dmc))
    assert 0<=result['value']<=max(r['conductivity_mS_cm'] for r in records)
    assert result['unit']=='mS/cm' and result['supporting_formulations']==38
    assert sum(result['solvent_percent'].values())==pytest.approx(100)


def test_exact_composition_reference_is_distinct_from_prediction(records):
    source=records[0]
    result=predict_electrolyte(records,{k:source[k] for k in ('salt_molality','ec_fraction','dmc_ratio')})
    assert result['matched_observation_mS_cm']==source['conductivity_mS_cm']
    assert result['kind']=='模型估算'


@pytest.mark.parametrize('gap,expected',[(50,88),(75,99),(100,110),(150,119.5),(200,129)])
def test_coating_interpolation_preserves_measured_units_and_control(gap,expected):
    result=predict_coating_thickness(load_separator_dataset(),CoatingRecipe(applicator_gap_um=gap))
    assert result['value']==expected and result['unit']=='μm'
    assert len(result['supporting_records'])==3
    assert result['matched_observation_um']==(expected if gap in [50,100,200] else None)


@pytest.mark.parametrize('change',[{'salt_molality':float('nan')},{'dmc_ratio':float('inf')},
    {'ec_fraction':float('-inf')},{'temperature_c':25},{'prompt':'invent an answer'}])
def test_nonfinite_or_unmodeled_inputs_rejected(change):
    with pytest.raises(ValueError):
        ElectrolyteRecipe(**change)


def test_planner_has_no_access_to_unknown_outcomes(records):
    candidates=[{k:r[k] for k in ('candidate_id','salt_molality','ec_fraction','dmc_ratio')} for r in records]
    observed={r['candidate_id']:r['conductivity_mS_cm'] for r in records[:5]}
    expected=suggest_experiments(candidates,observed,limit=3)
    changed=deepcopy(records)
    for r in changed[5:]:
        r['conductivity_mS_cm']=9999
    changed_inputs=[{k:r[k] for k in candidates[0]} for r in changed]
    assert suggest_experiments(changed_inputs,observed,limit=3)==expected
    with pytest.raises(ValueError,match='only identifiers'):
        suggest_experiments(records,observed)
    assert not set(observed)&{r['candidate_id'] for r in expected}
    with pytest.raises(ValueError,match='identity'):
        suggest_experiments(candidates,{'unknown':1})
    with pytest.raises(ValueError,match='permutation'):
        suggest_experiments(candidates,observed,tie_order=['CLIO-01']*len(candidates))
    with pytest.raises(ValueError,match='At least two'):
        suggest_experiments(candidates,{'CLIO-01':1})
    with pytest.raises(ValueError):
        ExperimentPlan(observations={'CLIO-01':1,'CLIO-02':float('nan')})


def test_every_published_trial_stops_on_third_hit_and_uses_paired_starts(records):
    directory=PROTOCOL_PATH.parent
    protocol=json.loads(PROTOCOL_PATH.read_text())
    report=json.loads((directory/'workflow_evaluation.json').read_text())
    trials=json.loads((directory/'workflow_trials.json').read_text())
    truth={r['candidate_id']:r['conductivity_mS_cm'] for r in records}
    assert len(trials)==100*4*3
    assert report['protocol_sha256']==hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    assert report['source_sha256']==hashlib.sha256((SOURCE_DIR/'clio_2022_measurements.csv').read_bytes()).hexdigest()
    assert report['actual_lab_experiments_saved'] is None
    for trial in trials:
        sequence=trial['query_sequence']
        assert len(sequence)==trial['experiments']==len(set(sequence))
        hits=[truth[key]>=trial['target_mS_cm'] for key in sequence]
        assert sum(hits)==3 and sum(hits[:-1])==2
        order=np.random.default_rng(trial['seed']).permutation(list(truth)).tolist()
        warm=min(5,len(sequence))
        assert sequence[:warm]==order[:warm]
        if trial['method']=='random':
            assert sequence==order[:len(sequence)]
    for summary in report['summaries']:
        counts={method:[t['experiments'] for t in trials if t['target_mS_cm']==summary['target_mS_cm']
                        and t['method']==method] for method in protocol['methods']}
        for item in summary['methods']:
            assert item['mean_experiments']==pytest.approx(np.mean(counts[item['method']]))
        for item in summary['comparisons']:
            before=np.asarray(counts[item['baseline']]);after=np.asarray(counts['adaptive_forest'])
            assert item['reduction_pct']==pytest.approx(100*(1-after.mean()/before.mean()))
            assert item['wins']==sum(after<before) and item['losses']==sum(after>before)


def test_recomputed_seed_matches_published_queries(records,tmp_path):
    protocol=json.loads(PROTOCOL_PATH.read_text())
    protocol['seed_values']=[17]
    result=evaluate_experiment_planner(records,protocol,tmp_path)
    actual=json.loads((tmp_path/'workflow_trials.json').read_text())
    published=[r for r in json.loads((PROTOCOL_PATH.parent/'workflow_trials.json').read_text()) if r['seed']==17]
    assert actual==published
    assert result['grouped_leave_one_out_mae_mS_cm']==pytest.approx(
        json.loads((PROTOCOL_PATH.parent/'workflow_evaluation.json').read_text())['grouped_leave_one_out_mae_mS_cm'])
