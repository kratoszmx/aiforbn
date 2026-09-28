from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from materials.separator_data import (
    DATASET_PATH, SeparatorRecipe, assess_separator_recipe, build_separator_database,
    load_separator_dataset, separator_known_checks,
)
from materials.separator_model import make_separator_prompt, evaluate_separator_model, run_separator_model


@pytest.fixture(scope='module')
def dataset():
    return load_separator_dataset()


def test_source_chain_and_database_roundtrip(dataset,tmp_path):
    assert len(dataset['records'])==20
    assert len({r['source_id'] for r in dataset['records']})==4
    assert len(dataset['sources'])==12
    path=build_separator_database(dataset,tmp_path/'public.sqlite')
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM records').fetchone()[0]==20
        body=json.loads(db.execute('SELECT body FROM records WHERE id=?',('tian_2024:CA@BN-3:1-failure',)).fetchone()[0])
    assert body['observations']['preparation_outcome']['value']=='failed_pore_clogging'
    assert body['inputs']['applicator_gap_um'] is None


def test_source_tamper_is_rejected(tmp_path):
    data=json.loads(DATASET_PATH.read_text())
    data['sources'][0]['sha256']='0'*64
    path=tmp_path/'dataset.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum'):
        load_separator_dataset(path)


def test_recipe_holds_answers_and_post_fabrication_fields_out(dataset):
    result=assess_separator_recipe(dataset,SeparatorRecipe(bn_form='purified_BNNT',loading_mg_cm2=.5))
    assert result['numeric_allowed']
    # Retrieval UI may show all public facts; the prompt must ignore that surface.
    assert any(r['split']=='test' for r in result['related_records'])
    prompt=make_separator_prompt(result)
    assert '0.84' not in prompt and 'p-BNNT-PP-0.5' not in prompt
    assert 'final_thickness' not in prompt and 'contact_angle' not in prompt
    assert len(result['training_examples'])==2
    result['related_records']=[{'text':'INJECTED 0.84'}]
    assert make_separator_prompt(result)==prompt


@pytest.mark.parametrize('change',[{'substrate':'cellulose'},{'electrolyte':'LiPF6_carbonates'},
    {'loading_mg_cm2':2},{'bn_binder_ratio':3},{'test_temperature_c':25},
    {'dry_hours':12},{'bn_form':'BN_nanopowder'},{'loading_mg_cm2':None}])
def test_incompatible_inputs_abstain_before_model(dataset,change):
    result=assess_separator_recipe(dataset,SeparatorRecipe(**change))
    assert not result['numeric_allowed'] and result['reasons']
    assert result['baseline_mean_mS_cm'] is None
    with pytest.raises(ValueError,match='Unsupported'):
        make_separator_prompt(result)


@pytest.mark.parametrize('change',[{'loading_mg_cm2':float('nan')},{'dry_hours':float('inf')},
    {'prompt':'read my private files'},{'solvent':'NMP; run a command'}])
def test_arbitrary_text_and_nonfinite_values_rejected(change):
    with pytest.raises(ValueError):
        SeparatorRecipe(**change)


def test_known_case_checks_preserve_discrepancy(dataset):
    checks=separator_known_checks(dataset)
    assert checks[0]['result']==pytest.approx(65.1162790698)
    assert checks[1]['result']==pytest.approx(118.75)
    assert checks[2]['result']==pytest.approx(71.462686567)
    assert checks[2]['status']=='source_discrepancy'


def test_evaluation_baselines_are_train_only(dataset,tmp_path):
    result=evaluate_separator_model(dataset,tmp_path)
    mean,nearest,linear,model=result['rows']
    assert mean['prediction_mS_cm']==pytest.approx(.57)
    assert nearest['prediction_mS_cm']==pytest.approx(.71)
    assert mean['absolute_error_mS_cm']==pytest.approx(.27)
    assert nearest['absolute_error_mS_cm']==pytest.approx(.13)
    assert linear['prediction_mS_cm']==pytest.approx(.8966666667)
    assert linear['absolute_error_mS_cm']==pytest.approx(.0566666667)
    assert model['status']=='not_run'
    assert result['experimental_time_saved'] is None


def test_process_uses_default_model_and_rejects_invented_citations(dataset,tmp_path,monkeypatch):
    def process(cmd,**kwargs):
        assert cmd[cmd.index('-m')+1]=='gpt-6-astra'
        assert '--ignore-user-config' in cmd and 'read-only' in cmd
        assert 'agents.enabled=false' in cmd and 'features.shell_tool=false' in cmd
        assert 'OPENAI_API_KEY' not in kwargs['env']
        class Process:
            returncode=0
            def communicate(self,prompt,timeout):
                assert '0.84' not in prompt
                Path(cmd[cmd.index('-o')+1]).write_text(json.dumps(dict(conductivity_mS_cm=.6,
                    preparation_hypothesis='hypothesis',supporting_record_ids=['invented'],limitations=['limited'])))
                kwargs['stdout'].write(json.dumps({'type':'turn.completed','usage':{}})+'\n')
                kwargs['stdout'].flush()
        return Process()
    monkeypatch.setattr('materials.separator_model.subprocess.Popen',process)
    with pytest.raises(RuntimeError,match='citation_invalid'):
        run_separator_model(assess_separator_recipe(dataset,SeparatorRecipe()),'/fake/codex',tmp_path)


@pytest.mark.parametrize('form,loading,allowed',[
    ('raw_BNNT',.01,True),('raw_BNNT',.1,True),('raw_BNNT',.2,True),
    ('raw_BNNT',.3,True),('raw_BNNT',.31,False),('raw_BNNT',.4,False),
    ('purified_BNNT',.01,True),('purified_BNNT',.3,True),
    ('purified_BNNT',.4,True),('purified_BNNT',.5,True),('purified_BNNT',.51,False),
    ('raw_BNNT',0,False),('purified_BNNT',0,False),
])
def test_loading_domain_respects_published_raw_peak(dataset,form,loading,allowed):
    result=assess_separator_recipe(dataset,SeparatorRecipe(bn_form=form,loading_mg_cm2=loading))
    assert result['numeric_allowed'] is allowed


@pytest.mark.parametrize('mode',['success','above_peak','malformed_event','tool_activity','incomplete'])
def test_configurable_provider_preserves_execution_and_result_boundaries(dataset,tmp_path,monkeypatch,mode):
    def process(cmd,**kwargs):
        assert cmd[cmd.index('-m')+1]=='future-model'
        class Process:
            returncode=0
            def communicate(self,prompt,timeout):
                Path(cmd[cmd.index('-o')+1]).write_text(json.dumps(dict(conductivity_mS_cm=.8 if mode=='above_peak' else .6,
                    preparation_hypothesis='材料分析',supporting_record_ids=['kim_2022:BNNT-PP-0.3'],limitations=['limited'])))
                event={'type':'turn.completed','usage':{}}
                if mode=='tool_activity':
                    event={'type':'item.completed','item':{'type':'command_execution'}}
                if mode=='incomplete':
                    event={'type':'turn.started'}
                kwargs['stdout'].write('invalid\n' if mode=='malformed_event' else json.dumps(event)+'\n')
                kwargs['stdout'].flush()
        return Process()
    monkeypatch.setattr('materials.separator_model.subprocess.Popen',process)
    assessment=assess_separator_recipe(dataset,SeparatorRecipe())
    if mode=='success':
        result=run_separator_model(assessment,'/fake/codex',tmp_path,model_name='future-model')
        assert result['requested_model']=='future-model' and result['result']['conductivity_mS_cm']==.6
    else:
        with pytest.raises(RuntimeError):
            run_separator_model(assessment,'/fake/codex',tmp_path,model_name='future-model')
    with pytest.raises(ValueError,match='identifier'):
        run_separator_model(assessment,'/fake/codex',tmp_path,model_name='bad; shell')
