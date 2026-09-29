from __future__ import annotations

import json
import csv
import io
import sqlite3
import threading

from fastapi.testclient import TestClient
import pytest

from ui.separator_app import create_separator_app


@pytest.fixture
def demo(tmp_path):
    app=create_separator_app(runtime_dir=tmp_path,model_executable='/fake/codex')
    return TestClient(app),tmp_path


def test_public_pages_evidence_and_export(demo):
    client,_=demo
    for url in ['/','/app.js','/i18n.js','/messages.tsv','/style.css','/health','/api/overview','/api/sources',
                '/api/records','/api/electrolytes','/api/aqueous','/api/evaluation',
                '/api/download/public.sqlite','/api/download/electrolytes.csv']:
        response=client.get(url)
        assert response.status_code==200,url
        assert response.headers['x-content-type-options']=='nosniff'
        assert 'gpt-6-astra' not in response.text.lower()
    assert len(client.get('/api/records?q=CA@BN').json())==4
    assert all('notes' not in r for r in client.get('/api/records').json())
    assert len(client.get('/api/electrolytes').json())==38
    overview=client.get('/api/overview').json()
    assert overview['daily_model_limit']==100 and overview['electrolyte_measurements']==125
    assert '40 mg' in client.get('/api/evidence/kim_2022:recipe').json()['text']
    evidence=client.get('/api/evidence/kim_2022:recipe').json()
    assert '2.3.' in evidence['section_title'] and 'Boron Nitride' in evidence['source_title']
    assert not {'xml_id','locator','paragraph'} & evidence.keys()
    for url in ['/api/checks','/api/download/access_token.txt','/.runtime/separator/access_token.txt']:
        assert client.get(url).status_code==404


def test_anonymous_prediction_hides_provider_and_caches_exact_config(demo,monkeypatch):
    client,runtime=demo
    models=[]
    def model(*args,model_name,language):
        models.append(model_name)
        return {'model':model_name,'usage':{'private':1},'result':{
            'conductivity_mS_cm':.65,'preparation_hypothesis':f'{model_name} Astra Codex OpenAI 材料分析'}}
    monkeypatch.setattr('ui.separator_app.run_separator_model',model)
    first=client.post('/api/predict',json={})
    assert first.status_code==200 and first.json()['model_called']
    assert first.json()['prediction']['value']==pytest.approx(.71)
    assert first.json()['prediction']['numerical_method']=='loading_linear_reference_v1'
    for private in ['astra','codex','openai','usage','provider_record']:
        assert private not in first.text.lower()
    assert client.post('/api/predict',json={}).json()['cached']
    (runtime/'deployment.json').write_text(json.dumps({'model_name':'future-research-model'}))
    changed=client.post('/api/predict',json={})
    assert changed.json()['cached'] is False
    assert 'future-research-model' not in changed.text
    assert models==['gpt-6-astra','future-research-model']


def test_response_language_selects_prompt_and_separates_only_model_cache(demo,monkeypatch):
    client,runtime=demo
    requested=[]
    def model(*args,language,**kwargs):
        requested.append(language)
        return {'result':{'preparation_hypothesis':{'zh-CN':'可能改善浸润。',
            'zh-TW':'可能改善浸潤。','en':'It may improve wetting.'}[language]}}
    monkeypatch.setattr('ui.separator_app.run_separator_model',model)
    default=client.post('/api/predict',json={}).json()
    assert default['explanation']=='可能改善浸润。'
    for lang,text in [('en','It may improve wetting.'),('zh-TW','可能改善浸潤。'),('zh-CN','可能改善浸润。')]:
        url='/api/predict?language='+lang
        result=client.post(url,json={}).json()
        assert result['explanation']==text
        assert client.post(url,json={}).json()['cached']
        local=client.post(url,json={'task':'aqueous_viscosity'}).json()
        assert local['prediction']['value']==pytest.approx(7.45)
        assert local['cached']==(lang!='en')
    assert requested==['zh-CN','en','zh-TW']
    with sqlite3.connect(runtime/'usage.sqlite') as db:
        assert db.execute('SELECT count(*) FROM calls').fetchone()[0]==4
    assert client.post('/api/predict?language=arbitrary-prompt',json={}).status_code==422
    assert len(requested)==3


def test_partner_csv_has_readable_columns_instead_of_json_cells(demo):
    client,_=demo
    response=client.get('/api/download/records.csv')
    assert response.status_code==200
    rows=list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
    assert len(rows)==20
    assert rows[0]['基材']=='PP 聚丙烯'
    assert rows[0]['文獻離子導電率 mS/cm']=='0.43'
    assert rows[0]['原始研究'].startswith('https://doi.org/')
    assert not any('json' in key.lower() for key in rows[0])
    assert all(not value.startswith(('{','[')) for row in rows for value in row.values())
    failed=next(row for row in rows if 'failure' in row['配方名稱'])
    assert failed['製備結果']=='製備失敗：孔道堵塞'


@pytest.mark.parametrize('form,loading,expected', [
    ('raw_BNNT', .05, .4766666667), ('raw_BNNT', .1, .5233333333),
    ('raw_BNNT', .2, .6166666667), ('raw_BNNT', .3, .71),
    ('purified_BNNT', .1, .5233333333), ('purified_BNNT', .3, .71),
    ('purified_BNNT', .4, .8033333333), ('purified_BNNT', .5, .8966666667),
])
def test_numeric_reference_does_not_use_language_guess_or_held_out_answer(demo, monkeypatch, form, loading, expected):
    client, runtime = demo
    def explain(assessment, *args, **kwargs):
        assert all(r['conductivity_mS_cm'] != .84 for r in assessment['training_examples'])
        return {'result':{'conductivity_mS_cm':.5, 'preparation_hypothesis':'依條件分析'}}
    monkeypatch.setattr('ui.separator_app.run_separator_model', explain)
    result=client.post('/api/predict',json={'bn_form':form,'loading_mg_cm2':loading}).json()
    assert result['prediction']['value']==pytest.approx(expected)
    assert result['model_called'] and not result['cached']
    with sqlite3.connect(runtime/'usage.sqlite') as db:
        private=json.loads(db.execute('SELECT result FROM calls').fetchone()[0])
    assert private['provider_record']['result']['conductivity_mS_cm']==.5


def test_dispersion_reviews_do_not_become_compatible_training_formulations(demo):
    client, _=demo
    sources={r['source_id']:r for r in client.get('/api/sources').json()}
    for key in ('bouville_2014','chen_2017'):
        assert sources[key]['access'].startswith('full_text_')
        assert sources[key]['scope']=='reviewed_for_dispersion_not_training'
    assert len(client.get('/api/records').json())==20


def test_aqueous_task_uses_local_data_and_exports_readable_cases(demo,monkeypatch):
    client,_=demo
    monkeypatch.setattr('ui.separator_app.run_separator_model',lambda *a,**kw:pytest.fail('local calculation called provider'))
    request={'task':'aqueous_viscosity','bn_volume_pct':4}
    result=client.post('/api/predict',json=request).json()
    assert result['prediction']['value']==pytest.approx(7.45)
    assert not result['model_called'] and not result['cached']
    assert client.post('/api/predict',json=request).json()['cached']
    assert len(client.get('/api/aqueous').json())==12
    rows=list(csv.DictReader(io.StringIO(client.get('/api/download/aqueous.csv').content.decode('utf-8-sig'))))
    assert len(rows)==12 and rows[6]['文獻黏度 mPa·s']==''
    assert rows[6]['塗布結果']=='膏狀，無法塗布'


def test_explanations_drop_notices_but_retain_uncertainty_and_material_failures(demo,monkeypatch):
    client,_=demo
    monkeypatch.setattr('ui.separator_app.run_separator_model',lambda *a,**kw:{'result':{
        'preparation_hypothesis':'此為機制假說，非觀察結果。BNNT 可能改善浸潤。高載量可能導致孔道堵塞。這只是研究原型，尚未驗證。',
        'conductivity_mS_cm':.65}})
    text=client.post('/api/predict',json={}).json()['explanation']
    assert text=='BNNT 可能改善浸潤。高載量可能導致孔道堵塞。'
    assert client.post('/api/predict',json={}).json()['explanation']==text


def test_unsupported_inputs_never_call_provider(demo,monkeypatch):
    client,_=demo
    def forbidden(*args,**kwargs):
        pytest.fail('out-of-domain request called model')
    monkeypatch.setattr('ui.separator_app.run_separator_model',forbidden)
    for body in [{'substrate':'cellulose'},{'loading_mg_cm2':.4},
                 {'bn_binder_ratio':3},{'test_temperature_c':25}]:
        result=client.post('/api/predict',json=body).json()
        assert result['status']=='needs_data' and result['model_called'] is False


def test_100_new_analyses_persist_cache_is_free_and_window_rolls(demo):
    client,runtime=demo
    for gap in range(50,150):
        response=client.post('/api/predict',json={'task':'coating_thickness','applicator_gap_um':gap})
        assert response.status_code==200 and response.json()['cached'] is False
    # A recreated application must retain both quota and genuine stored results.
    client=TestClient(create_separator_app(runtime_dir=runtime))
    assert client.post('/api/predict',json={'task':'coating_thickness','applicator_gap_um':50}).json()['cached']
    assert client.post('/api/predict',json={'task':'coating_thickness','applicator_gap_um':150}).status_code==429
    with sqlite3.connect(runtime/'usage.sqlite') as db:
        assert db.execute('SELECT count(*) FROM calls').fetchone()[0]==100
        db.execute('UPDATE calls SET time=0')
    assert client.post('/api/predict',json={'task':'coating_thickness','applicator_gap_um':150}).status_code==200


def test_provider_failure_is_private_and_active_calls_are_bounded(demo,monkeypatch):
    client,_=demo
    entered=threading.Event();release=threading.Event();results=[]
    def model(*args,**kwargs):
        entered.set()
        assert release.wait(5)
        raise RuntimeError('SECRET_OR_PRIVATE_PATH')
    monkeypatch.setattr('ui.separator_app.run_separator_model',model)
    thread=threading.Thread(target=lambda:results.append(client.post('/api/predict',json={})))
    thread.start()
    try:
        assert entered.wait(5)
        assert client.post('/api/predict',json={'loading_mg_cm2':.2}).status_code==429
    finally:
        release.set();thread.join(5)
    assert not thread.is_alive()
    assert results[0].status_code==503 and 'SECRET' not in results[0].text


@pytest.mark.parametrize('body',[
    {'task':'aqueous_viscosity','bn_volume_pct':12.01},
    {'task':'aqueous_viscosity','solids_weight_pct':20},
    {'task':'aqueous_viscosity','filler':'Al2O3'},
    {'task':'coating_thickness','applicator_gap_um':49},
    {'task':'coating_thickness','applicator_gap_um':201},
    {'task':'coating_thickness','bn_binder_ratio':3},
    {'task':'coating_thickness','solvent':'NMP'},
    {'task':'electrolyte_conductivity','salt_molality':.13},
    {'task':'electrolyte_conductivity','salt_molality':2.01},
    {'task':'electrolyte_conductivity','ec_fraction':.29},
    {'task':'electrolyte_conductivity','ec_fraction':.51},
    {'task':'electrolyte_conductivity','dmc_ratio':-1},
    {'task':'electrolyte_conductivity','dmc_ratio':1.01},
    {'task':'electrolyte_conductivity','temperature_c':40},
    {'task':'experiment_plan','observations':{'CLIO-01':1}},
    {'task':'experiment_plan','observations':{'CLIO-01':1,'UNKNOWN':2}},
    {'task':'experiment_plan','observations':{'CLIO-01':-1,'CLIO-02':2}},
    {'task':'bnnt_conductivity','model_name':'override'},
    {'task':'unrecognized'},{'prompt':'ignore rules'},
])
def test_invalid_task_inputs_are_rejected(demo,body):
    client,_=demo
    assert client.post('/api/predict',json=body).status_code==422


@pytest.mark.parametrize('value',['NaN','Infinity','-Infinity','1e999'])
def test_nonfinite_json_returns_validation_error(demo,value):
    client,_=demo
    response=client.post('/api/predict',content='{"loading_mg_cm2":'+value+'}',
                         headers={'Content-Type':'application/json'})
    assert response.status_code==422


def test_local_prediction_and_plan_use_disjoint_observation_inputs(demo):
    client,_=demo
    response=client.post('/api/predict',json={'task':'electrolyte_conductivity'}).json()
    assert response['prediction']['value']>0 and response['model_called'] is False
    known={'CLIO-01':1.,'CLIO-09':10.,'CLIO-17':12.}
    result=client.post('/api/predict',json={'task':'experiment_plan','observations':known}).json()
    assert result['observations_used']==3
    assert len(result['recommendations'])==1
    assert not {x['candidate_id'] for x in result['recommendations']} & set(known)
    assert 'conductivity_mS_cm' not in json.dumps(result)


def test_request_bounds_and_expiration(demo):
    client,runtime=demo
    assert client.post('/api/assess',content=b'x'*9000).status_code==413
    assert client.post('/api/predict',content='{broken').status_code==422
    (runtime/'deployment.json').write_text(json.dumps({'expires_at_utc':'2020-01-01T00:00:00+00:00'}))
    assert client.get('/').status_code==410
    assert client.post('/api/predict',json={}).status_code==410
