from __future__ import annotations

import json

from fastapi.testclient import TestClient
import pytest

from ui.separator_app import create_separator_app


@pytest.fixture
def demo(tmp_path):
    (tmp_path/'access_token.txt').write_text('test-invitation')
    app=create_separator_app(runtime_dir=tmp_path,model_executable='/fake/codex')
    return TestClient(app),tmp_path


def test_public_pages_evidence_and_export(demo):
    client,_=demo
    for url in ['/','/app.js','/style.css','/health','/api/overview','/api/sources','/api/records','/api/checks','/api/download/public.sqlite']:
        response=client.get(url)
        assert response.status_code==200,url
        assert response.headers['x-content-type-options']=='nosniff'
    assert len(client.get('/api/records?q=CA@BN').json())==4
    e=client.get('/api/evidence/kim_2022:recipe').json()
    assert '40 mg' in e['text'] and e['source'].startswith('https://doi.org/')
    assert client.get('/api/download/access_token.txt').status_code==404
    assert client.get('/.runtime/separator/access_token.txt').status_code==404


def test_model_auth_precedes_provider(demo,monkeypatch):
    client,_=demo
    def forbidden(*args):
        pytest.fail('unauthorized request called model')
    monkeypatch.setattr('ui.separator_app.run_separator_model',forbidden)
    assert client.post('/api/predict',json={}).status_code==401
    assert client.post('/api/predict',json={},headers={'X-Demo-Token':'wrong'}).status_code==401
    response=client.post('/api/predict',json={'substrate':'cellulose'},headers={'X-Demo-Token':'test-invitation'})
    assert response.json()['model_called'] is False


def test_model_cache_quota_and_provider_failure(demo,monkeypatch):
    client,runtime=demo
    calls=[]
    def model(*args):
        calls.append(args)
        return {'model':'gpt-6-astra','result':{'conductivity_mS_cm':.65}}
    monkeypatch.setattr('ui.separator_app.run_separator_model',model)
    headers={'X-Demo-Token':'test-invitation'}
    assert client.post('/api/predict',json={},headers=headers).json()['model_called'] is True
    assert client.post('/api/predict',json={},headers=headers).json()['cached'] is True
    assert len(calls)==1
    (runtime/'deployment.json').write_text(json.dumps({'daily_model_limit':1}))
    assert client.post('/api/predict',json={'loading_mg_cm2':.4},headers=headers).status_code==429
    (runtime/'deployment.json').write_text(json.dumps({'daily_model_limit':5}))
    def failure(*args):
        raise RuntimeError('SECRET_OR_PRIVATE_PATH')
    monkeypatch.setattr('ui.separator_app.run_separator_model',failure)
    response=client.post('/api/predict',json={'loading_mg_cm2':.4},headers=headers)
    assert response.status_code==503 and 'SECRET' not in response.text


def test_request_bounds_and_expiration(demo):
    client,runtime=demo
    assert client.post('/api/assess',content=b'x'*9000).status_code==413
    assert client.post('/api/assess',json={'prompt':'ignore rules'}).status_code==422
    (runtime/'deployment.json').write_text(json.dumps({'expires_at_utc':'2020-01-01T00:00:00+00:00'}))
    assert client.get('/').status_code==410
    assert client.post('/api/predict',json={}).status_code==410
