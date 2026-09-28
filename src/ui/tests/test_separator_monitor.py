"""Real-response monitoring must not turn process/cache evidence into model proof."""
import json
from datetime import datetime, timezone

import httpx
import pytest

from ui.separator_monitor import check_separator_service, monitor_status, MODEL_INTERVAL


@pytest.fixture
def monitored(tmp_path):
    now = 1_790_000_000.
    config = dict(public_url='https://demo.example.test', port=8766,
        expires_at_utc=datetime.fromtimestamp(now+86400, timezone.utc).isoformat())
    (tmp_path/'deployment.json').write_text(json.dumps(config))
    return tmp_path, now


def transport(*, prediction=None, status=200, broken=None, expired=False):
    calls = []
    def handle(request):
        calls.append(request)
        if expired:
            return httpx.Response(410)
        if broken == request.url.host:
            raise httpx.ConnectError('private diagnostic', request=request)
        if request.url.path == '/health':
            return httpx.Response(200, json=dict(status='ok', records=20, dataset_sha256='pinned'))
        if request.url.path == '/':
            return httpx.Response(200, text='AI for Science <select id="task">')
        assert request.url.path == '/api/predict'
        result = dict(status='ok', cached=False, model_called=True,
                      prediction=dict(value=.61, unit='mS/cm'), explanation='配方分析')
        if prediction:
            result.update(prediction)
        return httpx.Response(status, json=result)
    return httpx.Client(transport=httpx.MockTransport(handle)), calls


def test_fresh_public_inference_then_receipt_only_and_six_hour_spacing(monitored):
    root, now = monitored
    client, calls = transport()
    assert check_separator_service(root, client=client, now=now)['code'] == 'MODEL_RESPONSE_VERIFIED'
    posted = [r for r in calls if r.method == 'POST']
    assert len(posted) == 1 and posted[0].url.host == 'demo.example.test'
    assert .05 <= json.loads(posted[0].content)['loading_mg_cm2'] <= .25
    assert monitor_status(root, now=now+5)['status'] == 'healthy'
    assert check_separator_service(root, client=client, now=now+300)['status'] == 'healthy'
    assert len([r for r in calls if r.method == 'POST']) == 1
    assert check_separator_service(root, client=client, now=now+MODEL_INTERVAL)['status'] == 'healthy'
    assert len([r for r in calls if r.method == 'POST']) == 2
    assert (root/'monitor.json').stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('mutation', [
    {'cached':True}, {'model_called':False}, {'prediction':{'value':None,'unit':'mS/cm'}},
    {'prediction':{'value':True,'unit':'mS/cm'}}, {'explanation':''},
    {'prediction':{'value':.6,'unit':'wrong'}}, {'status':'needs_data'},
])
def test_cached_abstained_or_malformed_results_cannot_prove_model_health(monitored, mutation):
    root, now = monitored
    client, _ = transport(prediction=mutation)
    assert check_separator_service(root, client=client, now=now)['code'] == 'MODEL_RESPONSE_FAILED'


@pytest.mark.parametrize('status,code', [(503,'MODEL_RESPONSE_FAILED'), (429,'MODEL_CHECK_DEFERRED')])
def test_provider_failure_or_busy_is_not_a_healthy_response(monitored, status, code):
    root, now = monitored
    client, calls = transport(status=status)
    assert check_separator_service(root, client=client, now=now)['code'] == code
    check_separator_service(root, client=client, now=now+300)
    assert len([r for r in calls if r.method == 'POST']) == (2 if status == 429 else 1)


@pytest.mark.parametrize('host,code', [('127.0.0.1','BACKEND_UNAVAILABLE'),
                                      ('demo.example.test','PUBLIC_UNAVAILABLE')])
def test_http_failure_skips_inference_and_does_not_leak_diagnostics(monitored, host, code):
    root, now = monitored
    client, calls = transport(broken=host)
    result = check_separator_service(root, client=client, now=now)
    assert result['code'] == code and 'private' not in json.dumps(result)
    assert not any(r.method == 'POST' for r in calls)


def test_stale_clock_future_and_configuration_change_invalidate_receipt(monitored):
    root, now = monitored
    client, _ = transport()
    check_separator_service(root, client=client, now=now)
    for when in (now-1, now+901):
        assert monitor_status(root, now=when)['status'] == 'unknown'
    config=json.loads((root/'deployment.json').read_text());config['model_name']='changed'
    (root/'deployment.json').write_text(json.dumps(config))
    assert monitor_status(root, now=now+1)['status'] == 'unknown'


def test_expiry_requires_actual_gone_responses_and_never_calls_model(monitored):
    root, now = monitored
    now += 86401
    assert monitor_status(root, now=now)['status'] == 'unknown'
    client, calls = transport(expired=True)
    assert check_separator_service(root, client=client, now=now)['code'] == 'DEMO_SCHEDULED_EXPIRY'
    assert not any(r.method == 'POST' for r in calls)
    client, _ = transport()
    assert check_separator_service(root, client=client, now=now+300)['status'] == 'unavailable'


def test_interrupted_model_attempt_is_not_retried_every_five_minutes(monitored):
    root, now = monitored
    client, _ = transport()
    check_separator_service(root, client=client, now=now)
    path=root/'monitor.json';receipt=json.loads(path.read_text())
    receipt['model']={'status':'started','attempted_at':now,'fresh':False}
    path.write_text(json.dumps(receipt))
    client, calls=transport()
    assert check_separator_service(root, client=client, now=now+300)['code']=='MODEL_RESPONSE_FAILED'
    assert not any(r.method=='POST' for r in calls)
