"""Periodic HTTP and fresh inference checks; receipts are consumed by Supervisor.

This collector never sends messages, restarts services or changes provider
settings. A real public /api/predict request shares the partner's quota/lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlsplit

import httpx

RUNTIME = Path(__file__).resolve().parents[2] / '.runtime/separator'
HTTP_INTERVAL = 300
MODEL_INTERVAL = 21600


def _save(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        stage = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    stage.chmod(0o600)
    stage.replace(path)


def _configuration(runtime):
    config = json.loads((runtime / 'deployment.json').read_text())
    url = config['public_url'].rstrip('/')
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
            or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError('Invalid configured public origin')
    expiry = datetime.fromisoformat(config['expires_at_utc'])
    if expiry.tzinfo is None:
        raise ValueError('Expiry must have timezone')
    port = config.get('port', 8766)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid backend port')
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return url, port, expiry.timestamp(), identity


def monitor_status(runtime_dir=RUNTIME, *, now=None):
    """Return result-v1 status from bounded, deployment-bound fresh receipts."""
    now = time.time() if now is None else now
    try:
        runtime = Path(runtime_dir)
        _, _, expiry, identity = _configuration(runtime)
        path = runtime / 'monitor.json'
        if path.stat().st_size > 16384:
            raise ValueError('Oversized receipt')
        receipt = json.loads(path.read_text())
        age = now - receipt['checked_at']
        if receipt['deployment_sha256'] != identity or not 0 <= age <= HTTP_INTERVAL * 3:
            raise ValueError('Stale receipt')
        if now >= expiry and all(receipt[k]['status'] == 'expired' for k in ('backend', 'public')):
            return dict(status='healthy', code='DEMO_SCHEDULED_EXPIRY', evidence='transport')
        for component in ('backend', 'public'):
            if receipt[component]['status'] != 'ok':
                return dict(status='unavailable', code=f'{component.upper()}_UNAVAILABLE', evidence='transport')
        model = receipt['model']
        age = now - model['attempted_at']
        if not 0 <= age <= MODEL_INTERVAL + HTTP_INTERVAL * 3:
            raise ValueError('Stale model evidence')
        if model['status'] == 'deferred':
            return dict(status='unknown', code='MODEL_CHECK_DEFERRED', evidence='transport')
        if model['status'] == 'started' and age <= 180:
            return dict(status='unknown', code='MODEL_CHECK_RUNNING', evidence='transport')
        if model['status'] != 'ok':
            return dict(status='unavailable', code='MODEL_RESPONSE_FAILED', evidence='transport')
        if model.get('fresh') is not True or not 0 <= now - model['completed_at'] <= MODEL_INTERVAL + HTTP_INTERVAL * 3:
            raise ValueError('No fresh model completion')
        return dict(status='healthy', code='MODEL_RESPONSE_VERIFIED', evidence='transport')
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return dict(status='unknown', code='MONITOR_EVIDENCE_STALE', evidence='none')


def check_separator_service(runtime_dir=RUNTIME, *, client=None, now=None):
    """Check both origins every run; call uncached public inference every six hours.

    Persist an attempt before sending, so interruption cannot cause a tight retry
    loop. Busy/quota replies defer to the next HTTP interval without consuming a
    provider call. Only bounded numeric inputs are sent. Cache hits never pass.
    """
    runtime = Path(runtime_dir)
    now = time.time() if now is None else now
    supplied_now = now
    runtime.mkdir(parents=True, exist_ok=True)
    lock_path = runtime / 'monitor.lock'
    with lock_path.open('a') as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='unknown', code='MONITOR_BUSY', evidence='none')
        try:
            public, port, expiry, identity = _configuration(runtime)
        except (OSError, ValueError, KeyError, TypeError):
            return dict(status='unknown', code='MONITOR_CONFIG_INVALID', evidence='none')
        receipt_path = runtime / 'monitor.json'
        try:
            previous = json.loads(receipt_path.read_text())
            model = previous['model'] if previous['deployment_sha256'] == identity else {}
            if not isinstance(model, dict):
                model = {}
        except (OSError, ValueError, KeyError, TypeError):
            model = {}
        receipt = dict(schema_version=1, checked_at=now, deployment_sha256=identity, model=model)
        owned_client = client is None
        client = client or httpx.Client(timeout=15, follow_redirects=False)
        try:
            snapshots = {}
            for name, origin in (('backend', f'http://127.0.0.1:{port}'), ('public', public)):
                try:
                    response = client.get(origin + '/health', timeout=15)
                    if now >= expiry and response.status_code == 410:
                        receipt[name] = dict(status='expired')
                        continue
                    response.raise_for_status()
                    value = response.json()
                    if (now >= expiry or not isinstance(value, dict)
                            or value.get('status') != 'ok' or value.get('records', 0) < 1
                            or not isinstance(value.get('dataset_sha256'), str)):
                        raise ValueError('Invalid health response')
                    if name == 'public':
                        page = client.get(origin + '/', timeout=15)
                        page.raise_for_status()
                        if 'AI for Science' not in page.text or 'id="task"' not in page.text:
                            raise ValueError('Invalid frontend response')
                    snapshots[name] = value['dataset_sha256']
                    receipt[name] = dict(status='ok')
                except (httpx.HTTPError, ValueError, TypeError):
                    receipt[name] = dict(status='failed')
            if len(snapshots) == 2 and snapshots['backend'] != snapshots['public']:
                receipt['public'] = dict(status='failed')
            interval = HTTP_INTERVAL if model.get('status') == 'deferred' else MODEL_INTERVAL
            attempted = model.get('attempted_at', 0)
            due = not isinstance(attempted, (int, float)) or not 0 <= now - attempted < interval
            _save(receipt_path, receipt)
            if now < expiry and due and all(receipt[k]['status'] == 'ok' for k in ('backend', 'public')):
                receipt['model'] = dict(status='started', attempted_at=now, fresh=False)
                _save(receipt_path, receipt)
                started = time.monotonic()
                try:
                    # Six decimal places avoid an ordinary user/cache collision.
                    loading = .05 + secrets.randbelow(200001) / 1_000_000
                    response = client.post(public + '/api/predict', timeout=150,
                        json=dict(task='bnnt_conductivity', bn_form='raw_BNNT', loading_mg_cm2=loading))
                    if response.status_code == 429:
                        receipt['model']['status'] = 'deferred'
                    else:
                        response.raise_for_status()
                        result = response.json()
                        if not isinstance(result, dict) or not isinstance(result.get('prediction'), dict):
                            raise ValueError('Invalid inference response')
                        value = result.get('prediction', {}).get('value')
                        if (result.get('status') != 'ok' or result.get('cached') is not False
                                or result.get('model_called') is not True
                                or type(value) not in (int, float) or not math.isfinite(value)
                                or not 0 <= value <= 10 or result['prediction'].get('unit') != 'mS/cm'
                                or not isinstance(result.get('explanation'), str) or not result['explanation'].strip()):
                            raise ValueError('No fresh valid numerical completion')
                        receipt['model'].update(status='ok', fresh=True,
                            completed_at=supplied_now + time.monotonic() - started)
                except (httpx.HTTPError, ValueError, TypeError):
                    receipt['model']['status'] = 'failed'
                receipt['model']['elapsed_seconds'] = round(time.monotonic() - started, 3)
                _save(receipt_path, receipt)
        finally:
            if owned_client:
                client.close()
        return monitor_status(runtime, now=max(now, receipt['model'].get('completed_at', now)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true', help='Read receipt only; never call model')
    parser.add_argument('--runtime-dir', type=Path, default=RUNTIME)
    args = parser.parse_args()
    result = monitor_status(args.runtime_dir) if args.status else check_separator_service(args.runtime_dir)
    print(json.dumps(dict(schema_version=1, **result)))
