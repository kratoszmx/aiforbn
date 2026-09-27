"""Public separator demonstration with bounded, invitation-only Astra inference."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

SRC_DIR=Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0,str(SRC_DIR))

from materials.separator_data import (
    DATASET_PATH, SeparatorRecipe, assess_separator_recipe, build_separator_database,
    load_separator_dataset, search_separator_records, separator_known_checks,
)
from materials.separator_model import MODEL, run_separator_model

PROJECT_ROOT=SRC_DIR.parent
WEB_DIR=Path(__file__).with_name('separator_web')


def create_separator_app(data_path=DATASET_PATH, runtime_dir=None, model_executable=None):
    """Construct an evidence-only public API plus a guarded model endpoint."""
    runtime_dir=Path(runtime_dir or PROJECT_ROOT/'.runtime/separator')
    runtime_dir.mkdir(parents=True,exist_ok=True)
    data=load_separator_dataset(data_path)
    db_path=build_separator_database(data,runtime_dir/'public.sqlite')
    dataset_digest=hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    app=FastAPI(title='BN 隔膜研究原型',docs_url=None,redoc_url=None,openapi_url=None)
    app.state.dataset=data
    app.state.runtime_dir=runtime_dir
    inference_lock=threading.Lock()
    with sqlite3.connect(runtime_dir/'usage.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, time REAL NOT NULL, request_hash TEXT NOT NULL, status TEXT NOT NULL, result TEXT)')

    def settings():
        path=runtime_dir/'deployment.json'
        return json.loads(path.read_text()) if path.is_file() else {}

    @app.middleware('http')
    async def boundaries(request,call_next):
        expiry=settings().get('expires_at_utc')
        if expiry and datetime.now(timezone.utc)>=datetime.fromisoformat(expiry):
            return JSONResponse({'error':'demonstration_expired'},status_code=410)
        if request.method=='POST':
            size=0; chunks=[]
            async for chunk in request.stream():
                size+=len(chunk)
                if size>8192:
                    return JSONResponse({'error':'request_too_large'},status_code=413)
                chunks.append(chunk)
            request._body=b''.join(chunks)
        response=await call_next(request)
        response.headers.update({
            'X-Content-Type-Options':'nosniff', 'Referrer-Policy':'no-referrer',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            'Cache-Control':'no-store',
        })
        return response

    @app.get('/')
    def index():
        return FileResponse(WEB_DIR/'index.html',media_type='text/html')

    @app.get('/app.js')
    def javascript():
        return FileResponse(WEB_DIR/'app.js',media_type='application/javascript')

    @app.get('/style.css')
    def stylesheet():
        return FileResponse(WEB_DIR/'style.css',media_type='text/css')

    @app.get('/health')
    def health():
        return dict(status='ok',dataset_version=data['dataset_version'],dataset_sha256=dataset_digest,
                    records=len(data['records']),model=MODEL,
                    model_ready=bool((runtime_dir/'model_verified.json').is_file()),expires_at_utc=settings().get('expires_at_utc'))

    @app.get('/api/overview')
    def overview():
        return dict(dataset_version=data['dataset_version'],dataset_sha256=dataset_digest,
                    source_count=len(data['sources']),study_count=sum(s['scope']=='primary_study_extracted' for s in data['sources']),
                    record_count=len(data['records']),observation_count=sum(len(r['observations']) for r in data['records']),
                    model=MODEL,expires_at_utc=settings().get('expires_at_utc'),
                    numeric_status='exploratory_only',test_records=1,training_records=2,
                    limits='20 個配方分屬不同材料體系，不能當作 20 個可合併的訓練樣本。PP 導電率任務只有一篇研究的 3 個數值。',
                    privacy='只輸入公開或已獲准提交的配方參數。模型請求送至 OpenAI；配方與回覆會保存在此原型的本機紀錄。')

    @app.get('/api/sources')
    def sources():
        # Internal archival paths are not useful to the partner.
        return [{k:v for k,v in s.items() if k not in ['source_file']} for s in data['sources']]

    @app.get('/api/records')
    def records(substrate: str='', q: str=''):
        try:
            return search_separator_records(data,substrate or None,q)
        except ValueError:
            raise HTTPException(422,'invalid_search') from None

    @app.get('/api/evidence/{evidence_id:path}')
    def evidence(evidence_id: str):
        if evidence_id not in data['evidence']:
            raise HTTPException(404,'evidence_not_found')
        result=dict(data['evidence'][evidence_id])
        result['source']=next(s['url'] for s in data['sources'] if s['source_id']==result['source_id'])
        return result

    @app.get('/api/checks')
    def checks():
        return dict(checks=separator_known_checks(data),quarantined=data['quarantined'])

    @app.get('/api/evaluation')
    def evaluation():
        path=PROJECT_ROOT/'docs/research/separator_prototype/evaluation.json'
        return json.loads(path.read_text()) if path.is_file() else {'status':'not_run'}

    @app.get('/api/download/{name}')
    def download(name: str):
        allowed={'sources.csv':Path(data_path).with_name('source_inventory.csv'),
                 'records.csv':Path(data_path).with_name('pilot_records.csv'),
                 'public.sqlite':db_path}
        if name not in allowed:
            raise HTTPException(404,'download_not_found')
        return FileResponse(allowed[name],filename=name)

    @app.post('/api/assess')
    def assess(recipe: SeparatorRecipe):
        return assess_separator_recipe(data,recipe)

    @app.post('/api/predict')
    async def predict(recipe: SeparatorRecipe,request: Request):
        token_file=runtime_dir/'access_token.txt'
        expected=token_file.read_text().strip() if token_file.is_file() else ''
        provided=request.headers.get('X-Demo-Token','')
        if not expected or not hmac.compare_digest(provided.encode(),expected.encode()):
            raise HTTPException(401,'需要合作方試用碼；公開資料庫不需要試用碼。')
        assessment=assess_separator_recipe(data,recipe)
        if not assessment['numeric_allowed']:
            return dict(status='unsupported',assessment=assessment,model_called=False)
        key=hashlib.sha256((dataset_digest+json.dumps(recipe.model_dump(),sort_keys=True)+MODEL).encode()).hexdigest()
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429,'模型正在處理另一個配方，請稍後再試。')
        call_id=None
        try:
            with sqlite3.connect(runtime_dir/'usage.sqlite') as db:
                hit=db.execute('SELECT result FROM calls WHERE request_hash=? AND status=? ORDER BY id DESC LIMIT 1',(key,'ok')).fetchone()
                if hit:
                    return dict(status='ok',cached=True,model_called=False,assessment=assessment,model_response=json.loads(hit[0]))
                daily_limit=int(settings().get('daily_model_limit',20))
                count=db.execute('SELECT count(*) FROM calls WHERE time>?',(time.time()-86400,)).fetchone()[0]
                if count>=daily_limit:
                    raise HTTPException(429,'已達 24 小時試用上限；仍可查閱資料、示範及評測。')
                cursor=db.execute('INSERT INTO calls (time,request_hash,status) VALUES (?,?,?)',(time.time(),key,'started'))
                call_id=cursor.lastrowid
            executable=model_executable or shutil.which('codex')
            if not executable:
                raise RuntimeError('model_executable_missing')
            response=await asyncio.to_thread(run_separator_model,assessment,executable,runtime_dir/'model')
            with sqlite3.connect(runtime_dir/'usage.sqlite') as db:
                db.execute('UPDATE calls SET status=?,result=? WHERE id=?',('ok',json.dumps(response,ensure_ascii=False),call_id))
            return dict(status='ok',cached=False,model_called=True,assessment=assessment,model_response=response)
        except RuntimeError:
            if call_id is not None:
                with sqlite3.connect(runtime_dir/'usage.sqlite') as db:
                    db.execute('UPDATE calls SET status=? WHERE id=?',('failed',call_id))
            raise HTTPException(503,'模型本次未完成，沒有產生預測；請稍後重試。') from None
        finally:
            inference_lock.release()

    return app


if __name__=='__main__':
    import uvicorn
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    uvicorn.run(create_separator_app(),host='127.0.0.1',port=args.port,access_log=False)
