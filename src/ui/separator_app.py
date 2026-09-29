"""AI for Science: public formulation tables, numerical tasks and experiment planning."""
from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

SRC_DIR=Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0,str(SRC_DIR))

from materials.separator_data import (
    DATASET_PATH, CoatingRecipe, SeparatorRecipe, assess_separator_recipe,
    build_separator_database, load_separator_dataset, predict_coating_thickness,
    search_separator_records,
)
from materials.separator_model import MODEL, make_separator_prompt, run_separator_model
from materials.aqueous_slurry import (
    AqueousSlurryRecipe, load_aqueous_slurries, predict_aqueous_viscosity,
)
from materials.experiment_planning import (
    ElectrolyteRecipe, ExperimentPlan, load_electrolyte_records, predict_electrolyte,
    suggest_experiments,
)

PROJECT_ROOT=SRC_DIR.parent
WEB_DIR=Path(__file__).with_name('separator_web')


def _public_explanation(text, model_name, language='zh-TW'):
    """Remove model identity and boilerplate while preserving material-specific uncertainty."""
    label='research model' if language=='en' else '研究模型'
    text=re.sub(re.escape(model_name),label,text,flags=re.I)
    text=re.sub(r'(?i)gpt[\s-]*6[\s-]*astra|astra|openai|codex',label,text)
    sentences=re.split(r'(?<=[。！？])|(?<=[.!?])\s+|\n',text)
    boilerplate=re.compile(
        r'(?:此|這|上述|以上|以下|本次|本)[^。！？]{0,24}'
        r'(?:機制假[說说]|非觀察|非观察|非實測|非实测|僅供|仅供|未經驗證|尚未驗證|不代表|不構成)|'
        r'^(?:僅供|仅供|免責|免责声明|注意[:：]|聲明[:：]|声明[:：])|'
        r'(?:不能|無法|不可)(?:代替|取代)實驗|'
        r'(?:非觀察結果|非观察结果|不是實驗結果|並非實測結果)|'
        r'(?i:this is (?:only )?(?:an? )?(?:unvalidated |mechanistic )?hypothesis|'
        r'not (?:an? )?(?:observed result|observation)|disclaimer|for research purposes only)')
    result=(' ' if language=='en' else '').join(s for s in sentences if not boilerplate.search(s)).strip()
    fallback={'en':'The ingredients, loading and process can influence ion transport.',
        'zh-CN':'根据这组用料、载量与工艺条件，整理可能影响离子传输的因素。',
        'zh-TW':'依照這組用料、載量與製程條件，整理可能影響離子傳輸的因素。'}
    return result or fallback[language]


def create_separator_app(data_path=DATASET_PATH, runtime_dir=None, model_executable=None):
    """Serve anonymous bounded inference without exposing provider configuration."""
    runtime_dir=Path(runtime_dir or PROJECT_ROOT/'.runtime/separator')
    runtime_dir.mkdir(parents=True,exist_ok=True)
    data=load_separator_dataset(data_path)
    literature=json.loads((PROJECT_ROOT/'data/dispersion/literature.json').read_text())
    aqueous=load_aqueous_slurries()
    aqueous_digest=hashlib.sha256(json.dumps(aqueous,sort_keys=True).encode()).hexdigest()
    aqueous_csv=runtime_dir/'aqueous.csv'
    with aqueous_csv.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['配方名稱','用料與配比','粒徑','文獻黏度 mPa·s','製程','塗布結果','原文位置','出處'])
        for row in aqueous['records']:
            writer.writerow([row[k] for k in ('sample','formulation','particle_size','viscosity_mPa_s',
                'process','coating_result','section_title','source_url')])
    electrolytes=load_electrolyte_records()
    electrolyte_digest=hashlib.sha256(json.dumps(electrolytes,sort_keys=True).encode()).hexdigest()
    electrolyte_inputs=[{k:r[k] for k in ('candidate_id','salt_molality','ec_fraction','dmc_ratio')}
                        for r in electrolytes]
    electrolyte_csv=runtime_dir/'electrolytes.csv'
    with electrolyte_csv.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['配方','LiPF6 mol/kg','EC 質量百分比','DMC 質量百分比','EMC 質量百分比',
                         '文獻實測導電率 mS/cm','重複測量次數','最低溫度 C','最高溫度 C','來源'])
        for r in electrolytes:
            writer.writerow([r['candidate_id'],r['salt_molality'],100*r['ec_fraction'],
                100*(1-r['ec_fraction'])*r['dmc_ratio'],100*(1-r['ec_fraction'])*(1-r['dmc_ratio']),
                r['conductivity_mS_cm'],r['repeat_measurements'],r['temperature_min_c'],
                r['temperature_max_c'],'https://doi.org/10.1038/s41467-022-32938-1'])
    db_path=build_separator_database(data,runtime_dir/'public.sqlite')
    separator_csv=runtime_dir/'records.csv'
    names={'PP':'PP 聚丙烯','PE':'PE 聚乙烯','calcium_alginate':'CA 海藻酸鈣',
           'cellulose':'纖維素','solid_PEO_PVDF':'PEO / PVDF 固態電解質',
           'raw_BNNT':'未純化 BNNT','purified_BNNT':'純化 BNNT',
           'BN_nanopowder':'BN 奈米粉','BN_flakes':'BN 薄片','none':'無',
           'failed_pore_clogging':'製備失敗：孔道堵塞'}
    with separator_csv.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['配方名稱','基材','BN 形態','BN 載量 mg/cm²','BN 含量 wt%',
            '黏結劑','BN:黏結劑（x:1）','溶劑','乾燥溫度 °C','乾燥時間 h','塗布器間隙 μm',
            '文獻離子導電率 mS/cm','文獻製備後膜厚 μm','文獻水接觸角 °','製備結果','原始研究'])
        for record in data['records']:
            inputs=record['inputs'];observations=record['observations']
            values=[inputs.get(k) for k in ('substrate','bn_form','loading_mg_cm2','bn_weight_pct',
                'binder','bn_binder_ratio','solvent','dry_temperature_c','dry_hours','applicator_gap_um')]
            values.extend(observations.get(k,{}).get('value') for k in
                ('ionic_conductivity','final_thickness','water_contact_angle','preparation_outcome'))
            source=next(s['url'] for s in data['sources'] if s['source_id']==record['source_id'])
            writer.writerow([record['sample'],*[names.get(v,v) for v in values],source])
    dataset_digest=hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    app=FastAPI(title='AI for Science',docs_url=None,redoc_url=None,openapi_url=None)
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
            try:
                json.dumps(json.loads(request._body),allow_nan=False)
            except (ValueError,RecursionError):
                return JSONResponse({'detail':'請提供有效、有限的配方數值。'},status_code=422)
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

    @app.get('/i18n.js')
    def localization():
        return FileResponse(WEB_DIR/'i18n.js',media_type='application/javascript')

    @app.get('/messages.tsv')
    def messages():
        return FileResponse(WEB_DIR/'messages.tsv',media_type='text/tab-separated-values')

    @app.get('/health')
    def health():
        return dict(status='ok',dataset_version=data['dataset_version'],dataset_sha256=dataset_digest,
                    records=len(data['records']),electrolyte_formulations=len(electrolytes),
                    aqueous_formulations=len(aqueous['records']),aqueous_sha256=aqueous_digest,
                    expires_at_utc=settings().get('expires_at_utc'))

    @app.get('/api/overview')
    def overview():
        return dict(dataset_version=data['dataset_version'],dataset_sha256=dataset_digest,
                    source_count=len(data['sources']),study_count=sum(s['scope']=='primary_study_extracted' for s in data['sources']),
                    record_count=len(data['records']),observation_count=sum(len(r['observations']) for r in data['records']),
                    electrolyte_formulations=len(electrolytes),
                    aqueous_formulations=len(aqueous['records']),
                    electrolyte_measurements=sum(r['repeat_measurements'] for r in electrolytes),
                    expires_at_utc=settings().get('expires_at_utc'),
                    daily_model_limit=int(settings().get('daily_model_limit',100)),
                    catalogue_note='案例數是已完成原文核對的配方／樣本數，會隨資料擴充增加。')

    @app.get('/api/sources')
    def sources():
        # Internal archival paths are not useful to the partner.
        reviewed={s['source_id']:s for s in literature}
        return [{**{k:v for k,v in s.items() if k!='source_file'},
                 **reviewed.get(s['source_id'],{})} for s in data['sources']]

    @app.get('/api/records')
    def records(substrate: str='', q: str=''):
        try:
            return [{k:v for k,v in record.items() if k!='notes'}
                    for record in search_separator_records(data,substrate or None,q)]
        except ValueError:
            raise HTTPException(422,'invalid_search') from None

    @app.get('/api/evidence/{evidence_id:path}')
    def evidence(evidence_id: str):
        if evidence_id not in data['evidence']:
            raise HTTPException(404,'evidence_not_found')
        entry=data['evidence'][evidence_id]
        source=next(s for s in data['sources'] if s['source_id']==entry['source_id'])
        return dict(text=entry['text'],source=source['url'],source_title=source['title'],
                    section_title=entry['section_title'])

    @app.get('/api/evaluation')
    def evaluation():
        path=PROJECT_ROOT/'docs/research/separator_prototype/workflow_evaluation.json'
        return json.loads(path.read_text()) if path.is_file() else {'status':'not_run'}

    @app.get('/api/electrolytes')
    def electrolyte_records():
        return electrolytes

    @app.get('/api/aqueous')
    def aqueous_records():
        return aqueous['records']

    @app.get('/api/download/{name}')
    def download(name: str):
        allowed={'sources.csv':Path(data_path).with_name('source_inventory.csv'),
                 'records.csv':separator_csv,
                 'electrolytes.csv':electrolyte_csv,
                 'aqueous.csv':aqueous_csv,
                 'public.sqlite':db_path}
        if name not in allowed:
            raise HTTPException(404,'download_not_found')
        return FileResponse(allowed[name],filename=name)

    @app.post('/api/assess')
    def assess(recipe: SeparatorRecipe):
        result=assess_separator_recipe(data,recipe)
        return dict(numeric_allowed=result['numeric_allowed'],reasons=result['reasons'])

    @app.post('/api/predict')
    async def predict(recipe: SeparatorRecipe | CoatingRecipe | ElectrolyteRecipe | ExperimentPlan | AqueousSlurryRecipe,
                      language: Literal['zh-CN','zh-TW','en']='zh-CN'):
        configured=settings()
        model_name=configured.get('model_name',MODEL)
        assessment=None
        if isinstance(recipe,ExperimentPlan) and not set(recipe.observations).issubset(
                r['candidate_id'] for r in electrolytes):
            raise HTTPException(422,'已測配方編號不在這個電解液資料集中。')
        if isinstance(recipe,SeparatorRecipe):
            assessment=assess_separator_recipe(data,recipe)
            if not assessment['numeric_allowed']:
                return dict(status='needs_data',reasons=assessment['reasons'],model_called=False)
        identity=dict(api_version=5,recipe=recipe.model_dump(),dataset=dataset_digest,
                      electrolyte_data=electrolyte_digest,aqueous_data=aqueous_digest,
                      engine=model_name if assessment else 'public_data_v1',
                      prompt=make_separator_prompt(assessment,language=language) if assessment else None)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429,'正在分析另一個配方，請稍後再試。')
        call_id=None
        try:
            with sqlite3.connect(runtime_dir/'usage.sqlite') as db:
                hit=db.execute('SELECT result FROM calls WHERE request_hash=? AND status=? ORDER BY id DESC LIMIT 1',(key,'ok')).fetchone()
                if hit:
                    return dict(status='ok',cached=True,model_called=False,**json.loads(hit[0])['public'])
                daily_limit=int(configured.get('daily_model_limit',100))
                count=db.execute('SELECT count(*) FROM calls WHERE time>?',(time.time()-86400,)).fetchone()[0]
                if count>=daily_limit:
                    raise HTTPException(429,f'已達全站 24 小時 {daily_limit} 次新分析上限；仍可查看資料和已有結果。')
                cursor=db.execute('INSERT INTO calls (time,request_hash,status) VALUES (?,?,?)',(time.time(),key,'started'))
                call_id=cursor.lastrowid
            private=None
            if isinstance(recipe,AqueousSlurryRecipe):
                prediction=predict_aqueous_viscosity(aqueous,recipe)
                public=dict(prediction=prediction,explanation='依同一製程的公開漿料數據，估算 BN 比例改變時的黏度；下方列出相近配方的塗布結果。')
            elif isinstance(recipe,ExperimentPlan):
                recommendations=await asyncio.to_thread(suggest_experiments,electrolyte_inputs,
                    recipe.observations,limit=1)
                public=dict(recommendations=recommendations,observations_used=len(recipe.observations),
                            explanation='依據已提供的實測值，兼顧預期導電率與尚待探索的配方。')
            elif isinstance(recipe,CoatingRecipe):
                prediction=predict_coating_thickness(data,recipe)
                public=dict(prediction=prediction,explanation='根據同製程的已發表塗布設定與製備後膜厚，估算這個設定的膜厚。')
            elif isinstance(recipe,ElectrolyteRecipe):
                prediction=await asyncio.to_thread(predict_electrolyte,electrolytes,recipe)
                public=dict(prediction=prediction,explanation='依據同一電解液體系的實測配方，估算指定溶劑比例與鹽濃度的導電率。')
            else:
                executable=model_executable or shutil.which('codex')
                if not executable:
                    raise RuntimeError('model_executable_missing')
                private=await asyncio.to_thread(run_separator_model,assessment,executable,
                                               runtime_dir/'model',model_name=model_name,language=language)
                result=private['result']
                explanation=_public_explanation(result.get('preparation_hypothesis',''),model_name,language)
                # The language model explains the recipe. Its free numerical guess
                # has not beaten the simple reference; publish that reference
                # without fitting a correction to the already known holdout.
                public=dict(prediction=dict(value=assessment['baseline_linear_mS_cm'],unit='mS/cm',
                    property='離子導電率',kind='模型估算',
                    numerical_method='loading_linear_reference_v1',
                    supporting_records=result.get('supporting_record_ids',[]),
                    source_url='https://doi.org/10.3390/nano12010011'),explanation=explanation)
            response={'public':public,'provider_record':private}
            with sqlite3.connect(runtime_dir/'usage.sqlite') as db:
                db.execute('UPDATE calls SET status=?,result=? WHERE id=?',('ok',json.dumps(response,ensure_ascii=False),call_id))
            return dict(status='ok',cached=False,model_called=private is not None,**public)
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
