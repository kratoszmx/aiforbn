"""Source-checked separator records, SQLite export and recipe-domain checks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Literal
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

DATASET_PATH = Path(__file__).resolve().parents[2] / 'data/separators/dataset.json'
PRIMARY_COHORT = 'PP_BNNT_LiTFSI_DOL_DME'


class SeparatorRecipe(BaseModel):
    """Only structured formulation inputs may reach the model; no free-form prompt."""
    model_config = {'extra': 'forbid', 'allow_inf_nan': False}
    substrate: Literal['PP', 'PE', 'calcium_alginate', 'cellulose', 'solid_PEO_PVDF', 'other'] = 'PP'
    bn_form: Literal['none', 'raw_BNNT', 'purified_BNNT', 'BNNT', 'BN_nanopowder', 'BN_flakes', 'other'] = 'raw_BNNT'
    loading_mg_cm2: float | None = Field(default=.3, ge=0, le=10)
    bn_binder_ratio: float | None = Field(default=4., gt=0, le=100)
    binder: Literal['PVDF', 'none', 'other'] = 'PVDF'
    solvent: Literal['NMP', 'DMF', 'water', 'IPA/water', 'none', 'other'] = 'NMP'
    dry_temperature_c: float | None = Field(default=50., ge=0, le=250)
    dry_hours: float | None = Field(default=24., gt=0, le=200)
    electrolyte: Literal['LiTFSI_DOL_DME_LiNO3', 'LiPF6_carbonates', 'solid', 'other'] = 'LiTFSI_DOL_DME_LiNO3'
    test_temperature_c: float | None = Field(default=None, ge=-40, le=150)


def load_separator_dataset(path=DATASET_PATH):
    """Load v1 and verify source bytes, source anchors, IDs and finite observations."""
    path = Path(path)
    payload = json.loads(path.read_text())
    root = DATASET_PATH.parents[2]
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported separator dataset version')
    sources = {s['source_id']: s for s in payload['sources']}
    if len(sources) != len(payload['sources']):
        raise ValueError('Duplicate source ID')
    trees = {}
    for sid, source in sources.items():
        if source['scope'] != 'primary_study_extracted':
            continue
        source_path = (root / source['source_file']).resolve()
        if not source_path.is_relative_to(root / 'official_docs/separators'):
            raise ValueError('Source path outside public source directory')
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'Source checksum mismatch: {sid}')
        trees[sid] = ET.parse(source_path).getroot()
    for eid, e in payload['evidence'].items():
        tree = trees[e['source_id']]
        if 'paragraph' in e:
            node = list(tree.iter('p'))[e['paragraph'] - 1]
        else:
            node = tree.find(f'.//*[@id="{e["xml_id"]}"]')
        text = ' '.join(''.join(node.itertext()).split())
        if text != e['text'] or e['anchor'] not in text:
            raise ValueError(f'Evidence mismatch: {eid}')
    ids = set()
    groups = {}
    for r in payload['records']:
        if r['record_id'] in ids or r['source_id'] not in trees:
            raise ValueError('Duplicate record or missing primary source')
        ids.add(r['record_id'])
        old_split = groups.setdefault(r['formulation_group'], r['split'])
        if old_split != r['split']:
            raise ValueError('Formulation crosses train/test boundary')
        for eid in r['evidence_ids']:
            if payload['evidence'][eid]['source_id'] != r['source_id']:
                raise ValueError('Record evidence belongs to another source')
        for observation in r['observations'].values():
            if observation['evidence_id'] not in r['evidence_ids']:
                raise ValueError('Missing observation evidence')
            value = observation['value']
            if isinstance(value, (float, int)) and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError('Invalid observation')
    return payload


def build_separator_database(dataset, path):
    """Transactionally populate a public-record SQLite database with fixed tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, substrate TEXT NOT NULL, body TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
        for table in ['sources', 'records', 'evidence']:
            db.execute(f'DELETE FROM {table}')
        db.executemany('INSERT INTO sources VALUES (?,?)', [(s['source_id'], json.dumps(s,ensure_ascii=False)) for s in dataset['sources']])
        db.executemany('INSERT INTO records VALUES (?,?,?,?)', [(r['record_id'],r['source_id'],r['inputs']['substrate'],json.dumps(r,ensure_ascii=False)) for r in dataset['records']])
        db.executemany('INSERT INTO evidence VALUES (?,?)', [(k,json.dumps(v,ensure_ascii=False)) for k,v in dataset['evidence'].items()])
    return path


def search_separator_records(dataset, substrate=None, query='', limit=20):
    """Return bounded public records with matching substrate and literal text."""
    if not 1 <= limit <= 100 or len(query) > 100:
        raise ValueError('Invalid search bounds')
    return [r for r in dataset['records']
            if (not substrate or r['inputs']['substrate'] == substrate)
            and query.casefold() in json.dumps(r,ensure_ascii=False).casefold()][:limit]


def assess_separator_recipe(dataset, recipe):
    """Gate quantitative inference and select training-only compatible evidence.

    Numeric permission means a bounded exploratory task, never demonstrated
    accuracy. Missing test temperature prevents transfer to a specified temperature.
    """
    if not isinstance(recipe, SeparatorRecipe):
        recipe = SeparatorRecipe.model_validate(recipe)
    inputs = recipe.model_dump()
    related = search_separator_records(dataset, substrate=recipe.substrate)
    reasons = []
    if recipe.substrate != 'PP':
        reasons.append('目前數值任務只涵蓋 PP 基材；其他基材只能查文獻。')
    if recipe.bn_form not in ['none','raw_BNNT','purified_BNNT']:
        reasons.append('BN 形態超出目前 PP/BNNT 任務。')
    if recipe.electrolyte != 'LiTFSI_DOL_DME_LiNO3':
        reasons.append('電解液不匹配；不同電解液的導電率不能直接套用。')
    if recipe.test_temperature_c is not None:
        reasons.append('訓練文獻未清楚標示這項測量的溫度，不能預測指定溫度。')
    if recipe.loading_mg_cm2 is None or not 0 <= recipe.loading_mg_cm2 <= .5:
        reasons.append('可供數值探索的載量範圍僅為 0–0.5 mg/cm²。')
    if recipe.bn_form == 'none':
        if recipe.loading_mg_cm2 != 0 or recipe.binder != 'none' or recipe.solvent != 'none':
            reasons.append('空白 PP 對照需設定 BN 載量為 0，黏結劑與溶劑為 none。')
    else:
        if recipe.loading_mg_cm2 == 0:
            reasons.append('含 BN 配方的載量必須大於 0。')
        if (recipe.binder,recipe.solvent,recipe.bn_binder_ratio,recipe.dry_temperature_c,recipe.dry_hours) != ('PVDF','NMP',4.,50.,24.):
            reasons.append('比例或製程超出文獻條件：BN:PVDF 4:1、NMP、50°C 真空乾燥 24 h。')
    train = [r for r in dataset['records'] if r['cohort']==PRIMARY_COHORT and r['split']=='train'
             and 'ionic_conductivity' in r['observations']]
    if len(train)<2:
        reasons.append('可用訓練標籤不足。')
    def distance(r):
        x=r['inputs']
        return abs(x['loading_mg_cm2']-(recipe.loading_mg_cm2 or 0))/.5 + (x['bn_form']!=recipe.bn_form)
    train.sort(key=lambda r:(distance(r),r['record_id']))
    examples = [dict(record_id=r['record_id'], source_id=r['source_id'],inputs=r['inputs'],
                     conductivity_mS_cm=r['observations']['ionic_conductivity']['value']) for r in train]
    linear_prediction=None
    if examples and not reasons:
        xs=[r['inputs']['loading_mg_cm2'] for r in examples]
        ys=[r['conductivity_mS_cm'] for r in examples]
        xmean,ymean=sum(xs)/len(xs),sum(ys)/len(ys)
        variance=sum((x-xmean)**2 for x in xs)
        if variance:
            slope=sum((x-xmean)*(y-ymean) for x,y in zip(xs,ys))/variance
            linear_prediction=ymean+slope*(recipe.loading_mg_cm2-xmean)
    warning='只有一篇研究、兩個訓練數值；未證明未知配方準確度，不能代替實驗。固定製程另含超聲 1 h、攪拌過夜和真空乾燥。'
    return dict(inputs=inputs,numeric_allowed=not reasons,reasons=reasons,training_examples=examples,
                related_records=related,warning=warning,
                preparation_hypothesis='可能形成 BNNT/PVDF 塗層；均勻性、附著、孔道與乾燥殘留仍需實驗檢查。',
                baseline_mean_mS_cm=sum(r['conductivity_mS_cm'] for r in examples)/len(examples) if examples and not reasons else None,
                baseline_linear_mS_cm=linear_prediction,
                baseline_nearest_mS_cm=examples[0]['conductivity_mS_cm'] if examples and not reasons else None)


def separator_known_checks(dataset):
    """Recalculate source-supported comparisons, including a published discrepancy."""
    values={r['record_id']:r for r in dataset['records']}
    k=values['kim_2022:BNNT-PP-0.3']['observations']['ionic_conductivity']['value']
    pp=values['kim_2022:PP']['observations']['ionic_conductivity']['value']
    ca=values['tian_2024:CA@BN-100']['observations']['ionic_conductivity']['value']
    ca_pp=values['tian_2024:PP-control']['observations']['ionic_conductivity']['value']
    return [dict(name='PP 對照與 BNNT-PP-0.3 導電率',calculation=f'({k}/{pp}-1)*100',result=100*(k/pp-1),unit='%',status='reconstructed',source_id='kim_2022'),
            dict(name='65°C CA@BN-100 與 PP 導電率',calculation=f'({ca}/{ca_pp}-1)*100',result=100*(ca/ca_pp-1),unit='%',status='reconstructed',source_id='tian_2024'),
            dict(name='PP 硫利用率算術差異',calculation='1197/1675*100',result=1197/1675*100,reported=72.6,
                 discrepancy_percentage_points=1197/1675*100-72.6,unit='%',status='source_discrepancy',source_id='kim_2022')]
