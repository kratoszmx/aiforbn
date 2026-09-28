"""Source-checked water-based BN recipes and within-series viscosity interpolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Literal
import xml.etree.ElementTree as ET

import numpy as np
from pydantic import BaseModel, Field

AQUEOUS_DATA_PATH = Path(__file__).resolve().parents[2] / 'data/dispersion/aqueous_sources.json'


class AqueousSlurryRecipe(BaseModel):
    """Vary BN fraction only; other settings belong to one published series."""
    model_config = {'extra': 'forbid', 'allow_inf_nan': False}
    task: Literal['aqueous_viscosity'] = 'aqueous_viscosity'
    bn_volume_pct: float = Field(default=4., ge=0, le=12)
    solids_weight_pct: Literal[30] = 30
    binder_volume_pct: Literal[15] = 15
    binder: Literal['styrene_butyl_acrylate'] = 'styrene_butyl_acrylate'
    filler: Literal['BaTiO3'] = 'BaTiO3'
    solvent: Literal['water'] = 'water'
    bn_diameter_nm: Literal[150] = 150
    filler_diameter_nm: Literal[500] = 500
    preparation_temperature_c: Literal[25] = 25
    mixing_minutes: Literal[60] = 60


def load_aqueous_slurries(path=AQUEOUS_DATA_PATH):
    """Extract twelve cases from pinned patent paragraphs/tables, without pooling cohorts."""
    root = AQUEOUS_DATA_PATH.parents[2]
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported aqueous source manifest')
    sources = {s['source_id']: s for s in data['sources']}
    if set(sources) != {'lg_2026', 'cn105206783'} or len(data['sources']) != 2:
        raise ValueError('Incomplete or duplicate aqueous sources')
    trees = {}
    for sid, source in sources.items():
        file = (root / source['source_file']).resolve()
        if not file.is_relative_to(root / 'official_docs/dispersion'):
            raise ValueError('Aqueous source outside archive')
        if hashlib.sha256(file.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Aqueous source checksum mismatch')
        trees[sid] = ET.parse(file).getroot()
    records = []
    tree = trees['lg_2026']
    rows = {r.attrib['name']: [c.text for c in r.findall('cell')]
            for r in tree.findall('./table/row')}
    if not rows or any(len(r) != 8 for r in rows.values()):
        raise ValueError('Incomplete patent measurement table')
    for i in range(8):
        bn = float(rows['Hexagonal boron nitride'][i])
        filler = float(rows['Inorganic particles (vol %)'][i].split()[0])
        binder = float(rows['Polymer binder (vol %)'][i])
        solids = float(rows['Solid content (wt %)'][i])
        raw_viscosity = rows['Slurry viscosity (cps)'][i]
        viscosity = None if raw_viscosity == '—' else float(raw_viscosity)
        if bn + filler + binder != 100 or binder != 15 or solids != 30:
            raise ValueError('Patent series composition changed')
        if viscosity is not None and (not np.isfinite(viscosity) or viscosity <= 0):
            raise ValueError('Invalid source viscosity')
        sample = f'實施例 {i+1}' if i < 4 else f'比較例 {i-3}'
        coating = ('未達到 1.5 μm 塗層' if i == 5 else
                   '膏狀，無法塗布' if i == 6 else '雙面各 1.5 μm 塗層')
        records.append(dict(record_id=f'LG-{i+1:02}', sample=sample, source_id='lg_2026',
            cohort='BaTiO3_BN_SBA_water' if i != 7 else 'Al2O3_SBA_water',
            formulation=f'{"氧化鋁" if i == 7 else "鈦酸鋇"} / BN / 黏結劑 = {filler:g}:{bn:g}:{binder:g}（固體體積比）',
            particle_size='BN 150 nm；共填料 500 nm' if i != 7 else '氧化鋁 500 nm',
            bn_volume_pct=bn, filler_volume_pct=filler, binder_volume_pct=binder,
            solids_weight_pct=solids, viscosity_mPa_s=viscosity,
            process='總固含量 30 wt%；水；25°C 振盪混合 60 分鐘；苯乙烯－丙烯酸丁酯黏結劑',
            coating_result=coating, section_title='實施例與比較例；實驗例 1「漿料物性評估」、表 1',
            source_url=sources['lg_2026']['url'], source_title=sources['lg_2026']['title'],
            viscometer='TV-22 錐板黏度計', measurement_temperature_c=None, shear_rate_s_inv=None,
            source_column=i+1))
    tree = trees['cn105206783']
    for i, paragraph_id in enumerate(('p0042', 'p0048', 'p0054', 'p0060'), 1):
        text = tree.find(f'./paragraph[@id="{paragraph_id}"]').text
        patterns = {'particle_nm': r'分布在(\d+)nm', 'bn_g': r'\)([\d.]+)克，聚丙烯酸銨',
                    'dispersant_g': r'聚丙烯酸銨([\d.]+)克', 'binder_g': r'异辛酯([\d.]+)克',
                    'water_g': r'蒸馏水([\d.]+)克', 'rpm': r'每分钟(\d+)转',
                    'viscosity': r'绝对粘度为([\d.]+)mPa'}
        matches = {k: re.search(v, text) for k, v in patterns.items()}
        if not all(matches.values()):
            raise ValueError('Incomplete aqueous recipe paragraph')
        values = {k: float(m.group(1)) for k, m in matches.items()}
        records.append(dict(record_id=f'PI-{i:02}', sample=f'BN/H₂O-{i}', source_id='cn105206783',
            cohort='BN_polyacrylate_water', particle_size=f'BN {values["particle_nm"]:g} nm',
            formulation=f'BN {values["bn_g"]:g} g；聚丙烯酸銨 {values["dispersant_g"]:g} g；黏結劑 {values["binder_g"]:g} g；水 {values["water_g"]:g} g',
            viscosity_mPa_s=values['viscosity'], bn_volume_pct=None,
            # Supplier solution solids are not reported; mass inputs are not dry solids.
            solids_weight_pct=None, process=f'水；聚丙烯酸丁酯－異辛酯黏結劑；{values["rpm"]:g} rpm 乳化',
            coating_result='浸潤 PI 纖維基膜；100°C 乾燥後 200°C 熱處理',
            section_title=f'具體實施方式 · 實施例 {i} · 水基懸浮液配製',
            source_url=sources['cn105206783']['url'], source_title=sources['cn105206783']['title'],
            source_paragraph=paragraph_id, measurement_temperature_c=None, shear_rate_s_inv=None))
    return dict(sources=data['sources'], records=records)


def predict_aqueous_viscosity(dataset, recipe):
    """Linearly interpolate six matched recipes; controls/failures remain explicit records."""
    recipe = recipe if isinstance(recipe, AqueousSlurryRecipe) else AqueousSlurryRecipe.model_validate(recipe)
    rows = sorted((r for r in dataset['records'] if r['cohort'] == 'BaTiO3_BN_SBA_water'
                   and r['viscosity_mPa_s'] is not None), key=lambda r: r['bn_volume_pct'])
    x = [r['bn_volume_pct'] for r in rows]
    y = [r['viscosity_mPa_s'] for r in rows]
    if x != [0., 1., 3., 5., 8., 12.]:
        raise ValueError('Incomplete compatible viscosity series')
    match = next((r for r in rows if r['bn_volume_pct'] == recipe.bn_volume_pct), None)
    neighbours = sorted(rows, key=lambda r: abs(r['bn_volume_pct'] - recipe.bn_volume_pct))[:2]
    return dict(value=float(np.interp(recipe.bn_volume_pct, x, y)), unit='mPa·s',
        property='水性漿料黏度', kind='同體系插值估算', numerical_method='within_series_linear_interpolation_v1',
        matched_observation_mPa_s=match['viscosity_mPa_s'] if match else None,
        supporting_records=[r['record_id'] for r in rows], source_url=rows[0]['source_url'],
        conditions=f'BN {recipe.bn_volume_pct:g} vol%／鈦酸鋇 {85-recipe.bn_volume_pct:g} vol%／黏結劑 15 vol%；總固含量 30 wt%；TV-22 測法',
        nearby_cases=[{k: r[k] for k in ('sample', 'bn_volume_pct', 'viscosity_mPa_s', 'coating_result')}
                      for r in neighbours])


def evaluate_aqueous_viscosity(dataset):
    """Development leave-one-out comparison; each target label is removed before fitting."""
    rows = sorted((r for r in dataset['records'] if r['cohort'] == 'BaTiO3_BN_SBA_water'
                   and r['viscosity_mPa_s'] is not None), key=lambda r: r['bn_volume_pct'])
    cases = []
    # Endpoints cannot be checked by interpolation; do not silently extrapolate.
    for target in rows[1:-1]:
        train = [r for r in rows if r['record_id'] != target['record_id']]
        x = np.array([r['bn_volume_pct'] for r in train]); y = np.array([r['viscosity_mPa_s'] for r in train])
        query = target['bn_volume_pct']
        estimates = dict(interpolation=float(np.interp(query, x, y)),
                         nearest=float(y[np.argmin(abs(x-query))]),
                         linear=float(np.polyval(np.polyfit(x, y, 1), query)))
        cases.append(dict(record_id=target['record_id'], train_ids=[r['record_id'] for r in train],
                          truth_mPa_s=target['viscosity_mPa_s'], predictions_mPa_s=estimates))
    return dict(protocol='development_leave_one_formula_out_interior_only', cases=cases,
                independent_validation=False, experimental_savings_measured=False,
                mae_mPa_s={method: float(np.mean([abs(c['predictions_mPa_s'][method]-c['truth_mPa_s'])
                    for c in cases])) for method in ('interpolation', 'nearest', 'linear')})
