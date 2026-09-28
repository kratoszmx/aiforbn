"""Composition-only electrolyte prediction and answer-blind experiment replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field

from materials.modeling import make_model

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / 'official_docs/experiment_planning'
PROTOCOL_PATH = ROOT / 'docs/research/separator_prototype/replay_protocol.json'
FOREST_CONFIG = {'model': {'type': 'random_forest', 'random_forest': {
    'n_estimators': 64, 'min_samples_leaf': 1, 'max_features': 1.0,
    'random_state': 2026, 'n_jobs': 1,
}}}


class ElectrolyteRecipe(BaseModel):
    """LiPF6 in EC/DMC/EMC, matching the public liquid-electrolyte domain."""
    model_config = {'extra': 'forbid', 'allow_inf_nan': False}
    task: Literal['electrolyte_conductivity'] = 'electrolyte_conductivity'
    salt_molality: float = Field(default=1.0, ge=0.14, le=2.0)
    ec_fraction: float = Field(default=0.4, ge=0.3, le=0.5)
    dmc_ratio: float = Field(default=0.8, ge=0, le=1)
    temperature_c: float = Field(default=27., ge=26, le=28)


class ExperimentPlan(BaseModel):
    """Already measured outcomes for choosing the next public-pool recipes."""
    model_config = {'extra': 'forbid', 'allow_inf_nan': False}
    task: Literal['experiment_plan'] = 'experiment_plan'
    observations: dict[str, Annotated[float, Field(ge=0, le=100)]] = Field(min_length=2, max_length=38)


def load_electrolyte_records(source_dir=SOURCE_DIR):
    """Verify raw CSV bytes and collapse repeat measurements by composition."""
    source_dir = Path(source_dir)
    manifest = json.loads((source_dir / 'source_manifest.json').read_text())
    path = source_dir / 'clio_2022_measurements.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Electrolyte source checksum mismatch')
    grouped = {}
    with path.open(newline='') as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            key = tuple(float(row[k]) for k in (
                'LiPF6_molalitt', 'EC_mass_fraction', 'DMC_cosolvent_ratio'))
            value, temperature = float(row['conductivity']), float(row['Temp (C) 2'])
            if not all(math.isfinite(x) for x in (*key, value, temperature)) or value < 0:
                raise ValueError(f'Invalid source measurement at CSV line {line}')
            grouped.setdefault(key, []).append((line, value, temperature))
    records = []
    for index, (key, measurements) in enumerate(sorted(grouped.items()), 1):
        values = [x[1] for x in measurements]
        records.append(dict(
            candidate_id=f'CLIO-{index:02d}', salt_molality=key[0],
            ec_fraction=key[1], dmc_ratio=key[2],
            conductivity_mS_cm=float(np.mean(values)),
            repeat_measurements=len(values), measurement_std_mS_cm=float(np.std(values)),
            temperature_min_c=min(x[2] for x in measurements),
            temperature_max_c=max(x[2] for x in measurements),
            source_csv_lines=[x[0] for x in measurements], source_id='clio_2022',
        ))
    return records


def _feature_rows(recipes):
    # Scale using the predefined chemical domain, never future measured outcomes.
    return np.asarray([[r['salt_molality'] / 2., (r['ec_fraction'] - .3) / .2,
                        r['dmc_ratio']] for r in recipes], dtype=float)


def suggest_experiments(candidates, observations, *, method='adaptive_forest',
                        tie_order=None, limit=5):
    """Rank unmeasured inputs using ONLY supplied, already-observed responses.

    Candidates contain IDs and composition inputs; conductivity in a candidate
    is rejected to make leakage a caller-visible error. Observations map IDs to
    the results already paid for in the current experiment sequence.
    """
    allowed = {'candidate_id', 'salt_molality', 'ec_fraction', 'dmc_ratio'}
    if not candidates or any(set(r) != allowed for r in candidates):
        raise ValueError('Candidates must contain only identifiers and composition inputs')
    ids = [r['candidate_id'] for r in candidates]
    if len(set(ids)) != len(ids) or not set(observations).issubset(ids):
        raise ValueError('Invalid candidate or observation identity')
    for row in candidates:
        ElectrolyteRecipe.model_validate({k:row[k] for k in allowed if k!='candidate_id'})
    if not 1 <= limit <= len(ids) or method not in {'random', 'nearest', 'linear', 'adaptive_forest'}:
        raise ValueError('Invalid planner bounds or method')
    if not all(math.isfinite(v) and v >= 0 for v in observations.values()):
        raise ValueError('Invalid observed conductivity')
    order = ids if tie_order is None else list(tie_order)
    if len(order) != len(ids) or set(order) != set(ids):
        raise ValueError('Tie order must be a permutation of candidates')
    remaining = [i for i, key in enumerate(ids) if key not in observations]
    if not remaining:
        return []
    known = [i for i, key in enumerate(ids) if key in observations]
    x = _feature_rows(candidates)
    predictions = np.zeros(len(remaining))
    spreads = np.zeros(len(remaining))
    if method != 'random':
        if len(known) < 2:
            raise ValueError('At least two measured formulations are required')
        y = np.asarray([observations[ids[i]] for i in known])
        if method == 'nearest':
            distances = ((x[remaining, None, :] - x[None, known, :]) ** 2).sum(axis=2)
            predictions = y[np.argmin(distances, axis=1)]
        else:
            model = make_model(FOREST_CONFIG if method == 'adaptive_forest'
                               else {'model': {'type': 'linear_regression'}})
            model.fit(x[known], y)
            predictions = model.predict(x[remaining])
            if method == 'adaptive_forest':
                spreads = np.std([tree.predict(x[remaining]) for tree in model.estimators_], axis=0)
    positions = {key: index for index, key in enumerate(order)}
    ranked = [dict(candidate_id=ids[i], prediction_mS_cm=float(max(0., predictions[j])),
                   acquisition_score=float(predictions[j] + .5 * spreads[j]))
              for j, i in enumerate(remaining)]
    ranked.sort(key=lambda item: (-item['acquisition_score'], positions[item['candidate_id']]))
    return ranked[:limit]


def predict_electrolyte(records, recipe):
    """Estimate an in-domain, unmeasured composition from grouped public data."""
    recipe = recipe if isinstance(recipe, ElectrolyteRecipe) else ElectrolyteRecipe.model_validate(recipe)
    candidates = [{k: r[k] for k in ('candidate_id', 'salt_molality', 'ec_fraction', 'dmc_ratio')}
                  for r in records]
    target = dict(candidate_id='requested', **{k: getattr(recipe, k)
                  for k in ('salt_molality', 'ec_fraction', 'dmc_ratio')})
    candidates.append(target)
    observations = {r['candidate_id']: r['conductivity_mS_cm'] for r in records}
    predicted = suggest_experiments(candidates, observations, limit=1)[0]
    matches = [r for r in records if all(abs(r[k] - target[k]) < 1e-10
               for k in ('salt_molality', 'ec_fraction', 'dmc_ratio'))]
    return dict(value=predicted['prediction_mS_cm'], unit='mS/cm',
                property='離子導電率', kind='模型估算',
                matched_observation_mS_cm=matches[0]['conductivity_mS_cm'] if matches else None,
                supporting_formulations=len(records), source_url='https://doi.org/10.1038/s41467-022-32938-1',
                conditions='LiPF₆／EC／DMC／EMC 液態電解液，26–28°C',
                solvent_percent={'EC':100*recipe.ec_fraction,
                    'DMC':100*(1-recipe.ec_fraction)*recipe.dmc_ratio,
                    'EMC':100*(1-recipe.ec_fraction)*(1-recipe.dmc_ratio)})


def evaluate_experiment_planner(records, protocol, output_dir):
    """Replay fixed paired starts; count all queried formulations until success.

    Future answers remain in the evaluator. The planner receives composition-only
    candidates and observations revealed one at a time. No surrogate truth or
    generated measurement is substituted for the published experimental values.
    """
    if protocol['forest'] != FOREST_CONFIG['model']['random_forest']:
        raise ValueError('Frozen model parameters do not match implementation')
    candidates = [{k:r[k] for k in ('candidate_id','salt_molality','ec_fraction','dmc_ratio')}
                  for r in records]
    truth = {r['candidate_id']:r['conductivity_mS_cm'] for r in records}
    ids = list(truth)
    thresholds = [protocol['primary_target_mS_cm'], *protocol['secondary_thresholds_mS_cm']]
    needed = protocol['target_hits']
    if any(sum(v >= threshold for v in truth.values()) < needed for threshold in thresholds):
        raise ValueError('The measured pool cannot satisfy the frozen objective')
    trials = []
    for seed in protocol['seed_values']:
        order = np.random.default_rng(seed).permutation(ids).tolist()
        for method in protocol['methods']:
            observed = {}
            sequence = []
            counts = {}
            for step in range(len(ids)):
                if step < protocol['initial_experiments']:
                    selected = order[step]
                else:
                    selected = suggest_experiments(candidates, observed, method=method,
                                                   tie_order=order, limit=1)[0]['candidate_id']
                observed[selected] = truth[selected]
                sequence.append(selected)
                for threshold in thresholds:
                    if threshold not in counts and sum(v >= threshold for v in observed.values()) >= needed:
                        counts[threshold] = step + 1
                if len(counts) == len(thresholds):
                    break
            for threshold in thresholds:
                trials.append(dict(seed=seed, method=method, target_mS_cm=threshold,
                                   experiments=counts[threshold],
                                   query_sequence=sequence[:counts[threshold]]))
    summaries = []
    for threshold in thresholds:
        counts = {method: np.asarray([r['experiments'] for r in trials
                  if r['method']==method and r['target_mS_cm']==threshold], dtype=float)
                  for method in protocol['methods']}
        comparisons = []
        adaptive = counts['adaptive_forest']
        for method in ('random', 'nearest', 'linear'):
            baseline = counts[method]
            resample = np.random.default_rng(2026).integers(0, len(adaptive), (2000, len(adaptive)))
            reductions = 1-adaptive[resample].mean(axis=1)/baseline[resample].mean(axis=1)
            comparisons.append(dict(baseline=method,
                saved_experiments=float(baseline.mean()-adaptive.mean()),
                reduction_pct=float(100*(1-adaptive.mean()/baseline.mean())),
                paired_start_bootstrap_95_pct=[float(x) for x in np.quantile(reductions*100,[.025,.975])],
                wins=int((adaptive<baseline).sum()),ties=int((adaptive==baseline).sum()),
                losses=int((adaptive>baseline).sum())))
        summaries.append(dict(target_mS_cm=threshold, target_hits=needed,
            eligible_formulations=sum(v>=threshold for v in truth.values()),
            methods=[dict(method=method,mean_experiments=float(v.mean()),median_experiments=float(np.median(v)),
                          min_experiments=int(v.min()),max_experiments=int(v.max())) for method,v in counts.items()],
            comparisons=comparisons))
    # A different question: unseen-formulation numerical accuracy, grouped by recipe.
    errors = {'adaptive_forest': [], 'nearest': [], 'linear': []}
    for held in ids:
        observed = {key:value for key,value in truth.items() if key!=held}
        for method in errors:
            value=suggest_experiments(candidates,observed,method=method,limit=1)[0]['prediction_mS_cm']
            errors[method].append(abs(value-truth[held]))
    result=dict(protocol_version=protocol['version'], evaluation='measured_pool_sequential_replay',
        formulation_count=len(records),raw_measurement_count=sum(r['repeat_measurements'] for r in records),
        independent_studies=1,seeds=len(protocol['seed_values']),initial_experiments=protocol['initial_experiments'],
        primary_target_mS_cm=protocol['primary_target_mS_cm'],summaries=summaries,
        grouped_leave_one_out_mae_mS_cm={k:float(np.mean(v)) for k,v in errors.items()},
        protocol_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
        source_sha256=json.loads((SOURCE_DIR/'source_manifest.json').read_text())['data_sha256'],
        scope='公開液態電解液資料回放；不能換算為 BN 隔膜已節省的實驗。',
        actual_lab_experiments_saved=None,actual_calendar_time_saved=None)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'workflow_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output_dir/'workflow_trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/research/separator_prototype')
    args=parser.parse_args()
    report=evaluate_experiment_planner(load_electrolyte_records(),json.loads(PROTOCOL_PATH.read_text()),args.output_dir)
    print(json.dumps(report,ensure_ascii=False,indent=2))
