"""Bounded configurable inference and a leakage-aware separator benchmark.

The CLI is used with existing Codex authentication. Public callers supply a
validated recipe, never commands, paths, model IDs, or arbitrary instructions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

from pydantic import BaseModel, Field

from materials.separator_data import SeparatorRecipe, assess_separator_recipe

MODEL = 'gpt-6-astra'


class SeparatorModelResult(BaseModel):
    model_config = {'extra': 'forbid', 'allow_inf_nan': False}
    conductivity_mS_cm: float | None = Field(ge=0, le=10)
    preparation_hypothesis: str = Field(max_length=500)
    supporting_record_ids: list[str] = Field(min_length=1, max_length=2)
    limitations: list[str] = Field(min_length=1, max_length=5)


def make_separator_prompt(assessment):
    """Create an answer-blind prompt using only approved training input/labels."""
    if not assessment['numeric_allowed']:
        raise ValueError('Unsupported recipes cannot be sent for quantitative inference')
    context = dict(recipe=assessment['inputs'],training_examples=assessment['training_examples'])
    return (
        'You are a materials formulation research assistant. Use only the supplied data. '
        'No tools, browsing, file access, commands, memory, or outside numerical values. '
        'Predict ionic conductivity in mS/cm for the requested PP/BNNT separator, under the '
        'same unspecified measurement temperature as the supplied experiment. This is an '
        'exploratory prediction with two training labels from one study, not established accuracy. '
        'Do not infer a precise uncertainty interval. You may abstain with null. '
        'Do not reproduce a memorized paper answer. Distinguish a preparation hypothesis from an observation. '
        'Fixed process: 1 hour sonication, overnight stirring, vacuum drying. '
        'Return only schema-compliant JSON with conductivity_mS_cm, preparation_hypothesis, '
        'supporting_record_ids, limitations. Cite only record IDs in training_examples. '
        'Write explanatory strings in Traditional Chinese, under 200 words total. '
        'Do not mention model names or providers. Keep preparation_hypothesis focused on '
        'the material/process explanation; put data limitations only in limitations.\n'
        + json.dumps(context, ensure_ascii=False, allow_nan=False)
    )


def run_separator_model(assessment, executable, runtime_dir, timeout_seconds=120, *, model_name=MODEL):
    """Run one isolated ephemeral CLI inference; validate output and citation IDs.

    No retries or provider/model fallback. A timed-out process group is terminated.
    Only sanitized statistics are returned; CLI diagnostics stay out of responses.
    """
    prompt = make_separator_prompt(assessment)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}',model_name):
        raise ValueError('Invalid configured model identifier')
    runtime_dir = Path(runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= timeout_seconds <= 180:
        raise ValueError('Inference timeout must be between 1 and 180 seconds')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='inference-', dir=runtime_dir) as scratch:
        work = Path(scratch)
        schema = work/'response_schema.json'
        schema.write_text(json.dumps(SeparatorModelResult.model_json_schema()))
        output = work/'answer.json'
        cmd = [str(executable),'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
               '-C',str(work),'-m',model_name,'-s','read-only','--color','never','--json',
               '--output-schema',str(schema),'-o',str(output)]
        config = {
            'model_provider':'openai', 'model_reasoning_effort':'low', 'web_search':'disabled',
            'project_doc_max_bytes':0, 'agents.enabled':False,
            'features.multi_agent':False, 'features.multi_agent_v2':False,
            'features.shell_tool':False, 'features.unified_exec':False,
            'features.view_image':False, 'features.apps':False,
            'features.skill_search':False, 'features.sleep_tool':False,
            'features.goals':False, 'skills.include_instructions':False,
        }
        for key,value in config.items():
            cmd.extend(['-c',f'{key}={json.dumps(value)}'])
        cmd.append('-')
        env = {k:v for k,v in os.environ.items() if k not in ['OPENAI_API_KEY','AIFORBN_DEMO_TOKEN']}
        with (work/'events.jsonl').open('w+') as events, (work/'stderr.log').open('w+') as errors:
            try:
                proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=events,stderr=errors,
                                        text=True,env=env,start_new_session=True)
            except OSError:
                raise RuntimeError('model_start_failed') from None
            try:
                proc.communicate(prompt,timeout=timeout_seconds)
            except BaseException as exc:
                try:
                    os.killpg(proc.pid,signal.SIGTERM)
                    proc.wait(timeout=3)
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
                if isinstance(exc,subprocess.TimeoutExpired):
                    raise RuntimeError('model_timeout') from None
                raise
            if proc.returncode != 0 or not output.is_file() or output.stat().st_size > 16000:
                errors.seek(0)
                diagnostic=runtime_dir/'last_error.log'
                diagnostic.write_text(errors.read()[-12000:])
                diagnostic.chmod(0o600)
                raise RuntimeError('model_failed')
            events.seek(0)
            usage={}
            completed=False
            for line in events:
                try:
                    event=json.loads(line)
                    if not isinstance(event,dict) or not isinstance(event.get('item',{}),dict):
                        raise ValueError('invalid event')
                except ValueError:
                    raise RuntimeError('model_event_invalid') from None
                if event.get('type') == 'turn.completed':
                    completed=True
                    usage=event.get('usage',{})
                item=event.get('item',{})
                if item.get('type') not in [None,'agent_message','reasoning','error']:
                    raise RuntimeError('unexpected_model_tool_activity')
            if not completed:
                raise RuntimeError('model_turn_incomplete')
            try:
                result=SeparatorModelResult.model_validate_json(output.read_text())
            except ValueError:
                raise RuntimeError('model_output_invalid') from None
    allowed={r['record_id'] for r in assessment['training_examples']}
    if not set(result.supporting_record_ids).issubset(allowed):
        raise RuntimeError('model_citation_invalid')
    if assessment['inputs']['bn_form']=='raw_BNNT' and result.conductivity_mS_cm is not None:
        supported_peak=max(r['conductivity_mS_cm'] for r in assessment['training_examples'])
        if result.conductivity_mS_cm>supported_peak+1e-9:
            raise RuntimeError('model_exceeds_published_raw_bnnt_peak')
    return dict(model=model_name,requested_model=model_name,transport='codex_exec',result=result.model_dump(),
                elapsed_seconds=round(time.monotonic()-started,3),usage=usage,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                evidence_status='exploratory_unvalidated',warning=assessment['warning'])


def evaluate_separator_model(dataset, output_dir, executable=None):
    """Freeze two train recipes / one unused recipe, then compare before revealing.

    This is a within-study smoke benchmark, not a source-held-out accuracy claim.
    No model selection, calibration, or tuning occurs on the test recipe.
    """
    test=next(r for r in dataset['records'] if r['split']=='test')
    recipe=SeparatorRecipe(bn_form='purified_BNNT',loading_mg_cm2=.5)
    assessment=assess_separator_recipe(dataset,recipe)
    prompt=make_separator_prompt(assessment)
    if test['record_id'] in prompt or '0.84' in prompt:
        raise ValueError('Held-out answer leaked into prompt')
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    # Write the exact frozen inputs before contacting the provider.
    (output_dir/'frozen_prompt.txt').write_text(prompt+'\n')
    response=run_separator_model(assessment,executable,output_dir/'private-runtime') if executable else None
    truth=test['observations']['ionic_conductivity']['value']
    rows=[]
    for method,prediction in [('training_mean',assessment['baseline_mean_mS_cm']),
                              ('similar_formulation',assessment['baseline_nearest_mS_cm']),
                              ('loading_linear_regression',assessment['baseline_linear_mS_cm']),
                              (MODEL,response['result']['conductivity_mS_cm'] if response else None)]:
        rows.append(dict(method=method,record_id=test['record_id'],source_id=test['source_id'],
                         prediction_mS_cm=prediction,measured_mS_cm=truth,
                         absolute_error_mS_cm=abs(prediction-truth) if prediction is not None else None,
                         status='predicted' if prediction is not None else 'abstained' if response else 'not_run'))
    summary=dict(evaluation='frozen_within_study_single_formulation',train_records=2,test_records=1,
                 independent_studies=1,source_holdout='not_estimable: no compatible independent training study',
                 pretrained_paper_exposure='possible; cannot be excluded', rows=rows,model_response=response,
                 experimental_time_saved=None,experimental_cost_saved=None,
                 limitations=['One test recipe cannot establish prediction accuracy or superiority.',
                              'No prospective lab comparison or measured manual workflow timing exists.',
                              'Public-paper pretraining exposure remains possible despite prompt isolation.'])
    (output_dir/'evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (output_dir/'prediction_vs_measurement.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return summary
