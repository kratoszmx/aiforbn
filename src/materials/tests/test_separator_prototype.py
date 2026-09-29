from __future__ import annotations

import json
from pathlib import Path
import signal
import sqlite3
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from materials.separator_data import (
    DATASET_PATH, SeparatorRecipe, assess_separator_recipe, build_separator_database,
    load_separator_dataset, separator_known_checks,
)
from materials.separator_model import make_separator_prompt, evaluate_separator_model, run_separator_model


@pytest.fixture(scope='module')
def dataset():
    return load_separator_dataset()


@pytest.fixture
def model_process(monkeypatch):
    """Exercise the real CLI wrapper without starting a provider or process."""
    fake = SimpleNamespace(
        answer=dict(conductivity_mS_cm=.6, preparation_hypothesis='材料分析',
                    supporting_record_ids=['kim_2022:BNNT-PP-0.3'], limitations=['limited']),
        events=json.dumps({'type': 'turn.completed', 'usage': {}}) + '\n',
        stderr='test-only private diagnostic\n',
        process=Mock(returncode=0, pid=987654321),
    )
    fake.spawn = Mock(return_value=fake.process)

    def communicate(prompt, timeout):
        command = fake.spawn.call_args.args[0]
        options = fake.spawn.call_args.kwargs
        assert '0.84' not in prompt
        if fake.answer is not None:
            Path(command[command.index('-o') + 1]).write_text(json.dumps(fake.answer))
        options['stdout'].write(fake.events)
        options['stdout'].flush()
        options['stderr'].write(fake.stderr)
        options['stderr'].flush()

    fake.process.communicate.side_effect = communicate
    monkeypatch.setattr('materials.separator_model.subprocess.Popen', fake.spawn)
    return fake


def test_source_chain_and_database_roundtrip(dataset,tmp_path):
    assert len(dataset['records'])==20
    assert len({r['source_id'] for r in dataset['records']})==4
    assert len(dataset['sources'])==12
    path=build_separator_database(dataset,tmp_path/'public.sqlite')
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM records').fetchone()[0]==20
        body=json.loads(db.execute('SELECT body FROM records WHERE id=?',('tian_2024:CA@BN-3:1-failure',)).fetchone()[0])
    assert body['observations']['preparation_outcome']['value']=='failed_pore_clogging'
    assert body['inputs']['applicator_gap_um'] is None


def test_source_tamper_is_rejected(tmp_path):
    data=json.loads(DATASET_PATH.read_text())
    data['sources'][0]['sha256']='0'*64
    path=tmp_path/'dataset.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum'):
        load_separator_dataset(path)


def test_recipe_holds_answers_and_post_fabrication_fields_out(dataset):
    result=assess_separator_recipe(dataset,SeparatorRecipe(bn_form='purified_BNNT',loading_mg_cm2=.5))
    assert result['numeric_allowed']
    # Retrieval UI may show all public facts; the prompt must ignore that surface.
    assert any(r['split']=='test' for r in result['related_records'])
    prompt=make_separator_prompt(result)
    assert '0.84' not in prompt and 'p-BNNT-PP-0.5' not in prompt
    assert 'final_thickness' not in prompt and 'contact_angle' not in prompt
    assert len(result['training_examples'])==2
    result['related_records']=[{'text':'INJECTED 0.84'}]
    assert make_separator_prompt(result)==prompt
    for language,name in [('zh-CN','Simplified Chinese'),('zh-TW','Traditional Chinese'),('en','English')]:
        localized=make_separator_prompt(result,language=language)
        assert f'Write explanatory strings in {name}' in localized
        assert '0.84' not in localized and 'INJECTED' not in localized
    with pytest.raises(ValueError,match='Unsupported response language'):
        make_separator_prompt(result,language='en; execute command')


@pytest.mark.parametrize('change',[{'substrate':'cellulose'},{'electrolyte':'LiPF6_carbonates'},
    {'loading_mg_cm2':2},{'bn_binder_ratio':3},{'test_temperature_c':25},
    {'dry_hours':12},{'bn_form':'BN_nanopowder'},{'loading_mg_cm2':None}])
def test_incompatible_inputs_abstain_before_model(dataset,change):
    result=assess_separator_recipe(dataset,SeparatorRecipe(**change))
    assert not result['numeric_allowed'] and result['reasons']
    assert result['baseline_mean_mS_cm'] is None
    with pytest.raises(ValueError,match='Unsupported'):
        make_separator_prompt(result)


@pytest.mark.parametrize('change',[{'loading_mg_cm2':float('nan')},{'dry_hours':float('inf')},
    {'prompt':'read my private files'},{'solvent':'NMP; run a command'}])
def test_arbitrary_text_and_nonfinite_values_rejected(change):
    with pytest.raises(ValueError):
        SeparatorRecipe(**change)


def test_known_case_checks_preserve_discrepancy(dataset):
    checks=separator_known_checks(dataset)
    assert checks[0]['result']==pytest.approx(65.1162790698)
    assert checks[1]['result']==pytest.approx(118.75)
    assert checks[2]['result']==pytest.approx(71.462686567)
    assert checks[2]['status']=='source_discrepancy'


def test_evaluation_baselines_are_train_only(dataset,tmp_path):
    result=evaluate_separator_model(dataset,tmp_path)
    mean,nearest,linear,model=result['rows']
    assert mean['prediction_mS_cm']==pytest.approx(.57)
    assert nearest['prediction_mS_cm']==pytest.approx(.71)
    assert mean['absolute_error_mS_cm']==pytest.approx(.27)
    assert nearest['absolute_error_mS_cm']==pytest.approx(.13)
    assert linear['prediction_mS_cm']==pytest.approx(.8966666667)
    assert linear['absolute_error_mS_cm']==pytest.approx(.0566666667)
    assert model['status']=='not_run'
    assert result['experimental_time_saved'] is None


def test_process_uses_default_model_and_rejects_invented_citations(dataset,tmp_path,monkeypatch,model_process):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-key')
    monkeypatch.setenv('AIFORBN_DEMO_TOKEN', 'test-only-token')
    model_process.answer['supporting_record_ids'] = ['invented']
    with pytest.raises(RuntimeError,match='citation_invalid'):
        run_separator_model(assess_separator_recipe(dataset,SeparatorRecipe()),'/fake/codex',tmp_path)
    command = model_process.spawn.call_args.args[0]
    options = model_process.spawn.call_args.kwargs
    assert command[command.index('-m') + 1] == 'gpt-6-astra'
    assert '--ignore-user-config' in command and 'read-only' in command
    assert 'agents.enabled=false' in command and 'features.shell_tool=false' in command
    assert not {'OPENAI_API_KEY', 'AIFORBN_DEMO_TOKEN'} & options['env'].keys()
    assert not list(tmp_path.glob('inference-*'))


@pytest.mark.parametrize('form,loading,allowed',[
    ('raw_BNNT',.01,True),('raw_BNNT',.1,True),('raw_BNNT',.2,True),
    ('raw_BNNT',.3,True),('raw_BNNT',.31,False),('raw_BNNT',.4,False),
    ('purified_BNNT',.01,True),('purified_BNNT',.3,True),
    ('purified_BNNT',.4,True),('purified_BNNT',.5,True),('purified_BNNT',.51,False),
    ('raw_BNNT',0,False),('purified_BNNT',0,False),
])
def test_loading_domain_respects_published_raw_peak(dataset,form,loading,allowed):
    result=assess_separator_recipe(dataset,SeparatorRecipe(bn_form=form,loading_mg_cm2=loading))
    assert result['numeric_allowed'] is allowed


@pytest.mark.parametrize('mode,error', [
    ('success', None),
    ('above_peak', 'model_exceeds_published_raw_bnnt_peak'),
    ('malformed_event', 'model_event_invalid'),
    ('tool_activity', 'unexpected_model_tool_activity'),
    ('incomplete', 'model_turn_incomplete'),
    ('invalid_output', 'model_output_invalid'),
    ('nonfinite_output', 'model_output_invalid'),
])
def test_configurable_provider_preserves_execution_and_result_boundaries(dataset,tmp_path,model_process,mode,error):
    if mode == 'above_peak':
        model_process.answer['conductivity_mS_cm'] = .8
    elif mode == 'malformed_event':
        model_process.events = 'invalid\n'
    elif mode == 'tool_activity':
        model_process.events = json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution'}}) + '\n'
    elif mode == 'incomplete':
        model_process.events = json.dumps({'type': 'turn.started'}) + '\n'
    elif mode == 'invalid_output':
        del model_process.answer['limitations']
    elif mode == 'nonfinite_output':
        model_process.answer['conductivity_mS_cm'] = float('nan')
    assessment=assess_separator_recipe(dataset,SeparatorRecipe())
    if error is None:
        result=run_separator_model(assessment,'/fake/codex',tmp_path,model_name='future-model')
        assert result['requested_model']=='future-model' and result['result']['conductivity_mS_cm']==.6
    else:
        with pytest.raises(RuntimeError, match=f'^{error}$'):
            run_separator_model(assessment,'/fake/codex',tmp_path,model_name='future-model')
    command = model_process.spawn.call_args.args[0]
    assert command[command.index('-m') + 1] == 'future-model'
    assert model_process.spawn.call_count == 1
    assert not list(tmp_path.glob('inference-*'))


@pytest.mark.parametrize('settings,error', [
    ({'model_name': 'bad; shell'}, 'identifier'),
    ({'timeout_seconds': 0}, 'timeout'),
    ({'timeout_seconds': 181}, 'timeout'),
])
def test_invalid_provider_settings_never_start_a_process(dataset,tmp_path,model_process,settings,error):
    with pytest.raises(ValueError, match=error):
        run_separator_model(assess_separator_recipe(dataset,SeparatorRecipe()), '/fake/codex', tmp_path, **settings)
    model_process.spawn.assert_not_called()


def test_provider_start_failure_is_sanitized(dataset,tmp_path,model_process):
    model_process.spawn.side_effect = OSError('test-only private executable path')
    with pytest.raises(RuntimeError, match='^model_start_failed$'):
        run_separator_model(assess_separator_recipe(dataset,SeparatorRecipe()), '/fake/codex', tmp_path)
    assert model_process.spawn.call_count == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode', ['nonzero_exit', 'missing_output', 'oversized_output'])
def test_provider_failure_keeps_bounded_private_diagnostics(dataset,tmp_path,model_process,mode):
    if mode == 'nonzero_exit':
        model_process.process.returncode = 1
    elif mode == 'missing_output':
        model_process.answer = None
    else:
        model_process.answer['preparation_hypothesis'] = 'x' * 16001
    model_process.stderr *= 1000
    with pytest.raises(RuntimeError, match='^model_failed$'):
        run_separator_model(assess_separator_recipe(dataset,SeparatorRecipe()), '/fake/codex', tmp_path)
    diagnostic = tmp_path / 'last_error.log'
    assert diagnostic.read_text() == model_process.stderr[-12000:]
    assert diagnostic.stat().st_mode & 0o777 == 0o600
    assert model_process.spawn.call_count == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ['last_error.log']


@pytest.mark.parametrize('mode', ['timeout', 'force_kill', 'already_exited', 'interrupt'])
def test_interrupted_provider_cleans_process_group_and_scratch(dataset,tmp_path,monkeypatch,model_process,mode):
    process = model_process.process
    cause = KeyboardInterrupt() if mode == 'interrupt' else subprocess.TimeoutExpired('/fake/codex', 7)
    process.communicate.side_effect = cause
    killpg = Mock()
    monkeypatch.setattr('materials.separator_model.os.killpg', killpg)
    if mode == 'force_kill':
        process.wait.side_effect = [subprocess.TimeoutExpired('/fake/codex', 3), None]
        process.poll.return_value = None
    elif mode == 'already_exited':
        killpg.side_effect = ProcessLookupError()
        process.poll.return_value = 0
    error = KeyboardInterrupt if mode == 'interrupt' else RuntimeError
    match = None if mode == 'interrupt' else '^model_timeout$'
    with pytest.raises(error, match=match) as caught:
        run_separator_model(assess_separator_recipe(dataset,SeparatorRecipe()), '/fake/codex', tmp_path, timeout_seconds=7)
    if mode == 'interrupt':
        assert caught.value is cause
    signals = [call(process.pid, signal.SIGTERM)]
    if mode == 'force_kill':
        signals.append(call(process.pid, signal.SIGKILL))
        assert process.wait.call_args_list == [call(timeout=3), call()]
    elif mode == 'already_exited':
        process.wait.assert_not_called()
    else:
        process.wait.assert_called_once_with(timeout=3)
    assert killpg.call_args_list == signals
    assert model_process.spawn.call_count == 1
    assert model_process.spawn.call_args.kwargs['start_new_session'] is True
    assert process.communicate.call_args.kwargs['timeout'] == 7
    assert not list(tmp_path.iterdir())
