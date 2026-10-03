"""Billing isolation covers aliases, midnight retries, and interrupted writers."""
import json
import pytest

from scripts import collection_billing as billing
from scripts.run_billing_queue import ready_jobs


@pytest.fixture
def ledger(tmp_path):
    conn=billing.open_ledger(tmp_path/'ledger.db')
    yield conn
    conn.close()


def test_aliases_and_midnight_reservation(ledger):
    assert billing.billing_group('gpt-5.4-high')==billing.billing_group('gpt-5.4-2026-03-05')
    token=billing.reserve(ledger,'key','gpt-5.4','default','2026-10-02T15:59:59+00:00',123)
    billing.finish(ledger,token,'2026-10-02T16:01:00+00:00')
    assert billing.blockers(ledger,'key','gpt-5.4','high','2026-10-03T12:00:00+08:00')==['default']
    assert not billing.blockers(ledger,'key','gpt-5.4','high','2026-10-04T00:00:00+08:00')
    assert not billing.blockers(ledger,'key','other-model','high','2026-10-03T12:00:00+08:00')


def test_open_batch_fails_closed_and_conflicting_reserve_rolls_back(ledger):
    token=billing.reserve(ledger,'key','model','low','2026-10-02T12:00:00+00:00',123)
    with pytest.raises(ValueError,match='occupied'):
        billing.reserve(ledger,'key','model','high','2026-10-04T12:00:00+00:00',456)
    billing.finish(ledger,token,'2026-10-03T12:00:00+00:00')
    with pytest.raises(ValueError,match='occupied'):
        billing.reserve(ledger,'key','model','high','2026-10-03T12:01:00+00:00',456)
    assert ledger.execute('SELECT count(*) FROM batches').fetchone()[0]==1


def test_other_key_independent_after_drain(ledger):
    token=billing.reserve(ledger,'key-a','model','low','2026-10-02T12:00:00+00:00',1)
    billing.finish(ledger,token,'2026-10-02T12:01:00+00:00')
    assert not billing.blockers(ledger,'key-b','model','high','2026-10-02T12:02:00+00:00')


def test_request_outcome_and_filter_cost_are_not_duplicated(ledger,tmp_path,monkeypatch):
    monkeypatch.setattr(billing,'DIRECTORY',tmp_path)
    for rid,role,profile,model,attempt in [('a','forecast','low','gpt-oss-120b',1),('b','forecast','low','gpt-oss-120b',2),('c','filter','filter-fixed','qwen3.8-flash',1)]:
        billing.register_request(ledger,'db',rid,'key',role,profile,'run','new300',{'model':model},'2026-10-02T15:59:59+00:00',attempt,'event:'+rid)
        billing.register_outcome(ledger,'db',rid,'error' if rid=='a' else 'response',{'model':model,'usage':{'prompt_tokens':5,'completion_tokens':7}},'2026-10-02T16:00:01+00:00')
    ledger.commit()
    result=billing.export(ledger,'key')
    forecast=next(x for x in result['daily_requests'] if x['role']=='forecast')
    filtering=next(x for x in result['daily_requests'] if x['role']=='filter')
    assert forecast['requests']==2 and forecast['errors']==1 and forecast['retries']==1
    assert forecast['cross_day_requests']==2 and forecast['actual_cost'] is None
    assert filtering['requests']==1 and filtering['prompt_tokens']==5
    assert billing.blockers(ledger,'key','gpt-oss-120b','high','2026-10-03T08:00:00+08:00')==['low']


def test_ready_jobs_preserve_tiers_and_fill_other_groups(ledger):
    jobs=[dict(profile_id=n,billing_group=g,tier=t,not_before_billing_date='2026-10-03',expected_samples=3) for n,g,t in [('low','model',0),('other','other',0),('high','model',1)]]
    spec={'key_alias':'key','jobs':jobs};counts={n:{'completed':0,'refusals':0} for n in ('low','other','high')}
    billing.add_claims(ledger,'key','model','default','2026-10-03T01:00:00+08:00','2026-10-03T01:00:01+08:00','test');ledger.commit()
    assert [x['profile_id'] for x in ready_jobs(spec,ledger,'2026-10-03T08:00:00+08:00',counts)]==['other']
    counts['low']['completed']=3;counts['other']['refusals']=3
    assert [x['profile_id'] for x in ready_jobs(spec,ledger,'2026-10-04T00:00:00+08:00',counts)]==['high']


def test_semantic_deduplication_and_holds_never_become_observations(ledger):
    jobs=[dict(profile_id=n,billing_group=n,tier=0,not_before_billing_date='2026-10-03',expected_samples=3,
               dispatch_status=s) for n,s in [('off','runnable'),('on','deduplicated'),('unknown','semantic_verification_required')]]
    spec={'key_alias':'key','jobs':jobs}
    counts={n:{'completed':0,'refusals':0} for n in ('off','on','unknown')}
    assert [x['profile_id'] for x in ready_jobs(spec,ledger,'2026-10-03T08:00:00+08:00',counts)]==['off']
    coverage={n:{'expected':3,'collected':0,'refused':0} for n in counts}
    assert not billing.scope_coverage(spec,coverage)['attempts_complete']
    coverage['off'].update(collected=2,refused=1)
    result=billing.scope_coverage(spec,coverage)
    assert result['attempts_complete'] and result['expected']==3 and result['collected']==2
    assert [x['dispatch_status'] for x in result['uncollected_profiles']]==['deduplicated','semantic_verification_required']
    assert coverage['on']['collected']==coverage['unknown']['collected']==0
    counts['off'].update(completed=2,refusals=1)
    assert ready_jobs(spec,ledger,'2026-10-04T08:00:00+08:00',counts)==[]
    coverage['off']['expected']=4
    with pytest.raises(ValueError,match='target differ'):billing.scope_coverage(spec,coverage)
    jobs[0]['dispatch_status']='typo'
    with pytest.raises(ValueError,match='Unknown billing'):billing.runnable_jobs(spec)


def test_journal_incremental_sync_retains_midnight_outcomes(ledger,tmp_path):
    import sqlite3
    path=tmp_path/'raw.db';raw=sqlite3.connect(path)
    raw.execute('CREATE TABLE request_events(event_id INTEGER,run_id TEXT,model TEXT,request_id TEXT,kind TEXT,created_at TEXT,payload TEXT)')
    raw.execute('INSERT INTO request_events VALUES (?,?,?,?,?,?,?)',(1,'run','arm::r5::c4','r','llm.request','2026-10-02T15:59:59+00:00',json.dumps({'body':{'model':'model'},'attempt':1})))
    raw.commit();billing.sync_database(ledger,path,'key','new300')
    raw.execute('INSERT INTO request_events VALUES (?,?,?,?,?,?,?)',(2,'run','arm::r5::c4','r','llm.response','2026-10-02T16:00:01+00:00',json.dumps({'body':json.dumps({'model':'model','usage':{'prompt_tokens':2}})})))
    raw.commit();raw.close()
    billing.sync_database(ledger,path,'key','new300');billing.sync_database(ledger,path,'key','new300')
    assert ledger.execute('SELECT count(*) FROM requests').fetchone()[0]==1
    assert ledger.execute('SELECT outcome FROM requests').fetchone()[0]=='response'
    assert billing.blockers(ledger,'key','model','other','2026-10-03T10:00:00+08:00')==['arm']


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal_failure',[False,True])
async def test_queue_drains_one_batch_and_stops_on_terminal_failure(ledger,tmp_path,monkeypatch,terminal_failure):
    import asyncio
    import os
    import signal
    from types import SimpleNamespace
    from scripts import run_billing_queue as queue
    collection=tmp_path/'collection';collection.mkdir()
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    monkeypatch.setattr(queue,'COLLECTION',collection)
    monkeypatch.setattr(queue,'STATE',collection/'state.json')
    monkeypatch.setattr(queue,'digest',lambda p:'hash')
    monkeypatch.setattr(queue,'prior_ready',lambda:True)
    monkeypatch.setattr(queue,'verify_prior_catalog',lambda:None)
    monkeypatch.setattr(queue,'check_external',lambda:None)
    monkeypatch.setattr(queue,'excluded_profiles',lambda p:set())
    monkeypatch.setattr(queue,'capture_execution',lambda p:None)
    monkeypatch.setattr(billing,'sync_all',lambda c,k:None)
    monkeypatch.setattr(billing,'export',lambda c,k:{})
    monkeypatch.setattr(queue.db,'snapshot_settings',lambda s:{})
    monkeypatch.setattr(queue.shutil,'disk_usage',lambda p:SimpleNamespace(free=20*1024**3))
    cfg=SimpleNamespace(TAVILY_API_KEY=['test'],COLLECTION_BATCH_SAMPLES=240,COLLECTION_EXPENSIVE_CONCURRENCY=2,LLM_MAX_CONCURRENCY=20)
    cfg.model_copy=lambda update:SimpleNamespace(**{**cfg.__dict__,**update})
    monkeypatch.setattr(queue,'validate',lambda *args:cfg)
    async def quota(*args):return 1000
    monkeypatch.setattr(queue,'available_searches',quota)
    observed={'completed':0,'refusals':0};events=[];calls=[]
    monkeypatch.setattr(queue,'status',lambda p:{'profiles':{'arm':dict(observed)}})
    monkeypatch.setattr(queue,'record_dispatch',lambda p,e,**fields:events.append(e))
    async def evaluate(settings,*args,**kwargs):
        calls.append(settings.COLLECTION_SAMPLE_LIMIT)
        assert kwargs['skip_analysis'] is True
        if terminal_failure:return 4
        os.kill(os.getpid(),signal.SIGTERM)
        await asyncio.sleep(0.01)
        observed['completed']=3
        return 6
    monkeypatch.setattr(queue.evaluation,'_run_async',evaluate)
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *args,**kwargs:SimpleNamespace(pid=999999,wait=lambda:0))
    plan={'run_id':'test','runtime':{},'status':'prepared'}
    spec={'key_alias':'test','plan_path':'plan.json','jobs':[{'profile_id':'arm','requested_model':'gpt-5.4','billing_group':'gpt-5.4','tier':0,'not_before_billing_date':'2026-01-01','expected_samples':6}]}
    code=await queue.run(spec,plan,{},ledger)
    saved=json.loads((collection/'state.json').read_text())
    assert calls==[3] and events==['start','finish']
    assert saved['status']==('blocked' if terminal_failure else 'stopped_at_batch_boundary')
    assert code==(4 if terminal_failure else 0)
    assert ledger.execute('SELECT count(*) FROM batches WHERE ended_at IS NULL').fetchone()[0]==0


def test_historical_usage_preserves_dates_and_idempotence(ledger,tmp_path,monkeypatch):
    import sqlite3
    path=tmp_path/'historical.db';raw=sqlite3.connect(path)
    raw.execute('CREATE TABLE run_meta(run_id TEXT,model TEXT,sampling_n INTEGER)')
    raw.execute("INSERT INTO run_meta VALUES ('original','gpt-5.4::r5::c4',1)")
    raw.execute('CREATE TABLE run_results(question_id TEXT,s0_created_at TEXT,s0_error TEXT,s0_prompt_tokens INTEGER,s0_completion_tokens INTEGER)')
    raw.executemany('INSERT INTO run_results VALUES (?,?,?,?,?)',[
        ('b','2026-05-06T16:00:01+00:00',None,5,7),
        ('a','2026-05-06T15:59:59+00:00','unknown',3,4)])
    raw.commit();raw.close()
    billing.sync_database(ledger,path,'key','original80')
    billing.sync_database(ledger,path,'key','original80')
    assert ledger.execute('SELECT count(*) FROM samples').fetchone()[0]==2
    assert ledger.execute('SELECT count(*) FROM historical_usage').fetchone()[0]==2
    monkeypatch.setattr(billing,'DIRECTORY',tmp_path)
    result=billing.export(ledger,'key')
    assert result['gaps'] and not result['daily_requests']
    assert {x['billing_date'] for x in result['sample_observations']}=={'2026-05-06','2026-05-07'}
    assert all(x['reasoning_tokens'] is None for x in result['historical_sample_usage'])
    assert sum(x['prompt_tokens'] for x in result['historical_sample_usage'])==8


def test_search_retry_count_remains_unknown(ledger,tmp_path,monkeypatch):
    monkeypatch.setattr(billing,'DIRECTORY',tmp_path)
    billing.register_request(ledger,'raw','search','tavily-pool','search','arm','run','new300',{},'2026-10-03T01:00:00+00:00',1,'event')
    ledger.commit()
    record=billing.export(ledger,'key')['daily_requests'][0]
    assert record['requests']==1 and record['retries'] is None
    assert record['actual_cost'] is None and record['pending']==1
