"""Dispatch authorized collection arms on separate billing days, after the active queue drains."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from dotenv import dotenv_values
from loguru import logger
import evaluation
from forecast_eval import db
from forecast_eval.types import QFilter
from scripts.prepare_collection import local,digest
from scripts.collect_forecast_panel import (available_searches,capture_execution,dispatch_settings,
    excluded_profiles,expensive,record_dispatch,sample_target,status,write_json)
from scripts import collection_billing as billing

COLLECTION=local(ROOT/'runs/collection_300')
SPEC=local(COLLECTION/'billing_schedule.json')
STATE=local(COLLECTION/'billing_schedule_state.json')


def now(): return datetime.now(timezone.utc).isoformat()


def read(path): return json.loads(local(path).read_text())


def key_alias(env): return 'aihubmix-'+hashlib.sha256(env['LLM_API_KEY'].encode()).hexdigest()[:16]


def ready_jobs(spec,ledger,at,counts):
    pending=[job for job in spec['jobs'] if counts[job['profile_id']]['completed']+counts[job['profile_id']].get('refusals',0)<job['expected_samples']]
    if not pending:return []
    tier=min(job['tier'] for job in pending)
    day=billing.timestamp(at).astimezone(billing.ZONE).date().isoformat()
    return [job for job in pending if job['tier']==tier and job['not_before_billing_date']<=day
            and not billing.blockers(ledger,spec['key_alias'],job['billing_group'],job['profile_id'],at)]


def validate(spec,plan,env):
    if spec['key_alias']!=key_alias(env):raise ValueError('Configured API key fingerprint changed')
    if spec['key_alias']!=key_alias(dotenv_values(local(ROOT/'.env'))):raise ValueError('Current API key differs from the scheduled key')
    if spec['billing_timezone']!='Asia/Shanghai' or spec['day_boundary']!='00:00':raise ValueError('Billing timezone is not confirmed')
    if digest(local(ROOT/plan['runtime']['SOURCE_DB']))!=plan['source_sha256']:raise ValueError('Source changed')
    if {job['profile_id'] for job in spec['jobs']}!=set(plan['runtime']['MODELS']):raise ValueError('Queue differs from authorized scope')
    settings=dispatch_settings(plan,env)
    if db.compute_collection_contract_hash(db.snapshot_settings(settings))!=spec['inference_contract_hash']:raise ValueError('Inference contract changed')
    if not (settings.SAMPLING_N==3 and settings.REACT_MAX_STEPS==6 and settings.REACT_MAX_SEARCH_CALLS==[4]
            and settings.TAVILY_MAX_RESULTS==[5] and settings.WRITE_REQUEST_AUDIT and settings.WRITE_MESSAGES_TRACE
            and settings.COLLECTION_DRAIN_ON_ERROR and not settings.SCORE_ANSWERS):raise ValueError('Collection protocol differs')
    probes=read(ROOT/'logs/collection_300/profile_probes.json')
    for job in spec['jobs']:
        name=job['profile_id']
        if sample_target(plan['runtime'],name)!=job['expected_samples']:raise ValueError('Sample scope changed')
        probe=probes.get(name,{})
        if not probe.get('ok') or probe.get('profile')!=plan['runtime']['MODEL_PROFILES'][name]:raise ValueError('Unverified profile '+name)
        if job['billing_group']!=billing.billing_group(plan['runtime']['MODEL_PROFILES'][name]['model']):raise ValueError('Billing alias mapping changed')
    return settings


def prior_ready():
    state=read(COLLECTION/'authorized_sequence_state.json')
    if state.get('status')!='selected_collection_complete' or state.get('catalog_exit')!=0 or not state.get('finished_at'):
        return False
    # The caller also holds authorized_sequence.lock; state alone cannot establish exit.
    if state.get('child_pid') is not None:return False
    process=subprocess.run(['ps','-p',str(state.get('pid',0)),'-o','args='],capture_output=True,text=True)
    if 'run_authorized_sequence.py' in process.stdout:return False
    return True


def verify_prior_catalog():
    catalog=read(COLLECTION/'catalog.json');spec=read(COLLECTION/'authorized_sequence.json')
    state=read(COLLECTION/'authorized_sequence_state.json')
    if billing.timestamp(catalog['built_at'])<billing.timestamp(state['collection_finished_at']):raise ValueError('Prior catalog is stale')
    for phase in spec['phases']:
        coverage=next(x for x in catalog['coverage'] if x['phase']==phase['phase'])
        for name in phase['selected_profiles']:
            row=coverage['profiles'][name]
            if row['collected']+row['refused']!=row['expected']:raise ValueError('Prior authorized profile has unresolved slots')
    archive=local(ROOT/catalog['observations_path'])
    if digest(archive)!=catalog['observations_sha256']:raise ValueError('Prior observation archive hash mismatch')
    sources={}
    selected={n for phase in spec['phases'] for n in phase['selected_profiles']}
    for source in catalog['sources']:
        if source['profile_id'] not in selected or source['cohort']=='reference':continue
        path=local(ROOT/source['path'])
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as connection:
            connection.row_factory=sqlite3.Row
            metadata=dict(connection.execute('SELECT * FROM run_meta').fetchone())
            high=connection.execute('SELECT coalesce(max(event_id),0) FROM request_events').fetchone()[0]
            if metadata!=source['metadata'] or high!=source['journal']['event_id_through']:
                raise ValueError('Prior catalog source snapshot changed: '+source['path'])
        sources[source['path']]=digest(path)
        if source.get('expected_sha256') and sources[source['path']]!=source['expected_sha256']:
            raise ValueError('Prior source hash mismatch: '+source['path'])
    write_json(local(ROOT/'logs/collection_300/billing_predecessor_catalog.json'),
               {'verified_at':now(),'catalog_sha256':digest(COLLECTION/'catalog.json'),
                'observations_sha256':catalog['observations_sha256'],'source_sha256':sources})


def check_external():
    registry=read(COLLECTION/'delegated_results.json')
    for item in registry['profiles']:
        if local(ROOT/item['expected_result_db']).exists():
            raise ValueError('External return requires verified intake before further collection: '+item['profile_id'])


async def run(spec,plan,env,ledger):
    stop=asyncio.Event(); loop=asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM,stop.set);loop.add_signal_handler(signal.SIGINT,stop.set)
    state={'status':'waiting_for_original_queue','pid':os.getpid(),'started_at':now(),'schedule_sha256':digest(SPEC)}
    collected=False
    predecessor_verified=False
    def save(**fields):
        state.update(fields,updated_at=now());write_json(STATE,state)
    save()
    try:
        while not stop.is_set():
            prior_lock=local(COLLECTION/'authorized_sequence.lock').open('a+')
            try:fcntl.flock(prior_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                prior_lock.close()
                try:await asyncio.wait_for(stop.wait(),30)
                except asyncio.TimeoutError:pass
                continue
            if not prior_ready():
                prior_lock.close();save(status='waiting_for_original_queue')
                try:await asyncio.wait_for(stop.wait(),30)
                except asyncio.TimeoutError:pass
                continue
            with prior_lock:
                with local(COLLECTION/'collection.lock').open('a+') as collect_lock:
                    fcntl.flock(collect_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    check_external()
                    if not predecessor_verified:
                        verify_prior_catalog();predecessor_verified=True
                    validate(spec,plan,env)
                    billing.sync_all(ledger,spec['key_alias']); billing.export(ledger,spec['key_alias'])
                    counts=status(plan)['profiles']; ready=ready_jobs(spec,ledger,now(),counts)
                    pending=sum(job['expected_samples']-counts[job['profile_id']]['completed']-counts[job['profile_id']].get('refusals',0) for job in spec['jobs'])
                    if pending<0:raise ValueError('Observed slots exceed the authorized target')
                    save(counts=counts,pending=pending)
                    if not pending:
                        save(status='collection_complete_pending_catalog');break
                    if not ready:
                        save(status='waiting_for_billing_day',active_profile=None)
                    else:
                        job=ready[0];name=job['profile_id']
                        if name in excluded_profiles(plan):raise ValueError('Selected profile is locally excluded: '+name)
                        settings=validate(spec,plan,env)
                        if shutil.disk_usage(ROOT).free<10*1024**3:raise ValueError('Disk reserve reached')
                        remaining=await available_searches(settings.TAVILY_API_KEY,plan)
                        done=counts[name]['completed']+counts[name].get('refusals',0)
                        costly=expensive(job['requested_model']) or job['requested_model'].startswith('gemini-3.1-pro')
                        cap=min(3 if done==0 else 15 if costly else settings.COLLECTION_BATCH_SAMPLES,
                                job['expected_samples']-done,max(0,(remaining-100)//12*3))
                        if cap<min(3,job['expected_samples']-done):raise ValueError(f'Tavily reserve reached: {remaining} cached/queried estimate')
                        if stop.is_set():break
                        # The date may have changed during quota checks.
                        if billing.blockers(ledger,spec['key_alias'],job['billing_group'],name,now()):continue
                        token=billing.reserve(ledger,spec['key_alias'],job['billing_group'],name,now(),os.getpid())
                        collected=True
                        settings=settings.model_copy(update={'COLLECTION_MODEL':name,'COLLECTION_SAMPLE_LIMIT':cap,
                            'LLM_MAX_CONCURRENCY':settings.COLLECTION_EXPENSIVE_CONCURRENCY if costly else settings.LLM_MAX_CONCURRENCY})
                        directory=local(ROOT/'runs'/plan['run_id']);directory.mkdir(parents=True,exist_ok=True)
                        capture_execution(directory)
                        save(status='running',active_profile=name,batch_limit=cap,tavily_estimate=remaining,reservation=token)
                        record_dispatch(plan,'start',profile=name,sample_limit=cap,config_snapshot=db.snapshot_settings(settings),
                                        billing_reservation=token,key_alias=spec['key_alias'],billing_group=job['billing_group'])
                        # SIGTERM sets stop; the bounded batch drains without cancelling its requests.
                        try:
                            code=await evaluation._run_async(settings,QFilter(),plan['run_id'],directory,skip_analysis=True)
                        finally:
                            billing.sync_all(ledger,spec['key_alias'])
                            billing.finish(ledger,token,now());billing.export(ledger,spec['key_alias'])
                        after=status(plan)['profiles'][name]
                        record_dispatch(plan,'finish',profile=name,exit_code=code,counts=after,billing_reservation=token)
                        if code not in (0,6):raise ValueError(f'evaluation exit {code}; inspect raw journal; no automatic retry')
                        if after['completed']+after.get('refusals',0)<=done:raise ValueError('No collection progress')
                        plan.update(status='running',active_profile=name,updated_at=now());write_json(local(ROOT/spec['plan_path']),plan)
            if state['status']=='waiting_for_billing_day':
                try:await asyncio.wait_for(stop.wait(),60)
                except asyncio.TimeoutError:pass
        if stop.is_set():save(status='stopped_at_batch_boundary')
    except Exception as exc:
        message=str(exc)
        for secret in (env.get('LLM_API_KEY'),env.get('LEAK_DETECTOR_API_KEY')):
            if secret:message=message.replace(secret,'<redacted>')
        save(status='blocked',block_reason=message,stopped_at=now());logger.error('Billing queue stopped: {}',message)
    finally:
        loop.remove_signal_handler(signal.SIGTERM);loop.remove_signal_handler(signal.SIGINT)
    if collected or state['status']=='collection_complete_pending_catalog':
        # Keep both writer locks while the catalog takes its final source snapshot.
        with local(COLLECTION/'authorized_sequence.lock').open('a+') as a, local(COLLECTION/'collection.lock').open('a+') as b:
            fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
            process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B','scripts/catalog_collection.py'],cwd=ROOT)
            save(catalog_pid=process.pid)
            code=await asyncio.to_thread(process.wait)
            save(catalog_pid=None,catalog_exit=code,finished_at=now())
            if code==0 and state['status']=='collection_complete_pending_catalog':
                catalog=read(COLLECTION/'catalog.json')
                coverage=next(row for row in catalog['coverage'] if row['phase']=='billing_extension')
                if not coverage['attempts_complete']:raise ValueError('Catalog has unresolved target slots')
                save(status='complete')
    return 4 if state['status']=='blocked' else 0


def main():
    if ROOT.resolve()!=Path('/Users/mayiding/Desktop/Git/Forecast'):raise ValueError('Unexpected project root')
    os.environ['NO_PROXY']=os.environ['no_proxy']='localhost,127.0.0.1'
    spec=read(SPEC);plan=read(ROOT/spec['plan_path']);env=dotenv_values(local(ROOT/'.env'))
    validate(spec,plan,env)
    if '--validate-only' in sys.argv:
        logger.info('Validated {} billing-isolated profiles; no network calls',len(spec['jobs']));return 0
    with local(COLLECTION/'billing_schedule.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=billing.open_ledger()
        if '--ledger-only' in sys.argv:
            billing.sync_all(ledger,spec['key_alias']);result=billing.export(ledger,spec['key_alias'])
            logger.info('Updated {} daily records and {} conflict days',len(result['daily_requests']),len(result['conflicts']));return 0
        directory=local(ROOT/'runs'/plan['run_id']/'billing_code');directory.mkdir(parents=True,exist_ok=True)
        evidence={}
        for name in ('scripts/run_billing_queue.py','scripts/collection_billing.py'):
            source=local(ROOT/name);sha=digest(source);target=local(directory/(sha+'--'+source.name))
            if not target.exists():shutil.copyfile(source,target)
            if digest(target)!=sha:raise ValueError('Billing code capture mismatch')
            evidence[name]={'sha256':sha,'copy':str(target.relative_to(ROOT))}
        write_json(local(directory/'manifest.json'),{'captured_at':now(),'files':evidence,'schedule_sha256':digest(SPEC)})
        return asyncio.run(run(spec,plan,env,ledger))


if __name__=='__main__':raise SystemExit(main())
