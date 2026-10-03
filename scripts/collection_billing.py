"""Retain observed request dates and enforce one arm per billing model and day."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.prepare_collection import local, ROOT

ZONE = ZoneInfo("Asia/Shanghai")
DIRECTORY = local(ROOT / "runs/collection_300/billing")


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("A billing timestamp must include its timezone")
    return dt


def billing_days(start, end):
    first, last = timestamp(start).astimezone(ZONE).date(), timestamp(end).astimezone(ZONE).date()
    if last < first:
        raise ValueError("Request ended before it started")
    return [(first + timedelta(days=i)).isoformat() for i in range((last-first).days+1)]


def billing_group(model):
    value = model.lower()
    for prefix, group in [
        ("gpt-5.4", "gpt-5.4"), ("gpt-5.3-codex", "gpt-5.3-codex"),
        ("gemini-3.1-pro", "gemini-3.1-pro"), ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite"),
        ("claude-opus-4-6", "claude-opus-4-6"), ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("doubao-seed-2-0-lite", "doubao-seed-2-0-lite"),
        ("alicloud-kimi-k2.5", "kimi-k2.5"), ("kimi-k2.5", "kimi-k2.5"),
        ("qwen3.5-flash", "qwen3.5-flash"), ("qwen3.5-plus", "qwen3.5-plus"),
        ("qwen3.5-35b-a3b", "qwen3.5-35b-a3b"), ("qwen3.8-flash", "qwen3.8-flash"),
        ("glm-5", "glm-5"), ("gpt-oss-120b", "gpt-oss-120b"), ("minimax-m2.5", "minimax-m2.5"),
    ]:
        if value.startswith(prefix):
            return group
    return value


def open_ledger(path=None):
    target = local(path or DIRECTORY / "requests.sqlite3")
    target.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(target)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS requests (
      source TEXT, request_id TEXT, key_alias TEXT, role TEXT, profile TEXT, run_id TEXT,
      cohort TEXT, requested_model TEXT, returned_model TEXT, billing_group TEXT,
      started_at TEXT, ended_at TEXT, provider_created TEXT, attempt INTEGER,
      outcome TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER,
      usage_json TEXT, evidence TEXT, PRIMARY KEY(source, request_id));
    CREATE TABLE IF NOT EXISTS samples (
      source TEXT, event_id INTEGER, profile TEXT, run_id TEXT, cohort TEXT, at TEXT,
      outcome TEXT, PRIMARY KEY(source,event_id));
    CREATE TABLE IF NOT EXISTS cursors (source TEXT PRIMARY KEY, through INTEGER);
    CREATE TABLE IF NOT EXISTS claims (
      key_alias TEXT, billing_group TEXT, day TEXT, profile TEXT, evidence TEXT,
      PRIMARY KEY(key_alias,billing_group,day,profile,evidence));
    CREATE TABLE IF NOT EXISTS batches (
      id TEXT PRIMARY KEY, key_alias TEXT, billing_group TEXT, profile TEXT,
      started_at TEXT, ended_at TEXT, pid INTEGER);
    CREATE TABLE IF NOT EXISTS gaps (source TEXT PRIMARY KEY, detail TEXT);
    CREATE TABLE IF NOT EXISTS historical_usage (
      source TEXT, question_id TEXT, sample_idx INTEGER, profile TEXT, at TEXT,
      prompt_tokens INTEGER,completion_tokens INTEGER,reasoning_tokens INTEGER,
      PRIMARY KEY(source,question_id,sample_idx));
    CREATE INDEX IF NOT EXISTS claim_day ON claims(billing_group,day);
    """)
    return c


def add_claims(c, key, group, profile, start, end, evidence):
    c.executemany("INSERT OR IGNORE INTO claims VALUES (?,?,?,?,?)",
                  [(key, group, day, profile, evidence) for day in billing_days(start, end)])


def blockers(c, key, group, profile, at):
    day = timestamp(at).astimezone(ZONE).date().isoformat()
    conflicts = c.execute("SELECT DISTINCT profile FROM claims WHERE key_alias=? AND billing_group=? AND day=? AND profile!=?",
                          (key, group, day, profile)).fetchall()
    open_batches = c.execute("SELECT profile FROM batches WHERE ended_at IS NULL").fetchall()
    return sorted({row[0] for row in conflicts + open_batches})


def reserve(c, key, group, profile, at, pid):
    c.execute("BEGIN IMMEDIATE")
    try:
        conflict = blockers(c, key, group, profile, at)
        if conflict:
            raise ValueError("Billing day occupied: " + ", ".join(conflict))
        identifier = hashlib.sha256(f"{key}/{group}/{profile}/{at}/{pid}".encode()).hexdigest()
        c.execute("INSERT INTO batches VALUES (?,?,?,?,?,?,?)", (identifier,key,group,profile,at,None,pid))
        add_claims(c,key,group,profile,at,at,"batch:"+identifier)
        c.commit()
        return identifier
    except BaseException:
        c.rollback()
        raise


def finish(c, identifier, at):
    c.execute("BEGIN IMMEDIATE")
    try:
        row = c.execute("SELECT * FROM batches WHERE id=?", (identifier,)).fetchone()
        if not row or row['ended_at'] is not None:
            raise ValueError("No open billing reservation")
        add_claims(c,row['key_alias'],row['billing_group'],row['profile'],row['started_at'],at,"batch:"+identifier)
        c.execute("UPDATE batches SET ended_at=? WHERE id=?", (at,identifier))
        c.commit()
    except BaseException:
        c.rollback()
        raise


def decoded(value):
    if isinstance(value,str):
        try: return json.loads(value)
        except (ValueError,TypeError): return {}
    return value if isinstance(value,dict) else {}


def arm_from_request(body, fallback, role):
    if role == "forecast" and not fallback.startswith("probe"):
        return fallback.split("::")[0]
    params = {k: body[k] for k in ("model","reasoning_effort","reasoning","thinking","output_config",
              "max_tokens","max_completion_tokens","temperature","top_p","response_format") if k in body}
    sig = hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:16]
    return role + ":" + body.get("model","unknown") + ":" + sig


def register_request(c, source, rid, key, role, profile, run_id, cohort, body, at, attempt, evidence):
    model=body.get("model", "tavily" if role=="search" else "unknown")
    group=billing_group(model)
    c.execute("INSERT OR IGNORE INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (source,rid,key,role,profile,run_id,cohort,model,None,group,at,None,None,attempt,"pending",None,None,None,None,evidence))
    if role != "search": add_claims(c,key,group,profile,at,at,evidence)


def register_outcome(c,source,rid,outcome,body,at):
    row=c.execute("SELECT * FROM requests WHERE source=? AND request_id=?",(source,rid)).fetchone()
    if row is None: return
    body=decoded(body); usage=body.get("usage") or {}; details=usage.get("completion_tokens_details") or {}
    returned=body.get("model"); created=body.get("created")
    c.execute("""UPDATE requests SET ended_at=?, outcome=?, returned_model=?,provider_created=?,
      prompt_tokens=?,completion_tokens=?,reasoning_tokens=?,usage_json=? WHERE source=? AND request_id=?""",
      (at,outcome,returned,str(created) if created is not None else None,
       usage.get("prompt_tokens",usage.get("input_tokens")), usage.get("completion_tokens",usage.get("output_tokens")),
       details.get("reasoning_tokens",usage.get("reasoning_tokens")),json.dumps(usage) if usage else None,source,rid))
    if row['role']!='search':
        groups={row['billing_group']}
        if returned: groups.add(billing_group(returned))
        for group in groups: add_claims(c,row['key_alias'],group,row['profile'],row['started_at'],at,row['evidence'])


def sync_database(c, path, key, cohort):
    path=local(path); source=str(path.relative_to(ROOT))
    ro=sqlite3.connect(path.as_uri()+"?mode=ro",uri=True); ro.row_factory=sqlite3.Row
    tables={row[0] for row in ro.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'request_events' not in tables:
        c.execute("INSERT OR REPLACE INTO gaps VALUES (?,?)", (source,"No per-request journal; historical retries, key identity and daily cost require original invoice/logs."))
        if 'run_meta' in tables and 'run_results' in tables:
            meta=ro.execute("SELECT run_id,model,sampling_n FROM run_meta LIMIT 1").fetchone()
            profile=meta['model'].split('::')[0]+'--provider-default'
            columns={row[1] for row in ro.execute('PRAGMA table_info(run_results)')}
            c.execute("DELETE FROM samples WHERE source=?", (source,))
            c.execute("DELETE FROM historical_usage WHERE source=?", (source,))
            for i in range(meta['sampling_n']):
                usage_columns=[f's{i}_{field}' if f's{i}_{field}' in columns else 'NULL'
                               for field in ('prompt_tokens','completion_tokens','reasoning_tokens')]
                for n,row in enumerate(ro.execute(f"SELECT question_id,s{i}_created_at,s{i}_error,{','.join(usage_columns)} FROM run_results WHERE s{i}_created_at IS NOT NULL ORDER BY question_id")):
                    at=row[1]
                    if timestamp(at).tzinfo is None: continue
                    c.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?)",(source,n*10+i,profile,meta['run_id'],cohort,at,'success' if not row[2] else 'failure'))
                    c.execute("INSERT OR REPLACE INTO historical_usage VALUES (?,?,?,?,?,?,?,?)",(source,row[0],i,profile,at,*tuple(row)[3:]))
                    add_claims(c,key,billing_group(meta['model'].split('::')[0]),profile,at,at,source+':sample-time-only')
        ro.close();c.commit();return
    cur=c.execute("SELECT through FROM cursors WHERE source=?",(source,)).fetchone(); after=cur[0] if cur else 0
    high=ro.execute("SELECT coalesce(max(event_id),0) FROM request_events").fetchone()[0]
    if high<after: raise ValueError("Request journal cursor regressed: "+source)
    # Search response pages and sample traces stay in their raw databases.
    query="""SELECT event_id,run_id,model,request_id,kind,created_at,
      CASE WHEN kind='sample.result' THEN json_object('error',json_extract(payload,'$.error'))
           WHEN kind LIKE 'search.%' THEN '{}' ELSE payload END AS payload
      FROM request_events WHERE event_id>? AND event_id<=? AND
      (kind LIKE 'llm.%' OR kind IN ('detector.request','detector.response','detector.error','detector.cancelled',
       'search.request','search.response','search.error','search.cancelled','sample.result')) ORDER BY event_id"""
    for row in ro.execute(query,(after,high)):
        d=decoded(row['payload']); kind=row['kind']; at=row['created_at']; evidence=source+':event:'+str(row['event_id'])
        fallback=row['model']; role='forecast' if kind.startswith('llm.') else 'filter' if kind.startswith('detector.') else 'search'
        observed_cohort='additional220' if cohort=='new300' and fallback.startswith('gpt-5.4-high--provider-default') else cohort
        is_probe='probe' in row['run_id'] or cohort=='probe'
        if kind=='sample.result':
            err=d.get('error'); outcome='refusal' if err=='content_policy' else 'failure' if err else 'success'
            c.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?)",(source,row['event_id'],fallback.split('::')[0],row['run_id'],observed_cohort,at,outcome))
        elif kind.endswith('.request'):
            body=decoded(d.get('body')); actual_role='probe' if is_probe and role=='forecast' else role
            profile=arm_from_request(body,'probe' if is_probe else fallback,actual_role)
            register_request(c,source,row['request_id'],'tavily-pool' if role=='search' else key,actual_role,profile,row['run_id'],observed_cohort,body,at,d.get('attempt',1),evidence)
        else: register_outcome(c,source,row['request_id'],kind.split('.')[1],d.get('body'),at)
    ro.close()
    c.execute("INSERT OR REPLACE INTO cursors VALUES (?,?)",(source,high)); c.commit()


def source_databases():
    collection=local(ROOT/'runs/collection_300'); found={}
    inventory=json.loads(local(collection/'inventory.json').read_text())
    for item in inventory['reference_inventory']: found[local(ROOT/item['copy_path'])]='original80'
    for phase in ('continuation','reasoning','repair','billing_extension'):
        p=local(collection/(phase+'.json'))
        if not p.exists():continue
        plan=json.loads(p.read_text()); ids={plan['run_id'],*plan.get('prior_run_ids',[])}
        for rid in ids:
            directory=local(ROOT/'runs'/rid/'db')
            for f in directory.glob('*.db'): found[local(f)]='additional220' if phase=='continuation' else 'new300' if phase in ('reasoning','billing_extension') else 'repair'
        for paths in plan['runtime'].get('COLLECTION_REFERENCE_DBS',{}).values():
            for ref in paths:found.setdefault(local(ROOT/ref),phase)
    for name in ('probe.db','detector_format_probe.db'): found[local(ROOT/'logs/collection_300'/name)]='probe'
    return {p:cohort for p,cohort in found.items() if p.exists()}


def sync_all(c,key):
    for path,cohort in source_databases().items(): sync_database(c,path,key,cohort)
    directory=local(ROOT/'logs/aihubmix_probe_20260929')
    for path in sorted(directory.glob('*.result.json')):
        path=local(path); source=str(path.relative_to(ROOT))
        if c.execute("SELECT 1 FROM cursors WHERE source=?",(source,)).fetchone():continue
        d=json.loads(path.read_text()); at=d.get('started_at_utc'); body=decoded(d.get('request'))
        if not at or not body.get('model'):continue
        profile=arm_from_request(body,'probe','probe')
        register_request(c,source,'probe',key,'probe',profile,'aihubmix_probe_20260929','probe',body,at,1,source)
        end=(timestamp(at)+timedelta(seconds=d.get('latency_s') or 0)).isoformat()
        register_outcome(c,source,'probe','response' if d.get('http_status')==200 else 'error',d.get('response'),end)
        c.execute("INSERT INTO cursors VALUES (?,?)",(source,1))
    c.commit()


def export(c,key):
    records={}
    for row in c.execute("SELECT * FROM requests ORDER BY started_at"):
        day=timestamp(row['started_at']).astimezone(ZONE).date().isoformat()
        identity=(row['key_alias'],row['role'],row['profile'],row['billing_group'],day,row['cohort'])
        record=records.setdefault(identity,dict(zip(('key_alias','role','profile','billing_group','billing_date','cohort'),identity)))
        if 'requests' not in record:
            record.update(requests=0,responses=0,errors=0,retries=0,pending=0,prompt_tokens=0,completion_tokens=0,reasoning_tokens=0,
                          usage_observed=0,first_request=row['started_at'],last_request=row['started_at'],last_outcome=None,
                          requested_models=[],returned_models=[],run_ids=[],cross_day_requests=0,actual_cost=None,invoice_source=None)
        record['requests']+=1; record['last_request']=row['started_at']
        if row['role']=='search':record['retries']=None
        else:record['retries']+=int((row['attempt'] or 1)>1)
        record['responses']+=int(row['outcome']=='response');record['errors']+=int(row['outcome'] not in ('response','pending'))
        record['pending']+=int(row['outcome']=='pending');record['usage_observed']+=int(row['usage_json'] is not None)
        for f in ('prompt_tokens','completion_tokens','reasoning_tokens'):record[f]+=row[f] or 0
        for f,rf in (('requested_model','requested_models'),('returned_model','returned_models'),('run_id','run_ids')):
            if row[f] and row[f] not in record[rf]:record[rf].append(row[f])
        if row['ended_at']:
            record['last_outcome']=max(record['last_outcome'] or row['ended_at'],row['ended_at'])
            record['cross_day_requests']+=int(len(billing_days(row['started_at'],row['ended_at']))>1)
    conflicts=[]
    for row in c.execute("SELECT key_alias,billing_group,day,group_concat(DISTINCT profile) AS profiles FROM claims GROUP BY key_alias,billing_group,day HAVING count(DISTINCT profile)>1"):
        conflicts.append(dict(row))
    samples={}
    for row in c.execute('SELECT * FROM samples'):
        identity=(row['profile'],row['cohort'],timestamp(row['at']).astimezone(ZONE).date().isoformat())
        entry=samples.setdefault(identity,dict(profile=identity[0],cohort=identity[1],billing_date=identity[2],success=0,refusal=0,failure=0))
        entry[row['outcome']]+=1
    result={'observed_at':datetime.now(timezone.utc).isoformat(),'billing_timezone':'Asia/Shanghai','boundary':'00:00',
      'timezone_evidence':'User confirmed in this chat on 2026-10-02','key_alias':key,
      'key_provenance':'Session identifies the same original key; historical journals do not cryptographically bind keys.',
      'external_consumption':'Cannot exclude other callers using the same key; reconcile invoice before assigning daily totals.',
      'date_basis':'Request-start day for counts; all request-to-outcome days reserved. Provider billing timestamp requires invoice reconciliation.',
      'daily_requests':list(records.values()),'sample_observations':list(samples.values()),'conflicts':conflicts,
      'historical_sample_usage':[dict(row) for row in c.execute('SELECT * FROM historical_usage')],
      'search_evidence_limit':'Search request count and outcomes are observed; per-key identity and logical retry ordinal are absent from the raw journal. Retry count remains null.',
      'gaps':[dict(row) for row in c.execute('SELECT * FROM gaps')],
      'cost_rule':'Original80 actual historical cost plus additional220 actual cost, each counted once; new300 separate. No extrapolation. Filter/search/probe costs separate.',
      'source_cursors':[dict(row) for row in c.execute('SELECT * FROM cursors')]}
    target=local(DIRECTORY/'daily_usage.json');temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2));temp.replace(target)
    lines=['# 账单日期对照','', '账单时区：北京时间 UTC+8，00:00 日切（用户确认）。金额待账单填写；不按 tokens 或题数比例臆分。',
           '原80题历史费用只计一次，再加新增220题实际费用；新模型和新档位的300题单列。过滤、搜索及探测费用分别留存。',
           '同日混档标记不可按每日总账直接拆分；同 Key 外部消费尚不能排除。跨日请求需对照供应商实际入账时间。',
           'original80 仅有样本落库日期，不能代表全部请求日期；历史重试及实际入账日期待原日志或账单补齐。','',
           '| 模型 / 档位 | 样本范围 | 可见日期（北京时间） | 费用 |','|---|---|---|---|']
    arms={}
    for row in c.execute("SELECT profile,cohort,started_at,ended_at FROM requests WHERE role='forecast'"):
        arms.setdefault((row['profile'],row['cohort']),set()).update(billing_days(row['started_at'],row['ended_at'] or row['started_at']))
    for row in c.execute("SELECT profile,cohort,at FROM samples WHERE cohort='original80'"):
        arms.setdefault((row['profile'],row['cohort']),set()).update(billing_days(row['at'],row['at']))
    for (profile,cohort),days in sorted(arms.items()):lines.append(f"| {profile} | {cohort} | {', '.join(sorted(days))} | 待补实际账单 |")
    lines+=['','## 同日多档记录','', '| 计费归组 | 日期 | 档位 | 归因 |','|---|---|---|---|']
    for row in conflicts:lines.append(f"| {row['billing_group']} | {row['day']} | {row['profiles']} | 不可按每日总账直接拆分 |")
    local(DIRECTORY/'DATES.md').write_text('\n'.join(lines)+'\n')
    return result
