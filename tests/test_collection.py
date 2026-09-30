"""Collection contracts: identity, raw evidence, recovery, and deferred scoring."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
import respx
from openai import AsyncOpenAI, APITimeoutError, BadRequestError

from forecast_eval import db, leak_filter, llm, loader, react, runner
from forecast_eval.config import ModelProfile, Settings
from forecast_eval.errors import CollectionBlockedError, ErrorKind, classify
from forecast_eval.prompts import DEFAULT_PROMPT_TEMPLATES, render_user_prompt
from forecast_eval.search import SearchResult, SearchResultItem, _parse_tavily_response, _single_request, tavily_search
from forecast_eval.tavily_keys import TavilyKeyPool
from tests.test_evaluation import _call_init_model_db
from tests.test_leak_filter import _envelope, _scripted_client, _capturing_client
from tests.test_llm_no_browsing import _success_body
from tests.test_react import _ScriptedLLM, _yes_no_question
from forecast_eval.types import QFilter
from scripts.catalog_collection import sample_coverage
from scripts import collect_forecast_panel


def settings(**kwargs):
    return Settings(_env_file=None, **{
        "LLM_API_KEY": "sk-collection-secret", "TAVILY_API_KEY": ["tvly-collection-secret"],
        "LEAK_DETECTOR_API_KEY": "sk-detector-secret", "MODELS": ["test-arm"],
        "MODEL_PROFILES": {"test-arm": {"model": "provider-model", "reasoning_effort": "high"}},
        "ENABLE_SEARCH_LEAK_FILTER": False, **kwargs,
    })


def test_collection_plan_keeps_continuation_and_orders_twelve_reasoning_arms(tmp_path, monkeypatch):
    selected = [
        "gpt-5.4--reasoning-none", "gpt-5.4--reasoning-low",
        "gpt-5.4--reasoning-medium", "gpt-5.4--reasoning-high",
        "claude-opus-4-6--reasoning-low", "claude-opus-4-6--reasoning-medium",
        "claude-opus-4-6--reasoning-high", "gpt-5.3-codex--reasoning-xhigh",
        "claude-sonnet-4-6--reasoning-xhigh", "gemini-3.1-pro-preview-customtools--reasoning-high",
        "glm-5--reasoning-high", "alicloud-kimi-k2.5--reasoning-high",
    ]
    continuation = {model + "--provider-default": {"model": model}
                    for model in collect_forecast_panel.PRIORITY}
    candidates = collect_forecast_panel.profile_candidates()
    assert list(candidates) == selected
    profiles = {**continuation, **candidates}
    runtime = {"MODELS": list(profiles), "MODEL_PROFILES": profiles, "SAMPLING_N": 3,
               "MODEL_QUESTION_IDS": {name: list(range(220)) for name in continuation},
               "MODEL_TRAINING_CUTOFFS": {name: "2020-01-01" for name in profiles}}
    monkeypatch.setattr(collect_forecast_panel, "COLLECTION", tmp_path)
    monkeypatch.setattr(collect_forecast_panel, "PLAN", tmp_path / "plan.json")
    monkeypatch.setattr(collect_forecast_panel, "local", lambda path: path)
    run_ids = iter(["continuation-run", "reasoning-run"])
    monkeypatch.setattr(collect_forecast_panel.runner, "generate_run_id", lambda: next(run_ids))
    collect_forecast_panel.prepare_batches({"runtime": runtime, "source_sha256": "source", "reference_lineage": {}})
    first = json.loads((tmp_path / "continuation.json").read_text())
    second = json.loads((tmp_path / "reasoning.json").read_text())
    assert first["runtime"]["MODELS"] == list(continuation)
    assert first["runtime"]["MODEL_PROFILES"] == continuation
    assert first["runtime"]["MODEL_QUESTION_IDS"] == runtime["MODEL_QUESTION_IDS"]
    assert (first["new_sample_count"], first["search_call_ceiling"]) == (9240, 36960)
    assert second["runtime"]["MODELS"] == selected
    assert second["runtime"]["MODEL_QUESTION_IDS"] == {}
    assert (second["new_sample_count"], second["search_call_ceiling"]) == (10800, 43200)
    assert second["status"] == "awaiting_additional_quota"
    for name in selected:
        assert collect_forecast_panel.sample_target(second["runtime"], name) == 900
        assert second["runtime"]["MODEL_PROFILES"][name]["reasoning_effort"] == name.rsplit("-", 1)[1]


def test_collection_coverage_counts_only_requested_slots_and_resolved_repairs():
    successes = {("arm", "anchor", 0): "repair", ("arm", "anchor", 1): "reference",
                 ("arm", "added", 0): "collected"}
    failures = {("arm", "anchor", 0): {"unknown"}, ("arm", "added", 1): {"network"}}
    runtime = {"MODELS": ["arm"], "SAMPLING_N": 2, "MODEL_QUESTION_IDS": {"arm": ["added"]}}
    plan = {"phase": "continuation", "runtime": runtime, "new_sample_count": 2}
    coverage = sample_coverage(plan, {"anchor", "added"}, successes, failures)
    assert (coverage["collected"], coverage["failed"], coverage["missing"], coverage["complete"]) == (1, 1, 0, False)
    assert coverage["profiles"]["arm"]["pending_slots"] == [
        {"question_id": "added", "sample_idx": 1, "state": "failed", "errors": ["network"]}]
    repair = {"phase": "repair", "new_sample_count": 1,
              "runtime": {**runtime, "MODEL_QUESTION_IDS": {"arm": ["anchor"]},
                          "MODEL_SAMPLE_INDICES": {"arm": {"anchor": [0]}}}}
    coverage = sample_coverage(repair, {"anchor", "added"}, successes, failures)
    assert coverage["complete"] and coverage["collected"] == 1 and coverage["failed"] == 0
    with pytest.raises(ValueError, match="sample count"):
        sample_coverage({**plan, "new_sample_count": 4}, {"anchor", "added"}, successes, failures)
    coverage = sample_coverage(plan, {"anchor", "added"}, {}, {})
    assert coverage["missing"] == 2 and not coverage["complete"]


async def test_collection_stops_before_billable_calls_when_disk_reserve_is_reached(monkeypatch):
    plan = {"phase": "continuation", "runtime": {"MODELS": ["arm"], "SAMPLING_N": 3,
                                                 "MODEL_QUESTION_IDS": {"arm": ["question"]}}}
    monkeypatch.setattr(collect_forecast_panel, "status", lambda _: {"profiles": {"arm": {"completed": 0}}})
    monkeypatch.setattr(collect_forecast_panel.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3))
    with pytest.raises(CollectionBlockedError, match="disk space"):
        await collect_forecast_panel.dispatch_batches(plan, {})


async def test_cancelled_search_keeps_request_identity_without_a_fabricated_response(tmp_path):
    config = settings()
    async def cancel(request):
        raise asyncio.CancelledError()
    conn = db.connect(tmp_path / "cancelled.db")
    db.init_schema(conn, 1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(cancel)) as client:
        with db.capture_requests(conn, settings=config, run_id="r", model="test-arm", question_id="q", sample_idx=0):
            with pytest.raises(asyncio.CancelledError):
                await _single_request(client, query="historical query", end_date="2020-01-01",
                                      settings=config, api_key=config.TAVILY_API_KEY[0])
    events = conn.execute("SELECT * FROM request_events ORDER BY event_id").fetchall()
    assert [row["kind"] for row in events] == ["search.request", "search.cancelled"]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert config.TAVILY_API_KEY[0] not in events[0]["payload"]
    conn.close()


@respx.mock
async def test_profile_wire_identity_and_request_journal(tmp_path):
    config = settings(LLM_BASE_URL="https://provider.test/v1")
    route = respx.post("https://provider.test/v1/chat/completions").respond(200, json=_success_body())
    conn = db.connect(tmp_path / "model.db")
    db.init_schema(conn, 1)
    async with AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=0) as client:
        with db.capture_requests(conn, settings=config, run_id="r", model="test-arm", question_id="q", sample_idx=0):
            await llm.chat(model="test-arm", messages=[{"role": "user", "content": "fixture"}],
                           settings=config, tools=[], client=client)
            db.audit_event("fixture.error", {"message": config.LLM_API_KEY})
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "provider-model" and body["reasoning_effort"] == "high"
    assert "tool_choice" not in body and "plugins" not in body
    events = conn.execute("SELECT * FROM request_events ORDER BY event_id").fetchall()
    assert [r["kind"] for r in events] == ["llm.request", "llm.response", "fixture.error"]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert json.loads(json.loads(events[1]["payload"])["body"])["choices"]
    assert config.LLM_API_KEY not in events[2]["payload"]
    conn.close()
    reopened = db.connect(tmp_path / "model.db")
    assert reopened.execute("SELECT count(*) FROM request_events").fetchone()[0] == 3
    reopened.close()


@pytest.mark.parametrize("output_cap", [512, 2048])
async def test_detector_json_constraint_is_explicit_and_preserves_verdict(output_cap):
    config = settings(LEAK_DETECTOR_RESPONSE_FORMAT="json_object", LEAK_DETECTOR_MAX_TOKENS=output_cap)
    client = _capturing_client([_envelope('{"verdict":"drop","reason":"after cutoff"}')])
    verdict = await leak_filter._detect_one(SearchResultItem("page", "https://example.org", "text"), "2020-01-01", config, client)
    assert verdict == ("drop", "after cutoff")
    assert client.captured[0]["response_format"] == {"type": "json_object"}
    assert client.captured[0]["max_tokens"] == output_cap
    assert "tools" not in client.captured[0]
    assert db.collection_contract(db.snapshot_settings(config))["LEAK_DETECTOR_RESPONSE_FORMAT"] == "json_object"
    assert db.collection_contract(db.snapshot_settings(config))["LEAK_DETECTOR_MAX_TOKENS"] == output_cap
    assert db.compute_collection_contract_hash({"LEAK_DETECTOR_MAX_TOKENS": 512}) != db.compute_collection_contract_hash({"LEAK_DETECTOR_MAX_TOKENS": 2048})


@pytest.mark.parametrize(("reference_drop", "target_drop", "allowed"), [(False, False, True), (False, True, True), (True, False, False)])
@pytest.mark.parametrize(("reference_cap", "target_cap", "cap_allowed"), [(512, 512, True), (512, 2048, True), (2048, 512, False)])
async def test_linked_collection_stratum_skips_successes_without_overwriting_sources(tmp_path, monkeypatch, reference_drop, target_drop, allowed, reference_cap, target_cap, cap_allowed):
    question = _yes_no_question()
    old = settings(SAMPLING_N=3, RUNS_ROOT=str(tmp_path), LEAK_DETECTOR_DROP_CONTENT_POLICY=reference_drop, LEAK_DETECTOR_MAX_TOKENS=reference_cap)
    source_path = tmp_path / "reference.db"
    current = settings(SAMPLING_N=3, RUNS_ROOT=str(tmp_path), LEAK_DETECTOR_RESPONSE_FORMAT="json_object",
                       LEAK_DETECTOR_DROP_CONTENT_POLICY=target_drop,
                       LEAK_DETECTOR_MAX_TOKENS=target_cap,
                       COLLECTION_SAMPLE_LIMIT=1, COLLECTION_REFERENCE_DBS={"test-arm": [str(source_path)]})
    def prepare(path, config):
        conn = db.connect(path)
        db.init_schema(conn, 3)
        conn.execute("INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (question.id, question.choice_type, question.question_type, question.event, question.options, question.answer, question.end_time, db.utcnow_iso()))
        db.register_run_meta(conn, run_id=path.stem, model="test-arm", sampling_n=3, filters_snapshot={},
                             config_snapshot=db.snapshot_settings(config), source_db_hash="source", metadata_hash="metadata", prompt_templates_hash="prompt")
        return conn
    source = prepare(source_path, old)
    db.upsert_sample_sync(source, 3, {**runner._error_row(question, 0, "fixture"), "error": None})
    source.close()
    target = prepare(tmp_path / "target.db", current)
    calls = []
    async def completed(task, **kwargs):
        calls.append(task.sample_idx)
        return {**runner._error_row(task.question, task.sample_idx, "fixture"), "error": None}
    monkeypatch.setattr(runner, "_run_task_with_retry", completed)
    if allowed and cap_allowed:
        stats = await runner.run(settings=current, filters=QFilter(), questions=[question], templates={}, run_id="target", conns={"test-arm": target})
        assert calls == [1] and stats.completed_preexisting == 1 and stats.deferred == 1
    else:
        with pytest.raises(ValueError, match="inference contract differs"):
            await runner.run(settings=current, filters=QFilter(), questions=[question], templates={}, run_id="target", conns={"test-arm": target})
        assert calls == []
    target.close()
    source = db.connect(source_path)
    assert db.load_completed_samples(source, 3) == {(question.id, 0)}
    source.close()


@pytest.mark.parametrize("allow_drop", [False, True])
async def test_content_refusal_is_retained_and_never_presented_as_a_leak_verdict(allow_drop):
    config = settings(REQUIRE_HEALTHY_RETRIEVAL=True, LEAK_DETECTOR_DROP_CONTENT_POLICY=allow_drop,
                      LEAK_DETECTOR_FAIL_ACTION="keep")
    response = httpx.Response(400, request=httpx.Request("POST", "https://provider.test/v1/chat/completions"))
    refusal = BadRequestError("data_inspection_failed: Input text data may contain inappropriate content.",
                              response=response, body={"code": "data_inspection_failed"})
    client = _scripted_client([refusal, _envelope('{"verdict":"keep","reason":"before cutoff"}')])
    result = SearchResult("query", "2020-01-01", results=[SearchResultItem("refused", "https://example.org/refused", "raw text"),
                                                        SearchResultItem("safe", "https://example.org/safe", "dated text")])
    if allow_drop:
        await leak_filter.filter_search_result(result, end_date=result.end_date, settings=config, client=client)
        assert [item.title for item in result.results] == ["safe"]
        assert "refused" not in json.dumps(result.to_llm_payload())
    else:
        with pytest.raises(CollectionBlockedError):
            await leak_filter.filter_search_result(result, end_date=result.end_date, settings=config, client=client)
    assert result.audit["detector_verdicts"][0].startswith("failed:content_policy")
    assert result.audit["detector_error_kind"] == "content_policy"
    assert result.audit["results_raw"][0]["content"] == "raw text"
    assert "data_inspection_failed" in result.audit["detector_reasons"][0]
    assert db.collection_contract({"LEAK_DETECTOR_DROP_CONTENT_POLICY": False}) == db.collection_contract({})
    assert db.collection_contract({"LEAK_DETECTOR_DROP_CONTENT_POLICY": True}) != db.collection_contract({})
    assert db.compute_collection_contract_hash({"LEAK_DETECTOR_DROP_CONTENT_POLICY": False}) == db.compute_collection_contract_hash({})
    assert db.compute_collection_contract_hash({"LEAK_DETECTOR_DROP_CONTENT_POLICY": True}) != db.compute_collection_contract_hash({})


@pytest.mark.parametrize("failure", ["auth", "network", "rate_limit", "unknown"])
async def test_content_refusal_does_not_hide_another_detector_failure(monkeypatch, failure):
    config = settings(REQUIRE_HEALTHY_RETRIEVAL=True, LEAK_DETECTOR_DROP_CONTENT_POLICY=True)
    async def detect(item, *_):
        return ("failed:" + ("content_policy" if item.title == "refused" else failure), "failure retained")
    monkeypatch.setattr(leak_filter, "_detect_one", detect)
    result = SearchResult("query", "2020-01-01", results=[SearchResultItem("refused", "https://example.org/1", "text"),
                                                        SearchResultItem("unavailable", "https://example.org/2", "text")])
    with pytest.raises(CollectionBlockedError):
        await leak_filter.filter_search_result(result, end_date=result.end_date, settings=config, client=object())
    assert len(result.audit["detector_verdicts"]) == 2


@respx.mock
async def test_sampling_overrides_preserve_archived_settings_and_reasoning(tmp_path):
    config = settings(LLM_BASE_URL="https://provider.test/v1", MODEL_PROFILES={"test-arm": {
        "model": "provider-model", "reasoning_effort": "high", "temperature": 1.0,
        "omit_sampling_fields": ["top_p"], "max_tokens_param": "max_completion_tokens",
        "replay_reasoning": False,
    }})
    route = respx.post("https://provider.test/v1/chat/completions").respond(200, json=_success_body())
    messages = [{"role": "assistant", "content": "hello", "reasoning_content": "retained observation"}]
    async with AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=0) as client:
        await llm.chat(model="test-arm", messages=messages, settings=config, client=client)
    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 1.0 and "top_p" not in body
    assert body["max_completion_tokens"] == 12000 and "max_tokens" not in body
    assert "reasoning_content" not in body["messages"][0]
    assert messages[0]["reasoning_content"] == "retained observation"


def test_profile_parameters_are_fingerprinted_and_browsing_is_rejected():
    a = settings()
    b = settings(MODEL_PROFILES={"test-arm": {"model": "provider-model", "reasoning_effort": "none"}})
    assert db.snapshot_settings(a)["inference_profiles_hash"] != db.snapshot_settings(b)["inference_profiles_hash"]
    for profile in [{"model": "provider:online"}, {"model": "provider", "plugins": []}]:
        with pytest.raises(ValueError):
            ModelProfile(**profile)


@respx.mock
async def test_sdk_timeout_retries_same_request_and_journals_both_attempts(tmp_path):
    config = settings(LLM_BASE_URL="https://provider.test/v1", LLM_BACKOFF_NETWORK_S=[0])
    assert classify(APITimeoutError(request=httpx.Request("POST", "https://provider.test"))) is ErrorKind.NETWORK
    route = respx.post("https://provider.test/v1/chat/completions").mock(side_effect=[
        httpx.ReadTimeout("fixture timeout"), httpx.Response(200, json=_success_body())])
    conn = db.connect(tmp_path / "model.db")
    db.init_schema(conn, 1)
    async with AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=0) as client:
        with db.capture_requests(conn, settings=config, run_id="r", model="test-arm", question_id="q", sample_idx=0):
            await llm.chat(model="test-arm", messages=[{"role": "user", "content": "same context"}], settings=config, client=client)
    assert route.call_count == 2
    assert route.calls[0].request.content == route.calls[1].request.content
    assert [row[0] for row in conn.execute("SELECT kind FROM request_events ORDER BY event_id")] == ["llm.request", "llm.error", "llm.request", "llm.response"]
    conn.close()


async def test_dropped_evidence_and_reasons_never_enter_model_context():
    config = settings(LEAK_DETECTOR_MODEL="detector")
    source = {"results": [{"title": "page", "url": "https://example.org", "content": "future outcome",
                           "raw_content": "x" * 10000}]}
    result = _parse_tavily_response(source, "query", "2020-01-01", raw_content_max_chars=8000)
    client = _scripted_client([_envelope('{"verdict":"drop","reason":"post-cutoff event"}')])
    result = await leak_filter.filter_search_result(result, end_date="2020-01-01", settings=config, client=client)
    calls = []
    react._record_search_call(calls, query="query", end_date="2020-01-01", result=result)
    assert calls[0]["detector_reasons"] == ["post-cutoff event"]
    assert calls[0]["results_raw"][0]["content"] == "future outcome"
    assert len(calls[0]["raw_response"]["results"][0]["raw_content"]) == 10000
    assert result.to_llm_payload() == {"results": []}


async def test_raw_collection_does_not_read_gold_for_scoring(monkeypatch):
    config = settings(SCORE_ANSWERS=False, REACT_REFLECTION_PROTOCOL=False).model_copy(
        update={"TAVILY_MAX_RESULTS": 5, "REACT_MAX_SEARCH_CALLS": 4})
    question = replace(_yes_no_question(), answer="not a valid gold value")
    scripted = _ScriptedLLM([({"role": "assistant", "content": "\\boxed{Yes}"}, "stop", {})])
    monkeypatch.setattr(react, "llm_chat", scripted)
    monkeypatch.setattr(react, "parse_gt", lambda _: pytest.fail("collection must not score gold"))
    result = await react.run_react(question, model="test-arm", sample_idx=0, settings=config,
                                    templates=DEFAULT_PROMPT_TEMPLATES, run_id="r")
    assert result.correct is None and result.final_answer_raw and result.messages_trace


def test_question_scope_preserves_completed_anchors():
    a = _yes_no_question()
    b = replace(a, id="additional-question")
    config = settings(MODEL_QUESTION_IDS={"test-arm": [b.id]}, SAMPLING_N=3)
    tasks, _, stats = runner.build_task_plan(questions=[a, b], settings=config, completed={}, run_id="r")
    assert len(tasks) == 3 and all(task.question.id == b.id for task in tasks)
    assert stats.total == 3


def test_repair_plans_only_selected_sample_slot():
    question = _yes_no_question()
    config = settings(SAMPLING_N=3, MODEL_QUESTION_IDS={"test-arm": [question.id]},
                      MODEL_SAMPLE_INDICES={"test-arm": {question.id: [0]}})
    tasks, _, stats = runner.build_task_plan(questions=[question], settings=config, completed={}, run_id="repair")
    assert len(tasks) == 1 and tasks[0].sample_idx == 0 and stats.total == 1
    empty_contract = db.collection_contract({"MODEL_SAMPLE_INDICES": {}})
    assert empty_contract == db.collection_contract({})
    assert db.collection_contract({"MODEL_SAMPLE_INDICES": {"test-arm": {question.id: [0]}}}) != empty_contract
    for indices in [[-1], [3], [0, 0], []]:
        with pytest.raises(ValueError, match="sample slots"):
            settings(SAMPLING_N=3, MODEL_SAMPLE_INDICES={"test-arm": {question.id: indices}})


async def test_exhausted_search_pool_blocks_collection():
    config = settings(REQUIRE_HEALTHY_RETRIEVAL=True)
    pool = TavilyKeyPool.from_keys(config.TAVILY_API_KEY)
    await pool.report_failure(config.TAVILY_API_KEY[0], "auth")
    async with httpx.AsyncClient() as client:
        with pytest.raises(CollectionBlockedError):
            await tavily_search("query", "2020-01-01", config, pool=pool, client=client)


def test_resume_rejects_reasoning_change_before_overwriting_metadata(tmp_path):
    config = settings()
    path = tmp_path / "model.db"
    _call_init_model_db(config, path, capture={})
    changed = settings(MODEL_PROFILES={"test-arm": {"model": "provider-model", "reasoning_effort": "none"}})
    with pytest.raises(ValueError, match="collection_contract"):
        _call_init_model_db(changed, path, capture={})
    conn = db.connect(path)
    snapshot = json.loads(conn.execute("SELECT config_snapshot FROM run_meta").fetchone()[0])
    assert snapshot["MODEL_PROFILES"]["test-arm"]["reasoning_effort"] == "high"
    conn.close()


async def test_cancellation_is_not_stored_as_unknown(monkeypatch):
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError
    monkeypatch.setattr(runner, "run_react", cancelled)
    config = settings()
    task = runner.Task(_yes_no_question(), "test-arm", 0, config)
    with pytest.raises(asyncio.CancelledError):
        await runner._run_task_with_retry(task, _global_settings=config, templates={}, run_id="r",
                                         llm_semaphore=asyncio.Semaphore(1), search_semaphore=asyncio.Semaphore(1))


def test_shared_prompt_schema_requires_explicit_selection_and_preserves_text(tmp_path):
    templates = {k: v for k, v in DEFAULT_PROMPT_TEMPLATES.items() if not k.endswith("_prompt_template")}
    templates["prompt_template"] = ('{agent_role} Event: "{event} ({end_time}).{outcomes_block}"\n'
                                    'Your final answer MUST end with this exact format:\n{output_format}\n{guidance}')
    source = db.connect(tmp_path / "source.db")
    source.execute("CREATE TABLE dataset_metadata (features_json TEXT)")
    source.execute("INSERT INTO dataset_metadata VALUES (?)", (json.dumps({"prompt_reconstruction": templates}),))
    source.close()
    target = db.connect(tmp_path / "result.db")
    db.init_schema(target, 1)
    with pytest.raises(ValueError, match="missing required keys"):
        loader.sync_prompt_templates(tmp_path / "source.db", target)
    loaded = loader.sync_prompt_templates(tmp_path / "source.db", target, template_style="shared")
    assert loaded == templates
    question = _yes_no_question()
    expected = templates["prompt_template"].format(agent_role=templates["agent_role"], event=question.event,
        end_time=question.end_time, outcomes_block="", output_format=templates["yes_no_output_format"], guidance=templates["guidance"])
    assert render_user_prompt(question, loaded, template_style="shared") == expected
    assert db.collection_contract({"PROMPT_TEMPLATE_STYLE": "shared"}) != db.collection_contract({"PROMPT_TEMPLATE_STYLE": "typed"})
    target.close()


async def test_dispatch_limit_leaves_pending_samples_and_resume_skips_completed(tmp_path, monkeypatch):
    config = settings(SAMPLING_N=3, COLLECTION_MODEL="test-arm", COLLECTION_SAMPLE_LIMIT=1, DB_COMMIT_BATCH=1)
    conn = db.connect(tmp_path / "result.db")
    db.init_schema(conn, 3)
    question = _yes_no_question()
    conn.execute("INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (question.id, question.choice_type, question.question_type, question.event, question.options, question.answer, question.end_time, db.utcnow_iso()))
    calls = []
    async def completed(task, **kwargs):
        calls.append(task.sample_idx)
        return {**runner._error_row(task.question, task.sample_idx, "fixture"), "error": None}
    monkeypatch.setattr(runner, "_run_task_with_retry", completed)
    for expected in (0, 1):
        stats = await runner.run(settings=config, filters=QFilter(), questions=[question], templates={}, run_id="r", conns={"test-arm": conn})
        assert stats.planned == 1 and stats.deferred == 2 - expected and not stats.aborted
    assert calls == [0, 1]
    conn.close()


@pytest.mark.parametrize("retain,error,terminal", [(False, "content_policy", False),
    (True, "content_policy", True), (True, "network", False), (True, "bad_request", False)])
async def test_model_refusal_retention_preserves_error_and_resume_boundary(tmp_path, monkeypatch, retain, error, terminal):
    config = settings(SAMPLING_N=1, REQUIRE_HEALTHY_RETRIEVAL=True,
                      COLLECTION_RETAIN_MODEL_REFUSALS=retain, DB_COMMIT_BATCH=1)
    conn = db.connect(tmp_path / "refused.db")
    db.init_schema(conn, 1)
    question = _yes_no_question()
    conn.execute("INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (question.id, question.choice_type, question.question_type, question.event, question.options, question.answer, question.end_time, db.utcnow_iso()))
    calls = []
    async def fail(task, **kwargs):
        calls.append(task.sample_idx)
        db.audit_event("llm.error", {"type": "fixture", "body": {"code": error}})
        return runner._error_row(task.question, task.sample_idx, error)
    monkeypatch.setattr(runner, "_run_task_with_retry", fail)
    stats = await runner.run(settings=config, filters=QFilter(), questions=[question], templates={}, run_id="r", conns={"test-arm": conn})
    assert stats.aborted is not terminal and stats.errors == {error: 1}
    row = dict(conn.execute("SELECT * FROM run_results").fetchone())
    assert row["s0_error"] == error and row["s0_correct"] is None and row["s0_final_answer_raw"] is None
    assert bool(db.load_completed_samples(conn, 1, retain_model_refusals=retain)) == terminal
    stats = await runner.run(settings=config, filters=QFilter(), questions=[question], templates={}, run_id="r", conns={"test-arm": conn})
    assert len(calls) == (1 if terminal else 2)
    if terminal:
        assert dict(conn.execute("SELECT * FROM run_results").fetchone()) == row
        assert stats.planned == 0 and not stats.aborted
    assert conn.execute("SELECT count(*) FROM request_events WHERE kind='llm.error'").fetchone()[0] == len(calls)
    conn.close()


def test_refusal_retention_requires_evidence_and_does_not_change_sample_inference():
    config = settings()
    retained = settings(COLLECTION_RETAIN_MODEL_REFUSALS=True)
    assert db.snapshot_settings(retained)["COLLECTION_RETAIN_MODEL_REFUSALS"] is True
    assert db.compute_collection_contract_hash(db.snapshot_settings(config)) == db.compute_collection_contract_hash(db.snapshot_settings(retained))
    with pytest.raises(ValueError, match="requires WRITE_REQUEST_AUDIT"):
        settings(COLLECTION_RETAIN_MODEL_REFUSALS=True, WRITE_REQUEST_AUDIT=False)


def test_coverage_separates_retained_refusals_from_predictions_and_pending_failures():
    runtime = {"MODELS": ["arm"], "SAMPLING_N": 2, "MODEL_QUESTION_IDS": {"arm": ["q"]},
               "COLLECTION_RETAIN_MODEL_REFUSALS": True}
    plan = {"phase": "continuation", "runtime": runtime, "new_sample_count": 2}
    failures = {("arm", "q", 0): {"content_policy"}}
    coverage = sample_coverage(plan, {"q"}, {("arm", "q", 1): "source"}, failures)
    assert coverage["refused"] == 1 and coverage["collected"] == 1
    assert coverage["attempts_complete"] and not coverage["complete"]
    assert coverage["profiles"]["arm"]["pending_slots"] == []
    assert coverage["profiles"]["arm"]["refused_slots"] == [
        {"question_id": "q", "sample_idx": 0, "state": "refused", "errors": ["content_policy"]}]
    failures[("arm", "q", 1)] = {"network"}
    coverage = sample_coverage(plan, {"q"}, {}, failures)
    assert coverage["failed"] == 1 and coverage["refused"] == 1 and not coverage["attempts_complete"]
    runtime["COLLECTION_RETAIN_MODEL_REFUSALS"] = False
    coverage = sample_coverage(plan, {"q"}, {}, failures)
    assert coverage["failed"] == 2 and coverage["refused"] == 0


async def test_batch_dispatch_finishes_with_refusals_without_repeating_calls(monkeypatch):
    plan = {"phase": "continuation", "runtime": {"MODELS": ["arm"], "SAMPLING_N": 3,
            "MODEL_QUESTION_IDS": {"arm": ["q"]}, "COLLECTION_RETAIN_MODEL_REFUSALS": True}}
    monkeypatch.setattr(collect_forecast_panel, "status", lambda _: {
        "profiles": {"arm": {"completed": 2, "refusals": 1}}, "refusals": 1})
    saved = []
    monkeypatch.setattr(collect_forecast_panel, "write_json", lambda path, value: saved.append(dict(value)))
    assert await collect_forecast_panel.dispatch_batches(plan, {}) == 0
    assert saved[-1]["status"] == "complete_with_refusals" and saved[-1]["model_refusals"] == 1


async def test_batch_dispatch_counts_a_refusal_as_progress_without_counting_a_prediction(tmp_path, monkeypatch):
    runtime = settings(SAMPLING_N=1, COLLECTION_RETAIN_MODEL_REFUSALS=True,
                       MODEL_QUESTION_IDS={"test-arm": ["q"]}).model_dump()
    runtime.pop("LEAK_DETECTOR_API_KEY")
    plan = {"phase": "continuation", "run_id": "fixture", "runtime": runtime}
    calls = []
    def counts(_):
        return {"profiles": {"test-arm": {"completed": 0, "refusals": len(calls)}}, "refusals": len(calls)}
    async def quota(*args):
        return 1000
    async def evaluate(config, *args, **kwargs):
        assert config.COLLECTION_RETAIN_MODEL_REFUSALS and config.COLLECTION_SAMPLE_LIMIT == 1
        assert kwargs["skip_analysis"] is True
        calls.append(config.COLLECTION_MODEL)
        return 0
    monkeypatch.setattr(collect_forecast_panel, "status", counts)
    monkeypatch.setattr(collect_forecast_panel, "available_searches", quota)
    monkeypatch.setattr(collect_forecast_panel.evaluation, "_run_async", evaluate)
    monkeypatch.setattr(collect_forecast_panel.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(collect_forecast_panel, "write_json", lambda *args: None)
    monkeypatch.setattr(collect_forecast_panel, "COLLECTION", tmp_path)
    monkeypatch.setattr(collect_forecast_panel, "record_dispatch", lambda *args, **kwargs: None)
    monkeypatch.setattr(collect_forecast_panel, "local", lambda path: path)
    assert await collect_forecast_panel.dispatch_batches(plan, {"LLM_API_KEY": "sk-fixture"}) == 0
    assert calls == ["test-arm"] and plan["status"] == "complete_with_refusals"
    assert counts(plan)["profiles"]["test-arm"]["completed"] == 0


@pytest.mark.parametrize("costly,stop_after_first", [(False, False), (True, False), (False, True)])
async def test_dispatch_tuning_is_applied_between_batches_and_stop_drains(tmp_path, monkeypatch, costly, stop_after_first):
    runtime = settings(SAMPLING_N=3, MODEL_QUESTION_IDS={"test-arm": ["q1", "q2"]},
                       MODEL_PROFILES={"test-arm": {"model": "gpt-5.4" if costly else "qwen3.5-plus"}}).model_dump()
    runtime.pop("LEAK_DETECTOR_API_KEY")
    plan = {"phase": "continuation", "run_id": "fixture", "runtime": runtime}
    monkeypatch.setattr(collect_forecast_panel, "ROOT", tmp_path)
    monkeypatch.setattr(collect_forecast_panel, "COLLECTION", tmp_path)
    monkeypatch.setattr(collect_forecast_panel, "local", lambda path: path)
    (tmp_path / "runs/fixture").mkdir(parents=True)
    control = tmp_path / "dispatch_control.json"
    control.write_text(json.dumps({"LEAK_DETECTOR_CONCURRENCY": 10, "LLM_MAX_CONCURRENCY": 8}))
    completed = 0
    stop = asyncio.Event()
    configs = []
    async def quota(*args):
        return 1000
    async def evaluate(config, *args, **kwargs):
        nonlocal completed
        assert kwargs["skip_analysis"] is True
        configs.append(config)
        completed += config.COLLECTION_SAMPLE_LIMIT
        control.write_text(json.dumps({"LEAK_DETECTOR_CONCURRENCY": 15, "LLM_MAX_CONCURRENCY": 10}))
        if stop_after_first:
            stop.set()
        return 0
    monkeypatch.setattr(collect_forecast_panel, "status", lambda _: {
        "profiles": {"test-arm": {"completed": completed, "refusals": 0}}, "refusals": 0})
    monkeypatch.setattr(collect_forecast_panel, "available_searches", quota)
    monkeypatch.setattr(collect_forecast_panel.evaluation, "_run_async", evaluate)
    monkeypatch.setattr(collect_forecast_panel.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    assert await collect_forecast_panel.dispatch_batches(plan, {"LLM_API_KEY": "sk-fixture-private"}, stop) == 0
    assert completed == (3 if stop_after_first else 6)
    assert [cfg.LEAK_DETECTOR_CONCURRENCY for cfg in configs] == ([10] if stop_after_first else [10, 15])
    assert [cfg.LLM_MAX_CONCURRENCY for cfg in configs] == ([1] * len(configs) if costly else [8, 10][:len(configs)])
    expected_contract = db.collection_contract(runtime)
    assert all(db.collection_contract(db.snapshot_settings(cfg)) == expected_contract for cfg in configs)
    assert plan["status"] == ("stopped_at_batch_boundary" if stop_after_first else "complete")
    text = (tmp_path / "runs/fixture/dispatches.jsonl").read_text()
    assert "sk-fixture-private" not in text and "tvly-collection-secret" not in text
    journal = [json.loads(line) for line in text.splitlines()]
    assert [entry["event"] for entry in journal] == (
        ["start", "finish", "stopped_at_batch_boundary"] if stop_after_first else ["start", "finish", "start", "finish"])
    assert journal[0]["config_snapshot"]["LEAK_DETECTOR_CONCURRENCY"] == 10
    assert journal[1]["counts"]["completed"] == 3


@pytest.mark.parametrize("control", [{"LLM_MAX_TOKENS": 100}, {"LEAK_DETECTOR_CONCURRENCY": 0},
    {"LLM_MAX_CONCURRENCY": True}, {"LLM_MAX_CONCURRENCY": 21}, []])
def test_dispatch_tuning_rejects_contract_fields_and_invalid_limits(tmp_path, monkeypatch, control):
    monkeypatch.setattr(collect_forecast_panel, "COLLECTION", tmp_path)
    monkeypatch.setattr(collect_forecast_panel, "local", lambda path: path)
    (tmp_path / "dispatch_control.json").write_text(json.dumps(control))
    with pytest.raises(CollectionBlockedError, match="concurrency"):
        collect_forecast_panel.dispatch_settings({"runtime": {}}, {})


async def test_delegated_profiles_leave_local_queue_without_becoming_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_forecast_panel, "COLLECTION", tmp_path)
    monkeypatch.setattr(collect_forecast_panel, "local", lambda path: path)
    plan = {"phase": "continuation", "run_id": "fixture", "runtime": {"MODELS": ["arm"],
            "SAMPLING_N": 3, "MODEL_QUESTION_IDS": {"arm": ["q"]}}}
    (tmp_path / "local_queue_exclusions.json").write_text(json.dumps({"run_id": "fixture", "profiles": ["arm"]}))
    monkeypatch.setattr(collect_forecast_panel, "status", lambda _: {
        "profiles": {"arm": {"completed": 0, "refusals": 0}}, "refusals": 0})
    async def forbidden(*args):
        raise AssertionError("delegated profile must not issue paid requests")
    monkeypatch.setattr(collect_forecast_panel, "available_searches", forbidden)
    monkeypatch.setattr(collect_forecast_panel.evaluation, "_run_async", forbidden)
    assert await collect_forecast_panel.dispatch_batches(plan, {}) == 0
    assert plan["status"] == "local_queue_complete_pending_delegation"
    assert collect_forecast_panel.excluded_profiles({**plan, "run_id": "another-run"}) == set()
    (tmp_path / "local_queue_exclusions.json").write_text(json.dumps({"run_id": "fixture", "profiles": ["unknown"]}))
    with pytest.raises(CollectionBlockedError, match="unknown profile"):
        collect_forecast_panel.excluded_profiles(plan)


async def test_quota_refresh_retries_only_the_failed_key_and_uses_start_time(monkeypatch):
    delays, saved, called = [], [], []
    async def sleep(delay):
        delays.append(float(delay))
    times = iter(["2026-09-30T23:59:00+00:00", "2026-10-01T00:01:00+00:00"])
    monkeypatch.setattr(collect_forecast_panel.asyncio, "sleep", sleep)
    monkeypatch.setattr(collect_forecast_panel.db, "utcnow_iso", lambda: next(times))
    monkeypatch.setattr(collect_forecast_panel, "write_json", lambda path, value: saved.append(value))
    replies = iter([
        httpx.Response(200, json={"account": {"plan_limit": 1000, "plan_usage": 200}, "key": {"limit": None}}),
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(200, json={"account": {"plan_limit": 1000, "plan_usage": 100}, "key": {"limit": 500, "usage": 100}}),
        httpx.Response(401),
    ])
    def respond(request):
        called.append(request.headers["Authorization"])
        return next(replies)
    with respx.mock() as router:
        router.get("https://api.tavily.com/usage").mock(side_effect=respond)
        assert await collect_forecast_panel.query_search_quota(["key-a", "key-b", "key-c", "key-a"]) == 1200
    assert called == ["Bearer key-a", "Bearer key-b", "Bearer key-b", "Bearer key-b", "Bearer key-c"]
    assert delays == [7, 60]
    assert len(saved) == 1
    assert saved[0]["checked_at"] == "2026-09-30T23:59:00+00:00"
    assert saved[0]["finished_at"] == "2026-10-01T00:01:00+00:00"
    assert [key["remaining"] for key in saved[0]["keys"]] == [800, 400, 0]
    assert all(key not in json.dumps(saved) for key in ["key-a", "key-b", "key-c"])


async def test_quota_retry_exhaustion_preserves_snapshot_and_stops_other_queries(monkeypatch):
    delays, called = [], []
    async def sleep(delay):
        delays.append(float(delay))
    monkeypatch.setattr(collect_forecast_panel.asyncio, "sleep", sleep)
    monkeypatch.setattr(collect_forecast_panel, "write_json", lambda *args: pytest.fail("incomplete quota must not replace snapshot"))
    def respond(request):
        key = request.headers["Authorization"]
        called.append(key)
        if key == "Bearer key-a":
            return httpx.Response(200, json={"account": {"plan_limit": 1000, "plan_usage": 200}})
        return httpx.Response(429)
    with respx.mock() as router:
        router.get("https://api.tavily.com/usage").mock(side_effect=respond)
        with pytest.raises(httpx.HTTPStatusError):
            await collect_forecast_panel.query_search_quota(["key-a", "key-b", "key-c"])
    assert called == ["Bearer key-a", "Bearer key-b", "Bearer key-b", "Bearer key-b"]
    assert delays == [30, 60]


async def test_quota_refresh_recovers_transport_timeout(monkeypatch):
    delays, saved = [], []
    async def sleep(delay):
        delays.append(float(delay))
    monkeypatch.setattr(collect_forecast_panel.asyncio, "sleep", sleep)
    monkeypatch.setattr(collect_forecast_panel, "write_json", lambda path, value: saved.append(value))
    with respx.mock() as router:
        route = router.get("https://api.tavily.com/usage").mock(side_effect=[
            httpx.ReadTimeout("usage endpoint"),
            httpx.Response(200, json={"account": {"plan_limit": 1000, "plan_usage": 700}}),
        ])
        assert await collect_forecast_panel.query_search_quota(["key-a"]) == 300
        assert route.call_count == 2
    assert delays == [30] and len(saved) == 1


async def test_quota_unknown_limit_cannot_replace_confirmed_snapshot(monkeypatch):
    monkeypatch.setattr(collect_forecast_panel, "write_json", lambda *args: pytest.fail("unknown quota must not replace snapshot"))
    with respx.mock() as router:
        route = router.get("https://api.tavily.com/usage").respond(200, json={"account": {}})
        with pytest.raises(CollectionBlockedError):
            await collect_forecast_panel.query_search_quota(["key-a"])
        assert route.call_count == 1
