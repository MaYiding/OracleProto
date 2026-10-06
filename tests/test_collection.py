"""Collection contracts: identity, raw evidence, recovery, and deferred scoring."""
import asyncio
import json
from dataclasses import replace

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
from tests.test_react import _ScriptedLLM, _yes_no_question, _tool_msg, _final_msg
from forecast_eval.types import QFilter


def settings(**kwargs):
    return Settings(_env_file=None, **{
        "LLM_API_KEY": "sk-collection-secret", "TAVILY_API_KEY": ["tvly-collection-secret"],
        "LEAK_DETECTOR_API_KEY": "sk-detector-secret", "MODELS": ["test-arm"],
        "MODEL_PROFILES": {"test-arm": {"model": "provider-model", "reasoning_effort": "high"}},
        "ENABLE_SEARCH_LEAK_FILTER": False, **kwargs,
    })


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


@pytest.mark.parametrize("function", [{"arguments": "{}"}, {"name": None, "arguments": "{}"},
                                     {"name": "", "arguments": "{}"}])
@respx.mock
async def test_nameless_tool_call_preserves_raw_response_and_fixed_budget(tmp_path, monkeypatch, function):
    config = settings(
        SCORE_ANSWERS=False, WRITE_REQUEST_AUDIT=True, WRITE_MESSAGES_TRACE=True,
        REACT_REFLECTION_PROTOCOL=False, REACT_MAX_STEPS=6, REACT_MAX_SEARCH_CALLS=[4],
        REACT_MIN_SEARCH_CALLS=0, REACT_MAX_NUDGES=0, REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=False,
        LLM_BASE_URL="https://provider.test/v1",
    ).model_copy(update={"TAVILY_MAX_RESULTS": 5, "REACT_MAX_SEARCH_CALLS": 4})
    malformed = {"id": "call_bad", "type": "function", "function": function}
    first = _success_body()
    first["choices"][0]["message"] = _tool_msg("call_good", "bounded evidence query")
    first["choices"][0]["message"]["tool_calls"].append(malformed)
    first["choices"][0]["finish_reason"] = "tool_calls"
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json=first)
        assistant = next(m for m in body["messages"] if m.get("tool_calls"))
        calls = assistant["tool_calls"]
        assert calls[0]["function"]["name"] == "web_search"
        assert calls[1]["function"] == {**function, "name": "invalid_tool_call"}
        tool_results = [m for m in body["messages"] if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_results] == ["call_good", "call_bad"]
        assert json.loads(tool_results[1]["content"])["error"].startswith("unknown tool:")
        return httpx.Response(200, json=_success_body())

    model = respx.post("https://provider.test/v1/chat/completions").mock(side_effect=respond)
    search = respx.post("https://api.tavily.com/search").respond(200, json={"results": []})
    monkeypatch.setattr(react, "llm_chat", llm.chat)
    monkeypatch.setattr(react, "parse_gt", lambda _: pytest.fail("collection must not score gold"))
    conn = db.connect(tmp_path / "nameless.db")
    db.init_schema(conn, 1)
    async with AsyncOpenAI(api_key="test-key", base_url=config.LLM_BASE_URL, max_retries=0) as client:
        monkeypatch.setattr(llm, "get_client", lambda _: client)
        with db.capture_requests(conn, settings=config, run_id="r", model="test-arm", question_id="q", sample_idx=0):
            result = await react.run_react(_yes_no_question(), model="test-arm", sample_idx=0,
                                          settings=config, templates=DEFAULT_PROMPT_TEMPLATES, run_id="r")
    assert model.call_count == 2 and search.call_count == 1
    assert result.react_steps == 2 and result.tool_calls_count == 1 and result.correct is None
    assert result.error is None and result.final_answer_retry_used == 0
    trace = json.loads(result.messages_trace)
    assert next(m for m in trace if m.get("tool_calls"))["tool_calls"][1] == malformed
    assert first["choices"][0]["message"]["tool_calls"][1] == malformed
    events = conn.execute("SELECT kind,payload FROM request_events ORDER BY event_id").fetchall()
    responses = [json.loads(row["payload"]) for row in events if row["kind"] == "llm.response"]
    assert json.loads(responses[0]["body"])["choices"][0]["message"]["tool_calls"][1] == malformed
    outbound = [json.loads(row["payload"]) for row in events if row["kind"] == "llm.request"]
    assert "message_normalizations" not in outbound[0]
    assert outbound[1]["message_normalizations"] == [{
        "message_index": 1, "tool_call_index": 1, "tool_call_id": "call_bad", "field": "function.name",
        "original_present": "name" in function, "original_value": function.get("name"),
        "wire_value": "invalid_tool_call",
    }]
    assert all(request["tools"][0]["function"]["name"] == "web_search" for request in requests)
    conn.close()


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


@pytest.mark.parametrize("invalid_count", [1, 4])
@respx.mock
async def test_invalid_query_preserves_audit_and_fixed_search_budget(tmp_path, monkeypatch, invalid_count):
    config = settings(
        SCORE_ANSWERS=False, REQUIRE_HEALTHY_RETRIEVAL=True, WRITE_REQUEST_AUDIT=True,
        WRITE_MESSAGES_TRACE=True, REACT_REFLECTION_PROTOCOL=False, REACT_MAX_STEPS=6,
        REACT_MAX_SEARCH_CALLS=[4], REACT_MIN_SEARCH_CALLS=0, REACT_MAX_NUDGES=0,
        REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=False, SEARCH_BACKOFF_S=[0, 0, 0],
    ).model_copy(update={"TAVILY_MAX_RESULTS": 5, "REACT_MAX_SEARCH_CALLS": 4})
    rejection = {"detail": {"error": "Query cannot consist only of site: operators. Please provide search terms."}}
    queries = ["site:androidcentral.com/phones/google-pixel-11"] * invalid_count
    queries += [f"Google Pixel 11 launch evidence {i}" for i in range(4 - invalid_count)]
    scripted = _ScriptedLLM([(_tool_msg(f"call_{i}", query), "tool_calls", {}) for i, query in enumerate(queries)]
                            + [(_final_msg(), "stop", {})])
    tools_seen = []

    async def predict(**kwargs):
        tools_seen.append(kwargs["tools"])
        return await scripted(**kwargs)

    def search_response(request):
        query = json.loads(request.content)["query"]
        return httpx.Response(400, json=rejection) if query.startswith("site:") else httpx.Response(200, json={"results": []})

    route = respx.post("https://api.tavily.com/search").mock(side_effect=search_response)
    monkeypatch.setattr(react, "llm_chat", predict)
    monkeypatch.setattr(react, "parse_gt", lambda _: pytest.fail("collection must not score gold"))
    conn = db.connect(tmp_path / "query.db")
    db.init_schema(conn, 1)
    with db.capture_requests(conn, settings=config, run_id="r", model="test-arm", question_id="q", sample_idx=0):
        result = await react.run_react(_yes_no_question(), model="test-arm", sample_idx=0,
                                      settings=config, templates=DEFAULT_PROMPT_TEMPLATES, run_id="r")
    assert route.call_count == 4
    calls = json.loads(result.search_calls)
    assert len(calls) == 4 and [row["query"] for row in calls] == queries
    assert sum(row.get("error_kind") == "invalid_query" for row in calls) == invalid_count
    assert result.react_steps == 5 and result.correct is None
    assert tools_seen[-1] == [] and all(tools_seen[:-1])
    tool_payloads = [json.loads(row["content"]) for row in json.loads(result.messages_trace) if row["role"] == "tool"]
    assert len(tool_payloads) == 4
    assert all(row["error"] == "invalid_query" and row["message"] == rejection["detail"]["error"]
               for row in tool_payloads[:invalid_count])
    assert all(row == {"results": []} for row in tool_payloads[invalid_count:])
    events = conn.execute("SELECT * FROM request_events ORDER BY event_id").fetchall()
    assert [row["kind"] for row in events] == ["search.request", "search.response"] * 4
    for i in range(4):
        request, response = events[2 * i:2 * i + 2]
        assert request["request_id"] == response["request_id"]
        body = json.loads(request["payload"])["body"]
        assert body["query"] == queries[i] and body["end_date"] == calls[i]["end_date"]
        assert "api_key" not in body
        raw = json.loads(response["payload"])
        assert raw["status"] == (400 if i < invalid_count else 200)
        if i < invalid_count:
            assert json.loads(raw["body"]) == rejection
    conn.close()


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


@pytest.mark.parametrize("drain", [False, True])
@pytest.mark.parametrize("retain,error,terminal", [(False, "content_policy", False),
    (True, "content_policy", True), (True, "network", False), (True, "bad_request", False)])
async def test_model_refusal_retention_preserves_error_and_resume_boundary(tmp_path, monkeypatch, retain, error, terminal, drain):
    config = settings(SAMPLING_N=1, REQUIRE_HEALTHY_RETRIEVAL=True,
                      COLLECTION_RETAIN_MODEL_REFUSALS=retain, COLLECTION_DRAIN_ON_ERROR=drain, DB_COMMIT_BATCH=1)
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


@pytest.mark.parametrize("failure", ["row", "auth", "retrieval"])
async def test_collection_failure_drains_started_samples_without_admitting_pending(tmp_path, monkeypatch, failure):
    config = settings(SAMPLING_N=3, LLM_MAX_CONCURRENCY=2, COLLECTION_DRAIN_ON_ERROR=True,
                      REQUIRE_HEALTHY_RETRIEVAL=True, DB_COMMIT_BATCH=1)
    conn = db.connect(tmp_path / "drain.db")
    db.init_schema(conn, 3)
    questions = [replace(_yes_no_question(), id=qid) for qid in ("q1", "q2")]
    for q in questions:
        conn.execute("INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (q.id, q.choice_type, q.question_type, q.event, q.options, q.answer, q.end_time, db.utcnow_iso()))
    both_started = asyncio.Event()
    failure_returned = asyncio.Event()
    finish_started = asyncio.Event()
    release_writer = asyncio.Event()
    started, cancelled = [], []
    original_enqueue = runner.AsyncWriter.enqueue_result
    async def delayed_error_write(writer, row):
        if row.get("error"):
            await release_writer.wait()
        await original_enqueue(writer, row)
    monkeypatch.setattr(runner.AsyncWriter, "enqueue_result", delayed_error_write)
    async def execute(task, **kwargs):
        started.append((task.question.id, task.sample_idx))
        request = db.audit_event("llm.request", {"body": {"model": "fixture"}})
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        if task.question.id == "q1" and task.sample_idx == 0:
            db.audit_event("llm.error", {"type": failure}, request_id=request)
            failure_returned.set()
            if failure == "auth":
                raise runner.AuthError("fixture auth failure")
            if failure == "retrieval":
                raise CollectionBlockedError("fixture retrieval failure")
            return runner._error_row(task.question, task.sample_idx, "network")
        try:
            await finish_started.wait()
        except asyncio.CancelledError:
            cancelled.append(task.sample_idx)
            db.audit_event("llm.cancelled", {}, request_id=request)
            raise
        db.audit_event("llm.response", {"status": 200}, request_id=request)
        return {**runner._error_row(task.question, task.sample_idx, "fixture"), "error": None}
    monkeypatch.setattr(runner, "_run_task_with_retry", execute)
    running = asyncio.create_task(runner.run(settings=config, filters=QFilter(), questions=questions,
        templates={}, run_id="r", conns={"test-arm": conn}))
    await asyncio.wait_for(failure_returned.wait(), 2)
    finish_started.set()
    await asyncio.sleep(0.02)
    assert started == [("q1", 0), ("q1", 1)]
    assert not cancelled
    release_writer.set()
    stats = await asyncio.wait_for(running, 2)
    assert stats.aborted and stats.abort_reason
    assert stats.planned == 2 and stats.deferred == 4
    assert db.load_completed_samples(conn, 3) == {("q1", 1)}
    assert conn.execute("SELECT count(*) FROM request_events WHERE kind='llm.cancelled'").fetchone()[0] == 0
    resumed = []
    async def complete(task, **kwargs):
        resumed.append((task.question.id, task.sample_idx))
        return {**runner._error_row(task.question, task.sample_idx, "fixture"), "error": None}
    monkeypatch.setattr(runner, "_run_task_with_retry", complete)
    stats = await runner.run(settings=config, filters=QFilter(), questions=questions, templates={},
                             run_id="r", conns={"test-arm": conn})
    assert not stats.aborted and stats.planned == 5 and stats.deferred == 0
    assert ("q1", 1) not in resumed and len(resumed) == 5
    assert len(db.load_completed_samples(conn, 3)) == 6
    conn.close()


@pytest.mark.parametrize("searches_before,error_count,fallback", [(0, 1, False), (4, 1, False), (0, 6, True)])
@respx.mock
async def test_generated_tool_error_consumes_round_and_preserves_http_failure(
        tmp_path, monkeypatch, searches_before, error_count, fallback):
    config = settings(SCORE_ANSWERS=False, WRITE_REQUEST_AUDIT=True, WRITE_MESSAGES_TRACE=True,
        REACT_REFLECTION_PROTOCOL=False, REACT_MAX_STEPS=6, REACT_MAX_SEARCH_CALLS=[4],
        REACT_MIN_SEARCH_CALLS=0, REACT_MAX_NUDGES=0, REACT_FINAL_ANSWER_RETRY=fallback,
        LLM_BASE_URL="https://provider.test/v1").model_copy(
            update={"TAVILY_MAX_RESULTS": 5, "REACT_MAX_SEARCH_CALLS": 4})
    error = {"error": {"type": "invalid_request_error", "code": "tool_use_failed", "message":
        "Tool choice is none, but model called a tool" if searches_before else
        "Tool call validation failed: parameters for tool web_search did not match schema: missing properties: 'query'"}}
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) <= searches_before:
            reply = _success_body()
            reply["choices"][0].update(message=_tool_msg(str(len(requests)), "historical evidence"), finish_reason="tool_calls")
            return httpx.Response(200, json=reply)
        if searches_before:
            assert "tools" not in body
        if len(requests) <= searches_before + error_count:
            return httpx.Response(400, json=error)
        assert any(m["role"] == "user" and '"error":' in m["content"] for m in body["messages"])
        return httpx.Response(200, json=_success_body())
    model = respx.post("https://provider.test/v1/chat/completions").mock(side_effect=respond)
    search = respx.post("https://api.tavily.com/search").respond(200, json={"results": []})
    monkeypatch.setattr(react, "llm_chat", llm.chat)
    monkeypatch.setattr(react, "parse_gt", lambda _: pytest.fail("must not score"))
    conn = db.connect(tmp_path / "tool-errors.db")
    db.init_schema(conn, 1)
    async with AsyncOpenAI(api_key="test-key", base_url=config.LLM_BASE_URL, max_retries=0) as client:
        monkeypatch.setattr(llm, "get_client", lambda _: client)
        with db.capture_requests(conn, settings=config, run_id="r", model="test-arm", question_id="q", sample_idx=0):
            result = await react.run_react(_yes_no_question(), model="test-arm", sample_idx=0,
                settings=config, templates=DEFAULT_PROMPT_TEMPLATES, run_id="r")
    exhausted = error_count == 6
    assert result.error == ("tool_use_failed" if exhausted else None)
    assert model.call_count == result.react_steps == min(6, searches_before + error_count + 1)
    assert result.final_answer_retry_used == 0 and result.correct is None
    assert search.call_count == result.tool_calls_count == searches_before
    metrics = json.loads(result.step_metrics)
    failures = [m for m in metrics if m.get("error") == "tool_use_failed"]
    assert len(failures) == error_count
    assert all(m["prompt"] is None and m["completion"] is None and not m["usage_reported"] for m in failures)
    events = conn.execute("SELECT kind,request_id,payload FROM request_events ORDER BY event_id").fetchall()
    errors = [e for e in events if e["kind"] == "llm.error"]
    assert len(errors) == error_count
    assert {e["request_id"] for e in errors} == {m["request_id"] for m in failures}
    assert all(json.loads(e["payload"])["status"] == 400 and json.loads(json.loads(e["payload"])["body"]) == error for e in errors)
    trace = json.loads(result.messages_trace)
    assert sum(m["role"] == "assistant" for m in trace) == searches_before + (not exhausted)
    assert not exhausted or result.final_answer_raw == ""
    conn.close()


def test_tool_failure_deferral_keeps_inference_hash_and_requires_audit():
    config = settings()
    deferred = settings(COLLECTION_DEFER_TOOL_FAILURES=True)
    assert db.snapshot_settings(deferred)["COLLECTION_DEFER_TOOL_FAILURES"] is True
    assert db.compute_collection_contract_hash(db.snapshot_settings(config)) == db.compute_collection_contract_hash(db.snapshot_settings(deferred))
    with pytest.raises(ValueError, match="requires WRITE_REQUEST_AUDIT"):
        settings(COLLECTION_DEFER_TOOL_FAILURES=True, WRITE_REQUEST_AUDIT=False)


@pytest.mark.parametrize("virtual", [False, True])
async def test_paused_profile_keeps_pending_slots_and_can_resume(tmp_path, monkeypatch, virtual):
    config=settings(SAMPLING_N=1, SCORE_ANSWERS=False, COLLECTION_PAUSED_PROFILES=["test-arm"])
    slug="test-arm::r5::c4" if virtual else "test-arm"
    config=config.model_copy(update={"MODELS":[slug]})
    conn=db.connect(tmp_path / "paused.db");db.init_schema(conn,1)
    q=_yes_no_question()
    conn.execute("INSERT INTO questions (id,choice_type,question_type,event,options,answer,end_time,imported_at) VALUES (?,?,?,?,?,?,?,?)",
        (q.id,q.choice_type,q.question_type,q.event,q.options,q.answer,q.end_time,db.utcnow_iso()))
    calls=[]
    async def generate(task, **kwargs):
        calls.append(task.model)
        return runner._error_row(task.question,task.sample_idx,None)
    monkeypatch.setattr(runner,"_run_task_with_retry",generate)
    stats=await runner.run(settings=config,filters=QFilter(),questions=[q],templates={},run_id="r",conns={slug:conn})
    assert not calls and stats.planned==0 and stats.deferred==1 and stats.completed_preexisting==0
    assert not db.load_completed_samples(conn,1)
    config=config.model_copy(update={"COLLECTION_PAUSED_PROFILES":[]})
    stats=await runner.run(settings=config,filters=QFilter(),questions=[q],templates={},run_id="r",conns={slug:conn})
    assert calls==[slug] and stats.deferred==0 and db.load_completed_samples(conn,1)=={(q.id,0)}
    conn.close()


def test_profile_pause_is_dispatch_only_and_validates_names():
    original=db.snapshot_settings(settings())
    paused=db.snapshot_settings(settings(COLLECTION_PAUSED_PROFILES=["test-arm"]))
    assert paused["COLLECTION_PAUSED_PROFILES"]==["test-arm"]
    assert db.compute_collection_contract_hash(original)==db.compute_collection_contract_hash(paused)
    for names, reason in [(["other"],"declared in MODELS"),(["test-arm","test-arm"],"unique")]:
        with pytest.raises(ValueError,match=reason):settings(COLLECTION_PAUSED_PROFILES=names)
