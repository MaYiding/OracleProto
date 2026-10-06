from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import httpx
import pytest
import respx

from forecast_eval.errors import CollectionBlockedError
from forecast_eval.search import (
    SearchResult,
    SearchResultItem,
    TAVILY_ENDPOINT,
    _build_request_payload,
    _truncate_raw_content,
    tavily_search,
)
from forecast_eval.tavily_keys import (
    AllKeysExhausted,
    TavilyKeyPool,
    reset_pool_cache,
)
from forecast_eval.tools import (
    WEB_SEARCH_SCHEMA,
    extract_query,
    parse_tool_arguments,
    tool_error_message,
    tool_result_message,
)


@dataclass
class _StubSettings:
    # After the upgrade, settings.TAVILY_API_KEY is list[str] (CSV multi-key);
    # the test stub follows the same semantics.
    TAVILY_API_KEY: list[str] = field(default_factory=lambda: ["tvly-TEST"])
    TAVILY_KEY_COOLDOWN_S: float = 60.0
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "basic"
    TAVILY_INCLUDE_RAW_CONTENT: str = "false"
    TAVILY_RAW_CONTENT_MAX_CHARS: int = 8000
    TAVILY_INCLUDE_ANSWER: str = "false"
    SEARCH_RETRY_MAX: int = 3
    SEARCH_BACKOFF_S: list[int] = None  # type: ignore[assignment]
    LLM_TIMEOUT_S: int = 30
    # search-leak-filter-v1: off by default, take the non-detector path; tests
    # explicitly enable it when needed.
    ENABLE_SEARCH_LEAK_FILTER: bool = False
    REQUIRE_HEALTHY_RETRIEVAL: bool = False

    def __post_init__(self) -> None:
        if self.SEARCH_BACKOFF_S is None:
            self.SEARCH_BACKOFF_S = [0, 0, 0]  # tests run fast, no real backoff


@pytest.fixture(autouse=True)
def _reset_pool_cache() -> None:
    """Each test gets its own pool cache, preventing key blacklisting in one
    test from leaking into the next.

    The module-level cache being shared across grid cells is a feature in prod;
    in test isolation it's a bug source.
    """
    reset_pool_cache()


# ==================== tools schema/argument ====================

def test_web_search_schema_whitelist() -> None:
    props = WEB_SEARCH_SCHEMA["function"]["parameters"]["properties"]
    assert set(props.keys()) == {"query"}
    assert WEB_SEARCH_SCHEMA["function"]["parameters"]["required"] == ["query"]
    # no end_date / date / max_results leaked into LLM-visible schema
    for forbidden in ("end_date", "date", "time", "max_results"):
        assert forbidden not in props


def test_parse_tool_arguments_happy_path() -> None:
    args, err = parse_tool_arguments('{"query": "foo"}')
    assert err is None and args == {"query": "foo"}

    args, err = parse_tool_arguments(None)
    assert err is None and args == {}


def test_parse_tool_arguments_invalid_json() -> None:
    args, err = parse_tool_arguments("{not json")
    assert args is None
    assert err is not None and "invalid arguments JSON" in err


def test_parse_tool_arguments_not_object() -> None:
    args, err = parse_tool_arguments("[1, 2]")
    assert args is None
    assert err is not None


def test_extract_query_strips_extra_fields() -> None:
    args = {"query": "who won?", "end_date": "2099-01-01"}
    q, err = extract_query(args)
    assert err is None
    assert q == "who won?"


def test_extract_query_missing() -> None:
    q, err = extract_query({})
    assert q is None and err is not None


def test_tool_error_and_result_messages() -> None:
    err = tool_error_message("call-1", "bad")
    assert err["role"] == "tool" and err["tool_call_id"] == "call-1"
    parsed = json.loads(err["content"])
    assert parsed == {"error": "bad"}

    ok = tool_result_message("call-2", {"answer": "yes", "results": []})
    parsed = json.loads(ok["content"])
    assert parsed == {"answer": "yes", "results": []}


# ==================== _build_request_payload (enum mapping) ====================

def test_payload_default_uses_bool_false_and_omits_answer() -> None:
    settings = _StubSettings()
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-PAYLOAD"
    )
    # api_key comes from the caller-injected pool, not settings.TAVILY_API_KEY[0].
    assert p["api_key"] == "tvly-PAYLOAD"
    assert p["search_depth"] == "basic"
    # The "false" string must map to JSON bool false (Tavily protocol: bool | "markdown" | "text")
    assert p["include_raw_content"] is False
    # When include_answer is off by default, the field must not appear in the payload (Tavily default is false)
    assert "include_answer" not in p


def test_payload_include_raw_content_markdown() -> None:
    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="markdown")
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
    assert p["include_raw_content"] == "markdown"


def test_payload_include_raw_content_text() -> None:
    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="text")
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
    assert p["include_raw_content"] == "text"


def test_payload_search_depth_advanced() -> None:
    settings = _StubSettings(TAVILY_SEARCH_DEPTH="advanced")
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
    assert p["search_depth"] == "advanced"


@pytest.mark.parametrize("answer_mode", ["basic", "advanced"])
def test_payload_emits_include_answer_when_enabled(answer_mode: str) -> None:
    settings = _StubSettings(TAVILY_INCLUDE_ANSWER=answer_mode)
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
    assert p["include_answer"] == answer_mode


# ==================== _truncate_raw_content ====================

def test_truncate_raw_content_under_limit_returns_intact() -> None:
    assert _truncate_raw_content("hello", 10) == "hello"
    assert _truncate_raw_content("hello", 5) == "hello"  # equal length, no truncation


def test_truncate_raw_content_over_limit_appends_marker() -> None:
    out = _truncate_raw_content("a" * 100, 10)
    assert out is not None
    assert out.startswith("a" * 10)
    assert "truncated to 10 chars" in out


def test_truncate_raw_content_zero_means_no_truncation() -> None:
    long = "a" * 50000
    assert _truncate_raw_content(long, 0) == long


def test_truncate_raw_content_none_passthrough() -> None:
    assert _truncate_raw_content(None, 1000) is None


# ==================== tavily_search end-to-end ====================

@respx.mock
async def test_tavily_search_success_injects_end_date() -> None:
    route = respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "Team A won.",
                "results": [
                    {
                        "title": "Match recap",
                        "url": "https://example.com/a",
                        "content": "Short summary",
                        "published_date": "2026-01-16",
                        "score": 0.71,
                        "raw_content": "## Heading\nfull markdown body...",
                    },
                    {
                        "title": "Follow-up",
                        "url": "https://example.com/b",
                        "content": "details",
                    },
                ],
            },
        )
    )

    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="markdown")
    result = await tavily_search("who won yesterday?", "2026-01-17", settings)

    assert route.called
    body = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert body["query"] == "who won yesterday?"
    assert body["end_date"] == "2026-01-17"
    assert body["max_results"] == 5
    assert body["search_depth"] == "basic"
    assert body["include_raw_content"] == "markdown"
    assert "include_answer" not in body  # default false -> not sent

    assert result.ok
    assert result.answer == "Team A won."
    assert len(result.results) == 2
    assert result.results[0].published_date == "2026-01-16"
    assert result.results[0].score == 0.71
    assert result.results[0].raw_content == "## Heading\nfull markdown body..."
    assert result.results[1].published_date is None
    assert result.results[1].score is None
    assert result.results[1].raw_content is None

    payload = result.to_llm_payload()
    assert payload["answer"] == "Team A won."
    items = payload["results"]
    assert items[0]["score"] == 0.71
    assert items[0]["raw_content"].startswith("## Heading")
    # The second item lacks score / raw_content / published_date and must not appear as null fields in the payload
    assert "score" not in items[1]
    assert "raw_content" not in items[1]
    assert "published_date" not in items[1]


@respx.mock
async def test_tavily_search_truncates_long_raw_content() -> None:
    huge = "x" * 50000
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": None,
                "results": [
                    {
                        "title": "t",
                        "url": "https://example.com/a",
                        "content": "c",
                        "raw_content": huge,
                    }
                ],
            },
        )
    )
    settings = _StubSettings(
        TAVILY_INCLUDE_RAW_CONTENT="markdown",
        TAVILY_RAW_CONTENT_MAX_CHARS=8000,
    )
    result = await tavily_search("q", "2026-01-17", settings)
    rc = result.results[0].raw_content
    assert rc is not None
    assert rc.startswith("x" * 8000)
    assert "truncated to 8000 chars" in rc


@respx.mock
async def test_tavily_search_retries_then_succeeds() -> None:
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.Response(503, text="unavailable"),
            httpx.Response(
                200,
                json={"answer": None, "results": []},
            ),
        ]
    )
    settings = _StubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert route.call_count == 2
    assert result.ok
    assert result.answer is None


@respx.mock
async def test_tavily_search_exhausted_returns_error_payload() -> None:
    respx.post(TAVILY_ENDPOINT).mock(return_value=httpx.Response(503, text="down"))
    settings = _StubSettings(SEARCH_RETRY_MAX=2, SEARCH_BACKOFF_S=[0, 0])
    result = await tavily_search("q", "2026-01-17", settings)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    payload = result.to_llm_payload()
    assert payload["error"] == "tavily_error"
    assert payload["results"] == []


@pytest.mark.parametrize("strict", [False, True])
@respx.mock
async def test_site_only_query_returns_tool_error_without_retry_or_key_penalty(strict, monkeypatch):
    body = {"detail": {"error": "Query cannot consist only of site: operators. Please provide search terms."}}
    route = respx.post(TAVILY_ENDPOINT).respond(400, json=body)
    settings = _StubSettings(REQUIRE_HEALTHY_RETRIEVAL=strict, ENABLE_SEARCH_LEAK_FILTER=True)
    pool = TavilyKeyPool.from_keys(settings.TAVILY_API_KEY)
    from forecast_eval import leak_filter
    monkeypatch.setattr(leak_filter, "filter_search_result", lambda *a, **k: pytest.fail("no results to filter"))
    query = "site:androidcentral.com/phones/google-pixel-11"
    result = await tavily_search(query, "2026-08-19", settings, pool=pool)
    assert route.call_count == 1
    assert not result.ok and result.error_kind == "invalid_query"
    assert result.raw_response == body
    assert result.to_llm_payload() == {"error": "invalid_query", "message": body["detail"]["error"],
                                       "results": [], "answer": None}
    request = json.loads(route.calls.last.request.content)
    assert request["query"] == query and request["end_date"] == "2026-08-19"
    assert pool.states[0].used == 1 and not pool.states[0].blacklisted
    assert pool.states[0].cooldown_until == 0


@pytest.mark.parametrize("status,body,attempts", [
    (400, {"detail": {"error": "Account quota exhausted"}}, 3),
    (400, {"detail": ["Unexpected query validation schema"]}, 3),
    (400, None, 3),
    (401, {"detail": {"error": "Query cannot consist only of site: operators. Please provide search terms."}}, 1),
    (429, {"detail": {"error": "Query cannot consist only of site: operators. Please provide search terms."}}, 1),
    (503, {"detail": {"error": "Query cannot consist only of site: operators. Please provide search terms."}}, 3),
])
@respx.mock
async def test_other_tavily_failures_still_block_strict_collection(status, body, attempts):
    response = httpx.Response(status, json=body) if body is not None else httpx.Response(status, text="invalid JSON")
    route = respx.post(TAVILY_ENDPOINT).mock(return_value=response)
    with pytest.raises(CollectionBlockedError):
        await tavily_search("site:example.org", "2026-08-19", _StubSettings(REQUIRE_HEALTHY_RETRIEVAL=True))
    assert route.call_count == attempts


@respx.mock
async def test_tavily_search_network_error_retries() -> None:
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.ConnectError("boom"),
            httpx.Response(200, json={"answer": "x", "results": []}),
        ]
    )
    settings = _StubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert route.call_count == 2
    assert result.ok


@respx.mock
async def test_tavily_search_remote_protocol_error_retries() -> None:
    """v5.1 (harness-resilience): Tavily server hangs up mid-response. Pre-v5.1
    except clause (ConnectError / ReadTimeout / ConnectTimeout / asyncio.TimeoutError)
    let RemoteProtocolError bubble out — sample failed without retry. New
    clause treats it as a normal network blip and retries."""
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
            httpx.Response(200, json={"answer": "ok", "results": []}),
        ]
    )
    settings = _StubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert route.call_count == 2
    assert result.ok
    assert result.answer == "ok"


@respx.mock
async def test_tavily_search_remote_protocol_error_exhausts_to_payload() -> None:
    """RemoteProtocolError repeating up to SEARCH_RETRY_MAX MUST return an
    error payload (not raise). The ReAct loop relies on this so the LLM sees
    a tool_result error and the sample stays alive."""
    respx.post(TAVILY_ENDPOINT).mock(
        side_effect=httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        )
    )
    settings = _StubSettings(SEARCH_RETRY_MAX=2, SEARCH_BACKOFF_S=[0, 0])
    result = await tavily_search("q", "2026-01-17", settings)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    payload = result.to_llm_payload()
    assert payload["error"] == "tavily_error"
    assert payload["results"] == []


# ==================== to_llm_payload conditional emission ====================

def test_payload_emits_score_and_raw_content_when_present() -> None:
    r = SearchResult(
        query="q",
        end_date="2026-01-17",
        answer="cached",
        results=[
            SearchResultItem(
                title="t",
                url="u",
                content="c",
                published_date="2026-01-16",
                score=0.6,
                raw_content="page body",
            )
        ],
    )
    p = r.to_llm_payload()
    assert p["answer"] == "cached"
    item = p["results"][0]
    assert item["score"] == 0.6
    assert item["raw_content"] == "page body"
    assert item["published_date"] == "2026-01-16"


def test_payload_omits_optional_fields_when_none() -> None:
    r = SearchResult(
        query="q",
        end_date="2026-01-17",
        answer=None,
        results=[SearchResultItem(title="t", url="u", content="c")],
    )
    p = r.to_llm_payload()
    # answer=None -> the entire answer field must not appear
    assert "answer" not in p
    item = p["results"][0]
    assert "score" not in item
    assert "raw_content" not in item
    assert "published_date" not in item


# ==================== TavilyKeyPool unit tests ====================


def _used_for(pool: TavilyKeyPool) -> dict[str, int]:
    return {st.key: st.used for st in pool.states}


def _key_state(pool: TavilyKeyPool, key: str):
    for st in pool.states:
        if st.key == key:
            return st
    raise AssertionError(f"key {key!r} not in pool")


async def test_pool_least_used_balances_load() -> None:
    # Force offset=0 so counts start at 0,0,0; tests that least-used selection is symmetric.
    pool = TavilyKeyPool.from_keys(
        ["k-a", "k-b", "k-c"], rng=random.Random(0), cooldown_s=10.0
    )
    # Erase the random starting point so initial counts are all 0 -- tests the
    # least-used decision itself.
    for st in pool.states:
        st.used = 0

    counts: dict[str, int] = {"k-a": 0, "k-b": 0, "k-c": 0}
    for _ in range(9):
        k = await pool.acquire()
        counts[k] += 1
    # 9 / 3 keys -> each key 3 times (least-used + sequential tie-breaker determines balance).
    assert counts == {"k-a": 3, "k-b": 3, "k-c": 3}


async def test_pool_random_starting_offset_rotates_first_key() -> None:
    # Different seeds should produce different initial usage distributions
    # (starting offset works).
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    first_keys = []
    for seed in seeds:
        pool = TavilyKeyPool.from_keys(
            ["k-a", "k-b", "k-c", "k-d"], rng=random.Random(seed), cooldown_s=10.0
        )
        first_keys.append(await pool.acquire())
    # Don't require each seed to differ, but at least shouldn't hit keys[0] all
    # 8 times (that would mean no randomness).
    assert len(set(first_keys)) >= 2


async def test_pool_auth_failure_blacklists_permanently() -> None:
    pool = TavilyKeyPool.from_keys(
        ["k-a", "k-b"], rng=random.Random(0), cooldown_s=10.0
    )
    for st in pool.states:
        st.used = 0
    bad = await pool.acquire()  # k-a (sequential tie-breaker)
    await pool.report_failure(bad, "auth")
    assert _key_state(pool, bad).blacklisted is True

    # No matter how many times we acquire afterwards, bad will never be returned.
    for _ in range(10):
        k = await pool.acquire()
        assert k != bad


async def test_pool_rate_limit_cooldown_then_recover() -> None:
    fake_now = [1000.0]

    pool = TavilyKeyPool.from_keys(
        ["k-a", "k-b"], rng=random.Random(0), cooldown_s=30.0
    )
    pool._now = lambda: fake_now[0]
    for st in pool.states:
        st.used = 0

    rate_limited = await pool.acquire()  # k-a
    await pool.report_failure(rate_limited, "rate_limit")
    # During cooldown, this key must not be returned again.
    for _ in range(5):
        k = await pool.acquire()
        assert k != rate_limited

    # After cooldown, the key should recover (least-used selection prefers it
    # because the other key has already been used 6 times).
    fake_now[0] += 31.0
    saw_recovered = False
    for _ in range(20):
        k = await pool.acquire()
        if k == rate_limited:
            saw_recovered = True
            break
    assert saw_recovered, "key did not recover after cooldown"


async def test_pool_all_blacklisted_raises() -> None:
    pool = TavilyKeyPool.from_keys(["k-a", "k-b"], rng=random.Random(0))
    for k in ["k-a", "k-b"]:
        await pool.report_failure(k, "auth")
    with pytest.raises(AllKeysExhausted):
        await pool.acquire()


async def test_pool_other_failure_does_not_blacklist() -> None:
    pool = TavilyKeyPool.from_keys(["k-a"], rng=random.Random(0))
    await pool.report_failure("k-a", "other")
    # 5xx / network errors must not blacklist a key -- retries can still acquire it.
    k = await pool.acquire()
    assert k == "k-a"


async def test_pool_dedups_keys() -> None:
    pool = TavilyKeyPool.from_keys(["k-a", "k-a", "k-b"], rng=random.Random(0))
    keys_in_pool = [s.key for s in pool.states]
    assert keys_in_pool == ["k-a", "k-b"]


# ==================== tavily_search × multi-key integration ====================


@respx.mock
async def test_tavily_search_distributes_across_multiple_keys() -> None:
    seen_keys: list[str] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_keys.append(body["api_key"])
        return httpx.Response(200, json={"answer": None, "results": []})

    respx.post(TAVILY_ENDPOINT).mock(side_effect=_capture)

    settings = _StubSettings(
        TAVILY_API_KEY=["tvly-K1", "tvly-K2", "tvly-K3"],
    )
    pool = TavilyKeyPool.from_keys(
        list(settings.TAVILY_API_KEY), rng=random.Random(0), cooldown_s=10.0
    )
    # Erase the random starting point so the 6 calls split stably into 2/2/2.
    for st in pool.states:
        st.used = 0

    for _ in range(6):
        result = await tavily_search("q", "2026-01-17", settings, pool=pool)
        assert result.ok

    counts = {k: seen_keys.count(k) for k in ["tvly-K1", "tvly-K2", "tvly-K3"]}
    assert counts == {"tvly-K1": 2, "tvly-K2": 2, "tvly-K3": 2}


@respx.mock
async def test_tavily_search_401_blacklists_then_uses_other_key() -> None:
    seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body["api_key"])
        if body["api_key"] == "tvly-BAD":
            return httpx.Response(401, text="invalid api key")
        return httpx.Response(200, json={"answer": "ok", "results": []})

    respx.post(TAVILY_ENDPOINT).mock(side_effect=_handler)

    pool = TavilyKeyPool.from_keys(
        ["tvly-BAD", "tvly-GOOD"], rng=random.Random(0), cooldown_s=10.0
    )
    for st in pool.states:
        st.used = 0  # tie-breaker -> tvly-BAD picked first

    settings = _StubSettings(
        TAVILY_API_KEY=["tvly-BAD", "tvly-GOOD"], SEARCH_RETRY_MAX=2
    )
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert result.ok
    assert result.answer == "ok"
    # The first 401 doesn't count as a network retry; switch keys and retry
    # immediately; 2 requests in total.
    assert seen == ["tvly-BAD", "tvly-GOOD"]
    assert _key_state(pool, "tvly-BAD").blacklisted is True

    # Subsequent calls should go straight to tvly-GOOD, never trying the blacklisted key.
    seen.clear()
    result2 = await tavily_search("q2", "2026-01-17", settings, pool=pool)
    assert result2.ok
    assert seen == ["tvly-GOOD"]


@respx.mock
async def test_tavily_search_429_cools_down_then_uses_other_key() -> None:
    seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body["api_key"])
        if body["api_key"] == "tvly-FULL":
            return httpx.Response(429, text="rate limit exceeded")
        return httpx.Response(200, json={"answer": None, "results": []})

    respx.post(TAVILY_ENDPOINT).mock(side_effect=_handler)

    pool = TavilyKeyPool.from_keys(
        ["tvly-FULL", "tvly-OK"], rng=random.Random(0), cooldown_s=60.0
    )
    for st in pool.states:
        st.used = 0

    settings = _StubSettings(TAVILY_API_KEY=["tvly-FULL", "tvly-OK"])
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert result.ok
    # 429 doesn't count as a network retry; switch immediately to tvly-OK and succeed.
    assert seen == ["tvly-FULL", "tvly-OK"]
    # tvly-FULL should be in cooldown.
    assert _key_state(pool, "tvly-FULL").cooldown_until > pool._now()


@respx.mock
async def test_tavily_search_all_keys_exhausted_returns_error() -> None:
    respx.post(TAVILY_ENDPOINT).mock(return_value=httpx.Response(401, text="invalid"))
    pool = TavilyKeyPool.from_keys(
        ["tvly-A", "tvly-B"], rng=random.Random(0), cooldown_s=10.0
    )
    settings = _StubSettings(
        TAVILY_API_KEY=["tvly-A", "tvly-B"], SEARCH_RETRY_MAX=1
    )
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    # Both keys should be permanently blacklisted.
    for st in pool.states:
        assert st.blacklisted is True


@respx.mock
async def test_tavily_search_5xx_uses_network_retry_does_not_blacklist() -> None:
    """5xx still uses SEARCH_BACKOFF_S retry, key not blacklisted (server issue, not a key issue)."""
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.Response(503, text="down"),
            httpx.Response(200, json={"answer": None, "results": []}),
        ]
    )
    pool = TavilyKeyPool.from_keys(["tvly-only"], rng=random.Random(0))
    settings = _StubSettings(TAVILY_API_KEY=["tvly-only"])
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert route.call_count == 2
    assert result.ok
    assert _key_state(pool, "tvly-only").blacklisted is False


# ==================== search-leak-filter-v1: detector wiring ====================


@dataclass
class _DetectorStubSettings(_StubSettings):
    """Extend the search _StubSettings with detector knobs the leak_filter reads."""

    ENABLE_SEARCH_LEAK_FILTER: bool = True
    LEAK_DETECTOR_API_KEY: str = "sk-detector-test-1234"
    LEAK_DETECTOR_BASE_URL: str = "https://api.detector.test/v1"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LEAK_DETECTOR_MODEL: str = "anthropic/claude-sonnet-4.6"
    LEAK_DETECTOR_TIMEOUT_S: int = 60
    LEAK_DETECTOR_TEMPERATURE: float = 0.0
    LEAK_DETECTOR_MAX_TOKENS: int = 512
    LEAK_DETECTOR_RETRY_MAX: int = 0
    LEAK_DETECTOR_BACKOFF_S: list[int] = field(default_factory=lambda: [0, 0, 0])
    LEAK_DETECTOR_FAIL_ACTION: str = "drop"
    LEAK_DETECTOR_CONCURRENCY: int = 5


def _patch_detector(monkeypatch: pytest.MonkeyPatch, verdicts: list[str]) -> dict:
    """Install a fake detector by monkey-patching ``leak_filter._detect_one``.

    Returns a dict containing ``calls`` (list of (item.title, cutoff_date)) so
    tests can inspect ordering / count.
    """
    from forecast_eval import leak_filter

    state = {"calls": [], "verdicts": list(verdicts), "i": 0}

    async def fake_detect_one(item, cutoff_date, settings, client):  # noqa: ANN001
        state["calls"].append((item.title, cutoff_date))
        v = state["verdicts"][state["i"]]
        state["i"] += 1
        return v, "stub"

    monkeypatch.setattr(leak_filter, "_detect_one", fake_detect_one)
    return state


def _five_results_payload() -> dict:
    return {
        "answer": "synthesised answer",
        "results": [
            {
                "title": f"t{i}",
                "url": f"https://example.com/{i}",
                "content": f"snippet {i}",
                "published_date": "2026-01-16",
            }
            for i in range(5)
        ],
    }


@respx.mock
async def test_tavily_search_with_leak_filter_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leak filter enabled: 5 items + verdicts [keep,drop,keep,keep,drop] ->
    returns 3 items; result.audit contains 5 fields."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    state = _patch_detector(monkeypatch, ["keep", "drop", "keep", "keep", "drop"])
    settings = _DetectorStubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert result.ok
    assert len(result.results) == 3
    assert [r.title for r in result.results] == ["t0", "t2", "t3"]
    # Full audit fields: the five spec fields + published_dates_raw (used by
    # react to write search_calls.published_dates, length == n_results_raw).
    assert result.audit is not None
    assert {
        "n_results_raw",
        "n_results_kept",
        "detector_verdicts",
        "detector_latency_ms",
        "detector_error_kind",
    } <= set(result.audit.keys())
    assert "published_dates_raw" in result.audit
    assert len(result.audit["published_dates_raw"]) == 5
    assert result.audit["n_results_raw"] == 5
    assert result.audit["n_results_kept"] == 3
    # The cutoff_date the detector receives shares the same source as Tavily's end_date.
    assert all(c[1] == "2026-01-17" for c in state["calls"])


@respx.mock
async def test_tavily_search_with_leak_filter_disabled() -> None:
    """Switch off: byte-identical to the pre-proposal behaviour; audit is None; detector is not called."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    settings = _DetectorStubSettings(ENABLE_SEARCH_LEAK_FILTER=False)
    result = await tavily_search("q", "2026-01-17", settings)
    assert result.ok
    assert len(result.results) == 5  # original count, not trimmed by detector
    assert result.audit is None  # audit defaults to None when the switch is off


@respx.mock
async def test_tavily_search_failed_skips_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tavily retries exhausted -> leak_filter is not called, error_kind=tavily_error."""
    from forecast_eval import leak_filter

    respx.post(TAVILY_ENDPOINT).mock(return_value=httpx.Response(503, text="down"))
    detector_called = {"n": 0}

    async def fake_filter(*args, **kwargs):  # noqa: ANN001
        detector_called["n"] += 1
        raise AssertionError("filter_search_result MUST NOT be called on Tavily failure")

    monkeypatch.setattr(leak_filter, "filter_search_result", fake_filter)
    settings = _DetectorStubSettings(SEARCH_RETRY_MAX=1, SEARCH_BACKOFF_S=[0])
    result = await tavily_search("q", "2026-01-17", settings)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    assert result.audit is None
    assert detector_called["n"] == 0


@respx.mock
async def test_tavily_search_all_dropped_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All 5 dropped -> ok=True, results=[], answer=None (answer is synthesised from dropped content)."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    _patch_detector(monkeypatch, ["drop"] * 5)
    settings = _DetectorStubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert result.ok
    assert result.error_kind is None
    assert result.results == []
    assert result.answer is None
    assert result.audit["n_results_kept"] == 0


@respx.mock
async def test_tavily_search_partial_drop_keeps_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial drop -> answer retained (known trade-off, see spec)."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    _patch_detector(monkeypatch, ["keep", "drop", "keep", "drop", "keep"])
    settings = _DetectorStubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert len(result.results) == 3
    assert result.answer == "synthesised answer"


def test_to_llm_payload_excludes_audit() -> None:
    """to_llm_payload() MUST NOT emit the audit field, preventing it from reaching the main LLM-visible path."""
    r = SearchResult(
        query="q",
        end_date="2026-01-17",
        answer="ok",
        results=[SearchResultItem(title="t", url="u", content="c")],
        audit={
            "n_results_raw": 1,
            "n_results_kept": 1,
            "detector_verdicts": ["keep"],
            "detector_latency_ms": 12,
            "detector_error_kind": None,
        },
    )
    payload = r.to_llm_payload()
    assert "audit" not in payload
    # Also check that every item inside results lacks audit (defensive).
    for item in payload["results"]:
        assert "audit" not in item
