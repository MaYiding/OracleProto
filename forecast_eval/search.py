from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

from .config import Settings
from .db import audit_event
from .errors import CollectionBlockedError
from .tavily_keys import AllKeysExhausted, TavilyKeyPool, get_pool


TAVILY_ENDPOINT = "https://api.tavily.com/search"
# Tavily uses 401 / 403 to signal an invalid key or permission issue (permanent
# blacklist); 429 / quota-related responses also use 429 (temporary cooldown).
# Recognized query validation failures belong to the tool caller, not the key.
_AUTH_STATUS = frozenset({401, 403})
_RATE_LIMIT_STATUS = frozenset({429})
_SITE_ONLY_QUERY_ERROR = (
    "Query cannot consist only of site: operators. Please provide search terms."
)


@dataclass
class SearchResultItem:
    title: str
    url: str
    content: str
    published_date: str | None = None
    # Relevance score returned by Tavily (0-1, higher is more relevant). None
    # when missing.
    score: float | None = None
    # Full page body (markdown / text). Tavily does not return this when
    # include_raw_content="false"; left as None. The length cap is enforced by
    # settings.TAVILY_RAW_CONTENT_MAX_CHARS at _parse_tavily_response.
    raw_content: str | None = None


@dataclass
class SearchResult:
    query: str
    end_date: str
    answer: str | None = None
    results: list[SearchResultItem] = field(default_factory=list)
    error_kind: str | None = None
    error_message: str | None = None
    # search-leak-filter-v1: detector audit metadata (None when leak filter is
    # disabled or when Tavily failed). MUST appear after every other default
    # field so dataclass field-ordering rules stay satisfied. NOT exposed via
    # to_llm_payload() — the audit dict is for search_calls JSON only.
    audit: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error_kind is None

    def to_llm_payload(self) -> dict[str, Any]:
        """Compact payload handed back to the LLM.

        Optional fields (published_date / score / raw_content / answer) are
        only emitted when non-None, to avoid the LLM seeing a pile of null
        placeholders that dilute the judgment signal. The error path keeps
        answer=None so consumers can tell success / failure apart at a glance.
        """
        if self.error_kind is not None:
            return {
                "error": self.error_kind,
                "message": self.error_message or "",
                "results": [],
                "answer": None,
            }
        items: list[dict[str, Any]] = []
        for r in self.results:
            item: dict[str, Any] = {
                "title": r.title,
                "url": r.url,
                "content": r.content,
            }
            if r.published_date is not None:
                item["published_date"] = r.published_date
            if r.score is not None:
                item["score"] = r.score
            if r.raw_content is not None:
                item["raw_content"] = r.raw_content
            items.append(item)
        payload: dict[str, Any] = {"results": items}
        if self.answer is not None:
            payload["answer"] = self.answer
        return payload


def _truncate_raw_content(text: str | None, max_chars: int) -> str | None:
    """`max_chars=0` means no truncation; appends a truncation hint when the
    threshold is hit."""
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…[truncated to {max_chars} chars]"


def _parse_tavily_response(
    data: dict[str, Any],
    query: str,
    end_date: str,
    *,
    raw_content_max_chars: int,
) -> SearchResult:
    answer = data.get("answer")
    items: list[SearchResultItem] = []
    for raw in data.get("results", []) or []:
        if not isinstance(raw, dict):
            continue
        score_raw = raw.get("score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        rc = raw.get("raw_content")
        rc_str = str(rc) if isinstance(rc, str) and rc else None
        rc_str = _truncate_raw_content(rc_str, raw_content_max_chars)
        items.append(
            SearchResultItem(
                title=str(raw.get("title") or ""),
                url=str(raw.get("url") or ""),
                content=str(raw.get("content") or ""),
                published_date=raw.get("published_date"),
                score=score,
                raw_content=rc_str,
            )
        )
    return SearchResult(query=query, end_date=end_date, answer=answer, results=items, raw_response=data)


def _build_request_payload(
    *,
    query: str,
    end_date: str,
    settings: Settings,
    api_key: str,
) -> dict[str, Any]:
    """Assemble the Tavily request body. Maps enum strings to the types the
    Tavily protocol accepts, and omits the include_answer field entirely when
    set to "false" (Tavily's default is already false).

    `api_key` is supplied by the caller from TavilyKeyPool (rather than read
    from settings.TAVILY_API_KEY, which is a list[str]); this lets each
    request swap keys to enable rotation + circuit breaking.
    """
    raw_setting = settings.TAVILY_INCLUDE_RAW_CONTENT
    # Tavily's include_raw_content accepts bool | "markdown" | "text"
    raw_param: bool | str = False if raw_setting == "false" else raw_setting

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "end_date": end_date,
        "max_results": settings.TAVILY_MAX_RESULTS,
        "search_depth": settings.TAVILY_SEARCH_DEPTH,
        "include_raw_content": raw_param,
    }
    if settings.TAVILY_INCLUDE_ANSWER != "false":
        payload["include_answer"] = settings.TAVILY_INCLUDE_ANSWER
    return payload


async def _single_request(
    client: httpx.AsyncClient,
    *,
    query: str,
    end_date: str,
    settings: Settings,
    api_key: str,
) -> httpx.Response:
    payload = _build_request_payload(query=query, end_date=end_date, settings=settings, api_key=api_key)
    request_id = audit_event("search.request", {
        "endpoint": TAVILY_ENDPOINT, "body": {k: v for k, v in payload.items() if k != "api_key"},
    })
    try:
        response = await client.post(TAVILY_ENDPOINT, json=payload, timeout=settings.LLM_TIMEOUT_S)
    except asyncio.CancelledError:
        audit_event("search.cancelled", {}, request_id=request_id)
        raise
    except Exception as exc:
        audit_event("search.error", {"type": type(exc).__name__, "message": str(exc)}, request_id=request_id)
        raise
    audit_event("search.response", {"status": response.status_code, "body": response.text}, request_id=request_id)
    return response


def _query_validation_error(response: httpx.Response) -> dict[str, Any] | None:
    """Only an explicit query rejection is safe to return as a tool error."""
    if response.status_code != 400:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and detail.get("error") == _SITE_ONLY_QUERY_ERROR:
        return body
    return None


async def tavily_search(
    query: str,
    end_date: str,
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    pool: TavilyKeyPool | None = None,
) -> SearchResult:
    """Call Tavily /search with the given `end_date` injected by the caller.

    Multi-key behaviour:
    - Before each attempt, acquire a least-used healthy key from `pool` (by
      default a process-wide cache keyed on settings.TAVILY_API_KEY).
    - 401/403 -> permanently blacklist that key, **does NOT count as a network
      retry**, immediately swap key and try again (no sleep).
    - 429 -> temporarily cool down that key, also immediately swap key and
      try again.
    - Explicit HTTP 400 site-only query rejection -> return `invalid_query`
      without retrying or penalizing the key; the ReAct search budget is consumed.
    - 5xx / network / bad-JSON -> consumes the `SEARCH_RETRY_MAX` network-retry
      quota, sleeps per `SEARCH_BACKOFF_S` then retries (key not blacklisted).
    - All keys unavailable (AllKeysExhausted) -> return the error immediately,
      no further rotation.

    Query validation errors return `SearchResult` for the ReAct tool cycle.
    Unavailable retrieval raises `CollectionBlockedError` when strict collection
    is enabled; otherwise it returns an error result.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    if pool is None:
        pool = get_pool(list(settings.TAVILY_API_KEY), float(settings.TAVILY_KEY_COOLDOWN_S))

    sequence = list(settings.SEARCH_BACKOFF_S)
    max_attempts = max(1, int(settings.SEARCH_RETRY_MAX))
    n_keys = max(1, len(pool.states))
    # Hard ceiling to guard against an infinite loop when cooldown_s=0 or all
    # keys keep returning 429; the normal path exits via AllKeysExhausted or
    # max_attempts.
    hard_limit = max_attempts + n_keys
    last_error_kind: str | None = None
    last_error_message: str | None = None
    raw_content_max_chars = int(settings.TAVILY_RAW_CONTENT_MAX_CHARS)

    network_attempts = 0
    total_iterations = 0

    try:
        while total_iterations < hard_limit:
            total_iterations += 1
            try:
                api_key = await pool.acquire()
            except AllKeysExhausted as e:
                if getattr(settings, "REQUIRE_HEALTHY_RETRIEVAL", False):
                    raise CollectionBlockedError("Tavily key pool exhausted; collection paused") from e
                last_error_kind = "all_keys_exhausted"
                last_error_message = str(e)
                logger.warning("tavily {}", last_error_message)
                break

            network_class_error = False
            try:
                resp = await _single_request(
                    client, query=query, end_date=end_date, settings=settings, api_key=api_key
                )
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                # Transient-network family aligned with errors.classify.
                # RemoteProtocolError (server disconnect
                # mid-response) is caught here so the runner does not see
                # UNKNOWN, which it never retries — see search-tool spec.
                httpx.WriteTimeout,
                httpx.WriteError,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
                asyncio.TimeoutError,
            ) as e:
                await pool.report_failure(api_key, "other")
                last_error_kind = "network"
                last_error_message = f"{type(e).__name__}: {e}"
                logger.debug(
                    "tavily network error iter={} err={}", total_iterations, last_error_message
                )
                network_class_error = True
            else:
                status = resp.status_code
                query_error = _query_validation_error(resp)
                if query_error is not None:
                    await pool.report_ok(api_key)
                    return SearchResult(
                        query=query,
                        end_date=end_date,
                        error_kind="invalid_query",
                        error_message=_SITE_ONLY_QUERY_ERROR,
                        raw_response=query_error,
                    )
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        # 200 but not JSON: not a key problem, consumes the
                        # network-retry quota; key is not blacklisted.
                        await pool.report_ok(api_key)
                        last_error_kind = "bad_response"
                        last_error_message = f"Tavily returned non-JSON body: {e}"
                        network_class_error = True
                    else:
                        await pool.report_ok(api_key)
                        result = _parse_tavily_response(
                            data,
                            query=query,
                            end_date=end_date,
                            raw_content_max_chars=raw_content_max_chars,
                        )
                        # search-leak-filter-v1: Stage-2 detector pass. Local
                        # import to avoid an import cycle (leak_filter imports
                        # SearchResult / SearchResultItem from this module).
                        if settings.ENABLE_SEARCH_LEAK_FILTER:
                            from . import leak_filter  # noqa: PLC0415 — break import cycle
                            result = await leak_filter.filter_search_result(
                                result,
                                end_date=end_date,
                                settings=settings,
                            )
                        return result
                elif status in _AUTH_STATUS or status in (432, 433):
                    await pool.report_failure(api_key, "auth")
                    last_error_kind = "auth"
                    last_error_message = f"HTTP {status}: {resp.text[:500]}"
                    logger.debug("tavily auth error iter={} status={}", total_iterations, status)
                    # Does not count as a network retry, no sleep — just swap
                    # key and retry.
                elif status in _RATE_LIMIT_STATUS:
                    await pool.report_failure(api_key, "rate_limit")
                    last_error_kind = "rate_limit"
                    last_error_message = f"HTTP {status}: {resp.text[:500]}"
                    logger.debug(
                        "tavily rate limit iter={} status={}", total_iterations, status
                    )
                    # Same as above: does not count as an attempt, no sleep.
                else:
                    # 5xx / other 4xx: treat as a transient server fault, do
                    # not blacklist the key, consume the network-retry quota.
                    await pool.report_failure(api_key, "other")
                    last_error_kind = "http_error"
                    last_error_message = f"HTTP {status}: {resp.text[:500]}"
                    logger.debug(
                        "tavily http error iter={} status={}", total_iterations, status
                    )
                    network_class_error = True

            if network_class_error:
                network_attempts += 1
                if network_attempts >= max_attempts:
                    break
                idx = network_attempts - 1
                wait_s = float(sequence[idx]) if idx < len(sequence) else 0.0
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
            # Otherwise (auth / rate_limit): fall through to the next loop
            # iteration to swap key, no sleep.
    finally:
        if owns_client:
            await client.aclose()

    if getattr(settings, "REQUIRE_HEALTHY_RETRIEVAL", False):
        raise CollectionBlockedError(f"Tavily unavailable: {last_error_kind}; collection paused")
    return SearchResult(
        query=query,
        end_date=end_date,
        error_kind="tavily_error",
        error_message=last_error_message or last_error_kind or "Tavily failed",
    )
