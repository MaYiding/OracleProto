from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from .config import Settings
from .db import audit_event
from .errors import (
    AuthError,
    ErrorKind,
    backoff_seconds,
    classify,
    parse_retry_after,
    should_retry,
)
from .tools import WEB_SEARCH_SCHEMA


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class ChatResponse:
    """Minimal wrapper around an OpenAI-compatible chat completion response.

    Callers read `.message` (the serialisable assistant message dict), `.usage`
    (tokens, with reasoning defaulting to 0 when unsupported) and
    `.response_headers` (raw headers, needed for Retry-After parsing). The
    five trailing fields (`finish_reason` / `response_id` / `created` /
    `system_fingerprint` / `service_tier`) are extracted defensively from the
    `ChatCompletion` envelope for downstream observability — they MUST NOT
    drive retry/backoff and default to `None` when the provider omits them.
    """

    message: dict[str, Any]
    usage: Usage
    response_headers: httpx.Headers = field(default_factory=httpx.Headers)
    raw: Any = None
    finish_reason: str | None = None
    response_id: str | None = None
    created: int | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


_client: AsyncOpenAI | None = None


def get_client(settings: Settings) -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            max_retries=0,
        )
    return _client


def _is_reasoning_model(model: str, patterns: list[str]) -> bool:
    """Substring match against reasoning-model slugs. Matched models skip
    temperature / top_p when invoked."""
    lower = model.lower()
    return any(p.lower() in lower for p in patterns if p)


def _assert_no_browsing(*, model: str, tools: list[dict[str, Any]], extra_body: dict[str, Any] | None) -> None:
    """Hard gate: refuse to send a request that would leak information via
    provider-native browsing. These checks duplicate the startup-time Settings
    validation, but we still assert at send-time so a test fixture or partial
    config drift cannot bypass the barrier.
    """
    if model.endswith(":online"):
        raise ValueError(
            f"model {model!r} ends with ':online' — provider-native browsing is not allowed"
        )
    # Empty tools is allowed (ENABLE_WEB_SEARCH=false) — the LLM gets no tool
    # schema at all, which is strictly stricter than the single-tool baseline.
    if tools:
        if len(tools) != 1:
            raise ValueError("tools must contain at most one schema: web_search")
        schema = tools[0]
        if schema is not WEB_SEARCH_SCHEMA:
            # allow callers to pass the same dict by identity; otherwise check shape
            func = schema.get("function", {}) if isinstance(schema, dict) else {}
            if func.get("name") != "web_search":
                raise ValueError(
                    "the only allowed tool schema is web_search; provider-native retrieval is forbidden"
                )
    if extra_body and "plugins" in extra_body:
        raise ValueError("plugins field is forbidden (provider-native browsing)")


def _serialise_message(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_unset=True)
    if isinstance(msg, dict):
        return msg
    raise TypeError(f"cannot serialise message of type {type(msg)!r}")


def _extract_usage(raw_usage: Any) -> Usage:
    if raw_usage is None:
        return Usage()
    if hasattr(raw_usage, "model_dump"):
        d = raw_usage.model_dump()
    elif isinstance(raw_usage, dict):
        d = raw_usage
    else:
        return Usage()
    prompt = int(d.get("prompt_tokens") or 0)
    completion = int(d.get("completion_tokens") or 0)
    reasoning = 0
    ctd = d.get("completion_tokens_details") or {}
    if isinstance(ctd, dict):
        reasoning = int(ctd.get("reasoning_tokens") or 0)
    if not reasoning:
        reasoning = int(d.get("reasoning_tokens") or 0)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, reasoning_tokens=reasoning)


async def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    settings: Settings,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    client: AsyncOpenAI | None = None,
) -> ChatResponse:
    """Send one request to OpenRouter with the web_search tool attached.

    Handles layered retry by ErrorKind: NETWORK / RATE_LIMIT / SERVER_5XX get
    backed off, AUTH raises AuthError immediately, BAD_REQUEST / CONTENT_POLICY
    propagate so the runner can record the right `error` kind and move on.
    """
    tools = tools if tools is not None else [WEB_SEARCH_SCHEMA]
    extra_body: dict[str, Any] = {}
    profile = (getattr(settings, "MODEL_PROFILES", {}) or {}).get(model)
    if profile is not None:
        model = profile.model
        if not profile.replay_reasoning:
            messages = [{key: value for key, value in message.items()
                         if key not in ("reasoning_content", "reasoning_details")}
                        for message in messages]
        if profile.reasoning_effort is not None:
            extra_body["reasoning_effort"] = profile.reasoning_effort
        if profile.reasoning_max_tokens is not None:
            extra_body["reasoning"] = {"max_tokens": profile.reasoning_max_tokens}
    _assert_no_browsing(model=model, tools=tools, extra_body=extra_body)

    c = client or get_client(settings)
    attempt = 0
    last_exc: BaseException | None = None

    # Reasoning models do not accept custom temperature / top_p (o-series /
    # deepseek-r1 / qwq etc. return 400); when a pattern matches we explicitly
    # omit these two fields.
    reasoning_patterns = getattr(settings, "LLM_REASONING_MODEL_PATTERNS", []) or []
    skip_sampling = _is_reasoning_model(model, reasoning_patterns)
    # OpenAI o-series / GPT-5 deprecated `max_tokens` on /v1/chat/completions
    # and require `max_completion_tokens` instead. settings.MODEL_MAX_TOKENS_PARAM
    # overrides exactly per model slug; unlisted models keep the default
    # `max_tokens`.
    max_tokens_param = (
        getattr(settings, "MODEL_MAX_TOKENS_PARAM", {}) or {}
    ).get(model, "max_tokens")
    if profile is not None and profile.max_tokens_param is not None:
        max_tokens_param = profile.max_tokens_param
    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        max_tokens_param: max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        "timeout": timeout if timeout is not None else settings.LLM_TIMEOUT_S,
    }
    # Omit the `tools` key entirely when empty — some OpenAI-compatible providers
    # reject an empty list with 400, and "no tools" is exactly what we want when
    # ENABLE_WEB_SEARCH=false.
    if tools:
        base_kwargs["tools"] = tools
    elif profile is not None and profile.force_tool_choice_none:
        base_kwargs["tool_choice"] = "none"
    if extra_body:
        base_kwargs["extra_body"] = extra_body
    if not skip_sampling:
        omit_fields = set(
            (getattr(settings, "MODEL_OMIT_SAMPLING_FIELDS", {}) or {}).get(model, [])
        )
        if profile is not None and profile.omit_sampling_fields is not None:
            omit_fields = set(profile.omit_sampling_fields)
        profile_temperature = profile.temperature if profile is not None else None
        profile_top_p = profile.top_p if profile is not None else None
        if "temperature" not in omit_fields:
            base_kwargs["temperature"] = (
                temperature if temperature is not None else
                (profile_temperature if profile_temperature is not None else settings.LLM_TEMPERATURE)
            )
        if "top_p" not in omit_fields:
            base_kwargs["top_p"] = top_p if top_p is not None else (profile_top_p if profile_top_p is not None else settings.LLM_TOP_P)

    while True:
        attempt += 1
        wire_body = {k: v for k, v in base_kwargs.items() if k not in ("timeout", "extra_body")}
        request_id = audit_event("llm.request", {
            "endpoint": str(settings.LLM_BASE_URL).rstrip("/") + "/chat/completions",
            "body": {**wire_body, **extra_body}, "attempt": attempt,
        })
        try:
            raw = await c.chat.completions.with_raw_response.create(**base_kwargs)
        except asyncio.CancelledError:
            audit_event("llm.cancelled", {}, request_id=request_id)
            raise
        except Exception as exc:
            audit_event("llm.error", {"type": type(exc).__name__, "message": str(exc),
                        "status": getattr(exc, "status_code", None),
                        "body": getattr(getattr(exc, "response", None), "text", None)}, request_id=request_id)
            kind = classify(exc)
            if kind is ErrorKind.AUTH:
                raise AuthError(str(exc)) from exc
            last_exc = exc
            if not should_retry(kind):
                raise
            retry_after = parse_retry_after(getattr(getattr(exc, "response", None), "headers", None))
            wait = backoff_seconds(kind, attempt, settings, retry_after=retry_after)
            if wait is None:
                raise
            logger.warning(
                "llm.chat retry model={} attempt={} kind={} wait={}s err={}",
                model,
                attempt,
                kind,
                wait,
                exc,
            )
            await asyncio.sleep(wait)
            continue

        audit_event("llm.response", {"status": getattr(raw, "status_code", None),
                    "body": getattr(raw, "text", None)}, request_id=request_id)
        parsed = raw.parse()
        headers = getattr(raw, "headers", httpx.Headers())
        message = _serialise_message(parsed.choices[0].message)
        usage = _extract_usage(getattr(parsed, "usage", None))
        # All five envelope fields below are best-effort: providers (e.g.
        # Anthropic via OpenRouter) may omit `system_fingerprint` /
        # `service_tier`. `finish_reason` is per-choice, the rest are top-level
        # on the ChatCompletion. None values flow straight through to the DB.
        first_choice = parsed.choices[0] if parsed.choices else None
        finish_reason = getattr(first_choice, "finish_reason", None) if first_choice is not None else None
        response_id = getattr(parsed, "id", None)
        created_raw = getattr(parsed, "created", None)
        created = int(created_raw) if isinstance(created_raw, (int, float)) else None
        system_fingerprint = getattr(parsed, "system_fingerprint", None)
        service_tier = getattr(parsed, "service_tier", None)
        return ChatResponse(
            message=message,
            usage=usage,
            response_headers=headers,
            raw=parsed,
            finish_reason=finish_reason,
            response_id=response_id,
            created=created,
            system_fingerprint=system_fingerprint,
            service_tier=service_tier,
        )
