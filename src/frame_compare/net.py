# pyright: standard

"""Shared networking helpers for configurable retries/backoff."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Iterable
from urllib.parse import urlsplit

import httpx
from urllib3.util import Retry

__all__ = [
    "BackoffError",
    "ALLOWED_METHODS",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_READ_TIMEOUT",
    "RETRY_STATUS",
    "build_urllib3_retry",
    "default_requests_timeouts",
    "httpx_get_json_with_backoff",
    "log_backoff_attempt",
    "redact_url_for_logs",
]

logger = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
"""HTTP status codes treated as transient and eligible for backoff."""

ALLOWED_METHODS = frozenset({"GET", "POST"})
"""Request methods that participate in retry logic."""

DEFAULT_CONNECT_TIMEOUT = 10.0
"""Standard connect timeout (seconds) for outbound HTTP calls."""

DEFAULT_READ_TIMEOUT = 30.0
"""Standard read timeout (seconds) for outbound HTTP calls."""

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(
    DEFAULT_READ_TIMEOUT,
    connect=DEFAULT_CONNECT_TIMEOUT,
    read=DEFAULT_READ_TIMEOUT,
)
"""Default per-request timeout matching the project connect/read guidelines."""


class BackoffError(RuntimeError):
    """Raised when network retries are exhausted."""


def build_urllib3_retry(
    total: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: Iterable[int] | None = None,
    allowed_methods: Iterable[str] | None = None,
) -> Retry:
    """Return a configured urllib3 Retry object with project defaults."""

    statuses = frozenset(status_forcelist) if status_forcelist else RETRY_STATUS
    methods = frozenset(allowed_methods) if allowed_methods else ALLOWED_METHODS
    return Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=statuses,
        allowed_methods=methods,
        raise_on_status=False,
    )


def default_requests_timeouts(
    connect: float = DEFAULT_CONNECT_TIMEOUT, read: float = DEFAULT_READ_TIMEOUT
) -> tuple[float, float]:
    """Return standard connect/read timeout values for Requests sessions."""

    return (float(connect), float(read))


def log_backoff_attempt(host: str, attempt: int, delay: float) -> None:
    """Emit a concise log entry describing the next retry window."""

    logger.info("GET %s retry #%d scheduled in %.2f s", host, attempt, delay)


async def httpx_get_json_with_backoff(
    client: httpx.AsyncClient,
    path: str,
    params: Mapping[str, object],
    *,
    retries: int = 3,
    initial_backoff: float = 0.5,
    max_backoff: float = 4.0,
    retry_status: Iterable[int] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    on_backoff: Callable[[float, int], Awaitable[None]] | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> httpx.Response:
    """Perform a GET request with exponential backoff for transient status codes.

    Timeouts default to :data:`DEFAULT_HTTP_TIMEOUT` so requests cannot hang
    indefinitely (see HTTPX's timeout guidance:
    https://github.com/encode/httpx/blob/master/docs/advanced/timeouts.md).
    """

    retry_codes = frozenset(retry_status) if retry_status else RETRY_STATUS
    backoff = max(0.1, initial_backoff)
    upper_backoff = max(0.1, max_backoff)
    sleep_impl = sleep or asyncio.sleep
    last_network_error: httpx.RequestError | None = None
    last_response: httpx.Response | None = None
    max_attempts = max(0, retries) + 1
    base_url = getattr(client, "base_url", "") or ""
    host_label = redact_url_for_logs(str(base_url) or path)
    effective_timeout = timeout if timeout is not None else DEFAULT_HTTP_TIMEOUT

    for attempt_index in range(max_attempts):
        try:
            response = await client.get(path, params=params, timeout=effective_timeout)
        except httpx.RequestError as exc:
            last_network_error = exc
            last_response = None
            delay = backoff
        else:
            status = response.status_code
            if status in retry_codes:
                last_response = response
                delay = _retry_delay_from_response(response, backoff, upper_backoff)
            else:
                logger.info(
                    "GET %s completed after %d attempt%s",
                    host_label,
                    attempt_index + 1,
                    "" if attempt_index == 0 else "s",
                )
                return response

        if attempt_index >= max_attempts - 1:
            break

        if on_backoff is not None:
            await on_backoff(delay, attempt_index + 1)
        await sleep_impl(delay)
        backoff = min(backoff * 2, upper_backoff)

    if last_response is not None:
        raise BackoffError(f"Request failed with status {last_response.status_code}")
    if last_network_error is not None:
        raise last_network_error
    raise BackoffError("Request failed before receiving a response")


def redact_url_for_logs(url: str) -> str:
    """Return a safe identifier for URLs when logging sensitive endpoints."""

    try:
        parsed = urlsplit(url)
    except (ValueError, AttributeError):
        return "url"
    if parsed.netloc:
        if parsed.hostname:
            return parsed.hostname
        return parsed.netloc
    return parsed.path or "url"


def _retry_delay_from_response(response: httpx.Response, fallback: float, cap: float) -> float:
    """Compute the delay for the next retry using Retry-After when available."""

    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            delay = fallback
    else:
        delay = fallback
    return max(0.1, min(delay, cap))
