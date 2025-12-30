from __future__ import annotations

import asyncio
from typing import cast

import httpx
import pytest

from src.frame_compare import net

TimeoutException = httpx.TimeoutException  # type: ignore[attr-defined]
ReadTimeout = httpx.ReadTimeout  # type: ignore[attr-defined]
Request = httpx.Request  # type: ignore[attr-defined]


class HangingClient:
    def __init__(self) -> None:
        self.base_url = "https://example.com"
        self.timeouts: list[float | httpx.Timeout | None] = []

    async def get(
        self,
        path: str,
        params: dict[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        self.timeouts.append(timeout)
        seconds = _coerce_timeout(timeout)
        event = asyncio.Event()

        # Create a minimal request-like object for the ReadTimeout exception
        class FakeRequest:
            def __init__(self, url: str) -> None:
                self.url = url

        request = cast(httpx.Request, FakeRequest("https://example.com/api"))

        try:
            await asyncio.wait_for(event.wait(), timeout=seconds)
        except asyncio.TimeoutError as exc:
            raise ReadTimeout(f"Timeout after {seconds}s", request=request) from exc
        raise AssertionError("event wait unexpectedly completed")


class RecordingClient:
    def __init__(self) -> None:
        self.base_url = "https://example.com"
        self.timeouts: list[float | httpx.Timeout | None] = []

    async def get(
        self,
        path: str,
        params: dict[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        self.timeouts.append(timeout)
        return httpx.Response(status_code=200)


def _coerce_timeout(value: float | httpx.Timeout | None) -> float:
    if value is None:
        return 0.1
    if isinstance(value, (int, float)):
        return float(value)
    for attr in (getattr(value, "read", None), getattr(value, "connect", None)):
        if isinstance(attr, (int, float)):
            return float(attr)
    return 0.1


def test_httpx_backoff_propagates_timeout_errors() -> None:
    hanging_client = HangingClient()
    client = cast(httpx.AsyncClient, hanging_client)

    with pytest.raises(TimeoutException):
        asyncio.run(
            net.httpx_get_json_with_backoff(
                client,
                path="https://example.com/api",
                params={},
                retries=0,
                timeout=httpx.Timeout(0.01, connect=0.01, read=0.01),
            )
        )

    assert hanging_client.timeouts, "client should capture the configured timeout"


def test_httpx_backoff_uses_default_timeout_when_unspecified() -> None:
    recording_client = RecordingClient()
    client = cast(httpx.AsyncClient, recording_client)

    response = asyncio.run(
        net.httpx_get_json_with_backoff(
            client,
            path="https://example.com/api",
            params={},
            retries=0,
        )
    )

    assert response.status_code == 200
    assert recording_client.timeouts == [net.DEFAULT_HTTP_TIMEOUT]
