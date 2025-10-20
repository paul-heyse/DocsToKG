"""Shared fixtures for ContentDownload HTTPX tests."""

from __future__ import annotations

import contextlib
from collections import deque
from typing import Callable, Deque

import httpx
import pytest

from DocsToKG.ContentDownload import httpx_transport


@pytest.fixture
def install_mock_http_client(monkeypatch):
    """Install a deterministic HTTPX client backed by MockTransport."""

    created: Deque[httpx.Client] = deque()

    def _install(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
        transport = httpx.MockTransport(handler)
        httpx_transport.configure_http_client(transport=transport)
        client = httpx_transport.get_http_client()
        created.append(client)

        return client

    yield _install

    while created:
        client = created.pop()
        with contextlib.suppress(Exception):
            client.close()
    httpx_transport.reset_http_client_for_tests()


@pytest.fixture
def capture_sleep(monkeypatch):
    """Capture sleep durations used by Tenacity during retries."""

    calls: list[float] = []

    def _fake_sleep(duration: float) -> None:
        calls.append(float(duration))

    monkeypatch.setattr("DocsToKG.ContentDownload.networking.time.sleep", _fake_sleep)
    return calls
