"""Shared fixtures for ContentDownload HTTPX tests."""

from __future__ import annotations

import contextlib
import sys
import types
from collections import deque
from typing import Callable, Deque

import httpx
import pytest

from DocsToKG.ContentDownload import httpx_transport


def _install_requests_stub() -> None:
    """Install a lightweight ``requests`` stub backed by HTTPX primitives."""

    if "requests" in sys.modules:
        return

    requests_module = types.ModuleType("requests")

    class _RequestException(Exception):
        pass

    class _HTTPError(_RequestException):
        def __init__(self, response: httpx.Response | None = None) -> None:
            super().__init__("HTTP error")
            self.response = response

    class _Response:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers: dict[str, str] = {}
            self._content = b""
            self.raw = types.SimpleNamespace(close=lambda: None)

        @property
        def content(self) -> bytes:  # pragma: no cover - legacy stub
            return self._content

        @content.setter
        def content(self, value: bytes) -> None:
            self._content = value

        @property
        def text(self) -> str:
            return self._content.decode("utf-8", "ignore")

        def close(self) -> None:  # pragma: no cover - legacy stub
            return None

        def iter_content(self, chunk_size: int = 1024):  # pragma: no cover - legacy stub
            yield self._content

        def json(self):  # pragma: no cover - legacy stub
            return {}

        def raise_for_status(self) -> None:  # pragma: no cover - legacy stub
            return None

    class _Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def request(self, *_args, **_kwargs):  # pragma: no cover - legacy stub
            raise NotImplementedError("requests.Session is not available in HTTPX mode")

        def close(self) -> None:  # pragma: no cover - legacy stub
            return None

    adapters_module = types.ModuleType("requests.adapters")

    class _HTTPAdapter:  # pragma: no cover - legacy stub
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    adapters_module.HTTPAdapter = _HTTPAdapter

    requests_module.Session = _Session
    requests_module.Response = _Response
    requests_module.RequestException = _RequestException
    requests_module.HTTPError = _HTTPError
    requests_module.ConnectionError = _RequestException
    requests_module.Timeout = _RequestException
    requests_module.adapters = adapters_module
    requests_module.exceptions = types.SimpleNamespace(
        SSLError=_RequestException,
        ChunkedEncodingError=_RequestException,
    )

    sys.modules["requests"] = requests_module
    sys.modules["requests.adapters"] = adapters_module


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


_install_requests_stub()
