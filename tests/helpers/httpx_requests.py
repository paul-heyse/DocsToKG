"""Adapters bridging requests/`responses` to httpx clients for tests.

This allows legacy `responses` fixtures to continue working while the
runtime migrates to `httpx`-based transports. Each test can construct an
`httpx.Client` backed by a provided `requests.Session`, ensuring existing
`responses` mocks intercept outbound calls.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import httpx
import requests


class RequestsTransport(httpx.BaseTransport):
    """Minimal HTTPX transport that delegates to a `requests.Session`.

    The transport executes requests via the provided session. Responses are
    converted back into `httpx.Response` instances so call sites observe the
    expected API. Streaming is not required for current tests; bodies are
    read eagerly.
    """

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        data = request.read()
        response = self._session.request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=data,
            timeout=None,
            allow_redirects=False,
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
            extensions={"from_responses": True},
        )

    def close(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(Exception):
            self._session.close()


def make_httpx_client_from_requests(session: Optional[requests.Session] = None) -> httpx.Client:
    """Return an HTTPX client backed by a requests session.

    Args:
        session: Optional pre-constructed `requests.Session`. When omitted,
            a new session is created.

    Returns:
        httpx.Client: client suitable for passing into resolver/pipeline code.
    """

    transport = RequestsTransport(session=session)
    return httpx.Client(transport=transport, timeout=None)
