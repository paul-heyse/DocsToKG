# {
#   "module": "tests.ontology_download.test_polite_client",
#   "purpose": "Validate polite client resets when the underlying HTTP client is refreshed.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Regression tests for the polite HTTP client wrapper."""

from __future__ import annotations

import pytest

from DocsToKG.OntologyDownload.network.client import reset_http_client
from DocsToKG.OntologyDownload.network.polite_client import (
    get_polite_http_client,
    reset_polite_http_client,
)

httpx = pytest.importorskip("httpx")


def test_polite_client_recovers_after_http_client_reset(monkeypatch) -> None:
    """The polite client should transparently refresh its HTTP binding after a reset."""

    reset_http_client()
    reset_polite_http_client()

    responses: list[str] = []

    def _handler_factory(tag: str):
        def _handler(request: httpx.Request) -> httpx.Response:
            responses.append(tag)
            return httpx.Response(200, text=tag)

        return _handler

    transports = [
        httpx.MockTransport(_handler_factory("first")),
        httpx.MockTransport(_handler_factory("second")),
    ]

    def _build_client() -> httpx.Client:
        transport = transports.pop(0)
        return httpx.Client(transport=transport)

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.network.client._create_http_client",
        _build_client,
    )

    polite = get_polite_http_client()
    first = polite.get("https://example.com/test")
    assert first.text == "first"

    reset_http_client()

    second = polite.get("https://example.com/test")
    assert second.text == "second"

    refreshed = get_polite_http_client()
    assert refreshed is not polite

    assert responses == ["first", "second"]

    reset_http_client()
