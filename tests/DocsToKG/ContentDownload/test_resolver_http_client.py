"""Tests for per-resolver HTTP client timeout overrides."""

from __future__ import annotations

import httpx
import pytest

from DocsToKG.ContentDownload.resolver_http_client import (
    PerResolverHttpClient,
    RetryConfig,
)


class _DummySession:
    """Minimal httpx.Client lookalike for exercising timeout logic."""

    def __init__(self, timeout: httpx.Timeout) -> None:
        self.timeout = timeout

    def request(self, method: str, url: str, *, timeout, **kwargs):  # type: ignore[override]
        # Persist the timeout that was chosen for later assertions and raise to exit.
        self.last_timeout = timeout  # type: ignore[attr-defined]
        raise httpx.ReadTimeout("request timed out")


def test_per_resolver_timeout_override_applies_when_not_explicit() -> None:
    session_timeout = httpx.Timeout(timeout=30.0, connect=5.0)
    session = _DummySession(timeout=session_timeout)
    client = PerResolverHttpClient(
        session=session,
        resolver_name="dummy",
        retry_config=RetryConfig(max_attempts=1, timeout_read_s=0.25),
    )

    with pytest.raises(httpx.ReadTimeout):
        client.get("https://example.invalid/slow")

    assert isinstance(session.last_timeout, httpx.Timeout)
    assert session.last_timeout.read == pytest.approx(0.25)
    # Ensure we preserved the existing connect timeout from the shared session.
    assert session.last_timeout.connect == session_timeout.connect
