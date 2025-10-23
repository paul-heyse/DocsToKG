# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_rate_limit",
#   "purpose": "Behavioural checks for the rate limiter adapter.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Behavioural checks for the rate limiter adapter."""

from __future__ import annotations

import logging
from typing import List

import pytest

from DocsToKG.OntologyDownload.io.rate_limit import _LimiterAdapter


class _StubLimiter:
    """Limiter double that simulates temporary exhaustion."""

    def __init__(self, failures: int) -> None:
        self._failures_remaining = failures
        self.calls: List[int] = []

    def try_acquire(self, name: str, weight: int) -> bool:
        self.calls.append(weight)
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            return False
        return True


def test_limiter_adapter_consume_waits_when_exhausted(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """``consume`` should keep retrying when the limiter is temporarily exhausted."""

    limiter = _StubLimiter(failures=2)
    adapter = _LimiterAdapter(limiter, "service:host")
    sleep_calls: List[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.io.rate_limit.time.sleep", _fake_sleep
    )

    with caplog.at_level(logging.DEBUG):
        adapter.consume(tokens=1.0)

    assert limiter.calls == [1, 1, 1]
    assert sleep_calls, "expected consume to sleep while waiting for tokens"
    messages = [record.message for record in caplog.records]
    assert any("throttled" in message for message in messages)
    assert any("admitted" in message for message in messages)
