"""Rate control regression tests for resolver orchestration.

These tests exercise the low-level throttling primitives used by
`ResolverPipeline._respect_domain_limit` and the networking `CircuitBreaker`.
They verify that domain-level token buckets enforce back-off via the monotonic
clock, and that circuit breakers honour cooldown windows before accepting
additional requests.
"""

import pytest

from DocsToKG.ContentDownload.networking import CircuitBreaker
from DocsToKG.ContentDownload.pipeline import (
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
)


class _DummyLogger:
    def log_attempt(self, record, *, timestamp=None):
        return None

    def log_manifest(self, entry):
        return None

    def log_summary(self, summary):
        return None

    def close(self):
        return None


def test_respect_domain_limit_enforces_bucket_and_interval(patcher):
    sleep_calls: list[float] = []
    fake_time = [0.0]

    def fake_sleep(amount: float) -> None:
        sleep_calls.append(amount)
        fake_time[0] += amount

    def fake_monotonic() -> float:
        return fake_time[0]

    patcher.setattr("DocsToKG.ContentDownload.pipeline._time.sleep", fake_sleep)
    patcher.setattr("DocsToKG.ContentDownload.pipeline._time.monotonic", fake_monotonic)
    patcher.setattr("DocsToKG.ContentDownload.pipeline.random.random", lambda: 0.0)

    config = ResolverConfig(
        domain_min_interval_s={"example.com": 1.0},
        domain_token_buckets={"example.com": {"rate_per_second": 1.0, "capacity": 1.0}},
    )
    pipeline = ResolverPipeline(
        [], config, lambda *args, **kwargs: None, _DummyLogger(), ResolverMetrics(), run_id="test"
    )

    waited_first = pipeline._respect_domain_limit("https://example.com/first")
    assert waited_first == 0.0
    assert sleep_calls == []

    waited_second = pipeline._respect_domain_limit("https://example.com/second")
    assert pytest.approx(waited_second, rel=1e-2) == 1.0
    assert pytest.approx(sleep_calls[0], rel=1e-2) == 1.0


def test_circuit_breaker_cooldown(patcher):
    fake_time = [0.0]

    def fake_monotonic() -> float:
        return fake_time[0]

    patcher.setattr("DocsToKG.ContentDownload.networking.time.monotonic", fake_monotonic)

    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5.0, name="test")
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is False

    fake_time[0] += 5.0
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.allow() is True
