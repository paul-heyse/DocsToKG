"""
Parallel Resolver Rate Limiting Tests

This module exercises the resolver pipeline rate limiter under concurrent
execution to ensure cross-thread coordination respects the configured
minimum interval between resolver calls.

Key Scenarios:
- Validates multiple workers share rate state when using ThreadPoolExecutor
- Confirms the enforced delay never falls below the configured threshold

Dependencies:
- pytest: Provides fixtures and assertions
- DocsToKG.ContentDownload.resolvers: Supplies ResolverPipeline utilities

Usage:
    pytest tests/test_parallel_execution.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

import requests

from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolvers.types import (
    DownloadOutcome,
    ResolverConfig,
    ResolverResult,
)


class _NullLogger:
    def log(self, record):  # pragma: no cover - no-op sink
        pass


def test_rate_limiting_with_parallel_workers():
    config = ResolverConfig()
    config.resolver_min_interval_s = {"test": 0.1}
    pipeline = ResolverPipeline([], config, lambda *args, **kwargs: None, _NullLogger())

    # Establish initial timestamp so subsequent calls respect the interval.
    pipeline._respect_rate_limit("test")

    def call_respect_limit() -> float:
        pipeline._respect_rate_limit("test")
        return time.monotonic()

    with ThreadPoolExecutor(max_workers=4) as executor:
        timestamps = list(executor.map(lambda _: call_respect_limit(), range(4)))

    timestamps.sort()
    for first, second in zip(timestamps, timestamps[1:]):
        assert second - first >= 0.09


class _SlowResolver:
    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self._delay = delay

    def is_enabled(self, config, artifact):  # noqa: D401 - protocol implementation
        return True

    def iter_urls(self, session, config, artifact):
        time.sleep(self._delay)
        yield ResolverResult(url=f"https://example.org/{self.name}.html")


class _MemoryLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, record):  # noqa: D401 - protocol implementation
        self.records.append(record)


class _StubArtifact:
    def __init__(self) -> None:
        self.work_id = "W-concurrency"
        self.failed_pdf_urls = []


def _download_stub(session, artifact, url, referer, timeout, context=None):
    del session, artifact, url, referer, timeout, context
    return DownloadOutcome(
        classification="html",
        http_status=200,
        content_type="text/html",
        elapsed_ms=5.0,
    )


def test_concurrent_pipeline_reduces_wall_time(monkeypatch):
    monkeypatch.setattr("DocsToKG.ContentDownload.resolvers.pipeline.random.random", lambda: 0.0)
    resolver_count = 4
    delay = 0.1
    resolvers = [_SlowResolver(f"slow-{idx}", delay) for idx in range(resolver_count)]
    resolver_names = [resolver.name for resolver in resolvers]

    def _make_config(max_workers: int) -> ResolverConfig:
        return ResolverConfig(
            resolver_order=list(resolver_names),
            resolver_toggles={name: True for name in resolver_names},
            max_concurrent_resolvers=max_workers,
            enable_head_precheck=False,
            sleep_jitter=0.0,
        )

    artifact = _StubArtifact()
    session = requests.Session()
    try:
        sequential_logger = _MemoryLogger()
        sequential = ResolverPipeline(
            resolvers=resolvers,
            config=_make_config(1),
            download_func=_download_stub,
            logger=sequential_logger,
        )
        concurrent_logger = _MemoryLogger()
        concurrent = ResolverPipeline(
            resolvers=resolvers,
            config=_make_config(resolver_count),
            download_func=_download_stub,
            logger=concurrent_logger,
        )

        start = time.perf_counter()
        sequential.run(session, artifact, context={"dry_run": False})
        sequential_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        concurrent.run(session, artifact, context={"dry_run": False})
        concurrent_elapsed = time.perf_counter() - start
    finally:
        session.close()

    assert sequential_elapsed > delay * resolver_count * 0.8
    assert concurrent_elapsed < sequential_elapsed / 2
    assert len(sequential_logger.records) == resolver_count
    assert len(concurrent_logger.records) == resolver_count
