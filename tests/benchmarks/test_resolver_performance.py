# === NAVMAP v1 ===
# {
#   "module": "tests.benchmarks.test_resolver_performance",
#   "purpose": "Pytest coverage for benchmarks resolver performance scenarios",
#   "sections": [
#     {
#       "id": "dummyartifact",
#       "name": "DummyArtifact",
#       "anchor": "class-dummyartifact",
#       "kind": "class"
#     },
#     {
#       "id": "nulllogger",
#       "name": "NullLogger",
#       "anchor": "class-nulllogger",
#       "kind": "class"
#     },
#     {
#       "id": "slowresolver",
#       "name": "SlowResolver",
#       "anchor": "class-slowresolver",
#       "kind": "class"
#     },
#     {
#       "id": "html-outcome",
#       "name": "_html_outcome",
#       "anchor": "function-html-outcome",
#       "kind": "function"
#     },
#     {
#       "id": "make-config",
#       "name": "_make_config",
#       "anchor": "function-make-config",
#       "kind": "function"
#     },
#     {
#       "id": "test-sequential-vs-concurrent-execution",
#       "name": "test_sequential_vs_concurrent_execution",
#       "anchor": "function-test-sequential-vs-concurrent-execution",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-precheck-overhead-vs-savings",
#       "name": "test_head_precheck_overhead_vs_savings",
#       "anchor": "function-test-head-precheck-overhead-vs-savings",
#       "kind": "function"
#     },
#     {
#       "id": "test-retry-backoff-timing",
#       "name": "test_retry_backoff_timing",
#       "anchor": "function-test-retry-backoff-timing",
#       "kind": "function"
#     },
#     {
#       "id": "test-memory-usage-large-batch",
#       "name": "test_memory_usage_large_batch",
#       "anchor": "function-test-memory-usage-large-batch",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Benchmark-style tests verifying resolver performance characteristics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, List

import pytest
import httpx

pytest.importorskip("pytest_benchmark")

from DocsToKG.ContentDownload.pipeline import (
    DownloadOutcome,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
)
from tests.conftest import PatchManager


@dataclass
class DummyArtifact:
    work_id: str
    failed_pdf_urls: List[str] = field(default_factory=list)


class NullLogger:
    def log(self, record) -> None:  # pragma: no cover - trivial sink
        return None

    def log_attempt(self, record, *, timestamp=None) -> None:  # pragma: no cover - benchmark helper
        return None


class SlowResolver:
    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self.delay = delay

    def is_enabled(self, config: ResolverConfig, artifact: DummyArtifact) -> bool:
        return True

    def iter_urls(
        self, session, config: ResolverConfig, artifact: DummyArtifact
    ) -> Iterable[ResolverResult]:
        time.sleep(self.delay)
        yield ResolverResult(url=f"https://{self.name}.example/pdf")


# --- Helper Functions ---


def _html_outcome() -> DownloadOutcome:
    return DownloadOutcome(
        classification="html",
        path=None,
        http_status=200,
        content_type="text/html",
        elapsed_ms=10.0,
    )


def _make_config(names: List[str], max_workers: int) -> ResolverConfig:
    return ResolverConfig(
        resolver_order=names,
        resolver_toggles={name: True for name in names},
        enable_head_precheck=False,
        max_concurrent_resolvers=max_workers,
    )


@pytest.mark.benchmark
# --- Test Cases ---


def test_sequential_vs_concurrent_execution() -> None:
    artifact = DummyArtifact("W-bench")
    resolvers = [SlowResolver(f"resolver_{i}", delay=0.1) for i in range(6)]
    logger = NullLogger()
    metrics = ResolverMetrics()

    def download_func(*args, **kwargs):
        return _html_outcome()

    sequential = ResolverPipeline(
        resolvers, _make_config([r.name for r in resolvers], 1), download_func, logger, metrics
    )
    concurrent = ResolverPipeline(
        resolvers, _make_config([r.name for r in resolvers], 3), download_func, logger, metrics
    )

    start_seq = time.perf_counter()
    sequential.run(object(), artifact)
    sequential_elapsed = time.perf_counter() - start_seq

    start_conc = time.perf_counter()
    concurrent.run(object(), artifact)
    concurrent_elapsed = time.perf_counter() - start_conc

    assert concurrent_elapsed < sequential_elapsed
    # Allow modest overhead; concurrency should still deliver a noticeable win.
    assert concurrent_elapsed <= sequential_elapsed * 0.9


def test_head_precheck_overhead_vs_savings() -> None:
    pdf_cost = 0.2
    html_cost = 1.0
    head_cost = 0.05
    pdf_urls = 85
    html_urls = 15

    no_precheck_time = pdf_urls * pdf_cost + html_urls * html_cost
    precheck_time = (pdf_urls + html_urls) * head_cost + pdf_urls * pdf_cost

    assert precheck_time < no_precheck_time * 0.9


def test_retry_backoff_timing(patcher: PatchManager) -> None:
    from DocsToKG.ContentDownload.networking import request_with_retries

    sleeps: List[float] = []

    def fake_sleep(value: float) -> None:
        sleeps.append(value)

    class _SequenceWait:
        def __init__(self, values: List[float]) -> None:
            self._values = values
            self._index = 0
            self._last = values[-1] if values else 0.0

        def __call__(self, retry_state) -> float:  # pragma: no cover - exercised via Tenacity
            if self._index < len(self._values):
                value = self._values[self._index]
                self._index += 1
                self._last = value
                return value
            return self._last

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.wait_random_exponential",
        lambda *args, **kwargs: _SequenceWait([1.0, 2.0, 4.0]),
    )

    response_codes = [503, 503, 503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        if not response_codes:
            raise AssertionError("unexpected extra request")
        code = response_codes.pop(0)
        return httpx.Response(code, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    patcher.setattr(
        "DocsToKG.ContentDownload.networking.time.sleep",
        fake_sleep,
    )

    response = request_with_retries(
        client, "GET", "https://example.org/test", max_retries=3, backoff_factor=1.0
    )
    assert response.status_code == 200
    assert sleeps == [1.0, 2.0, 4.0]
    assert sum(sleeps) == 7.0
    client.close()


def test_memory_usage_large_batch() -> None:
    import tracemalloc

    artifact = DummyArtifact("W-memory")
    resolver = SlowResolver("memory", delay=0.0)
    logger = NullLogger()
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline(
        [resolver],
        _make_config([resolver.name], 1),
        lambda *args, **kwargs: _html_outcome(),
        logger,
        metrics,
    )

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    for _ in range(200):
        pipeline.run(object(), artifact)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    growth_mb = (peak - baseline) / (1024 * 1024)
    assert growth_mb < 5.0
