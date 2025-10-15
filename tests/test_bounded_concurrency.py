"""Tests for bounded intra-work concurrency in the resolver pipeline."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolvers.types import (
    AttemptRecord,
    DownloadOutcome,
    ResolverConfig,
    ResolverMetrics,
    ResolverResult,
)


class RecordingLogger:
    """Collect attempt records emitted by the pipeline during tests."""

    def __init__(self) -> None:
        self.records: List[AttemptRecord] = []
        self._lock = threading.Lock()

    def log(self, record: AttemptRecord) -> None:
        with self._lock:
            self.records.append(record)


@dataclass
class DummyArtifact:
    work_id: str
    pdf_dir: Path
    html_dir: Path
    failed_pdf_urls: List[str] = field(default_factory=list)


class DelayResolver:
    """Resolver that sleeps before yielding URLs to simulate slow sources."""

    def __init__(self, name: str, urls: Iterable[str], delay: float = 0.0) -> None:
        self.name = name
        self._urls = list(urls)
        self.delay = delay
        self.start_times: List[float] = []

    def is_enabled(self, config: ResolverConfig, artifact: DummyArtifact) -> bool:
        return True

    def iter_urls(
        self,
        session,
        config: ResolverConfig,
        artifact: DummyArtifact,
    ) -> Iterable[ResolverResult]:
        start = time.monotonic()
        self.start_times.append(start)
        if self.delay:
            time.sleep(self.delay)
        for url in self._urls:
            yield ResolverResult(url=url)


class FailingResolver:
    """Resolver that raises an exception to test error isolation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def is_enabled(self, config: ResolverConfig, artifact: DummyArtifact) -> bool:
        return True

    def iter_urls(self, session, config: ResolverConfig, artifact: DummyArtifact):
        raise RuntimeError("boom")


def _make_artifact(tmp_path) -> DummyArtifact:
    return DummyArtifact(
        work_id="W-test",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )


def _make_config(names: Iterable[str], **kwargs) -> ResolverConfig:
    names = list(names)
    toggles = {name: True for name in names}
    return ResolverConfig(
        resolver_order=names,
        resolver_toggles=toggles,
        enable_head_precheck=False,
        **kwargs,
    )


def _html_outcome() -> DownloadOutcome:
    return DownloadOutcome(
        classification="html",
        path=None,
        http_status=200,
        content_type="text/html",
        elapsed_ms=100.0,
    )


def test_sequential_execution_when_max_concurrent_is_one(tmp_path):
    artifact = _make_artifact(tmp_path)
    resolvers = [
        DelayResolver("r1", ["https://r1.example/a"], delay=0.1),
        DelayResolver("r2", ["https://r2.example/a"], delay=0.1),
        DelayResolver("r3", ["https://r3.example/a"], delay=0.1),
    ]
    config = _make_config([r.name for r in resolvers], max_concurrent_resolvers=1)
    logger = RecordingLogger()
    metrics = ResolverMetrics()

    def download_func(session, artifact, url, referer, timeout):
        time.sleep(0.1)
        return _html_outcome()

    pipeline = ResolverPipeline(resolvers, config, download_func, logger, metrics)
    start = time.monotonic()
    result = pipeline.run(object(), artifact)
    elapsed = time.monotonic() - start

    assert result.success is False
    assert elapsed >= 0.5, f"Sequential execution finished too quickly: {elapsed:.3f}s"


def test_concurrent_execution_with_three_workers(tmp_path):
    artifact = _make_artifact(tmp_path)
    seq_resolvers = [
        DelayResolver("r1", ["https://r1.example/a"], delay=0.3),
        DelayResolver("r2", ["https://r2.example/a"], delay=0.3),
        DelayResolver("r3", ["https://r3.example/a"], delay=0.3),
    ]
    conc_resolvers = [
        DelayResolver("r1", ["https://r1.example/a"], delay=0.3),
        DelayResolver("r2", ["https://r2.example/a"], delay=0.3),
        DelayResolver("r3", ["https://r3.example/a"], delay=0.3),
    ]
    sequential_config = _make_config([r.name for r in seq_resolvers], max_concurrent_resolvers=1)
    concurrent_config = _make_config([r.name for r in conc_resolvers], max_concurrent_resolvers=3)
    logger = RecordingLogger()
    metrics = ResolverMetrics()

    def download_func(session, artifact, url, referer, timeout):
        time.sleep(0.05)
        return _html_outcome()

    sequential_pipeline = ResolverPipeline(
        seq_resolvers, sequential_config, download_func, logger, metrics
    )
    conc_pipeline = ResolverPipeline(
        conc_resolvers, concurrent_config, download_func, logger, metrics
    )

    seq_elapsed = time.monotonic()
    sequential_pipeline.run(object(), artifact)
    seq_elapsed = time.monotonic() - seq_elapsed

    conc_elapsed = time.monotonic()
    conc_pipeline.run(object(), artifact)
    conc_elapsed = time.monotonic() - conc_elapsed

    assert conc_elapsed < seq_elapsed * 0.8

    # All resolvers should have started close together when running concurrently.
    start_spread = max(r.start_times[0] for r in conc_resolvers) - min(
        r.start_times[0] for r in conc_resolvers
    )
    assert start_spread < 0.15


def test_rate_limits_enforced_under_concurrency(tmp_path):
    artifact = _make_artifact(tmp_path)
    resolvers = [
        DelayResolver("r1", ["https://r1.example/a"], delay=0.0),
        DelayResolver("r2", ["https://r2.example/a"], delay=0.0),
    ]
    config = _make_config(
        [r.name for r in resolvers],
        max_concurrent_resolvers=2,
        resolver_min_interval_s={r.name: 0.2 for r in resolvers},
    )
    logger = RecordingLogger()
    metrics = ResolverMetrics()

    def download_func(session, artifact, url, referer, timeout):
        return _html_outcome()

    pipeline = ResolverPipeline(resolvers, config, download_func, logger, metrics)
    now = time.monotonic()
    for resolver in resolvers:
        pipeline._last_invocation[resolver.name] = now

    for _ in range(3):
        pipeline.run(object(), artifact)

    for resolver in resolvers:
        assert resolver.start_times, "Resolver did not execute"
        for idx in range(1, len(resolver.start_times)):
            delta = resolver.start_times[idx] - resolver.start_times[idx - 1]
            assert delta >= 0.19, f"Rate limit gap too small: {delta:.3f}s for {resolver.name}"


def test_early_stop_cancels_remaining_resolvers(tmp_path):
    artifact = _make_artifact(tmp_path)
    fast = DelayResolver("fast", ["https://fast.example/pdf"], delay=0.0)
    slow1 = DelayResolver("slow1", ["https://slow1.example/html"], delay=0.4)
    slow2 = DelayResolver("slow2", ["https://slow2.example/html"], delay=0.4)
    resolvers = [fast, slow1, slow2]
    config = _make_config([r.name for r in resolvers], max_concurrent_resolvers=2)
    logger = RecordingLogger()
    metrics = ResolverMetrics()

    downloaded: List[str] = []

    def download_func(session, artifact, url, referer, timeout):
        downloaded.append(url)
        if "fast" in url:
            return DownloadOutcome(
                classification="pdf",
                path=str(artifact.pdf_dir / "fast.pdf"),
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=50.0,
            )
        time.sleep(0.3)
        return _html_outcome()

    pipeline = ResolverPipeline(resolvers, config, download_func, logger, metrics)
    result = pipeline.run(object(), artifact)

    assert result.success is True
    assert downloaded == ["https://fast.example/pdf"]


def test_resolver_failure_does_not_abort_concurrency(tmp_path):
    artifact = _make_artifact(tmp_path)
    failing = FailingResolver("failing")
    healthy = DelayResolver("healthy", ["https://healthy.example/html"], delay=0.0)
    resolvers = [failing, healthy]
    config = _make_config([r.name for r in resolvers], max_concurrent_resolvers=2)
    logger = RecordingLogger()
    metrics = ResolverMetrics()

    downloaded: List[str] = []

    def download_func(session, artifact, url, referer, timeout):
        downloaded.append(url)
        return _html_outcome()

    pipeline = ResolverPipeline(resolvers, config, download_func, logger, metrics)
    result = pipeline.run(object(), artifact)

    assert result.success is False
    assert downloaded == ["https://healthy.example/html"]
    error_records = [r for r in logger.records if r.status == "error"]
    assert error_records, "Expected resolver error to be logged"
    assert error_records[0].reason == "resolver-exception"
