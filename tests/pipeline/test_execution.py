# === NAVMAP v1 ===
# {
#   "module": "tests.pipeline.test_execution",
#   "purpose": "Pytest coverage for pipeline execution scenarios",
#   "sections": [
#     {
#       "id": "recordinglogger",
#       "name": "RecordingLogger",
#       "anchor": "class-recordinglogger",
#       "kind": "class"
#     },
#     {
#       "id": "dummyartifact",
#       "name": "DummyArtifact",
#       "anchor": "class-dummyartifact",
#       "kind": "class"
#     },
#     {
#       "id": "delayresolver",
#       "name": "DelayResolver",
#       "anchor": "class-delayresolver",
#       "kind": "class"
#     },
#     {
#       "id": "failingresolver",
#       "name": "FailingResolver",
#       "anchor": "class-failingresolver",
#       "kind": "class"
#     },
#     {
#       "id": "make-artifact",
#       "name": "_make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "make-config",
#       "name": "_make_config",
#       "anchor": "function-make-config",
#       "kind": "function"
#     },
#     {
#       "id": "html-outcome",
#       "name": "_html_outcome",
#       "anchor": "function-html-outcome",
#       "kind": "function"
#     },
#     {
#       "id": "test-sequential-execution-when-max-concurrent-is-one",
#       "name": "test_sequential_execution_when_max_concurrent_is_one",
#       "anchor": "function-test-sequential-execution-when-max-concurrent-is-one",
#       "kind": "function"
#     },
#     {
#       "id": "test-concurrent-execution-with-three-workers",
#       "name": "test_concurrent_execution_with_three_workers",
#       "anchor": "function-test-concurrent-execution-with-three-workers",
#       "kind": "function"
#     },
#     {
#       "id": "test-rate-limits-enforced-under-concurrency",
#       "name": "test_rate_limits_enforced_under_concurrency",
#       "anchor": "function-test-rate-limits-enforced-under-concurrency",
#       "kind": "function"
#     },
#     {
#       "id": "test-early-stop-cancels-remaining-resolvers",
#       "name": "test_early_stop_cancels_remaining_resolvers",
#       "anchor": "function-test-early-stop-cancels-remaining-resolvers",
#       "kind": "function"
#     },
#     {
#       "id": "test-resolver-failure-does-not-abort-concurrency",
#       "name": "test_resolver_failure_does_not_abort_concurrency",
#       "anchor": "function-test-resolver-failure-does-not-abort-concurrency",
#       "kind": "function"
#     },
#     {
#       "id": "nulllogger",
#       "name": "_NullLogger",
#       "anchor": "class-nulllogger",
#       "kind": "class"
#     },
#     {
#       "id": "test-rate-limiting-with-parallel-workers",
#       "name": "test_rate_limiting_with_parallel_workers",
#       "anchor": "function-test-rate-limiting-with-parallel-workers",
#       "kind": "function"
#     },
#     {
#       "id": "slowresolver",
#       "name": "_SlowResolver",
#       "anchor": "class-slowresolver",
#       "kind": "class"
#     },
#     {
#       "id": "memorylogger",
#       "name": "_MemoryLogger",
#       "anchor": "class-memorylogger",
#       "kind": "class"
#     },
#     {
#       "id": "stubartifact",
#       "name": "_StubArtifact",
#       "anchor": "class-stubartifact",
#       "kind": "class"
#     },
#     {
#       "id": "download-stub",
#       "name": "_download_stub",
#       "anchor": "function-download-stub",
#       "kind": "function"
#     },
#     {
#       "id": "test-concurrent-pipeline-reduces-wall-time",
#       "name": "test_concurrent_pipeline_reduces_wall_time",
#       "anchor": "function-test-concurrent-pipeline-reduces-wall-time",
#       "kind": "function"
#     },
#     {
#       "id": "memorylogger",
#       "name": "MemoryLogger",
#       "anchor": "class-memorylogger",
#       "kind": "class"
#     },
#     {
#       "id": "stubresolver",
#       "name": "StubResolver",
#       "anchor": "class-stubresolver",
#       "kind": "class"
#     },
#     {
#       "id": "make-artifact",
#       "name": "make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "build-outcome",
#       "name": "build_outcome",
#       "anchor": "function-build-outcome",
#       "kind": "function"
#     },
#     {
#       "id": "test-pipeline-respects-custom-order",
#       "name": "test_pipeline_respects_custom_order",
#       "anchor": "function-test-pipeline-respects-custom-order",
#       "kind": "function"
#     },
#     {
#       "id": "test-pipeline-stops-after-max-attempts",
#       "name": "test_pipeline_stops_after_max_attempts",
#       "anchor": "function-test-pipeline-stops-after-max-attempts",
#       "kind": "function"
#     },
#     {
#       "id": "test-pipeline-deduplicates-urls",
#       "name": "test_pipeline_deduplicates_urls",
#       "anchor": "function-test-pipeline-deduplicates-urls",
#       "kind": "function"
#     },
#     {
#       "id": "test-pipeline-collects-html-paths",
#       "name": "test_pipeline_collects_html_paths",
#       "anchor": "function-test-pipeline-collects-html-paths",
#       "kind": "function"
#     },
#     {
#       "id": "test-pipeline-rate-limit-enforced",
#       "name": "test_pipeline_rate_limit_enforced",
#       "anchor": "function-test-pipeline-rate-limit-enforced",
#       "kind": "function"
#     },
#     {
#       "id": "test-openalex-resolver-executes-first",
#       "name": "test_openalex_resolver_executes_first",
#       "anchor": "function-test-openalex-resolver-executes-first",
#       "kind": "function"
#     },
#     {
#       "id": "test-openalex-respects-rate-limit",
#       "name": "test_openalex_respects_rate_limit",
#       "anchor": "function-test-openalex-respects-rate-limit",
#       "kind": "function"
#     },
#     {
#       "id": "test-pipeline-records-failed-urls",
#       "name": "test_pipeline_records_failed_urls",
#       "anchor": "function-test-pipeline-records-failed-urls",
#       "kind": "function"
#     },
#     {
#       "id": "make-artifact",
#       "name": "_make_artifact",
#       "anchor": "function-make-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "test-pipeline-executes-resolvers-in-expected-order",
#       "name": "test_pipeline_executes_resolvers_in_expected_order",
#       "anchor": "function-test-pipeline-executes-resolvers-in-expected-order",
#       "kind": "function"
#     },
#     {
#       "id": "test-real-network-download",
#       "name": "test_real_network_download",
#       "anchor": "function-test-real-network-download",
#       "kind": "function"
#     },
#     {
#       "id": "test-resolver-pipeline-downloads-pdf-end-to-end",
#       "name": "test_resolver_pipeline_downloads_pdf_end_to_end",
#       "anchor": "function-test-resolver-pipeline-downloads-pdf-end-to-end",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-candidate-marks-corrupt-without-eof",
#       "name": "test_download_candidate_marks_corrupt_without_eof",
#       "anchor": "function-test-download-candidate-marks-corrupt-without-eof",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Consolidated pipeline execution tests."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Iterable, List, Optional

import pytest
import requests

from DocsToKG.ContentDownload import cli as downloader
from DocsToKG.ContentDownload import pipeline as resolvers
from DocsToKG.ContentDownload.core import Classification, WorkArtifact
from DocsToKG.ContentDownload.pipeline import (
    AttemptRecord,
    DownloadOutcome,
    OpenAlexResolver,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
    default_resolvers,
)

# --- test_bounded_concurrency.py ---


class RecordingLogger:
    """Collect attempt records emitted by the pipeline during tests."""

    def __init__(self) -> None:
        self.records: List[AttemptRecord] = []
        self._lock = threading.Lock()

    def log(self, record: AttemptRecord) -> None:
        with self._lock:
            self.records.append(record)

    def log_attempt(self, record: AttemptRecord, *, timestamp: Optional[str] = None) -> None:
        self.log(record)


# --- test_bounded_concurrency.py ---


@dataclass
class DummyArtifact:
    work_id: str
    pdf_dir: Path
    html_dir: Path
    failed_pdf_urls: List[str] = field(default_factory=list)


# --- test_bounded_concurrency.py ---


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


# --- test_bounded_concurrency.py ---


class FailingResolver:
    """Resolver that raises an exception to test error isolation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def is_enabled(self, config: ResolverConfig, artifact: DummyArtifact) -> bool:
        return True

    def iter_urls(self, session, config: ResolverConfig, artifact: DummyArtifact):
        raise RuntimeError("boom")


# --- test_bounded_concurrency.py ---


def _make_artifact(tmp_path) -> DummyArtifact:
    return DummyArtifact(
        work_id="W-test",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )


# --- test_bounded_concurrency.py ---


def _make_config(names: Iterable[str], **kwargs) -> ResolverConfig:
    names = list(names)
    toggles = {name: True for name in names}
    return ResolverConfig(
        resolver_order=names,
        resolver_toggles=toggles,
        enable_head_precheck=False,
        **kwargs,
    )


# --- test_bounded_concurrency.py ---


def _html_outcome() -> DownloadOutcome:
    return DownloadOutcome(
        classification="html",
        path=None,
        http_status=200,
        content_type="text/html",
        elapsed_ms=100.0,
    )


# --- test_bounded_concurrency.py ---


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


# --- test_bounded_concurrency.py ---


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


# --- test_bounded_concurrency.py ---


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


# --- test_bounded_concurrency.py ---


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


# --- test_bounded_concurrency.py ---


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


# --- test_parallel_execution.py ---

pytest.importorskip("requests")

# --- test_parallel_execution.py ---

pytest.importorskip("pyalex")


# --- test_parallel_execution.py ---


class _NullLogger:
    def log(self, record):  # pragma: no cover - no-op sink
        pass


# --- test_parallel_execution.py ---


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


# --- test_parallel_execution.py ---


class _SlowResolver:
    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self._delay = delay

    def is_enabled(self, config, artifact):  # noqa: D401 - protocol implementation
        return True

    def iter_urls(self, session, config, artifact):
        time.sleep(self._delay)
        yield ResolverResult(url=f"https://example.org/{self.name}.html")


# --- test_parallel_execution.py ---


class _MemoryLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, record):  # noqa: D401 - protocol implementation
        self.records.append(record)

    def log_attempt(self, record, *, timestamp=None):
        del timestamp
        self.log(record)


# --- test_parallel_execution.py ---


class _StubArtifact:
    def __init__(self) -> None:
        self.work_id = "W-concurrency"
        self.failed_pdf_urls = []


# --- test_parallel_execution.py ---


def _download_stub(session, artifact, url, referer, timeout, context=None):
    del session, artifact, url, referer, timeout, context
    return DownloadOutcome(
        classification="html",
        http_status=200,
        content_type="text/html",
        elapsed_ms=5.0,
    )


# --- test_parallel_execution.py ---


def test_concurrent_pipeline_reduces_wall_time(monkeypatch):
    monkeypatch.setattr("DocsToKG.ContentDownload.pipeline.random.random", lambda: 0.0)
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


# --- test_pipeline_behaviour.py ---

pytest.importorskip("pyalex")


# --- test_pipeline_behaviour.py ---


class MemoryLogger:
    def __init__(self, records: Optional[List[AttemptRecord]] = None) -> None:
        self.records = records or []

    def log(self, record: AttemptRecord) -> None:
        self.records.append(record)

    def log_attempt(self, record: AttemptRecord, *, timestamp: str | None = None) -> None:
        del timestamp
        self.log(record)


# --- test_pipeline_behaviour.py ---


class StubResolver:
    def __init__(self, name: str, results):
        self.name = name
        self._results = results

    def is_enabled(self, config, artifact):
        return True

    def iter_urls(self, session, config, artifact):
        for result in self._results:
            yield result


# --- test_pipeline_behaviour.py ---


def make_artifact(tmp_path: Path) -> downloader.WorkArtifact:
    return downloader.WorkArtifact(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi="10.1000/example",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )


# --- test_pipeline_behaviour.py ---


def build_outcome(classification: str, path: str | None = None) -> DownloadOutcome:
    return DownloadOutcome(
        classification=classification,
        path=path,
        http_status=200 if classification == "pdf" else 400,
        content_type="application/pdf" if classification.startswith("pdf") else "text/html",
        elapsed_ms=10.0,
        error=None,
    )


# --- test_pipeline_behaviour.py ---


def test_pipeline_respects_custom_order(tmp_path):
    artifact = make_artifact(tmp_path)
    attempts: List[str] = []

    def downloader_fn(session, art, url, referer, timeout):
        attempts.append(url)
        if "beta" in url:
            return build_outcome("pdf", path=str(art.pdf_dir / "beta.pdf"))
        return build_outcome("http_error")

    alpha = StubResolver("alpha", [ResolverResult(url="https://alpha.example/1")])
    beta = StubResolver("beta", [ResolverResult(url="https://beta.example/1")])

    config = ResolverConfig(
        resolver_order=["beta", "alpha"],
        resolver_toggles={"alpha": True, "beta": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = ResolverPipeline(
        resolvers=[alpha, beta],
        config=config,
        download_func=downloader_fn,
        logger=logger,
    )
    session = requests.Session()
    result = pipeline.run(session, artifact)
    assert result.success is True
    assert attempts == ["https://beta.example/1"]
    assert logger.records[0].resolver_name == "beta"


# --- test_pipeline_behaviour.py ---


def test_pipeline_stops_after_max_attempts(tmp_path):
    artifact = make_artifact(tmp_path)

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("http_error")

    resolver = StubResolver(
        "single",
        [
            ResolverResult(url="https://example.com/a"),
            ResolverResult(url="https://example.com/b"),
        ],
    )
    config = ResolverConfig(
        resolver_order=["single"],
        resolver_toggles={"single": True},
        max_attempts_per_work=1,
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=downloader_fn,
        logger=logger,
    )
    session = requests.Session()
    result = pipeline.run(session, artifact)
    assert result.success is False
    assert result.reason == "max-attempts-reached"
    assert len(logger.records) == 1


# --- test_pipeline_behaviour.py ---


def test_pipeline_deduplicates_urls(tmp_path):
    artifact = make_artifact(tmp_path)
    resolver = StubResolver(
        "dup",
        [
            ResolverResult(url="https://dup.example/x"),
            ResolverResult(url="https://dup.example/x"),
        ],
    )
    config = ResolverConfig(
        resolver_order=["dup"],
        resolver_toggles={"dup": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("http_error")

    pipeline = ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=downloader_fn,
        logger=logger,
    )
    session = requests.Session()
    pipeline.run(session, artifact)
    duplicates = [record.reason for record in logger.records if record.reason == "duplicate-url"]
    assert duplicates == ["duplicate-url"]


# --- test_pipeline_behaviour.py ---


def test_pipeline_collects_html_paths(tmp_path):
    artifact = make_artifact(tmp_path)

    def downloader_fn(session, art, url, referer, timeout):
        html_path = art.html_dir / "example.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text("<html></html>", encoding="utf-8")
        return build_outcome("html", path=str(html_path))

    resolver = StubResolver("html", [ResolverResult(url="https://example.com/html")])
    config = ResolverConfig(
        resolver_order=["html"],
        resolver_toggles={"html": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=downloader_fn,
        logger=logger,
    )
    session = requests.Session()
    result = pipeline.run(session, artifact)
    assert result.success is False
    assert result.html_paths and Path(result.html_paths[0]).name == "example.html"


# --- test_pipeline_behaviour.py ---


def test_pipeline_rate_limit_enforced(monkeypatch, tmp_path):
    artifact = make_artifact(tmp_path)
    timeline = [0.0, 0.2, 0.2, 1.2, 2.0, 2.8]

    def fake_monotonic():
        return timeline.pop(0)

    sleeps: List[float] = []

    def fake_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr("DocsToKG.ContentDownload.pipeline._time.monotonic", fake_monotonic)
    monkeypatch.setattr("DocsToKG.ContentDownload.pipeline._time.sleep", fake_sleep)

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("pdf", path=str(art.pdf_dir / "out.pdf"))

    resolver = StubResolver("limited", [ResolverResult(url="https://example.com/pdf")])
    config = ResolverConfig(
        resolver_order=["limited"],
        resolver_toggles={"limited": True},
        resolver_min_interval_s={"limited": 1.0},
        enable_head_precheck=False,
    )
    pipeline = ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=downloader_fn,
        logger=MemoryLogger(),
    )
    session = requests.Session()
    pipeline.run(session, artifact)
    pipeline.run(session, artifact)
    assert len(sleeps) == 2
    assert pytest.approx(sleeps[0], rel=0.01) == 1.0
    # The second invocation occurs 0.2s after the first due to the simulated
    # timeline, so the sleep duration reflects the remaining 0.8s window.
    assert pytest.approx(sleeps[1], rel=0.01) == 0.8


# --- test_pipeline_behaviour.py ---


def test_openalex_resolver_executes_first(tmp_path):
    artifact = make_artifact(tmp_path)
    artifact.pdf_urls = ["https://openalex.org/direct.pdf"]

    download_calls: List[str] = []

    def downloader_fn(session, art, url, referer, timeout):
        download_calls.append(url)
        pdf_path = art.pdf_dir / "openalex.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF")
        return build_outcome("pdf", path=str(pdf_path))

    fallback = StubResolver("fallback", [ResolverResult(url="https://fallback.example/pdf")])
    config = ResolverConfig(
        resolver_order=["openalex", "fallback"],
        resolver_toggles={"openalex": True, "fallback": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline(
        resolvers=[OpenAlexResolver(), fallback],
        config=config,
        download_func=downloader_fn,
        logger=logger,
        metrics=metrics,
    )
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is True
    assert result.resolver_name == "openalex"
    assert download_calls == ["https://openalex.org/direct.pdf"]
    assert [record.resolver_name for record in logger.records] == ["openalex"]
    assert metrics.successes["openalex"] == 1


# --- test_pipeline_behaviour.py ---


def test_openalex_respects_rate_limit(monkeypatch, tmp_path):
    artifact = make_artifact(tmp_path)
    artifact.pdf_urls = ["https://openalex.org/pdf-one"]

    timeline = [0.0, 0.4, 0.4, 1.2, 2.0, 2.8]

    def fake_monotonic():
        return timeline.pop(0)

    sleeps: List[float] = []

    def fake_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr("DocsToKG.ContentDownload.pipeline._time.monotonic", fake_monotonic)
    monkeypatch.setattr("DocsToKG.ContentDownload.pipeline._time.sleep", fake_sleep)

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("pdf", path=str(art.pdf_dir / "result.pdf"))

    config = ResolverConfig(
        resolver_order=["openalex"],
        resolver_toggles={"openalex": True},
        resolver_min_interval_s={"openalex": 0.8},
        enable_head_precheck=False,
    )
    pipeline = ResolverPipeline(
        resolvers=[OpenAlexResolver()],
        config=config,
        download_func=downloader_fn,
        logger=MemoryLogger(),
    )
    session = requests.Session()
    pipeline.run(session, artifact)
    pipeline.run(session, artifact)

    assert sleeps and pytest.approx(sleeps[0], rel=0.05) == 0.8


# --- test_pipeline_behaviour.py ---


def test_pipeline_records_failed_urls(tmp_path):
    artifact = make_artifact(tmp_path)
    artifact.pdf_urls = ["https://openalex.org/broken.pdf"]

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("http_error")

    config = ResolverConfig(
        resolver_order=["openalex"],
        resolver_toggles={"openalex": True},
        max_attempts_per_work=1,
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = ResolverPipeline(
        resolvers=[OpenAlexResolver()],
        config=config,
        download_func=downloader_fn,
        logger=logger,
    )
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is False
    assert result.failed_urls == ["https://openalex.org/broken.pdf"]
    assert artifact.failed_pdf_urls == ["https://openalex.org/broken.pdf"]


# --- test_full_pipeline_integration.py ---

pytest.importorskip("pyalex")


# --- test_full_pipeline_integration.py ---


def _make_artifact(tmp_path: Path) -> WorkArtifact:  # noqa: F811
    return WorkArtifact(
        work_id="W-integration",
        title="Integration Test",
        publication_year=2024,
        doi="10.1234/example",
        pmid="123456",
        pmcid="PMC123456",
        arxiv_id="arXiv:2101.00001",
        landing_urls=["https://example.org"],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=["Example"],
        base_stem="2024__Integration__W-integration",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )


# --- test_full_pipeline_integration.py ---


def test_pipeline_executes_resolvers_in_expected_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = _make_artifact(tmp_path)
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact.html_dir.mkdir(parents=True, exist_ok=True)
    artifact.pdf_urls = ["https://openalex.example/preprint.pdf"]
    artifact.open_access_url = "https://openalex.example/oa.pdf"

    resolvers = default_resolvers()
    call_order: List[str] = []

    def _make_stub(resolver_name: str):
        def _stub(self, session, config, art):
            call_order.append(resolver_name)
            yield ResolverResult(url=f"https://{resolver_name}.example/pdf")

        return _stub

    for resolver in resolvers:
        monkeypatch.setattr(
            resolver,
            "iter_urls",
            MethodType(_make_stub(resolver.name), resolver),
        )

    config = ResolverConfig(
        enable_head_precheck=False,
        unpaywall_email="ci@example.org",
        core_api_key="token",
    )
    logger = MemoryLogger([])
    metrics = ResolverMetrics()

    def download_func(session, art, url, referer, timeout, context=None):
        classification = "pdf" if "zenodo" in url else "html"
        return DownloadOutcome(
            classification=classification,
            path=str(artifact.pdf_dir / "test.pdf") if classification == "pdf" else None,
            http_status=200,
            content_type="application/pdf" if classification == "pdf" else "text/html",
            elapsed_ms=25.0,
            sha256="deadbeef" if classification == "pdf" else None,
            content_length=2048 if classification == "pdf" else None,
        )

    pipeline = ResolverPipeline(resolvers, config, download_func, logger, metrics)
    respect_calls: List[str] = []
    original_respect = pipeline._respect_rate_limit

    def _tracking_respect(name: str) -> None:
        respect_calls.append(name)
        original_respect(name)

    pipeline._respect_rate_limit = _tracking_respect  # type: ignore[assignment]

    result = pipeline.run(object(), artifact)

    expected_prefix = [
        "openalex",
        "unpaywall",
        "crossref",
        "landing_page",
        "arxiv",
        "pmc",
        "europe_pmc",
        "core",
        "zenodo",
    ]

    assert call_order[: len(expected_prefix)] == expected_prefix
    assert result.success is True
    assert result.resolver_name == "zenodo"
    assert metrics.successes["zenodo"] == 1
    assert respect_calls[: len(expected_prefix)] == expected_prefix


# --- test_full_pipeline_integration.py ---


@pytest.mark.integration
def test_real_network_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not os.environ.get("DOCSTOKG_RUN_NETWORK_TESTS"):
        pytest.skip("set DOCSTOKG_RUN_NETWORK_TESTS=1 to enable network integration test")

    pytest.importorskip("requests")

    artifact = _make_artifact(tmp_path)
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact.html_dir.mkdir(parents=True, exist_ok=True)

    resolvers = default_resolvers()
    config = ResolverConfig(enable_head_precheck=False)
    metrics = ResolverMetrics()
    logger = MemoryLogger([])

    from DocsToKG.ContentDownload.cli import download_candidate
    from DocsToKG.ContentDownload.network import create_session

    session = create_session({"User-Agent": "DocsToKG-Test/1.0"})
    try:
        result = ResolverPipeline(
            resolvers,
            config,
            download_candidate,
            logger,
            metrics,
        ).run(session, artifact)
    finally:
        session.close()

    assert isinstance(result.success, bool)


# --- test_end_to_end_offline.py ---

pytest.importorskip("pyalex")

# --- test_end_to_end_offline.py ---

responses = pytest.importorskip("responses")


# --- test_end_to_end_offline.py ---


@responses.activate
def test_resolver_pipeline_downloads_pdf_end_to_end(tmp_path):
    work = {
        "id": "https://openalex.org/W123",
        "title": "Resolver Demo",
        "publication_year": 2021,
        "ids": {"doi": "10.1000/resolver-demo"},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }
    artifact = downloader.create_artifact(
        work, pdf_dir=tmp_path / "pdf", html_dir=tmp_path / "html"
    )

    pdf_url = "https://cdn.example/resolver-demo.pdf"
    responses.add(
        responses.GET,
        "https://api.unpaywall.org/v2/10.1000/resolver-demo",
        json={"best_oa_location": {"url_for_pdf": pdf_url}},
        status=200,
    )
    responses.add(
        responses.HEAD,
        pdf_url,
        headers={"Content-Type": "application/pdf"},
        status=200,
    )
    responses.add(
        responses.GET,
        pdf_url,
        body=b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n%%EOF",
        headers={"Content-Type": "application/pdf"},
        status=200,
    )

    config = resolvers.ResolverConfig(
        resolver_order=["unpaywall"],
        resolver_toggles={"unpaywall": True},
        unpaywall_email="team@example.org",
    )
    logger = MemoryLogger()
    session = requests.Session()
    session.headers.update({"User-Agent": "pytest"})
    pipeline = resolvers.ResolverPipeline(
        resolvers=[resolvers.UnpaywallResolver()],
        config=config,
        download_func=downloader.download_candidate,
        logger=logger,
    )
    result = pipeline.run(session, artifact)
    assert result.success is True
    assert result.outcome and result.outcome.path
    pdf_path = Path(result.outcome.path)
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


# --- test_end_to_end_offline.py ---


@responses.activate
def test_download_candidate_marks_corrupt_without_eof(tmp_path):
    work = {
        "id": "https://openalex.org/W456",
        "title": "Corrupt PDF",
        "publication_year": 2022,
        "ids": {"doi": "10.1000/corrupt"},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }
    artifact = downloader.create_artifact(
        work, pdf_dir=tmp_path / "pdf", html_dir=tmp_path / "html"
    )
    pdf_url = "https://cdn.example/corrupt.pdf"
    responses.add(
        responses.GET,
        pdf_url,
        body=b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n",
        headers={"Content-Type": "application/pdf"},
        status=200,
    )
    responses.add(responses.HEAD, pdf_url, headers={"Content-Type": "application/pdf"}, status=200)
    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, pdf_url, referer=None, timeout=5.0)
    assert outcome.classification is Classification.MISS
    assert outcome.path is None
    assert outcome.error == "pdf-eof-missing"
