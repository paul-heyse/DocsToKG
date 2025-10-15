"""
Resolver Pipeline Behaviour Tests

This module verifies the orchestration logic within the resolver pipeline
including ordering, attempt limits, deduplication, artifact collection,
and rate limiting to ensure predictable download flows.

Key Scenarios:
- Enforces resolver execution order and configurable attempt limits
- Deduplicates resolver URLs while capturing HTML fallback paths
- Validates rate limiting and logging side effects across runs

Dependencies:
- pytest/requests: Simulation of HTTP sessions
- DocsToKG.ContentDownload: Resolver pipeline under test

Usage:
    pytest tests/test_pipeline_behaviour.py
"""

from pathlib import Path
from typing import List

import pytest
import requests

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload import resolvers
from DocsToKG.ContentDownload.resolvers.providers.openalex import OpenAlexResolver


class MemoryLogger:
    def __init__(self) -> None:
        self.records: List[resolvers.AttemptRecord] = []

    def log(self, record: resolvers.AttemptRecord) -> None:
        self.records.append(record)


class StubResolver:
    def __init__(self, name: str, results):
        self.name = name
        self._results = results

    def is_enabled(self, config, artifact):
        return True

    def iter_urls(self, session, config, artifact):
        for result in self._results:
            yield result


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


def build_outcome(classification: str, path: str | None = None) -> resolvers.DownloadOutcome:
    return resolvers.DownloadOutcome(
        classification=classification,
        path=path,
        http_status=200 if classification == "pdf" else 400,
        content_type="application/pdf" if classification.startswith("pdf") else "text/html",
        elapsed_ms=10.0,
        error=None,
    )


def test_pipeline_respects_custom_order(tmp_path):
    artifact = make_artifact(tmp_path)
    attempts: List[str] = []

    def downloader_fn(session, art, url, referer, timeout):
        attempts.append(url)
        if "beta" in url:
            return build_outcome("pdf", path=str(art.pdf_dir / "beta.pdf"))
        return build_outcome("http_error")

    alpha = StubResolver("alpha", [resolvers.ResolverResult(url="https://alpha.example/1")])
    beta = StubResolver("beta", [resolvers.ResolverResult(url="https://beta.example/1")])

    config = resolvers.ResolverConfig(
        resolver_order=["beta", "alpha"],
        resolver_toggles={"alpha": True, "beta": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = resolvers.ResolverPipeline(
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


def test_pipeline_stops_after_max_attempts(tmp_path):
    artifact = make_artifact(tmp_path)

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("http_error")

    resolver = StubResolver(
        "single",
        [
            resolvers.ResolverResult(url="https://example.com/a"),
            resolvers.ResolverResult(url="https://example.com/b"),
        ],
    )
    config = resolvers.ResolverConfig(
        resolver_order=["single"],
        resolver_toggles={"single": True},
        max_attempts_per_work=1,
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = resolvers.ResolverPipeline(
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


def test_pipeline_deduplicates_urls(tmp_path):
    artifact = make_artifact(tmp_path)
    resolver = StubResolver(
        "dup",
        [
            resolvers.ResolverResult(url="https://dup.example/x"),
            resolvers.ResolverResult(url="https://dup.example/x"),
        ],
    )
    config = resolvers.ResolverConfig(
        resolver_order=["dup"],
        resolver_toggles={"dup": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("http_error")

    pipeline = resolvers.ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=downloader_fn,
        logger=logger,
    )
    session = requests.Session()
    pipeline.run(session, artifact)
    duplicates = [record.reason for record in logger.records if record.reason == "duplicate-url"]
    assert duplicates == ["duplicate-url"]


def test_pipeline_collects_html_paths(tmp_path):
    artifact = make_artifact(tmp_path)

    def downloader_fn(session, art, url, referer, timeout):
        html_path = art.html_dir / "example.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text("<html></html>", encoding="utf-8")
        return build_outcome("html", path=str(html_path))

    resolver = StubResolver("html", [resolvers.ResolverResult(url="https://example.com/html")])
    config = resolvers.ResolverConfig(
        resolver_order=["html"],
        resolver_toggles={"html": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = resolvers.ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=downloader_fn,
        logger=logger,
    )
    session = requests.Session()
    result = pipeline.run(session, artifact)
    assert result.success is False
    assert result.html_paths and Path(result.html_paths[0]).name == "example.html"


def test_pipeline_rate_limit_enforced(monkeypatch, tmp_path):
    artifact = make_artifact(tmp_path)
    timeline = [0.0, 0.2, 0.2, 1.2]

    def fake_monotonic():
        return timeline.pop(0)

    sleeps: List[float] = []

    def fake_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr(resolvers.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(resolvers.time, "sleep", fake_sleep)

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("pdf", path=str(art.pdf_dir / "out.pdf"))

    resolver = StubResolver("limited", [resolvers.ResolverResult(url="https://example.com/pdf")])
    config = resolvers.ResolverConfig(
        resolver_order=["limited"],
        resolver_toggles={"limited": True},
        resolver_rate_limits={"limited": 1.0},
        enable_head_precheck=False,
    )
    pipeline = resolvers.ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=downloader_fn,
        logger=MemoryLogger(),
    )
    session = requests.Session()
    pipeline.run(session, artifact)
    pipeline.run(session, artifact)
    assert len(sleeps) == 2
    for delay in sleeps:
        assert pytest.approx(delay, rel=0.01) == 1.0


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

    fallback = StubResolver(
        "fallback", [resolvers.ResolverResult(url="https://fallback.example/pdf")]
    )
    config = resolvers.ResolverConfig(
        resolver_order=["openalex", "fallback"],
        resolver_toggles={"openalex": True, "fallback": True},
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    metrics = resolvers.ResolverMetrics()
    pipeline = resolvers.ResolverPipeline(
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


def test_openalex_respects_rate_limit(monkeypatch, tmp_path):
    artifact = make_artifact(tmp_path)
    artifact.pdf_urls = ["https://openalex.org/pdf-one"]

    timeline = [0.0, 0.4, 0.4, 1.2]

    def fake_monotonic():
        return timeline.pop(0)

    sleeps: List[float] = []

    def fake_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr(resolvers.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(resolvers.time, "sleep", fake_sleep)

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("pdf", path=str(art.pdf_dir / "result.pdf"))

    config = resolvers.ResolverConfig(
        resolver_order=["openalex"],
        resolver_toggles={"openalex": True},
        resolver_min_interval_s={"openalex": 0.8},
        enable_head_precheck=False,
    )
    pipeline = resolvers.ResolverPipeline(
        resolvers=[OpenAlexResolver()],
        config=config,
        download_func=downloader_fn,
        logger=MemoryLogger(),
    )
    session = requests.Session()
    pipeline.run(session, artifact)
    pipeline.run(session, artifact)

    assert sleeps and pytest.approx(sleeps[0], rel=0.05) == 0.8


def test_pipeline_records_failed_urls(tmp_path):
    artifact = make_artifact(tmp_path)
    artifact.pdf_urls = ["https://openalex.org/broken.pdf"]

    def downloader_fn(session, art, url, referer, timeout):
        return build_outcome("http_error")

    config = resolvers.ResolverConfig(
        resolver_order=["openalex"],
        resolver_toggles={"openalex": True},
        max_attempts_per_work=1,
        enable_head_precheck=False,
    )
    logger = MemoryLogger()
    pipeline = resolvers.ResolverPipeline(
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
