"""Integration tests covering the resolver pipeline orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import List

import pytest

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact
from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolvers.providers import default_resolvers
from DocsToKG.ContentDownload.resolvers.types import (
    AttemptRecord,
    DownloadOutcome,
    ResolverConfig,
    ResolverMetrics,
    ResolverResult,
)


@dataclass
class MemoryLogger:
    records: List[AttemptRecord]

    def log(self, record: AttemptRecord) -> None:
        self.records.append(record)


def _make_artifact(tmp_path: Path) -> WorkArtifact:
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

    from DocsToKG.ContentDownload.download_pyalex_pdfs import _make_session, download_candidate

    session = _make_session({"User-Agent": "DocsToKG-Test/1.0"})
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
