"""
Content Download Edge Case Tests

This module captures regression tests for less common scenarios within
the OpenAlex download pipeline, exercising HTML misclassification,
Wayback fallbacks, logging integrity, and retry budgeting.

Key Scenarios:
- Reclassifies misleading content types based on payload inspection
- Validates Wayback resolver behaviour when archives are unavailable
- Ensures manifest and attempt logs stay synchronized on success/failure
- Confirms retry budgeting halts processing after configured limits

Dependencies:
- pytest/responses: HTTP mocking for resolver interactions
- DocsToKG.ContentDownload: Resolver pipeline and helpers under test

Usage:
    pytest tests/test_edge_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pytest

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import (
    JsonlLogger,
    WorkArtifact,
    attempt_openalex_candidates,
    download_candidate,
    process_one_work,
)
from DocsToKG.ContentDownload.resolvers import (
    DownloadOutcome,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
    WaybackResolver,
)

requests = pytest.importorskip("requests")
responses = pytest.importorskip("responses")


def _make_artifact(tmp_path: Path, **overrides: Any) -> WorkArtifact:
    params: Dict[str, Any] = dict(
        work_id="WEDGE",
        title="Edge Case",
        publication_year=2024,
        doi="10.1/test",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/resource"],
        open_access_url=None,
        source_display_names=[],
        base_stem="edge-case",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    params.update(overrides)
    return WorkArtifact(**params)


@responses.activate
def test_html_classification_overrides_misleading_content_type(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    url = artifact.pdf_urls[0]
    responses.add(responses.HEAD, url, status=200, headers={"Content-Type": "application/pdf"})
    responses.add(responses.GET, url, status=200, body="<html><body>Fake PDF</body></html>")

    outcome = download_candidate(
        requests.Session(),
        artifact,
        url,
        referer=None,
        timeout=10.0,
        context={"dry_run": False, "extract_html_text": False, "previous": {}},
    )

    assert outcome.classification == "html"
    assert outcome.path and outcome.path.endswith(".html")


@responses.activate
def test_wayback_resolver_skips_unavailable_archives(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path, pdf_urls=[])
    artifact.failed_pdf_urls = ["https://example.org/missing.pdf"]
    session = requests.Session()
    config = ResolverConfig()
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={
            "archived_snapshots": {"closest": {"available": False, "url": "https://archive.org"}}
        },
        status=200,
    )

    results = list(WaybackResolver().iter_urls(session, config, artifact))
    assert results == []


def test_manifest_and_attempts_single_success(tmp_path: Path) -> None:
    work = {
        "id": "https://openalex.org/WEDGE",
        "title": "Edge Case",
        "publication_year": 2024,
        "ids": {"doi": "10.1/test"},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }

    artifact = _make_artifact(tmp_path)
    logger_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(logger_path)
    metrics = ResolverMetrics()

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterable[ResolverResult]:
            yield ResolverResult(url="https://resolver.example/paper.pdf")

    def fake_download(*args: Any, **kwargs: Any) -> DownloadOutcome:
        pdf_path = artifact.pdf_dir / "resolver.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
        return DownloadOutcome(
            classification="pdf",
            path=str(pdf_path),
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=5.0,
            sha256="deadbeef",
            content_length=12,
        )

    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
    )
    pipeline = ResolverPipeline([StubResolver()], config, fake_download, logger, metrics)

    result = process_one_work(
        work,
        requests.Session(),
        artifact.pdf_dir,
        artifact.html_dir,
        pipeline,
        logger,
        metrics,
        dry_run=False,
        extract_html_text=False,
        previous_lookup={},
        resume_completed=set(),
    )

    logger.close()

    assert result["saved"] is True
    records = [
        json.loads(line) for line in logger_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    attempts = [
        entry
        for entry in records
        if entry["record_type"] == "attempt" and entry["status"] in {"pdf", "pdf_unknown"}
    ]
    manifests = [
        entry
        for entry in records
        if entry["record_type"] == "manifest" and entry["classification"] in {"pdf", "pdf_unknown"}
    ]
    assert len(attempts) == 1
    assert len(manifests) == 1
    assert attempts[0]["work_id"] == manifests[0]["work_id"] == "WEDGE"
    assert attempts[0]["sha256"] == "deadbeef"
    assert manifests[0]["path"].endswith("resolver.pdf")
    assert Path(manifests[0]["path"]).exists()


@responses.activate
def test_openalex_attempts_use_session_headers(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    logger_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(logger_path)
    metrics = ResolverMetrics()
    session = requests.Session()
    session.headers.update({"User-Agent": "EdgeTester/1.0"})
    url = artifact.pdf_urls[0]
    responses.add(responses.HEAD, url, status=200, headers={"Content-Type": "application/pdf"})
    responses.add(responses.GET, url, status=503)

    attempt_openalex_candidates(
        session,
        artifact,
        logger,
        metrics,
        {"dry_run": True, "extract_html_text": False, "previous": {}},
    )

    assert any(
        call.request.headers.get("User-Agent") == "EdgeTester/1.0" for call in responses.calls
    )
    logger.close()


def test_retry_budget_honours_max_attempts(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
        max_attempts_per_work=3,
    )

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterator[ResolverResult]:
            for i in range(10):
                yield ResolverResult(url=f"https://resolver.example/{i}.pdf")

    calls: List[str] = []

    def failing_download(*args: Any, **kwargs: Any) -> DownloadOutcome:
        url = args[2]
        calls.append(url)
        return DownloadOutcome(
            classification="http_error",
            path=None,
            http_status=503,
            content_type="text/plain",
            elapsed_ms=1.0,
            error="server-error",
        )

    class ListLogger:
        def __init__(self) -> None:
            self.records: List[Any] = []

        def log(self, record) -> None:  # pragma: no cover - simple passthrough
            self.records.append(record)

    pipeline = ResolverPipeline(
        [StubResolver()],
        config,
        failing_download,
        ListLogger(),
        ResolverMetrics(),
    )

    result = pipeline.run(requests.Session(), artifact)
    assert result.success is False
    assert len(calls) == config.max_attempts_per_work
