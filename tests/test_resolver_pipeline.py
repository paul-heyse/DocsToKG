"""
Resolver Pipeline Integration Tests

This module validates the resolver pipeline orchestration, covering
candidate selection, resolver order, CLI integration, and classification
helpers that support the OpenAlex download CLI.

Key Scenarios:
- Confirms pipeline stops after first successful download
- Exercises Unpaywall and landing page resolvers in isolation
- Validates CLI happy path flows including manifest creation

Dependencies:
- pytest: Assertions and fixtures
- DocsToKG.ContentDownload: Resolver pipeline and CLI under test

Usage:
    pytest tests/test_resolver_pipeline.py
"""

from pathlib import Path
import sys

import pytest

from DocsToKG.ContentDownload.download_pyalex_pdfs import (
    WorkArtifact,
    classify_payload,
    ensure_dir,
)
from DocsToKG.ContentDownload.resolvers import (
    AttemptRecord,
    DownloadOutcome,
    ResolverConfig,
    ResolverPipeline,
    ResolverResult,
    UnpaywallResolver,
    LandingPageResolver,
    ResolverMetrics,
)


class DummySession:
    def __init__(self, responses):
        self._responses = responses

    def head(self, url, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError

    def get(self, url, **kwargs):
        try:
            response = self._responses[url]
        except KeyError as exc:  # pragma: no cover - safety
            raise AssertionError(f"Unexpected URL {url}") from exc
        return response


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class ListLogger:
    def __init__(self):
        self.records = []

    def log(self, record: AttemptRecord) -> None:
        self.records.append(record)


class StubResolver:
    def __init__(self, name, urls):
        self.name = name
        self.urls = urls

    def is_enabled(self, config, artifact):
        return True

    def iter_urls(self, session, config, artifact):
        for url in self.urls:
            yield ResolverResult(url=url)


def build_artifact(tmp_path: Path) -> WorkArtifact:
    return WorkArtifact(
        work_id="W123",
        title="Example",
        publication_year=2020,
        doi="10.123/example",
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


def test_classify_payload_detects_pdf_and_html():
    html = b"<html><body>Hello</body></html>"
    pdf = b"%PDF-sample"
    assert classify_payload(html, "text/html", "https://example.com") == "html"
    assert classify_payload(pdf, "application/pdf", "https://example.com/doc.pdf") == "pdf"


def test_pipeline_stops_on_first_success(tmp_path):
    artifact = build_artifact(tmp_path)

    def fake_download(session, artifact, url, referer, timeout):
        status = "pdf" if url == "https://b.example/pdf" else "http_error"
        return DownloadOutcome(
            classification=status,
            path=str(artifact.pdf_dir / "out.pdf") if status == "pdf" else None,
            http_status=200 if status == "pdf" else 404,
            content_type="application/pdf" if status == "pdf" else "text/plain",
            elapsed_ms=12.0,
        )

    resolver_a = StubResolver("resolver_a", ["https://a.example/1"])
    resolver_b = StubResolver("resolver_b", ["https://b.example/pdf"])
    config = ResolverConfig(resolver_order=["resolver_a", "resolver_b"], resolver_toggles={"resolver_a": True, "resolver_b": True})
    logger = ListLogger()
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline(
        resolvers=[resolver_a, resolver_b],
        config=config,
        download_func=fake_download,
        logger=logger,
        metrics=metrics,
    )
    session = DummySession({})
    result = pipeline.run(session, artifact)
    assert result.success is True
    assert result.resolver_name == "resolver_b"
    assert metrics.successes["resolver_b"] == 1


def test_unpaywall_resolver_extracts_candidates(tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = UnpaywallResolver()
    config = ResolverConfig()
    config.unpaywall_email = "user@example.com"
    response = DummyResponse(
        json_data={
            "best_oa_location": {"url_for_pdf": "https://example.com/best.pdf"},
            "oa_locations": [
                {"url_for_pdf": "https://example.com/extra.pdf"},
                {"url_for_pdf": "https://example.com/best.pdf"},
            ],
        }
    )
    session = DummySession({"https://api.unpaywall.org/v2/10.123/example": response})
    urls = list(resolver.iter_urls(session, config, artifact))
    assert [u.url for u in urls if not u.is_event] == [
        "https://example.com/best.pdf",
        "https://example.com/extra.pdf",
    ]


def test_landing_page_resolver_meta_parsing(tmp_path, monkeypatch):
    pytest.importorskip("bs4")
    artifact = build_artifact(tmp_path)
    artifact.landing_urls = ["https://example.com/article"]
    resolver = LandingPageResolver()
    html = """<html><head><meta name='citation_pdf_url' content='/files/paper.pdf'></head></html>"""
    response = DummyResponse(text=html)
    session = DummySession({"https://example.com/article": response})
    config = ResolverConfig()
    results = list(resolver.iter_urls(session, config, artifact))
    assert results[0].url == "https://example.com/files/paper.pdf"


def test_cli_integration_happy_path(monkeypatch, tmp_path):
    from DocsToKG.ContentDownload import download_pyalex_pdfs as module

    works = [
        {
            "id": "https://openalex.org/W1",
            "title": "Direct PDF",
            "publication_year": 2021,
            "ids": {"doi": "10.1/direct"},
            "best_oa_location": {"pdf_url": "https://direct/pdf1"},
            "primary_location": None,
            "locations": [],
            "open_access": {"oa_url": None},
        },
        {
            "id": "https://openalex.org/W2",
            "title": "Resolver Success",
            "publication_year": 2021,
            "ids": {"doi": "10.1/resolver"},
            "best_oa_location": {},
            "primary_location": {},
            "locations": [],
            "open_access": {"oa_url": None},
        },
        {
            "id": "https://openalex.org/W3",
            "title": "HTML Only",
            "publication_year": 2021,
            "ids": {"doi": "10.1/html"},
            "best_oa_location": {},
            "primary_location": {},
            "locations": [],
            "open_access": {"oa_url": None},
        },
    ]

    def fake_iterate(query, per_page, max_results):
        for work in works:
            yield work

    class FakeResolver:
        def __init__(self, name):
            self.name = name

        def is_enabled(self, config, artifact):
            return True

        def iter_urls(self, session, config, artifact):
            if artifact.work_id == "W2" and self.name == "unpaywall":
                yield ResolverResult(url="resolver://success")
            elif artifact.work_id == "W3" and self.name == "unpaywall":
                yield ResolverResult(url="resolver://html")

    def fake_download(session, artifact, url, referer, timeout):
        if "direct" in url or "success" in url:
            path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
            ensure_dir(path.parent)
            path.write_bytes(b"%PDF")
            return DownloadOutcome("pdf", str(path), 200, "application/pdf", 1.0)
        path = artifact.html_dir / f"{artifact.base_stem}.html"
        ensure_dir(path.parent)
        path.write_text("<html></html>")
        return DownloadOutcome("html", str(path), 200, "text/html", 1.0)

    class FakeSession:
        def __init__(self):
            self.headers = {}

    monkeypatch.setattr(module, "build_query", lambda args: None)
    monkeypatch.setattr(module, "iterate_openalex", fake_iterate)
    monkeypatch.setattr(module, "download_candidate", fake_download)
    monkeypatch.setattr(module, "default_resolvers", lambda: [FakeResolver("unpaywall")])
    monkeypatch.setattr(module.requests, "Session", FakeSession)

    log_csv = tmp_path / "attempts.csv"
    pdf_dir = tmp_path / "pdfs"
    args = [
        "prog",
        "--topic",
        "test",
        "--year-start",
        "2020",
        "--year-end",
        "2021",
        "--out",
        str(pdf_dir),
        "--log-csv",
        str(log_csv),
        "--max",
        "3",
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(sys, "argv", args)
    module.main()

    saved_files = list(pdf_dir.glob("*.pdf"))
    assert any(f.name.endswith("W1.pdf") or "Direct" in f.name for f in saved_files)
    assert any("Resolver" in f.name for f in saved_files)
    assert log_csv.exists()
    rows = [row for row in log_csv.read_text().strip().splitlines() if row]
    assert len(rows) >= 2  # header + attempts
