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

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

pytest.importorskip("pyalex")

import requests

from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact, classify_payload, ensure_dir
from DocsToKG.ContentDownload.resolvers import (
    AttemptRecord,
    DownloadOutcome,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
)
from DocsToKG.ContentDownload.resolvers.providers.landing_page import LandingPageResolver
from DocsToKG.ContentDownload.resolvers.providers.unpaywall import UnpaywallResolver


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
            if isinstance(url, ResolverResult):
                yield url
            else:
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
    config = ResolverConfig(
        resolver_order=["resolver_a", "resolver_b"],
        resolver_toggles={"resolver_a": True, "resolver_b": True},
    )
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


def test_head_precheck_allows_redirect(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", ["https://example.org/file.pdf"])
    config = ResolverConfig(
        resolver_order=["stub"], resolver_toggles={"stub": True}, enable_head_precheck=True
    )
    logger = ListLogger()
    metrics = ResolverMetrics()

    def download_func(session, artifact, url, referer, timeout):  # pragma: no cover - not used
        return DownloadOutcome(classification="http_error", http_status=404, elapsed_ms=1.0)

    pipeline = ResolverPipeline(
        resolvers=[resolver],
        config=config,
        download_func=download_func,
        logger=logger,
        metrics=metrics,
    )

    class _HeadResponse:
        def __init__(self, status_code: int, headers: Dict[str, str]):
            self.status_code = status_code
            self.headers = headers

        def close(self) -> None:
            return None

    def fake_head(session, method, url, **kwargs):
        assert method == "HEAD"
        assert kwargs["allow_redirects"] is True
        return _HeadResponse(302, {"Content-Type": "application/pdf", "Content-Length": "1234"})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.pipeline.request_with_retries",
        fake_head,
    )

    assert pipeline._head_precheck_url(object(), "https://example.org/file.pdf", timeout=5.0)

    def fake_html(session, method, url, **kwargs):
        return _HeadResponse(200, {"Content-Type": "text/html", "Content-Length": "128"})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.pipeline.request_with_retries",
        fake_html,
    )

    assert not pipeline._head_precheck_url(object(), "https://example.org/file.pdf", timeout=5.0)


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
            if self.name == "openalex":
                for direct in artifact.pdf_urls:
                    yield ResolverResult(url=direct)
                if artifact.open_access_url:
                    yield ResolverResult(url=artifact.open_access_url)
            elif artifact.work_id == "W2" and self.name == "unpaywall":
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

        def mount(self, *args, **kwargs):
            return None

    monkeypatch.setattr(module, "build_query", lambda args: None)
    monkeypatch.setattr(module, "iterate_openalex", fake_iterate)
    monkeypatch.setattr(module, "download_candidate", fake_download)
    monkeypatch.setattr(
        module,
        "default_resolvers",
        lambda: [FakeResolver("openalex"), FakeResolver("unpaywall")],
    )
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
        "--resolver-order",
        "openalex,unpaywall",
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


class DummyHeadResponse:
    def __init__(self, status_code: int = 200, headers: Optional[Dict[str, str]] = None):
        self.status_code = status_code
        self.headers = headers or {}

    def close(self) -> None:
        return None


def test_head_precheck_skips_html(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"], resolver_toggles={"stub": True})
    logger = ListLogger()
    download_calls: List[str] = []

    def fake_download(session, art, url, referer, timeout):
        download_calls.append(url)
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    def fake_request(session, method, url, **kwargs):
        assert method == "HEAD"
        return DummyHeadResponse(headers={"Content-Type": "text/html"})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.http.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert download_calls == []
    assert result.success is False
    assert any(record.reason == "head-precheck-failed" for record in logger.records)


def test_head_precheck_skips_zero_length(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"], resolver_toggles={"stub": True})
    logger = ListLogger()
    download_calls: List[str] = []

    def fake_download(session, art, url, referer, timeout):
        download_calls.append(url)
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    def fake_request(session, method, url, **kwargs):
        return DummyHeadResponse(headers={"Content-Length": "0"})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.http.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is False
    assert any(record.reason == "head-precheck-failed" for record in logger.records)
    assert download_calls == []


def test_head_precheck_skips_error_status(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"], resolver_toggles={"stub": True})
    logger = ListLogger()
    download_calls: List[str] = []

    def fake_download(session, art, url, referer, timeout):
        download_calls.append(url)
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    def fake_request(session, method, url, **kwargs):
        return DummyHeadResponse(status_code=404)

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.http.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is False
    assert any(record.reason == "head-precheck-failed" for record in logger.records)
    assert download_calls == []


def test_head_precheck_allows_pdf(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"], resolver_toggles={"stub": True})
    logger = ListLogger()
    download_calls: List[str] = []

    def fake_download(session, art, url, referer, timeout):
        download_calls.append(url)
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    def fake_request(session, method, url, **kwargs):
        return DummyHeadResponse(headers={"Content-Type": "application/pdf"})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.http.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is True
    assert download_calls == ["https://example.org/pdf"]


def test_head_precheck_failure_allows_download(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"], resolver_toggles={"stub": True})
    logger = ListLogger()
    download_calls: List[str] = []

    def fake_download(session, art, url, referer, timeout):
        download_calls.append(url)
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    def fake_request(session, method, url, **kwargs):
        raise requests.Timeout("HEAD timeout")

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.http.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is True
    assert download_calls == ["https://example.org/pdf"]


def test_head_precheck_respects_global_disable(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
        enable_head_precheck=False,
    )
    logger = ListLogger()
    head_calls: List[str] = []
    download_calls: List[str] = []

    def fake_request(session, method, url, **kwargs):
        head_calls.append(url)
        return DummyHeadResponse()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.http.request_with_retries",
        fake_request,
    )

    def fake_download(session, art, url, referer, timeout):
        download_calls.append(url)
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    pipeline.run(session, artifact)

    assert head_calls == []
    assert download_calls == ["https://example.org/pdf"]


def test_head_precheck_resolver_override(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    skip_resolver = StubResolver("skip", [ResolverResult(url="https://example.org/skip.pdf")])
    enforce_resolver = StubResolver(
        "enforce", [ResolverResult(url="https://example.org/enforce.pdf")]
    )
    config = ResolverConfig(
        resolver_order=["skip", "enforce"],
        resolver_toggles={"skip": True, "enforce": True},
        resolver_head_precheck={"skip": False},
    )
    logger = ListLogger()
    head_calls: List[str] = []

    def fake_request(session, method, url, **kwargs):
        head_calls.append(url)
        return DummyHeadResponse(headers={"Content-Type": "application/pdf"})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.http.request_with_retries",
        fake_request,
    )

    def fake_download(session, art, url, referer, timeout):
        return DownloadOutcome("http_error", None, 500, "text/html", 1.0)

    pipeline = ResolverPipeline(
        [skip_resolver, enforce_resolver],
        config,
        fake_download,
        logger,
        ResolverMetrics(),
    )
    session = requests.Session()
    pipeline.run(session, artifact)

    assert head_calls == ["https://example.org/enforce.pdf"]
