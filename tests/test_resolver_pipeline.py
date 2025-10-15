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
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("pyalex")

import requests

from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact, classify_payload, ensure_dir
from DocsToKG.ContentDownload.resolvers import pipeline as pipeline_module
from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolvers.providers.landing_page import LandingPageResolver
from DocsToKG.ContentDownload.resolvers.providers.unpaywall import UnpaywallResolver
from DocsToKG.ContentDownload.resolvers.types import (
    AttemptRecord,
    DownloadOutcome,
    ResolverConfig,
    ResolverMetrics,
    ResolverResult,
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


def test_pipeline_records_resolver_exception(tmp_path):
    artifact = build_artifact(tmp_path)

    class ExplodingResolver:
        name = "exploder"

        def is_enabled(self, config, art):  # noqa: D401
            return True

        def iter_urls(self, session, config, art):  # noqa: D401
            raise RuntimeError("boom")

    def fake_download(session, artifact, url, referer, timeout):  # pragma: no cover - not reached
        return DownloadOutcome(
            "pdf", str(artifact.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0
        )

    resolver = ExplodingResolver()
    config = ResolverConfig(resolver_order=["exploder"], resolver_toggles={"exploder": True})
    logger = ListLogger()
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline([resolver], config, fake_download, logger, metrics)
    session = DummySession({})

    result = pipeline.run(session, artifact)

    assert result.success is False
    assert any(record.reason == "resolver-exception" for record in logger.records)
    assert metrics.failures["exploder"] == 1


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


def test_head_precheck_allows_redirect_to_pdf(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"], resolver_toggles={"stub": True})
    logger = ListLogger()
    download_calls: List[str] = []

    def fake_download(session, art, url, referer, timeout):
        download_calls.append(url)
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    def fake_request(session, method, url, **kwargs):
        return DummyHeadResponse(
            status_code=302,
            headers={"Content-Type": "application/pdf", "Location": "https://example.org/pdf"},
        )

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


def test_callable_accepts_argument_handles_noncallable():
    from DocsToKG.ContentDownload.resolvers.pipeline import _callable_accepts_argument

    sentinel = object()
    # Non-callable should trigger TypeError path and default to True
    assert _callable_accepts_argument(sentinel, "context") is True


def test_pipeline_logs_missing_resolver(tmp_path):
    artifact = build_artifact(tmp_path)
    config = ResolverConfig(resolver_order=["ghost"], resolver_toggles={"ghost": True})
    logger = ListLogger()
    metrics = ResolverMetrics()

    pipeline = ResolverPipeline([], config, lambda *args, **kwargs: None, logger, metrics)
    session = DummySession({})
    result = pipeline.run(session, artifact)

    assert result.success is False
    assert logger.records[0].reason == "resolver-missing"
    assert metrics.skips["ghost:missing"] == 1


def test_pipeline_skips_disabled_resolver(tmp_path):
    artifact = build_artifact(tmp_path)

    class DisabledResolver(StubResolver):
        def __init__(self):
            super().__init__("disabled", [])

    resolver = DisabledResolver()
    config = ResolverConfig(resolver_order=["disabled"])
    config.resolver_toggles["disabled"] = False
    logger = ListLogger()
    metrics = ResolverMetrics()

    pipeline = ResolverPipeline([resolver], config, lambda *args, **kwargs: None, logger, metrics)
    session = DummySession({})
    pipeline.run(session, artifact)

    assert logger.records[0].reason == "resolver-disabled"
    assert metrics.skips["disabled:disabled"] == 1


def test_pipeline_skips_not_applicable_resolver(tmp_path):
    artifact = build_artifact(tmp_path)

    class InapplicableResolver(StubResolver):
        def __init__(self):
            super().__init__("inapplicable", [])

        def is_enabled(self, config, art):
            return False

    resolver = InapplicableResolver()
    config = ResolverConfig(resolver_order=["inapplicable"])
    logger = ListLogger()
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline([resolver], config, lambda *args, **kwargs: None, logger, metrics)

    pipeline.run(DummySession({}), artifact)

    assert logger.records[0].reason == "resolver-not-applicable"
    assert metrics.skips["inapplicable:not-applicable"] == 1


def test_collect_resolver_results_handles_exception(tmp_path):
    artifact = build_artifact(tmp_path)

    class Exploder(StubResolver):
        def __init__(self):
            super().__init__("boom", [])

        def iter_urls(self, session, config, art):
            raise RuntimeError("simulated failure")

    resolver = Exploder()
    config = ResolverConfig(resolver_order=["boom"])
    logger = ListLogger()
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline([resolver], config, lambda *args, **kwargs: None, logger, metrics)

    results, wall_ms = pipeline._collect_resolver_results(
        resolver.name, resolver, DummySession({}), artifact
    )

    assert wall_ms >= 0.0
    assert results[0].event_reason == "resolver-exception"
    assert metrics.failures["boom"] == 1


def test_pipeline_records_event_and_skip_reason(tmp_path):
    artifact = build_artifact(tmp_path)
    event_result = ResolverResult(url=None, event="info", event_reason="rate-limit")
    resolver = StubResolver("stub", [event_result])
    config = ResolverConfig(resolver_order=["stub"])
    logger = ListLogger()
    metrics = ResolverMetrics()

    pipeline = ResolverPipeline([resolver], config, lambda *args, **kwargs: None, logger, metrics)
    pipeline.run(DummySession({}), artifact)

    assert logger.records[0].status == "info"
    assert metrics.skips["stub:rate-limit"] == 1


def test_pipeline_skips_duplicate_urls(tmp_path):
    artifact = build_artifact(tmp_path)
    duplicated = ResolverResult(url="https://example.org/dup.pdf")
    resolver = StubResolver("dup", [duplicated, duplicated])
    config = ResolverConfig(resolver_order=["dup"])
    logger = ListLogger()
    metrics = ResolverMetrics()

    def fake_download(session, art, url, referer, timeout):
        return DownloadOutcome("html", None, 200, "text/html", 1.0)

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, metrics)
    pipeline.run(DummySession({}), artifact)

    reasons = [record.reason for record in logger.records if record.reason == "duplicate-url"]
    assert reasons and metrics.skips["dup:duplicate-url"] == 1


def test_pipeline_head_precheck_failure_skips_attempt(monkeypatch, tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"])
    logger = ListLogger()
    metrics = ResolverMetrics()
    download_mock = Mock()

    pipeline = ResolverPipeline([resolver], config, download_mock, logger, metrics)
    monkeypatch.setattr(pipeline, "_head_precheck_url", lambda *args, **kwargs: False)

    pipeline.run(DummySession({}), artifact)

    assert download_mock.called is False
    assert logger.records[0].reason == "head-precheck-failed"
    assert metrics.skips["stub:head-precheck-failed"] == 1


def test_pipeline_event_without_reason(tmp_path):
    artifact = build_artifact(tmp_path)
    event_result = ResolverResult(url=None, event="info", event_reason=None)
    resolver = StubResolver("stub", [event_result])
    config = ResolverConfig(resolver_order=["stub"])
    logger = ListLogger()
    metrics = ResolverMetrics()

    pipeline = ResolverPipeline([resolver], config, lambda *args, **kwargs: None, logger, metrics)
    pipeline.run(DummySession({}), artifact)

    assert logger.records[0].status == "info"
    assert metrics.skips == {}


def test_pipeline_downloads_with_context_argument(tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"])
    logger = ListLogger()
    metrics = ResolverMetrics()
    context_received = {}

    def download_with_context(session, art, url, referer, timeout, context):
        context_received["value"] = context
        return DownloadOutcome("http_error", None, 500, "text/html", 1.0)

    pipeline = ResolverPipeline([resolver], config, download_with_context, logger, metrics)
    pipeline.run(DummySession({}), artifact, context={"dry_run": False})

    assert context_received["value"] == {"dry_run": False}
    assert logger.records[-1].status == "http_error"


def test_pipeline_respects_max_attempts(tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("stub", [ResolverResult(url="https://example.org/pdf")])
    config = ResolverConfig(resolver_order=["stub"], max_attempts_per_work=1)
    logger = ListLogger()
    metrics = ResolverMetrics()

    def non_pdf_download(session, art, url, referer, timeout):
        return DownloadOutcome("html", None, 200, "text/html", 1.0)

    pipeline = ResolverPipeline([resolver], config, non_pdf_download, logger, metrics)
    result = pipeline.run(DummySession({}), artifact)

    assert result.success is False
    assert result.reason == "max-attempts-reached"


def test_pipeline_jitter_sleep_no_delay():
    config = ResolverConfig()
    config.sleep_jitter = 0.0
    pipeline = ResolverPipeline([], config, lambda *a, **k: None, ListLogger(), ResolverMetrics())

    with patch("DocsToKG.ContentDownload.resolvers.pipeline.time.sleep") as mock_sleep:
        pipeline._jitter_sleep()

    mock_sleep.assert_not_called()


def test_pipeline_concurrent_skips_missing_resolver(tmp_path):
    artifact = build_artifact(tmp_path)
    resolver = StubResolver("available", [ResolverResult(url="https://example.org/file.pdf")])
    config = ResolverConfig(
        resolver_order=["missing", "available"],
        resolver_toggles={"missing": True, "available": True},
        max_concurrent_resolvers=2,
    )
    logger = ListLogger()
    metrics = ResolverMetrics()

    def download(session, art, url, referer, timeout):
        return DownloadOutcome("pdf", str(art.pdf_dir / "result.pdf"), 200, "application/pdf", 1.0)

    pipeline = ResolverPipeline([resolver], config, download, logger, metrics)
    result = pipeline.run(DummySession({}), artifact)

    assert result.success is True
    assert any(record.reason == "resolver-missing" for record in logger.records)


def test_pipeline_ignores_empty_url(tmp_path):
    artifact = build_artifact(tmp_path)

    class NullResolver(StubResolver):
        def __init__(self):
            super().__init__("null", [ResolverResult(url="")])

    resolver = NullResolver()
    config = ResolverConfig(resolver_order=["null"])
    logger = ListLogger()
    metrics = ResolverMetrics()

    pipeline = ResolverPipeline(
        [resolver],
        config,
        lambda *a, **k: DownloadOutcome("html", None, 200, "text/html", 1.0),
        logger,
        metrics,
    )
    result = pipeline.run(DummySession({}), artifact)

    assert result.success is False


def test_pipeline_concurrent_execution(tmp_path):
    artifact = build_artifact(tmp_path)
    resolver_a = StubResolver("a", [ResolverResult(url="https://example.org/a.pdf")])
    resolver_b = StubResolver("b", [ResolverResult(url="https://example.org/b.pdf")])
    config = ResolverConfig(
        resolver_order=["a", "b"],
        resolver_toggles={"a": True, "b": True},
        max_concurrent_resolvers=2,
    )
    logger = ListLogger()
    metrics = ResolverMetrics()

    def download_pdf(session, art, url, referer, timeout):
        ensure_dir(art.pdf_dir)
        return DownloadOutcome(
            "pdf",
            path=str(art.pdf_dir / "result.pdf"),
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=1.0,
        )

    pipeline = ResolverPipeline([resolver_a, resolver_b], config, download_pdf, logger, metrics)
    result = pipeline.run(DummySession({}), artifact)

    assert result.success is True
    assert result.resolver_name in {"a", "b"}


def test_pipeline_global_deduplication_skips_repeat_urls(tmp_path):
    artifact = build_artifact(tmp_path)
    artifact.metadata["target_url"] = "https://example.org/shared.pdf"
    alt_pdf_dir = tmp_path / "pdf-second"
    alt_html_dir = tmp_path / "html-second"
    artifact_second = replace(
        artifact,
        work_id="W124",
        base_stem="example-two",
        pdf_dir=alt_pdf_dir,
        html_dir=alt_html_dir,
    )

    class StaticResolver(StubResolver):
        def __init__(self) -> None:
            super().__init__("static", [ResolverResult(url="https://example.org/shared.pdf")])

    resolver = StaticResolver()
    config = ResolverConfig(resolver_order=["static"], resolver_toggles={"static": True})
    config.enable_head_precheck = False
    config.enable_global_url_dedup = True
    logger = ListLogger()
    metrics = ResolverMetrics()
    download_calls: List[str] = []

    def download_pdf(session, art, url, referer, timeout):
        ensure_dir(art.pdf_dir)
        path = art.pdf_dir / f"{art.base_stem}.pdf"
        download_calls.append(art.work_id)
        return DownloadOutcome(
            "pdf",
            path=str(path),
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=1.0,
        )

    pipeline = ResolverPipeline([resolver], config, download_pdf, logger, metrics)
    result_first = pipeline.run(DummySession({}), artifact)
    result_second = pipeline.run(DummySession({}), artifact_second)

    assert result_first.success is True
    assert result_second.success is False
    assert download_calls == [artifact.work_id]
    assert any(record.reason == "duplicate-url-global" for record in logger.records)
    assert metrics.skips["static:duplicate-url-global"] == 1


def test_pipeline_domain_rate_limiting_enforces_interval(monkeypatch, tmp_path):
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: List[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            self.sleeps.append(duration)
            self.now += duration

    fake = FakeClock()
    monkeypatch.setattr(pipeline_module.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(pipeline_module.time, "sleep", fake.sleep)

    artifact = replace(
        build_artifact(tmp_path),
        metadata={"target_url": "https://example.org/shared.pdf"},
    )
    second = replace(
        artifact,
        work_id="W125",
        base_stem="example-two",
        pdf_dir=tmp_path / "pdf-second",
        html_dir=tmp_path / "html-second",
        metadata={"target_url": "https://example.org/shared.pdf"},
    )
    third = replace(
        artifact,
        work_id="W126",
        base_stem="example-three",
        pdf_dir=tmp_path / "pdf-third",
        html_dir=tmp_path / "html-third",
        metadata={"target_url": "https://other.org/alt.pdf"},
    )

    class DynamicResolver:
        name = "dynamic"

        def is_enabled(self, config, art):
            return True

        def iter_urls(self, session, config, art):
            yield ResolverResult(url=art.metadata["target_url"])

    config = ResolverConfig(resolver_order=["dynamic"], resolver_toggles={"dynamic": True})
    config.enable_head_precheck = False
    config.domain_min_interval_s = {"example.org": 0.5}
    logger = ListLogger()
    metrics = ResolverMetrics()
    download_calls: List[str] = []

    def download_pdf(session, art, url, referer, timeout):
        ensure_dir(art.pdf_dir)
        download_calls.append(url)
        return DownloadOutcome(
            "pdf",
            path=str(art.pdf_dir / f"{art.base_stem}.pdf"),
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=1.0,
        )

    pipeline = ResolverPipeline([DynamicResolver()], config, download_pdf, logger, metrics)

    first = pipeline.run(DummySession({}), artifact)
    fake.now += 0.1
    second_result = pipeline.run(DummySession({}), second)
    fake.now += 0.2
    third_result = pipeline.run(DummySession({}), third)

    assert first.success is True
    assert second_result.success is True
    assert third_result.success is True
    assert download_calls == [
        "https://example.org/shared.pdf",
        "https://example.org/shared.pdf",
        "https://other.org/alt.pdf",
    ]
    assert fake.sleeps
    assert fake.sleeps[0] == pytest.approx(0.4, rel=0.05)
    assert len(fake.sleeps) == 1
