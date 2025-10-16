"""Consolidated resolver core tests."""

from __future__ import annotations

import importlib
import json
import sys
import warnings
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import Mock, patch

import pytest

import DocsToKG.ContentDownload.resolvers as pipeline_module
import DocsToKG.ContentDownload.resolvers as providers_module
import DocsToKG.ContentDownload.resolvers as resolvers
from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload.download_pyalex_pdfs import (
    WorkArtifact,
    classify_payload,
    ensure_dir,
    load_resolver_config,
)
from DocsToKG.ContentDownload.resolvers import (
    ApiResolverBase,
    ArxivResolver,
    AttemptRecord,
    CoreResolver,
    CrossrefResolver,
    DoajResolver,
    DownloadOutcome,
    EuropePmcResolver,
    HalResolver,
    LandingPageResolver,
    OpenAireResolver,
    OpenAlexResolver,
    OsfResolver,
    PmcResolver,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
    SemanticScholarResolver,
    UnpaywallResolver,
    WaybackResolver,
    ZenodoResolver,
    clear_resolver_caches,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
)

try:  # pragma: no cover - optional dependency for some environments
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = pytest.importorskip("requests")  # type: ignore

# BeautifulSoup is optional in some environments; skip helper tests when absent.
_bs4 = pytest.importorskip("bs4")
from bs4 import BeautifulSoup  # type: ignore  # noqa: E402

# ---- test_resolver_caching.py -----------------------------
pytest.importorskip("requests")


# ---- test_resolver_caching.py -----------------------------
@pytest.fixture(autouse=True)
def clear_cache_between_tests():
    clear_resolver_caches()
    yield
    clear_resolver_caches()


class DummyApiResolver(ApiResolverBase, register=False):
    name = "dummy_api"

    def is_enabled(self, config: ResolverConfig, artifact: Any) -> bool:  # pragma: no cover - helper
        return True

    def iter_urls(
        self,
        session: Any,
        config: ResolverConfig,
        artifact: Any,
    ) -> Iterable[ResolverResult]:  # pragma: no cover - helper
        return []


def test_api_resolver_base_error_handling(monkeypatch: Any) -> None:
    resolver = DummyApiResolver()
    config = ResolverConfig()
    config.polite_headers = {"User-Agent": "Tester"}
    session = Mock(spec=requests.Session)
    timeout_value = config.get_timeout(resolver.name)

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=requests.Timeout("boom")),
    )
    data, error = resolver._request_json(
        session,
        "GET",
        "https://example.org/timeout",
        config=config,
    )
    assert data is None
    assert error is not None
    assert error.event_reason == "timeout"
    assert error.metadata["timeout"] == timeout_value

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=requests.ConnectionError("offline")),
    )
    _, error = resolver._request_json(
        session,
        "GET",
        "https://example.org/connection",
        config=config,
    )
    assert error is not None
    assert error.event_reason == "connection-error"

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=requests.RequestException("bad request")),
    )
    _, error = resolver._request_json(
        session,
        "GET",
        "https://example.org/request",
        config=config,
    )
    assert error is not None
    assert error.event_reason == "request-error"

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(return_value=_StubResponse(status_code=502, json_data={})),
    )
    _, error = resolver._request_json(
        session,
        "GET",
        "https://example.org/http",
        config=config,
    )
    assert error is not None
    assert error.event_reason == "http-error"
    assert error.http_status == 502

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(
            return_value=_StubResponse(
                json_data=ValueError("bad json"),
                text="not json",  # provides preview
            )
        ),
    )
    _, error = resolver._request_json(
        session,
        "GET",
        "https://example.org/json",
        config=config,
    )
    assert error is not None
    assert error.event_reason == "json-error"
    assert "content_preview" in error.metadata

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(return_value=_StubResponse(json_data={"ok": True})),
    )
    data, error = resolver._request_json(
        session,
        "GET",
        "https://example.org/success",
        config=config,
    )
    assert error is None
    assert data == {"ok": True}


def test_landing_page_helpers_extract_expected_urls() -> None:
    base = "https://example.org/articles/view"

    soup_meta = BeautifulSoup(
        "<html><head><meta name='citation_pdf_url' content='../files/paper.pdf'></head></html>",
        "html.parser",
    )
    assert find_pdf_via_meta(soup_meta, base) == "https://example.org/files/paper.pdf"

    soup_meta_missing = BeautifulSoup("<html></html>", "html.parser")
    assert find_pdf_via_meta(soup_meta_missing, base) is None

    soup_link = BeautifulSoup(
        """
        <html><head>
            <link rel='alternate' type='application/pdf' href='assets/download.pdf'>
        </head></html>
        """,
        "html.parser",
    )
    assert find_pdf_via_link(soup_link, base) == "https://example.org/assets/download.pdf"

    soup_link_missing = BeautifulSoup(
        "<html><head><link rel='stylesheet' href='style.css'></head></html>",
        "html.parser",
    )
    assert find_pdf_via_link(soup_link_missing, base) is None

    soup_anchor = BeautifulSoup(
        "<html><body><a href='pdfs/final.pdf'>Download</a></body></html>",
        "html.parser",
    )
    assert find_pdf_via_anchor(soup_anchor, base) == "https://example.org/pdfs/final.pdf"

    soup_anchor_text = BeautifulSoup(
        "<html><body><a href='pdf?id=123'>View PDF</a></body></html>",
        "html.parser",
    )
    assert find_pdf_via_anchor(soup_anchor_text, base) is None

# ---- test_resolver_caching.py -----------------------------
def test_resolvers_use_shared_retry_helper(monkeypatch):
    calls: list[str] = []

    def fake_request(session, method, url, **kwargs):
        calls.append(url)
        if "unpaywall" in url:
            return _StubResponse(
                json_data={
                    "best_oa_location": {"url_for_pdf": "https://example.org/unpaywall.pdf"}
                }
            )
        if "crossref" in url:
            return _StubResponse(
                json_data={
                    "message": {
                        "link": [
                            {
                                "URL": "https://example.org/crossref.pdf",
                                "content-type": "application/pdf",
                            }
                        ]
                    }
                }
            )
        if "semanticscholar" in url:
            return _StubResponse(
                json_data={"openAccessPdf": {"url": "https://example.org/s2.pdf"}}
            )
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        fake_request,
    )

    config = ResolverConfig()
    config.unpaywall_email = "test@example.org"
    config.mailto = "test@example.org"
    config.polite_headers = {"User-Agent": "Test"}
    config.semantic_scholar_api_key = "key"

    artifact = SimpleNamespace(doi="10.1234/test")

    session = Mock()

    unpaywall = UnpaywallResolver()
    list(unpaywall.iter_urls(session, config, artifact))
    list(unpaywall.iter_urls(session, config, artifact))
    crossref = CrossrefResolver()
    list(crossref.iter_urls(session, config, artifact))
    list(crossref.iter_urls(session, config, artifact))
    s2 = SemanticScholarResolver()
    list(s2.iter_urls(session, config, artifact))
    list(s2.iter_urls(session, config, artifact))

    assert calls == [
        "https://api.unpaywall.org/v2/10.1234/test",
        "https://api.unpaywall.org/v2/10.1234/test",
        "https://api.crossref.org/works/10.1234/test",
        "https://api.crossref.org/works/10.1234/test",
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/test",
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/test",
    ]

    clear_resolver_caches()
    list(unpaywall.iter_urls(session, config, artifact))
    assert calls[-1] == "https://api.unpaywall.org/v2/10.1234/test"


# ---- test_resolver_config.py -----------------------------
pytest.importorskip("requests")

# ---- test_resolver_config.py -----------------------------
pytest.importorskip("pyalex")


# ---- test_resolver_config.py -----------------------------
def test_deprecated_resolver_rate_limits_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{" '"resolver_rate_limits": {"unpaywall": 2.0}' "}")
    args = Namespace(
        resolver_config=str(config_path),
        unpaywall_email=None,
        core_api_key=None,
        semantic_scholar_api_key=None,
        doaj_api_key=None,
        mailto=None,
        max_resolver_attempts=None,
        resolver_timeout=None,
        disable_resolver=[],
        enable_resolver=[],
        resolver_order=None,
        log_jsonl=None,
        log_format="jsonl",
        resume_from=None,
    )
    caplog.set_level("WARNING")
    config = load_resolver_config(args, ["unpaywall"], None)
    assert config.resolver_min_interval_s["unpaywall"] == 2.0
    assert any("resolver_rate_limits deprecated" in record.message for record in caplog.records)


# ---- test_resolver_config.py -----------------------------
def test_user_agent_includes_mailto(tmp_path: Path) -> None:
    args = Namespace(
        resolver_config=None,
        unpaywall_email=None,
        core_api_key=None,
        semantic_scholar_api_key=None,
        doaj_api_key=None,
        mailto="ua-tester@example.org",
        max_resolver_attempts=None,
        resolver_timeout=None,
        disable_resolver=[],
        enable_resolver=["openaire"],
        resolver_order=None,
        log_jsonl=None,
        log_format="jsonl",
        resume_from=None,
    )

    config = load_resolver_config(args, ["unpaywall", "crossref", "openaire"], None)
    user_agent = config.polite_headers.get("User-Agent")
    assert (
        user_agent
        == "DocsToKGDownloader/1.0 (+ua-tester@example.org; mailto:ua-tester@example.org)"
    )
    assert config.polite_headers.get("mailto") == "ua-tester@example.org"
    assert config.resolver_toggles["openaire"] is True


def test_resolver_toggle_defaults_single_source(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {"openalex": False, "osf": True}
    monkeypatch.setattr(resolvers, "_DEFAULT_RESOLVER_TOGGLES", overrides.copy(), raising=False)
    monkeypatch.setattr(
        resolvers,
        "DEFAULT_RESOLVER_TOGGLES",
        MappingProxyType(resolvers._DEFAULT_RESOLVER_TOGGLES),  # type: ignore[attr-defined]
        raising=False,
    )

    args = Namespace(
        resolver_config=None,
        unpaywall_email=None,
        core_api_key=None,
        semantic_scholar_api_key=None,
        doaj_api_key=None,
        mailto=None,
        max_resolver_attempts=None,
        resolver_timeout=None,
        disable_resolver=[],
        enable_resolver=[],
        resolver_order=None,
        log_jsonl=None,
        log_format="jsonl",
        resume_from=None,
    )

    config = load_resolver_config(args, ["openalex", "osf"], None)

    assert config.resolver_toggles["openalex"] is False
    assert config.resolver_toggles["osf"] is True


@pytest.mark.parametrize(
    "resolver_cls, expected_url, extra_headers",
    [
        (
            UnpaywallResolver,
            "https://api.unpaywall.org/v2/10.1234/example",
            {},
        ),
        (
            CrossrefResolver,
            "https://api.crossref.org/works/10.1234/example",
            {},
        ),
        (
            CoreResolver,
            "https://api.core.ac.uk/v3/search/works",
            {"Authorization": "Bearer core-key"},
        ),
        (
            DoajResolver,
            "https://doaj.org/api/v2/search/articles/",
            {"X-API-KEY": "doaj-key"},
        ),
        (
            SemanticScholarResolver,
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/example",
            {"x-api-key": "s2-key"},
        ),
    ],
)
def test_resolvers_apply_polite_headers_and_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    resolver_cls,
    expected_url: str,
    extra_headers: Dict[str, str],
) -> None:
    captured: List[Dict[str, Any]] = []

    class _Response:
        def __init__(self, data: Any) -> None:
            self._data = data
            self.status_code = 200
            self.headers: Dict[str, str] = {}

        def json(self) -> Any:
            return self._data

        def close(self) -> None:  # pragma: no cover - compatibility shim
            return None

    def _payload() -> Any:
        if resolver_cls is UnpaywallResolver:
            return {"best_oa_location": {"url_for_pdf": "https://example.org/unpaywall.pdf"}}
        if resolver_cls is CrossrefResolver:
            return {
                "message": {
                    "link": [
                        {
                            "URL": "https://example.org/crossref.pdf",
                            "content-type": "application/pdf",
                        }
                    ]
                }
            }
        if resolver_cls is CoreResolver:
            return {
                "results": [
                    {
                        "downloadUrl": "https://example.org/core.pdf",
                        "fullTextLinks": [],
                    }
                ]
            }
        if resolver_cls is DoajResolver:
            return {
                "results": [
                    {
                        "bibjson": {
                            "link": [
                                {
                                    "url": "https://example.org/doaj.pdf",
                                }
                            ]
                        }
                    }
                ]
            }
        if resolver_cls is SemanticScholarResolver:
            return {"openAccessPdf": {"url": "https://example.org/s2.pdf"}}
        raise AssertionError("Unhandled resolver payload request")

    def fake_request(session, method, url, **kwargs):
        captured.append({"url": url, "kwargs": kwargs})
        return _Response(_payload())

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        fake_request,
    )

    config = ResolverConfig()
    config.polite_headers = {"User-Agent": "TestAgent/1.0", "mailto": "test@example.org"}
    config.mailto = "test@example.org"
    config.unpaywall_email = "test@example.org"
    config.core_api_key = "core-key"
    config.doaj_api_key = "doaj-key"
    config.semantic_scholar_api_key = "s2-key"

    resolver = resolver_cls()
    override = 17.5
    config.resolver_timeouts[resolver.name] = override

    artifact = SimpleNamespace(doi="10.1234/example")

    list(resolver.iter_urls(Mock(), config, artifact))

    assert captured, "resolver did not issue a request"
    record = captured[0]
    assert record["url"] == expected_url
    headers = record["kwargs"]["headers"]
    assert headers["User-Agent"] == "TestAgent/1.0"
    assert headers.get("mailto") == "test@example.org"
    for key, value in extra_headers.items():
        assert headers.get(key) == value
    assert record["kwargs"]["timeout"] == pytest.approx(override)


# ---- test_resolver_pipeline.py -----------------------------
pytest.importorskip("pyalex")


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
class ListLogger:
    def __init__(self):
        self.records = []

    def log(self, record: AttemptRecord) -> None:
        self.records.append(record)


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
def test_classify_payload_detects_pdf_and_html():
    html = b"<html><body>Hello</body></html>"
    pdf = b"%PDF-sample"
    assert classify_payload(html, "text/html", "https://example.com") == "html"
    assert classify_payload(pdf, "application/pdf", "https://example.com/doc.pdf") == "pdf"


def test_classify_payload_octet_stream_requires_sniff() -> None:
    data = b"binary data with no signature"
    assert (
        classify_payload(data, "application/octet-stream", "https://example.com/file.pdf")
        is None
    )


def test_classify_payload_octet_stream_with_pdf_signature() -> None:
    data = b"%PDF-1.7"
    assert (
        classify_payload(data, "application/octet-stream", "https://example.com/file.pdf")
        == "pdf"
    )


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.head_precheck",
        lambda *args, **kwargs: True,
    )

    assert pipeline._head_precheck_url(object(), "https://example.org/file.pdf", timeout=5.0)

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.head_precheck",
        lambda *args, **kwargs: False,
    )

    assert not pipeline._head_precheck_url(object(), "https://example.org/file.pdf", timeout=5.0)


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
class DummyHeadResponse:
    def __init__(self, status_code: int = 200, headers: Optional[Dict[str, str]] = None):
        self.status_code = status_code
        self.headers = headers or {}

    def close(self) -> None:
        return None


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert download_calls == []
    assert result.success is False
    assert any(record.reason == "head-precheck-failed" for record in logger.records)


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is False
    assert any(record.reason == "head-precheck-failed" for record in logger.records)
    assert download_calls == []


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is False
    assert any(record.reason == "head-precheck-failed" for record in logger.records)
    assert download_calls == []


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is True
    assert download_calls == ["https://example.org/pdf"]


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is True
    assert download_calls == ["https://example.org/pdf"]


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    pipeline = ResolverPipeline([resolver], config, fake_download, logger, ResolverMetrics())
    session = requests.Session()
    result = pipeline.run(session, artifact)

    assert result.success is True
    assert download_calls == ["https://example.org/pdf"]


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
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


# ---- test_resolver_pipeline.py -----------------------------
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
        "DocsToKG.ContentDownload.network.request_with_retries",
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


# ---- test_resolver_pipeline.py -----------------------------
def test_callable_accepts_argument_handles_noncallable():
    from DocsToKG.ContentDownload.resolvers import _callable_accepts_argument

    sentinel = object()
    # Non-callable should trigger TypeError path and default to True
    assert _callable_accepts_argument(sentinel, "context") is True


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
def test_pipeline_jitter_sleep_no_delay():
    config = ResolverConfig()
    config.sleep_jitter = 0.0
    pipeline = ResolverPipeline([], config, lambda *a, **k: None, ListLogger(), ResolverMetrics())

    with patch("DocsToKG.ContentDownload.resolvers._time.sleep") as mock_sleep:
        pipeline._jitter_sleep()

    mock_sleep.assert_not_called()


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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


# ---- test_resolver_pipeline.py -----------------------------
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
    monkeypatch.setattr(pipeline_module._time, "monotonic", fake.monotonic)
    monkeypatch.setattr(pipeline_module._time, "sleep", fake.sleep)

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
    assert 0.4 <= fake.sleeps[0] <= 0.45
    assert len(fake.sleeps) == 1


# ---- new jitter test -----------------------------
def test_domain_limit_includes_jitter_component(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(pipeline_module._time, "monotonic", fake.monotonic)
    monkeypatch.setattr(pipeline_module._time, "sleep", fake.sleep)
    monkeypatch.setattr(pipeline_module.random, "random", lambda: 0.5)

    config = ResolverConfig(resolver_order=[], resolver_toggles={})
    config.domain_min_interval_s = {"example.org": 0.1}
    logger = ListLogger()
    metrics = ResolverMetrics()
    pipeline = ResolverPipeline([], config, lambda *a, **k: None, logger, metrics)

    pipeline._last_host_hit["example.org"] = 0.0
    fake.now = 0.02
    pipeline._respect_domain_limit("https://example.org/resource")

    expected_wait = 0.1 - 0.02
    jitter = 0.5 * 0.05
    assert fake.sleeps == [pytest.approx(expected_wait + jitter)]


# ---- test_resolver_providers_additional.py -----------------------------
try:  # pragma: no cover - requests is an optional dependency in the test env
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = pytest.importorskip("requests")  # type: ignore

# ---- test_resolver_providers_additional.py -----------------------------
pytest.importorskip("bs4")


# ---- test_resolver_providers_additional.py -----------------------------
class _StubResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data

    def close(self) -> None:
        return None


# ---- test_resolver_providers_additional.py -----------------------------
def _artifact(tmp_path: Any, **overrides: Any) -> WorkArtifact:
    base_kwargs: Dict[str, Any] = dict(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi="10.1000/example",
        pmid="12345",
        pmcid="PMC123456",
        arxiv_id="arXiv:2101.00001",
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    base_kwargs.update(overrides)
    return WorkArtifact(**base_kwargs)


# ---- test_resolver_providers_additional.py -----------------------------
def test_arxiv_resolver_skips_missing_identifier(tmp_path) -> None:
    resolver = ArxivResolver()
    artifact = _artifact(tmp_path, arxiv_id=None)
    config = ResolverConfig()

    results = list(resolver.iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "no-arxiv-id"


# ---- test_resolver_providers_additional.py -----------------------------
def test_arxiv_resolver_strips_prefix(tmp_path) -> None:
    resolver = ArxivResolver()
    artifact = _artifact(tmp_path, arxiv_id="arXiv:2301.12345")
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.url.endswith("/2301.12345.pdf")


# ---- test_resolver_providers_additional.py -----------------------------
def test_openalex_resolver_skip(tmp_path) -> None:
    resolver = OpenAlexResolver()
    artifact = _artifact(tmp_path, pdf_urls=[], open_access_url=None)
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-openalex-urls"


# ---- test_resolver_providers_additional.py -----------------------------
def test_openalex_resolver_dedupes(tmp_path) -> None:
    resolver = OpenAlexResolver()
    artifact = _artifact(
        tmp_path,
        pdf_urls=["https://openalex.example/a.pdf", "https://openalex.example/a.pdf"],
        open_access_url="https://openalex.example/oa.pdf",
    )
    config = ResolverConfig()

    urls = [result.url for result in resolver.iter_urls(Mock(), config, artifact)]

    assert urls == ["https://openalex.example/a.pdf", "https://openalex.example/oa.pdf"]


# ---- test_resolver_providers_additional.py -----------------------------
def test_landing_page_resolver_meta_pattern(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.landing_urls = ["https://example.org/article"]
    config = ResolverConfig()

    html = """
    <html><head><meta name='citation_pdf_url' content='/files/paper.pdf'></head></html>
    """

    monkeypatch.setattr(
        providers_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(text=html),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.metadata["pattern"] == "meta"
    assert result.url.endswith("/files/paper.pdf")


# ---- test_resolver_providers_additional.py -----------------------------
def test_landing_page_resolver_link_pattern(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.landing_urls = ["https://example.org/article"]
    config = ResolverConfig()

    html = """
    <html><head>
    <link rel='alternate' type='application/pdf' href='/download/paper.pdf'>
    </head></html>
    """

    monkeypatch.setattr(
        providers_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(text=html),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.metadata["pattern"] == "link"


# ---- test_resolver_providers_additional.py -----------------------------
def test_landing_page_resolver_anchor_pattern(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.landing_urls = ["https://example.org/article"]
    config = ResolverConfig()

    html = """
    <html><body>
    <a href='/pdfs/paper.pdf'>Download PDF</a>
    </body></html>
    """

    monkeypatch.setattr(
        providers_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(text=html),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.metadata["pattern"] == "anchor"


# ---- test_resolver_providers_additional.py -----------------------------
def test_landing_page_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.landing_urls = ["https://example.org/article"]
    config = ResolverConfig()

    monkeypatch.setattr(
        providers_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=500),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_landing_page_resolver_request_errors(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    artifact.landing_urls = ["https://example.org/article"]
    config = ResolverConfig()

    monkeypatch.setattr(
        providers_module,
        "request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_core_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=503),
    )

    results = list(CoreResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "http-error"
    assert results[0].http_status == 503


# ---- test_resolver_providers_additional.py -----------------------------
def test_core_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad json"), text="oops"),
    )

    results = list(CoreResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_core_resolver_emits_results(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    core_payload = {
        "results": [
            {
                "downloadUrl": "https://core.example/primary.pdf",
                "fullTextLinks": [
                    {"url": "https://core.example/alternate.pdf"},
                    {"link": "https://core.example/alternate.pdf"},
                ],
            }
        ]
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=core_payload),
    )

    urls = [result.url for result in CoreResolver().iter_urls(Mock(), config, artifact)]

    assert urls[0] == "https://core.example/primary.pdf"
    assert "https://core.example/alternate.pdf" in urls


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_core_resolver_error_paths(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(CoreResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_core_resolver_skips_when_no_doi(tmp_path) -> None:
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()
    config.core_api_key = "token"

    result = next(CoreResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-doi"


# ---- test_resolver_providers_additional.py -----------------------------
def test_core_resolver_is_enabled_requires_key(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = None

    assert CoreResolver().is_enabled(config, artifact) is False


# ---- test_resolver_providers_additional.py -----------------------------
def test_core_resolver_ignores_non_dict_hits(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    payload = {"results": ["not-dict", {"downloadUrl": None, "fullTextLinks": ["oops"]}]}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = list(CoreResolver().iter_urls(Mock(), config, artifact))

    assert urls == []


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(return_value=_StubResponse(status_code=429)),
    )

    results = list(CrossrefResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "http-error"
    assert results[0].http_status == 429


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_success(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "message": {
            "link": [
                {"URL": "https://publisher.example/paper.pdf", "content-type": "application/pdf"},
                {"URL": "https://publisher.example/landing.html", "content-type": "text/html"},
                "not-a-dict",
            ]
        }
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(return_value=_StubResponse(json_data=payload)),
    )

    results = list(CrossrefResolver().iter_urls(Mock(), config, artifact))

    urls = [result.url for result in results]
    assert "https://publisher.example/paper.pdf" in urls
    assert "https://publisher.example/landing.html" in urls


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_link_not_list(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {"message": {"link": "not-a-list"}}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(return_value=_StubResponse(json_data=payload)),
    )

    results = list(CrossrefResolver().iter_urls(Mock(), config, artifact))

    assert results == []


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_skip_without_doi(tmp_path) -> None:
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()
    session = Mock()
    session.get = Mock()

    result = next(CrossrefResolver().iter_urls(session, config, artifact))

    assert result.event_reason == "no-doi"


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_is_enabled(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    resolver = CrossrefResolver()

    assert resolver.is_enabled(config, artifact) is True
    assert resolver.is_enabled(config, _artifact(tmp_path, doi=None)) is False


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_session_success(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.mailto = "user@example.org"

    response = _StubResponse(
        json_data={
            "message": {
                "link": [
                    {
                        "URL": "https://publisher.example/paper.pdf",
                        "content-type": "application/pdf",
                    },
                    {
                        "URL": "https://publisher.example/paper.pdf",
                        "content-type": "application/pdf",
                    },
                ]
            }
        }
    )
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        mock_request,
    )
    session = Mock()

    results = list(CrossrefResolver().iter_urls(session, config, artifact))

    mock_request.assert_called_once()
    called_session, method, endpoint = mock_request.call_args[0][:3]
    assert called_session is session
    assert method == "GET"
    assert endpoint.endswith("/10.1000/example")
    assert results[0].metadata["content_type"] == "application/pdf"
    assert len(results) == 1


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_cached_request_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=requests.RequestException("boom")),
    )

    result = next(CrossrefResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "request-error"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_crossref_resolver_session_errors(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    mock_request = Mock(side_effect=exception)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        mock_request,
    )
    session = Mock()
    session.get = Mock()

    result = next(CrossrefResolver().iter_urls(session, config, artifact))

    assert result.event_reason == reason
    mock_request.assert_called_once()


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_session_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    response = _StubResponse(status_code=504)
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        mock_request,
    )
    session = Mock()
    session.get = Mock()

    result = next(CrossrefResolver().iter_urls(session, config, artifact))

    assert result.event_reason == "http-error"
    mock_request.assert_called_once()


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_session_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    response = _StubResponse(json_data=ValueError("bad json"), text="oops")
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        mock_request,
    )
    session = Mock()
    session.get = Mock()

    result = next(CrossrefResolver().iter_urls(session, config, artifact))

    assert result.event_reason == "json-error"
    mock_request.assert_called_once()


# ---- test_resolver_providers_additional.py -----------------------------
def test_crossref_resolver_uses_central_retry_logic(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    class _Response:
        def __init__(self, status_code: int, headers=None, payload=None):
            self.status_code = status_code
            self.headers = headers or {}
            self._payload = payload

        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload or {"message": {"link": []}}

        def close(self):
            return None

        @property
        def text(self):
            return ""

    responses = [
        _Response(429, headers={"Retry-After": "1"}),
        _Response(429, headers={}),
        _Response(200, payload={"message": {"link": []}}),
    ]

    class _Session:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.get = Mock()

        def request(self, method: str, url: str, **kwargs):
            if not responses:
                raise AssertionError("unexpected extra request")
            response = responses.pop(0)
            self.calls.append(response.status_code)
            return response

    monkeypatch.setattr("DocsToKG.ContentDownload.network.random.random", lambda: 0.0)
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.time.sleep", lambda delay: sleep_calls.append(delay)
    )

    session = _Session()
    results = list(CrossrefResolver().iter_urls(session, config, artifact))

    assert results == []
    assert session.calls == [429, 429, 200]
    assert sleep_calls[0] == pytest.approx(1.0, abs=0.05)
    assert sleep_calls[1] == pytest.approx(1.5, abs=0.05)


# ---- test_resolver_providers_additional.py -----------------------------
def test_doaj_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=502),
    )

    result = next(DoajResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_doaj_resolver_emits_candidate(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "results": [
            {
                "bibjson": {
                    "link": [
                        {"url": "https://doaj.example/file.pdf"},
                        {"url": "https://doaj.example/file.pdf"},
                        {"url": "https://doaj.example/file.txt"},
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in DoajResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://doaj.example/file.pdf"]


# ---- test_resolver_providers_additional.py -----------------------------
def test_doaj_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("broken"), text="oops"),
    )

    result = next(DoajResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_doaj_resolver_error_paths(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(DoajResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_doaj_resolver_includes_api_key(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.doaj_api_key = "secret"

    captured_headers = {}

    def _fake_request(session, method, url, **kwargs):
        nonlocal captured_headers
        captured_headers = kwargs.get("headers", {})
        return _StubResponse(json_data={"results": []})

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        _fake_request,
    )

    list(DoajResolver().iter_urls(Mock(), config, artifact))

    assert captured_headers.get("X-API-KEY") == "secret"


# ---- test_resolver_providers_additional.py -----------------------------
def test_doaj_resolver_skip_no_doi(tmp_path) -> None:
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()

    result = next(DoajResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-doi"


# ---- test_resolver_providers_additional.py -----------------------------
def test_europe_pmc_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=502),
    )

    results = list(EuropePmcResolver().iter_urls(Mock(), config, artifact))

    assert results == []


# ---- test_resolver_providers_additional.py -----------------------------
def test_europe_pmc_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("oops"), text="bad"),
    )

    result = next(EuropePmcResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_europe_pmc_resolver_emits_pdf(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "resultList": {
            "result": [
                {
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {"documentStyle": "pdf", "url": "https://epmc.example/file.pdf"},
                            {"documentStyle": "html", "url": "https://epmc.example/file.html"},
                        ]
                    }
                }
            ]
        }
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in EuropePmcResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://epmc.example/file.pdf"]


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception",
    [requests.Timeout("slow"), requests.ConnectionError("down"), requests.RequestException("boom")],
)
def test_europe_pmc_resolver_error_paths(monkeypatch, tmp_path, exception) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    results = list(EuropePmcResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason in {"timeout", "connection-error", "request-error"}


# ---- test_resolver_providers_additional.py -----------------------------
def test_hal_resolver_emits_urls(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "response": {
            "docs": [
                {
                    "fileMain_s": "https://hal.example/main.pdf",
                    "file_s": [
                        "https://hal.example/supp.pdf",
                        "https://hal.example/notes.txt",
                    ],
                }
            ]
        }
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in HalResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://hal.example/main.pdf", "https://hal.example/supp.pdf"]


# ---- test_resolver_providers_additional.py -----------------------------
def test_hal_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(HalResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_hal_resolver_error_paths(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(HalResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_hal_resolver_is_enabled(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    resolver = HalResolver()

    assert resolver.is_enabled(config, artifact) is True
    assert resolver.is_enabled(config, _artifact(tmp_path, doi=None)) is False


# ---- test_resolver_providers_additional.py -----------------------------
def test_hal_resolver_skip_no_doi(tmp_path) -> None:
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()

    result = next(HalResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-doi"


# ---- test_resolver_providers_additional.py -----------------------------
def test_openaire_resolver_emits_pdf(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    complex_payload = {
        "response": {
            "results": {
                "result": [
                    {
                        "metadata": {
                            "instance": {
                                "url": "https://openaire.example/paper.pdf",
                                "extra": ["https://openaire.example/paper.pdf"],
                            }
                        }
                    }
                ]
            }
        }
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=complex_payload),
    )

    urls = [result.url for result in OpenAireResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://openaire.example/paper.pdf"]


# ---- test_resolver_providers_additional.py -----------------------------
def test_openaire_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(text="{", json_data=ValueError("bad")),
    )

    result = next(OpenAireResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_openaire_resolver_fallback_json_load(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "response": {
            "results": {
                "result": [
                    {
                        "metadata": {
                            "instance": {
                                "url": "https://openaire.example/alt.pdf",
                            }
                        }
                    }
                ]
            }
        }
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(
            json_data=ValueError("bad"), text=json.dumps(payload)
        ),
    )

    urls = [result.url for result in OpenAireResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://openaire.example/alt.pdf"]


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_openaire_resolver_error_paths(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(OpenAireResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_osf_resolver_emits_urls(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "data": [
            {
                "links": {"download": "https://osf.example/direct.pdf"},
                "attributes": {
                    "primary_file": {"links": {"download": "https://osf.example/primary.pdf"}}
                },
            }
        ]
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in OsfResolver().iter_urls(Mock(), config, artifact)]

    assert urls == [
        "https://osf.example/direct.pdf",
        "https://osf.example/primary.pdf",
    ]


# ---- test_resolver_providers_additional.py -----------------------------
def test_osf_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(OsfResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_osf_resolver_error_paths(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(OsfResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_osf_resolver_skip_no_doi(tmp_path) -> None:
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()

    result = next(OsfResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-doi"


# ---- test_resolver_providers_additional.py -----------------------------
def test_unpaywall_resolver_cached_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"

    http_error = requests.HTTPError("boom")
    http_error.response = Mock(status_code=404)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers._fetch_unpaywall_data",
        Mock(side_effect=http_error),
    )

    class _Session:
        pass

    results = list(UnpaywallResolver().iter_urls(_Session(), config, artifact))

    assert results[0].event_reason == "http-error"
    assert results[0].http_status == 404


# ---- test_resolver_providers_additional.py -----------------------------
def test_unpaywall_resolver_cached_success(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"

    payload = {
        "best_oa_location": {"url_for_pdf": "https://unpaywall.example/best.pdf"},
        "oa_locations": [
            {"url_for_pdf": "https://unpaywall.example/extra.pdf"},
            {"url_for_pdf": "https://unpaywall.example/best.pdf"},
            "not-a-dict",
        ],
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers._fetch_unpaywall_data",
        Mock(return_value=payload),
    )

    class _Session:
        pass

    urls = [result.url for result in UnpaywallResolver().iter_urls(_Session(), config, artifact)]

    assert urls == [
        "https://unpaywall.example/best.pdf",
        "https://unpaywall.example/extra.pdf",
    ]


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_unpaywall_resolver_session_errors(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"

    session = Mock()
    session.get = Mock(side_effect=exception)

    result = next(UnpaywallResolver().iter_urls(session, config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_unpaywall_resolver_session_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"

    session = Mock()
    session.get = Mock(return_value=_StubResponse(json_data=ValueError("bad"), text="oops"))

    result = next(UnpaywallResolver().iter_urls(session, config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_unpaywall_resolver_is_enabled(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"
    resolver = UnpaywallResolver()

    assert resolver.is_enabled(config, artifact) is True
    assert resolver.is_enabled(config, _artifact(tmp_path, doi=None)) is False
    config.unpaywall_email = None
    assert resolver.is_enabled(config, artifact) is False


# ---- test_resolver_providers_additional.py -----------------------------
def test_unpaywall_resolver_session_success(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"

    payload = {
        "best_oa_location": {"url_for_pdf": "https://unpaywall.example/best.pdf"},
        "oa_locations": [],
    }

    session = Mock()
    session.get = Mock(return_value=_StubResponse(json_data=payload))

    results = list(UnpaywallResolver().iter_urls(session, config, artifact))

    session.get.assert_called_once()
    assert results[0].url == "https://unpaywall.example/best.pdf"


# ---- test_resolver_providers_additional.py -----------------------------
def test_semantic_scholar_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    http_error = requests.HTTPError("boom")
    http_error.response = Mock(status_code=503)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers._fetch_semantic_scholar_data",
        Mock(side_effect=http_error),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == "http-error"
    assert result.http_status == 503


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_semantic_scholar_resolver_errors(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers._fetch_semantic_scholar_data",
        Mock(side_effect=exception),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_semantic_scholar_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers._fetch_semantic_scholar_data",
        Mock(side_effect=ValueError("bad")),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_semantic_scholar_resolver_no_open_access(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers._fetch_semantic_scholar_data",
        Mock(return_value={"openAccessPdf": {}}),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == "no-openaccess-pdf"


# ---- test_resolver_providers_additional.py -----------------------------
def test_pmc_resolver_no_identifiers(tmp_path) -> None:
    resolver = PmcResolver()
    artifact = _artifact(tmp_path, pmcid=None, pmid=None, doi=None)
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-pmcid"


# ---- test_resolver_providers_additional.py -----------------------------
def test_pmc_resolver_timeout_fallback(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=requests.Timeout("slow")),
    )

    results = list(PmcResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "timeout"
    assert results[1].metadata["source"] == "pdf-fallback"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_pmc_resolver_other_errors(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path, pmcid="PMC123456")
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=[exception, _StubResponse(text="")]),
    )

    results = list(PmcResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_pmc_resolver_success(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path, pmcid="PMC123456")
    config = ResolverConfig()

    # OA response containing relative hrefs should resolve to absolute URLs
    oa_html = '<a href="/pmc/articles/PMC123456/pdf/123.pdf">Download</a>'
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(return_value=_StubResponse(text=oa_html)),
    )

    urls = [result.url for result in PmcResolver().iter_urls(Mock(), config, artifact)]

    assert "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/123.pdf" in urls
    assert urls[-1] == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"


# ---- test_resolver_providers_additional.py -----------------------------
def test_pmc_lookup_pmcids_success(monkeypatch, tmp_path) -> None:
    resolver = PmcResolver()
    config = ResolverConfig()

    payload = {
        "records": [
            {"pmcid": "PMC123"},
            {"pmcid": "pmc456"},
            {"pmcid": None},
        ]
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    ids = resolver._lookup_pmcids(Mock(), ["10.1000/example"], config)

    assert ids == ["PMC123", "PMC456"]


# ---- test_resolver_providers_additional.py -----------------------------
def test_pmc_lookup_pmcids_handles_json_error(monkeypatch, tmp_path) -> None:
    resolver = PmcResolver()
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    ids = resolver._lookup_pmcids(Mock(), ["10.1000/example"], config)

    assert ids == []


# ---- test_resolver_providers_additional.py -----------------------------
def test_wayback_resolver_handles_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.failed_pdf_urls = ["https://example.org/pdf"]
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=500),
    )

    result = next(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_wayback_resolver_returns_archive(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.failed_pdf_urls = ["https://example.org/pdf"]
    config = ResolverConfig()

    payload = {
        "archived_snapshots": {
            "closest": {
                "available": True,
                "url": "https://web.archive.org/web/20200101/https://example.org/pdf",
                "timestamp": "20200101000000",
            }
        }
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    results = list(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert results[0].url.startswith("https://web.archive.org/")
    assert results[0].metadata["timestamp"] == "20200101000000"


# ---- test_resolver_providers_additional.py -----------------------------
def test_wayback_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.failed_pdf_urls = ["https://example.org/pdf"]
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.ConnectionError("down"), "connection-error"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_wayback_resolver_error_paths(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    artifact.failed_pdf_urls = ["https://example.org/pdf"]
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_wayback_resolver_no_snapshot(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.failed_pdf_urls = ["https://example.org/pdf"]
    config = ResolverConfig()

    payload = {"archived_snapshots": {}}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    results = list(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert results == []


# ---- test_resolver_providers_additional.py -----------------------------
def test_zenodo_resolver_no_doi(tmp_path) -> None:
    resolver = ZenodoResolver()
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-doi"


# ---- test_resolver_providers_additional.py -----------------------------
@pytest.mark.parametrize(
    "exception,reason",
    [
        (requests.Timeout("slow"), "timeout"),
        (requests.RequestException("boom"), "request-error"),
    ],
)
def test_zenodo_resolver_errors(monkeypatch, tmp_path, exception, reason) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(ZenodoResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# ---- test_resolver_providers_additional.py -----------------------------
def test_zenodo_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=502),
    )

    result = next(ZenodoResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_zenodo_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(ZenodoResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


# ---- test_resolver_providers_additional.py -----------------------------
def test_zenodo_resolver_emits_urls(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "hits": {
            "hits": [
                {
                    "id": "123",
                    "files": [
                        {
                            "type": "pdf",
                            "key": "paper.pdf",
                            "links": {"self": "https://zenodo.example/paper.pdf"},
                        },
                        {
                            "type": "txt",
                            "key": "paper.txt",
                            "links": {"self": "https://zenodo.example/paper.txt"},
                        },
                    ],
                }
            ]
        }
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in ZenodoResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://zenodo.example/paper.pdf"]


# ---- test_resolver_providers_additional.py -----------------------------
def test_zenodo_resolver_is_enabled(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    resolver = ZenodoResolver()
    config = ResolverConfig()

    assert resolver.is_enabled(config, artifact) is True
    assert resolver.is_enabled(config, _artifact(tmp_path, doi=None)) is False


# ---- test_resolver_providers_additional.py -----------------------------
def test_zenodo_resolver_malformed_hits(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data={"hits": "unexpected"}),
    )

    results = list(ZenodoResolver().iter_urls(Mock(), config, artifact))

    assert results == []


# ---- test_resolvers_namespace.py -----------------------------
def test_legacy_time_alias_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        module_time = resolvers.time  # type: ignore[attr-defined]

    assert module_time is importlib.import_module("time")
    assert any(w.category is DeprecationWarning for w in captured)


# ---- test_resolvers_namespace.py -----------------------------
def test_reloading_resolvers_preserves_deprecation_behaviour():
    module = importlib.reload(resolvers)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        module.requests  # type: ignore[attr-defined]

    assert any(w.category is DeprecationWarning for w in captured)


# ---- test_resolvers_unit.py -----------------------------
pytest.importorskip("pyalex")

# ---- test_resolvers_unit.py -----------------------------
requests = pytest.importorskip("requests")

# ---- test_resolvers_unit.py -----------------------------
responses = pytest.importorskip("responses")


# ---- test_resolvers_unit.py -----------------------------
def make_artifact(tmp_path: Path, **overrides: object) -> downloader.WorkArtifact:
    base_kwargs = dict(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi="10.1000/example",
        pmid="12345",
        pmcid="PMC67890",
        arxiv_id="2101.00001",
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    base_kwargs.update(overrides)
    return downloader.WorkArtifact(**base_kwargs)


# ---- test_resolvers_unit.py -----------------------------
def build_config(**overrides: object) -> ResolverConfig:
    config = ResolverConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_unpaywall_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.unpaywall.org/v2/10.1000/example",
        json={
            "best_oa_location": {"url_for_pdf": "https://oa.example/best.pdf"},
            "oa_locations": [{"url_for_pdf": "https://oa.example/extra.pdf"}],
        },
        status=200,
    )

    urls = [
        result.url
        for result in UnpaywallResolver().iter_urls(session, config, artifact)
        if not result.is_event
    ]
    assert urls == ["https://oa.example/best.pdf", "https://oa.example/extra.pdf"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_unpaywall_resolver_http_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.unpaywall.org/v2/10.1000/example",
        status=503,
    )
    results = list(UnpaywallResolver().iter_urls(session, config, artifact))
    assert results[0].event == "error"
    assert results[0].event_reason == "http-error"


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_crossref_resolver_includes_mailto(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org", mailto="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.1000/example",
        json={
            "message": {
                "link": [
                    {
                        "URL": "https://publisher.example/file.pdf",
                        "content-type": "application/pdf",
                    }
                ]
            }
        },
        status=200,
    )

    urls = [r.url for r in CrossrefResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://publisher.example/file.pdf"]
    assert "mailto=tester%40example.org" in responses.calls[0].request.url


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_crossref_resolver_handles_json_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://api.crossref.org/works/10.1000/example",
        body="not-json",
        status=200,
    )
    results = list(CrossrefResolver().iter_urls(session, config, artifact))
    assert results[0].event == "error"
    assert results[0].event_reason == "json-error"


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_landing_page_resolver_patterns(tmp_path):
    pytest.importorskip("bs4")
    artifact = make_artifact(tmp_path, landing_urls=["https://site.example/article"])
    config = build_config()
    html = """
    <html><head>
    <meta name="citation_pdf_url" content="/files/paper.pdf">
    </head></html>
    """
    responses.add(responses.GET, "https://site.example/article", body=html, status=200)
    session = requests.Session()
    results = [
        r for r in LandingPageResolver().iter_urls(session, config, artifact) if not r.is_event
    ]
    assert results[0].url == "https://site.example/files/paper.pdf"

    html_anchor = """
    <html><body><a href="/download/paper.pdf">Get PDF</a></body></html>
    """
    responses.add(responses.GET, "https://site.example/anchor", body=html_anchor, status=200)
    artifact.landing_urls = ["https://site.example/anchor"]
    results = [
        r for r in LandingPageResolver().iter_urls(session, config, artifact) if not r.is_event
    ]
    assert results[0].metadata["pattern"] == "anchor"


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_landing_page_resolver_http_error(tmp_path):  # noqa: F811
    pytest.importorskip("bs4")
    artifact = make_artifact(tmp_path, landing_urls=["https://site.example/error"])
    config = build_config()
    responses.add(responses.GET, "https://site.example/error", status=500)
    session = requests.Session()
    events = [r for r in LandingPageResolver().iter_urls(session, config, artifact) if r.is_event]
    assert events[0].event_reason == "http-error"


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_pmc_resolver_uses_id_converter(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, pmcid=None)
    config = build_config(unpaywall_email="tester@example.org")
    responses.add(
        responses.GET,
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
        match=[
            responses.matchers.query_param_matcher(
                {
                    "ids": "10.1000/example",
                    "format": "json",
                    "tool": "docs-to-kg",
                    "email": "tester@example.org",
                }
            )
        ],
        json={"records": [{"pmcid": "PMC777"}]},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC777",
        body='href="/articles/PMC777/pdf/foo.pdf"',
        status=200,
    )
    results = [r.url for r in PmcResolver().iter_urls(session, config, artifact)]
    assert "https://www.ncbi.nlm.nih.gov/articles/PMC777/pdf/foo.pdf" in results
    assert "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC777/pdf/" in results


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_pmc_resolver_handles_request_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, pmcid="PMC123")
    config = build_config()
    responses.add(
        responses.GET,
        "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC123",
        body=responses.ConnectionError("boom"),
    )
    results = list(PmcResolver().iter_urls(session, config, artifact))
    assert results[-1].url.endswith("PMC123/pdf/")


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_europe_pmc_resolver_filters_pdf(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        json={
            "resultList": {
                "result": [
                    {
                        "fullTextUrlList": {
                            "fullTextUrl": [
                                {"documentStyle": "pdf", "url": "https://epmc.org/pdf1"},
                                {"documentStyle": "html", "url": "https://epmc.org/html"},
                            ]
                        }
                    }
                ]
            }
        },
        status=200,
    )
    urls = [r.url for r in EuropePmcResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://epmc.org/pdf1"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_europe_pmc_resolver_http_error(tmp_path):  # noqa: F811
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        status=500,
    )
    urls = list(EuropePmcResolver().iter_urls(session, config, artifact))
    assert urls == []


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_openaire_resolver_collects_pdf_candidates(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.openaire.eu/search/publications",
        json={
            "response": {
                "results": {
                    "result": [
                        {
                            "metadata": {
                                "instance": {
                                    "url": "https://openaire.example/paper.pdf",
                                    "extra": [
                                        "https://openaire.example/ignore.txt",
                                        "https://openaire.example/paper.pdf",
                                    ],
                                }
                            }
                        }
                    ]
                }
            }
        },
        status=200,
    )

    urls = [r.url for r in OpenAireResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://openaire.example/paper.pdf"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_hal_resolver_uses_file_fields(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.archives-ouvertes.fr/search/",
        json={
            "response": {
                "docs": [
                    {
                        "fileMain_s": "https://hal.archives-ouvertes.fr/fileMain.pdf",
                        "file_s": [
                            "https://hal.archives-ouvertes.fr/alternate.pdf",
                            "https://hal.archives-ouvertes.fr/fileMain.pdf",
                        ],
                    }
                ]
            }
        },
        status=200,
    )

    urls = [r.url for r in HalResolver().iter_urls(session, config, artifact)]
    assert urls == [
        "https://hal.archives-ouvertes.fr/fileMain.pdf",
        "https://hal.archives-ouvertes.fr/alternate.pdf",
    ]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_osf_resolver_merges_download_links(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.osf.io/v2/preprints/",
        json={
            "data": [
                {
                    "links": {"download": "https://osf.io/download1"},
                    "attributes": {
                        "primary_file": {"links": {"download": "https://osf.io/download2"}}
                    },
                }
            ]
        },
        status=200,
    )

    urls = [r.url for r in OsfResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://osf.io/download1", "https://osf.io/download2"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_core_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(core_api_key="abc123")
    responses.add(
        responses.GET,
        "https://api.core.ac.uk/v3/search/works",
        json={
            "results": [
                {"downloadUrl": "https://core.org/paper.pdf"},
                {"fullTextLinks": [{"url": "https://core.org/extra.pdf"}]},
            ]
        },
        status=200,
    )
    urls = [r.url for r in CoreResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://core.org/paper.pdf", "https://core.org/extra.pdf"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_core_resolver_handles_failure(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config(core_api_key="abc123")
    responses.add(responses.GET, "https://api.core.ac.uk/v3/search/works", status=500)
    events = [
        result for result in CoreResolver().iter_urls(session, config, artifact) if result.is_event
    ]
    assert events[0].event_reason == "http-error"
    assert events[0].http_status == 500
    assert "CORE API returned" in events[0].metadata["error_detail"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_doaj_resolver_filters_pdf(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://doaj.org/api/v2/search/articles/",
        json={
            "results": [
                {
                    "bibjson": {
                        "link": [
                            {"type": "fulltext", "url": "https://doaj.org/paper.pdf"},
                            {"type": "landing", "url": "https://doaj.org/landing"},
                        ]
                    }
                }
            ]
        },
        status=200,
    )
    urls = [r.url for r in DoajResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://doaj.org/paper.pdf"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_doaj_resolver_handles_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(responses.GET, "https://doaj.org/api/v2/search/articles/", status=429)
    events = [
        result for result in DoajResolver().iter_urls(session, config, artifact) if result.is_event
    ]
    assert events[0].event_reason == "http-error"
    assert events[0].http_status == 429
    assert "DOAJ API returned" in events[0].metadata["error_detail"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_semantic_scholar_resolver_handles_error(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1000/example",
        status=500,
    )
    results = list(SemanticScholarResolver().iter_urls(session, config, artifact))
    error = next(result for result in results if result.is_event)
    assert error.event_reason == "http-error"
    assert error.http_status == 500
    assert "Semantic Scholar HTTPError" in error.metadata["error_detail"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_semantic_scholar_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path)
    config = build_config()
    responses.add(
        responses.GET,
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1000/example",
        json={"openAccessPdf": {"url": "https://s2.org/paper.pdf"}},
        status=200,
    )
    urls = [r.url for r in SemanticScholarResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://s2.org/paper.pdf"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_wayback_resolver_success(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, failed_pdf_urls=["https://dead.example/file.pdf"])
    config = build_config()
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "url": "https://web.archive.org/web/20200101/https://dead.example/file.pdf",
                    "timestamp": "20200101000000",
                }
            }
        },
        status=200,
    )
    urls = [r.url for r in WaybackResolver().iter_urls(session, config, artifact)]
    assert urls == ["https://web.archive.org/web/20200101/https://dead.example/file.pdf"]


# ---- test_resolvers_unit.py -----------------------------
@responses.activate
def test_wayback_resolver_handles_missing_snapshot(tmp_path):
    session = requests.Session()
    artifact = make_artifact(tmp_path, failed_pdf_urls=["https://dead.example/file.pdf"])
    config = build_config()
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={"archived_snapshots": {"closest": {"available": False}}},
        status=200,
    )
    urls = list(WaybackResolver().iter_urls(session, config, artifact))
    assert urls == []
