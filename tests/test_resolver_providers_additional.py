"""Additional resolver provider coverage tests.

These tests focus on execution paths that were previously uncovered in the
resolver provider implementations (fallback code paths, error handling, and
deduplication helpers). They are intentionally lightweight and rely purely on
mocked responses so they remain fast and deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

try:  # pragma: no cover - requests is an optional dependency in the test env
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = pytest.importorskip("requests")  # type: ignore

pytest.importorskip("bs4")

from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact
from DocsToKG.ContentDownload.resolvers.providers.arxiv import ArxivResolver
from DocsToKG.ContentDownload.resolvers.providers.core import CoreResolver
from DocsToKG.ContentDownload.resolvers.providers.crossref import CrossrefResolver
from DocsToKG.ContentDownload.resolvers.providers.doaj import DoajResolver
from DocsToKG.ContentDownload.resolvers.providers.europe_pmc import EuropePmcResolver
from DocsToKG.ContentDownload.resolvers.providers.pmc import PmcResolver
from DocsToKG.ContentDownload.resolvers.providers.unpaywall import UnpaywallResolver
from DocsToKG.ContentDownload.resolvers.providers.wayback import WaybackResolver
from DocsToKG.ContentDownload.resolvers.providers.zenodo import ZenodoResolver
from DocsToKG.ContentDownload.resolvers.providers.semantic_scholar import (
    SemanticScholarResolver,
    _fetch_semantic_scholar_data,
)
from DocsToKG.ContentDownload.resolvers.providers.hal import HalResolver
from DocsToKG.ContentDownload.resolvers.providers.openaire import OpenAireResolver
from DocsToKG.ContentDownload.resolvers.providers.osf import OsfResolver
from DocsToKG.ContentDownload.resolvers.providers.openalex import OpenAlexResolver
from DocsToKG.ContentDownload.resolvers.providers.landing_page import LandingPageResolver
from DocsToKG.ContentDownload.resolvers.providers import landing_page as landing_page_module
from DocsToKG.ContentDownload.resolvers.types import ResolverConfig


class _StubResponse:
    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self) -> Any:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


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


# --- ArxivResolver ------------------------------------------------------------------------


def test_arxiv_resolver_skips_missing_identifier(tmp_path) -> None:
    resolver = ArxivResolver()
    artifact = _artifact(tmp_path, arxiv_id=None)
    config = ResolverConfig()

    results = list(resolver.iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "no-arxiv-id"


def test_arxiv_resolver_strips_prefix(tmp_path) -> None:
    resolver = ArxivResolver()
    artifact = _artifact(tmp_path, arxiv_id="arXiv:2301.12345")
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.url.endswith("/2301.12345.pdf")


def test_openalex_resolver_skip(tmp_path) -> None:
    resolver = OpenAlexResolver()
    artifact = _artifact(tmp_path, pdf_urls=[], open_access_url=None)
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-openalex-urls"


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


def test_landing_page_resolver_meta_pattern(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.landing_urls = ["https://example.org/article"]
    config = ResolverConfig()

    html = """
    <html><head><meta name='citation_pdf_url' content='/files/paper.pdf'></head></html>
    """

    monkeypatch.setattr(
        landing_page_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(text=html),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.metadata["pattern"] == "meta"
    assert result.url.endswith("/files/paper.pdf")


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
        landing_page_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(text=html),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.metadata["pattern"] == "link"


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
        landing_page_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(text=html),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.metadata["pattern"] == "anchor"


def test_landing_page_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.landing_urls = ["https://example.org/article"]
    config = ResolverConfig()

    monkeypatch.setattr(
        landing_page_module,
        "request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=500),
    )

    result = next(LandingPageResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


# --- CoreResolver -------------------------------------------------------------------------


def test_core_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.core.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=503),
    )

    results = list(CoreResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "http-error"
    assert results[0].http_status == 503


def test_core_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.core.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad json"), text="oops"),
    )

    results = list(CoreResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.core.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=core_payload),
    )

    urls = [result.url for result in CoreResolver().iter_urls(Mock(), config, artifact)]

    assert urls[0] == "https://core.example/primary.pdf"
    assert "https://core.example/alternate.pdf" in urls


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
        "DocsToKG.ContentDownload.resolvers.providers.core.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(CoreResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


def test_core_resolver_skips_when_no_doi(tmp_path) -> None:
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()
    config.core_api_key = "token"

    result = next(CoreResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-doi"


def test_core_resolver_is_enabled_requires_key(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = None

    assert CoreResolver().is_enabled(config, artifact) is False


def test_core_resolver_ignores_non_dict_hits(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.core_api_key = "token"

    payload = {"results": ["not-dict", {"downloadUrl": None, "fullTextLinks": ["oops"]}]}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.core.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = list(CoreResolver().iter_urls(Mock(), config, artifact))

    assert urls == []


# --- CrossrefResolver ---------------------------------------------------------------------


def test_crossref_resolver_cached_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    http_error = requests.HTTPError("boom")
    http_error.response = Mock(status_code=429)

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.crossref._fetch_crossref_data",
        Mock(side_effect=http_error),
    )

    class _Session:  # lacks .get to force cached path
        pass

    results = list(CrossrefResolver().iter_urls(_Session(), config, artifact))

    assert results[0].event_reason == "http-error"
    assert results[0].http_status == 429


def test_crossref_resolver_cached_success(monkeypatch, tmp_path) -> None:
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
        "DocsToKG.ContentDownload.resolvers.providers.crossref._fetch_crossref_data",
        Mock(return_value=payload),
    )

    class _Session:
        pass

    results = list(CrossrefResolver().iter_urls(_Session(), config, artifact))

    urls = [result.url for result in results]
    assert "https://publisher.example/paper.pdf" in urls
    assert "https://publisher.example/landing.html" in urls


def test_crossref_resolver_link_not_list(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {"message": {"link": "not-a-list"}}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.crossref._fetch_crossref_data",
        Mock(return_value=payload),
    )

    class _Session:
        pass

    results = list(CrossrefResolver().iter_urls(_Session(), config, artifact))

    assert results == []


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

    session = Mock()
    session.get = Mock(side_effect=exception)

    result = next(CrossrefResolver().iter_urls(session, config, artifact))

    assert result.event_reason == reason


def test_crossref_resolver_session_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    response = _StubResponse(status_code=504)
    session = Mock()
    session.get = Mock(return_value=response)

    result = next(CrossrefResolver().iter_urls(session, config, artifact))

    assert result.event_reason == "http-error"


def test_crossref_resolver_session_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    response = _StubResponse(json_data=ValueError("bad json"), text="oops")
    session = Mock()
    session.get = Mock(return_value=response)

    result = next(CrossrefResolver().iter_urls(session, config, artifact))

    assert result.event_reason == "json-error"


# --- DoajResolver ------------------------------------------------------------------------


def test_doaj_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.doaj.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=502),
    )

    result = next(DoajResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.doaj.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in DoajResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://doaj.example/file.pdf"]


def test_doaj_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.doaj.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("broken"), text="oops"),
    )

    result = next(DoajResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.doaj.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(DoajResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


def test_europe_pmc_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.europe_pmc.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=502),
    )

    results = list(EuropePmcResolver().iter_urls(Mock(), config, artifact))

    assert results == []


def test_europe_pmc_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.europe_pmc.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("oops"), text="bad"),
    )

    result = next(EuropePmcResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.europe_pmc.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in EuropePmcResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://epmc.example/file.pdf"]


@pytest.mark.parametrize(
    "exception",
    [requests.Timeout("slow"), requests.ConnectionError("down"), requests.RequestException("boom")],
)
def test_europe_pmc_resolver_error_paths(monkeypatch, tmp_path, exception) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.europe_pmc.request_with_retries",
        Mock(side_effect=exception),
    )

    results = list(EuropePmcResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason in {"timeout", "connection-error", "request-error"}


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
        "DocsToKG.ContentDownload.resolvers.providers.hal.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in HalResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://hal.example/main.pdf", "https://hal.example/supp.pdf"]


def test_hal_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.hal.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(HalResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.hal.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(HalResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


def test_openaire_resolver_emits_pdf(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {"result": {"items": ["ignored"]}}
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
        "DocsToKG.ContentDownload.resolvers.providers.openaire.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=complex_payload),
    )

    urls = [result.url for result in OpenAireResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://openaire.example/paper.pdf"]


def test_openaire_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.openaire.request_with_retries",
        lambda *args, **kwargs: _StubResponse(text="{", json_data=ValueError("bad")),
    )

    result = next(OpenAireResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.openaire.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(OpenAireResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


def test_osf_resolver_emits_urls(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    payload = {
        "data": [
            {
                "links": {"download": "https://osf.example/direct.pdf"},
                "attributes": {
                    "primary_file": {
                        "links": {"download": "https://osf.example/primary.pdf"}
                    }
                },
            }
        ]
    }

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.osf.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in OsfResolver().iter_urls(Mock(), config, artifact)]

    assert urls == [
        "https://osf.example/direct.pdf",
        "https://osf.example/primary.pdf",
    ]


def test_osf_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.osf.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(OsfResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.osf.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(OsfResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


# --- UnpaywallResolver -------------------------------------------------------------------


def test_unpaywall_resolver_cached_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"

    http_error = requests.HTTPError("boom")
    http_error.response = Mock(status_code=404)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.unpaywall._fetch_unpaywall_data",
        Mock(side_effect=http_error),
    )

    class _Session:
        pass

    results = list(UnpaywallResolver().iter_urls(_Session(), config, artifact))

    assert results[0].event_reason == "http-error"
    assert results[0].http_status == 404


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
        "DocsToKG.ContentDownload.resolvers.providers.unpaywall._fetch_unpaywall_data",
        Mock(return_value=payload),
    )

    class _Session:
        pass

    urls = [result.url for result in UnpaywallResolver().iter_urls(_Session(), config, artifact)]

    assert urls == [
        "https://unpaywall.example/best.pdf",
        "https://unpaywall.example/extra.pdf",
    ]


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


def test_unpaywall_resolver_session_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()
    config.unpaywall_email = "user@example.org"

    session = Mock()
    session.get = Mock(return_value=_StubResponse(json_data=ValueError("bad"), text="oops"))

    result = next(UnpaywallResolver().iter_urls(session, config, artifact))

    assert result.event_reason == "json-error"


def test_semantic_scholar_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    http_error = requests.HTTPError("boom")
    http_error.response = Mock(status_code=503)
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.semantic_scholar._fetch_semantic_scholar_data",
        Mock(side_effect=http_error),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == "http-error"
    assert result.http_status == 503


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
        "DocsToKG.ContentDownload.resolvers.providers.semantic_scholar._fetch_semantic_scholar_data",
        Mock(side_effect=exception),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == reason


def test_semantic_scholar_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.semantic_scholar._fetch_semantic_scholar_data",
        Mock(side_effect=ValueError("bad")),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == "json-error"


def test_semantic_scholar_resolver_no_open_access(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.semantic_scholar._fetch_semantic_scholar_data",
        Mock(return_value={"openAccessPdf": {}}),
    )

    class _Session:
        pass

    result = next(SemanticScholarResolver().iter_urls(_Session(), config, artifact))

    assert result.event_reason == "no-openaccess-pdf"


# --- PmcResolver -------------------------------------------------------------------------


def test_pmc_resolver_no_identifiers(tmp_path) -> None:
    resolver = PmcResolver()
    artifact = _artifact(tmp_path, pmcid=None, pmid=None, doi=None)
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-pmcid"


def test_pmc_resolver_timeout_fallback(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.pmc.request_with_retries",
        Mock(side_effect=requests.Timeout("slow")),
    )

    results = list(PmcResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == "timeout"
    assert results[1].metadata["source"] == "pdf-fallback"


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
        "DocsToKG.ContentDownload.resolvers.providers.pmc.request_with_retries",
        Mock(side_effect=[exception, _StubResponse(text="")]),
    )

    results = list(PmcResolver().iter_urls(Mock(), config, artifact))

    assert results[0].event_reason == reason


def test_pmc_resolver_success(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path, pmcid="PMC123456")
    config = ResolverConfig()

    # OA response containing relative hrefs should resolve to absolute URLs
    oa_html = '<a href="/pmc/articles/PMC123456/pdf/123.pdf">Download</a>'
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.pmc.request_with_retries",
        Mock(return_value=_StubResponse(text=oa_html)),
    )

    urls = [result.url for result in PmcResolver().iter_urls(Mock(), config, artifact)]

    assert "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/123.pdf" in urls
    assert urls[-1] == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"


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
        "DocsToKG.ContentDownload.resolvers.providers.pmc.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    ids = resolver._lookup_pmcids(Mock(), ["10.1000/example"], config)

    assert ids == ["PMC123", "PMC456"]


def test_pmc_lookup_pmcids_handles_json_error(monkeypatch, tmp_path) -> None:
    resolver = PmcResolver()
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.pmc.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    ids = resolver._lookup_pmcids(Mock(), ["10.1000/example"], config)

    assert ids == []


# --- WaybackResolver ---------------------------------------------------------------------


def test_wayback_resolver_handles_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.failed_pdf_urls = ["https://example.org/pdf"]
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.wayback.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=500),
    )

    result = next(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.wayback.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    results = list(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert results[0].url.startswith("https://web.archive.org/")
    assert results[0].metadata["timestamp"] == "20200101000000"


def test_wayback_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    artifact.failed_pdf_urls = ["https://example.org/pdf"]
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.wayback.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.wayback.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(WaybackResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


def test_zenodo_resolver_no_doi(tmp_path) -> None:
    resolver = ZenodoResolver()
    artifact = _artifact(tmp_path, doi=None)
    config = ResolverConfig()

    result = next(resolver.iter_urls(Mock(), config, artifact))

    assert result.event_reason == "no-doi"


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
        "DocsToKG.ContentDownload.resolvers.providers.zenodo.request_with_retries",
        Mock(side_effect=exception),
    )

    result = next(ZenodoResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == reason


def test_zenodo_resolver_http_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.zenodo.request_with_retries",
        lambda *args, **kwargs: _StubResponse(status_code=502),
    )

    result = next(ZenodoResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "http-error"


def test_zenodo_resolver_json_error(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = ResolverConfig()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.resolvers.providers.zenodo.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=ValueError("bad"), text="oops"),
    )

    result = next(ZenodoResolver().iter_urls(Mock(), config, artifact))

    assert result.event_reason == "json-error"


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
        "DocsToKG.ContentDownload.resolvers.providers.zenodo.request_with_retries",
        lambda *args, **kwargs: _StubResponse(json_data=payload),
    )

    urls = [result.url for result in ZenodoResolver().iter_urls(Mock(), config, artifact)]

    assert urls == ["https://zenodo.example/paper.pdf"]
